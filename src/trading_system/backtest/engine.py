"""The event loop: one chronological stream, and a store that cannot look forward.

Three things live here, and the split matters.

:class:`DataHandler` merges every ``(symbol, timeframe)`` stream into one
chronological sequence **keyed on bar close**, and hands it out one instant at a
time. It never yields a bar twice and never yields one out of order.

:class:`BarStore` is the only way anything downstream reads a price. It holds a
cursor per stream and **raises** on a read past it. That is what makes lookahead
a stack trace rather than a convention: a strategy cannot reach the next bar,
because the object it is given does not have it yet. The P06 ``BarContext``
already guarantees this within a series; the store extends the same guarantee
across streams, where it was previously nobody's job.

:class:`BacktestEngine` runs the phases in the fixed order below and delegates
each to a :class:`Phases` implementation. The order is written out in one
readable place on purpose — it decides the equity curve and it decides whether
the Risk Engine sees a limit freed by an exit, and both are the kind of thing
that gets silently reordered by a well-meaning refactor.

**The order within one instant ``T``.** Everything is at one wall-clock instant;
``T`` is simultaneously ``close(t)`` for the bars that just closed and
``open(t+1)`` for the bars that just opened.

0. **Publish** every bar whose close is ``T``, finest timeframe first. Nothing
   is evaluated. Because publication and evaluation are separate phases, an H1
   bar closing at 21:00 and the D1 bar closing at the same 21:00 are both
   visible to each other's evaluation, which is correct: the D1 bar is an
   aggregate of bars ``<= T`` and carries no information the H1 evaluation is
   not already entitled to. It is also what a live run does at 21:00:00.
1. **Financing** over ``(T_prev, T]``. A fact about time already elapsed, so it
   is charged before anything can close and escape it.
2. **Fills of orders deferred from ``T_prev``**, at ``open(t)`` — **exits first,
   then entries**. An exit frees portfolio heat and the per-instrument, cluster,
   strategy and direction headroom, and phase 7 has to see that.
3. **Entry recognition** on ``close(t)``. Pure: it reads the bar and produces
   signals, touching no money. Ahead of the exits because
   ``SignalReverseExit`` needs to know on *this* bar that an opposing signal
   was recognised; deferring that by a bar would make a reversal exit fire one
   bar late for no reason other than phase ordering.
4. **Exit rules** on bar ``t`` — resting levels touched inside it — then
   **stop modifiers**, in that order, inside one ``ExitPlan.on_bar`` call. A
   position opened at ``open(t)`` in phase 2 is included: it really was exposed
   to bar ``t``'s range.
5. **Mark to market** at ``close(t)``, and recompute what each open position has
   at stake from its current stop.
6. **One equity row** for the instant, written after every fill and mark it
   contains.
7. **Risk sizing** of the signals from phase 3, against the snapshot from phase
   6. Each approval is fed straight back through ``AccountState.with_opened``,
   so a second signal at the same instant sees the headroom the first consumed
   — risk headroom, free margin and exposure alike.
8. **Queue** approved orders for ``open(t+1)``, which is phase 2 of the next
   instant for that stream.

One artefact is accepted rather than hidden: a signal sized at ``T_prev`` was
measured against an account that still held a position which stops out inside
bar ``t``, and its fill (phase 2) precedes that stop-out (phase 4) even though
the stop may have been hit seconds into the bar. The direction is conservative —
the new position was sized while the limit was still occupied — and the true
order inside a bar is exactly what OHLC does not record.
"""

import heapq
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol

from trading_system.backtest.clock import StreamKey, bar_close_ts
from trading_system.core.exceptions import TradingSystemError
from trading_system.core.types import Bar, Price, Volume
from trading_system.data.resample import DayOrigin
from trading_system.entry.context import PRICE_FIELDS, BarContext, BarSeries


class LookaheadError(TradingSystemError):
    """A bar was read before the merged clock reached its close.

    Raised, never softened. Every alternative — returning ``None``, clamping to
    the cursor, logging — turns a defect that invalidates a whole run into one
    that silently changes a few numbers.
    """


@dataclass(frozen=True)
class BarEvent:
    """One bar becoming visible.

    Attributes:
        key: Which stream it belongs to.
        index: Its position in that stream.
        close_ts: When it closed, which is when this event happens.
    """

    key: StreamKey
    index: int
    close_ts: datetime


@dataclass(frozen=True)
class Instant:
    """Every bar that closed at one wall-clock instant.

    Attributes:
        ts: The instant, tz-aware UTC.
        bars: The bars, finest timeframe first and then by symbol. The order is
            for reproducibility only — correctness does not rest on it, because
            all of them are published before any of them is evaluated.
    """

    ts: datetime
    bars: tuple[BarEvent, ...]


class DataHandler:
    """Every stream of the run, merged into one forward-only chronology."""

    __slots__ = ("_closes", "_day_origin", "_streams")

    def __init__(self, streams: Mapping[StreamKey, BarSeries], day_origin: DayOrigin) -> None:
        """Bind the handler to its streams and precompute their close instants.

        Args:
            streams: One series per instrument and timeframe.
            day_origin: Where the trading day starts, for daily close instants.

        Raises:
            ValueError: If a stream's key disagrees with its series, or a
                series' timestamps are not strictly ascending.
        """
        for key, series in streams.items():
            if series.symbol != key.symbol or series.timeframe is not key.timeframe:
                raise ValueError(
                    f"stream keyed {key} holds {series.symbol}@{series.timeframe.value}"
                )
        self._streams = dict(streams)
        self._day_origin = day_origin
        self._closes = {
            key: self._close_times(key, series) for key, series in self._streams.items()
        }

    def __repr__(self) -> str:
        """Compact description naming the streams and their sizes."""
        sizes = ", ".join(f"{key}:{len(series)}" for key, series in sorted(self._streams.items()))
        return f"DataHandler({sizes})"

    @property
    def streams(self) -> Mapping[StreamKey, BarSeries]:
        """The series being walked, keyed by stream."""
        return dict(self._streams)

    def close_times(self, key: StreamKey) -> Sequence[datetime]:
        """Precomputed close instants of one stream, by bar index.

        Args:
            key: Stream to read.

        Returns:
            One instant per bar.
        """
        return self._closes[key]

    def _close_times(self, key: StreamKey, series: BarSeries) -> list[datetime]:
        """Close instants of one series, checking it runs strictly forward.

        ``BarContext.bar_close_ts`` adds the timeframe's nominal duration, which
        is right for every intraday size and an hour out for ``D1`` across a DST
        transition. The open time is recovered from it and re-closed through
        :mod:`trading_system.backtest.clock`, which is the authority.
        """
        closes: list[datetime] = []
        previous: datetime | None = None
        duration = key.timeframe.duration
        for index in range(len(series)):
            opening = series.context(index).bar_close_ts - duration
            if previous is not None and opening <= previous:
                raise ValueError(
                    f"{key}: bar {index} opens at {opening:%F %T%z}, not after the previous "
                    f"bar's {previous:%F %T%z}; a stream must run strictly forward"
                )
            previous = opening
            closes.append(bar_close_ts(key.timeframe, opening, self._day_origin))
        return closes

    def _stream_events(self, key: StreamKey) -> Iterator[tuple[datetime, int, str, int, StreamKey]]:
        """One stream's bars as merge-sortable tuples.

        Args:
            key: Stream to walk.

        Yields:
            ``(close, timeframe rank, symbol, index, key)`` per bar. The three
            middle fields order several bars closing at one instant: finest
            timeframe first, then by symbol. Nothing downstream depends on that
            order for correctness — publication precedes evaluation — only on it
            being the same on every run.
        """
        for index, close in enumerate(self._closes[key]):
            yield close, key.rank, key.symbol, index, key

    def instants(self) -> Iterator[Instant]:
        """Walk every bar of every stream, oldest close first, batched by instant.

        Yields:
            One :class:`Instant` per distinct close time, carrying every bar
            that closed then.
        """
        # Each stream's generator is built by a function taking the key as an
        # argument, not by a nested generator expression closing over the loop
        # variable. A genexp would evaluate `key` lazily, at consumption time,
        # by which point the enclosing loop has moved on — so every event in the
        # merge would carry the LAST stream's key while carrying the right close
        # time and index. The result reads as a plausible chronology of the
        # wrong instruments.
        merged = heapq.merge(*(self._stream_events(key) for key in self._closes))
        batch: list[BarEvent] = []
        current: datetime | None = None
        for close, _rank, _symbol, index, key in merged:
            if current is not None and close != current:
                yield Instant(ts=current, bars=tuple(batch))
                batch = []
            current = close
            batch.append(BarEvent(key=key, index=index, close_ts=close))
        if current is not None:
            yield Instant(ts=current, bars=tuple(batch))


class BarStore:
    """Published bars, and a refusal to hand out any others.

    The cursor per stream starts before the first bar and only ever advances.
    Anything asking for a bar past it gets a :class:`LookaheadError`, which is
    the point: the guarantee is structural rather than a rule somebody has to
    remember while writing a new rule or filter.
    """

    __slots__ = ("_cursors", "_streams")

    def __init__(self, streams: Mapping[StreamKey, BarSeries]) -> None:
        """Wrap the run's series with a per-stream cursor.

        Args:
            streams: One series per stream.
        """
        self._streams = dict(streams)
        self._cursors = dict.fromkeys(streams, -1)

    def __repr__(self) -> str:
        """Compact description naming each stream's cursor."""
        cursors = ", ".join(f"{key}:{index}" for key, index in sorted(self._cursors.items()))
        return f"BarStore({cursors})"

    @property
    def streams(self) -> Mapping[StreamKey, BarSeries]:
        """The series held, keyed by stream."""
        return dict(self._streams)

    def cursor(self, key: StreamKey) -> int:
        """Index of the newest published bar of a stream, ``-1`` before any.

        Args:
            key: Stream to read.

        Returns:
            The cursor.
        """
        return self._cursors[key]

    def publish(self, event: BarEvent) -> None:
        """Make one bar visible.

        Args:
            event: The bar becoming visible.

        Raises:
            ValueError: If it is not the next bar of its stream. Publishing out
                of order or twice would move the cursor to a number that no
                longer means "everything up to here has been seen".
        """
        expected = self._cursors[event.key] + 1
        if event.index != expected:
            raise ValueError(
                f"{event.key}: published bar {event.index} but the next one is {expected}"
            )
        self._cursors[event.key] = event.index

    def has_published(self, key: StreamKey) -> bool:
        """Whether any bar of a stream has been published yet.

        Args:
            key: Stream to test.

        Returns:
            ``True`` once its first bar has been seen.
        """
        return self._cursors[key] >= 0

    def context(self, key: StreamKey, index: int | None = None) -> BarContext:
        """A no-lookahead view of one published bar.

        Args:
            key: Stream to read.
            index: Bar to view; the newest published one by default.

        Returns:
            The P06 context for that bar, which counts backwards only.

        Raises:
            LookaheadError: If the bar has not been published.
        """
        cursor = self._cursors[key]
        wanted = cursor if index is None else index
        if wanted > cursor:
            raise LookaheadError(
                f"{key}: bar {wanted} has not closed yet; the stream is at {cursor}"
            )
        if wanted < 0:
            raise LookaheadError(f"{key}: no bar has closed yet")
        return self._streams[key].context(wanted)

    def bar(self, key: StreamKey, index: int | None = None) -> Bar:
        """One published bar as an OHLCV record, for the execution models.

        Args:
            key: Stream to read.
            index: Bar to read; the newest published one by default.

        Returns:
            The bar, timestamped at its OPEN as
            :class:`~trading_system.core.types.Bar` requires.

        Raises:
            LookaheadError: If the bar has not been published.
            ValueError: If a price column is missing.
        """
        ctx = self.context(key, index)
        prices: dict[str, float] = {}
        for field in PRICE_FIELDS:
            value = ctx.price(field)
            if value is None:
                raise ValueError(f"{key}: bar {ctx.index} has no {field}")
            prices[field] = value
        return Bar(
            timestamp=ctx.bar_close_ts - key.timeframe.duration,
            open=Price(prices["open"]),
            high=Price(prices["high"]),
            low=Price(prices["low"]),
            close=Price(prices["close"]),
            volume=Volume(prices["volume"]),
        )

    def last_close(self, key: StreamKey) -> float | None:
        """Close of the newest published bar, for marking to market.

        Args:
            key: Stream to read.

        Returns:
            The close, or ``None`` if the stream has published nothing. Never a
            value forward-filled from a bar that has not closed — such a bar is
            not in the store to be read.
        """
        if not self.has_published(key):
            return None
        return self.context(key).price("close")


class Phases(Protocol):
    """What the engine calls, in the order it calls it.

    A protocol rather than a base class so the loop can be exercised against a
    recording double that asserts on the order alone.
    """

    def on_financing(self, ts: datetime) -> None:
        """Charge financing for the interval ending at ``ts``."""
        ...

    def on_deferred_exits(self, event: BarEvent) -> None:
        """Fill exits decided on the previous bar's close, at this bar's open."""
        ...

    def on_pending_entries(self, event: BarEvent) -> None:
        """Fill or expire entry orders resting from earlier bars."""
        ...

    def on_recognise(self, event: BarEvent) -> None:
        """Recognise entry signals on this bar's close. No money moves."""
        ...

    def on_exits(self, event: BarEvent) -> None:
        """Run exit rules then stop modifiers over this bar."""
        ...

    def on_mark(self, ts: datetime) -> None:
        """Mark open positions to the latest closes and restate their risk."""
        ...

    def on_snapshot(self, ts: datetime) -> None:
        """Write one equity row for this instant."""
        ...

    def on_sizing(self, event: BarEvent) -> None:
        """Size the signals recognised on this bar and queue what is approved."""
        ...


class BacktestEngine:
    """Drives the phases over the merged stream, in the fixed order."""

    __slots__ = ("_data", "_phases", "_store")

    def __init__(self, data: DataHandler, store: BarStore, phases: Phases) -> None:
        """Wire the loop.

        Args:
            data: The merged chronology.
            store: The published-bar view the phases read through.
            phases: What to do at each phase.
        """
        self._data = data
        self._store = store
        self._phases = phases

    def __repr__(self) -> str:
        """Compact description naming the data handler."""
        return f"BacktestEngine({self._data!r})"

    @property
    def store(self) -> BarStore:
        """The published-bar view this loop advances."""
        return self._store

    def run(self) -> int:
        """Walk every instant of the run.

        Returns:
            How many instants were processed.
        """
        instants = 0
        for instant in self._data.instants():
            instants += 1
            # 0 — publication precedes every evaluation, which is what makes the
            # order of bars inside one instant a reproducibility concern rather
            # than a correctness one.
            for event in instant.bars:
                self._store.publish(event)

            self._phases.on_financing(instant.ts)

            for event in instant.bars:
                self._phases.on_deferred_exits(event)
            for event in instant.bars:
                self._phases.on_pending_entries(event)
            for event in instant.bars:
                self._phases.on_recognise(event)
            for event in instant.bars:
                self._phases.on_exits(event)

            self._phases.on_mark(instant.ts)
            self._phases.on_snapshot(instant.ts)

            for event in instant.bars:
                self._phases.on_sizing(event)
        return instants
