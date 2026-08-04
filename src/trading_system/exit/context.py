"""What an exit rule is allowed to see on bar ``t``.

:class:`ExitContext` wraps P06's :class:`~trading_system.entry.context.BarContext`
rather than being it. The reason is the one stated in P07's requirements: the
rule interface must accept ``NewsExit``, ``VolumeExhaustionExit`` and
``TargetLevelExit`` later *without rework*, and it must accept intrabar ticks
now, for :attr:`~trading_system.exit.base.IntrabarPolicy.TICK_BASED`. None of
those belong in a bar context — ticks are not bars and news is not price — so a
bare ``BarContext`` would have forced a change to the ``ExitRule`` signature the
first time one of them arrived, which is exactly the rework the interface exists
to prevent. The wrapper grows a field; the protocol does not move.

The no-lookahead guarantee is inherited, not re-argued. Every price, feature and
label accessor delegates straight to the wrapped context, so the property proved
in P06 — a context at bar ``t`` over the full series answers exactly as one at
``t`` over the series truncated at ``t`` — carries over, and the same equivalence
test is run against this class too. The one genuinely new surface is
:attr:`ExitContext.ticks`, which is held to the same standard by construction:
a context carries the ticks of *its own bar only*, and a tick timestamped
outside ``[open(t), close(t))`` is rejected rather than stored. A tick from bar
``t+1`` is lookahead of the most direct possible kind.

:attr:`ExitContext.reverse_signal_side` is the same pattern applied to
``SignalReverseExit``: whether an opposing entry signal was recognised on this
bar's close. It is a plain ``Side | None`` — a fact about the bar, not a
reference to an :class:`~trading_system.entry.signal.EntrySignal` or anything
else that would pull the Entry compiler into this layer. Something that already
holds both engines — the backtest orchestrator — is what fills it in, the same
way a market data provider fills in ticks; Exit only has to accept the value,
not know where it came from.
"""

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime

from trading_system.core.types import Side, Timeframe, ensure_utc
from trading_system.entry.context import BarContext, BarSeries


@dataclass(frozen=True)
class Tick:
    """One traded price inside a bar.

    The minimum a tick has to be for the only thing this layer asks of ticks:
    which of two levels price reached first. Bid/ask, size and venue are
    execution-model concerns and are not needed to order two events.

    Attributes:
        ts: When it traded, tz-aware UTC.
        price: Traded price.
    """

    ts: datetime
    price: float


class ExitContext:
    """Bar ``t`` as an exit rule sees it: the bar context, plus its ticks.

    Every accessor is a delegation to the wrapped
    :class:`~trading_system.entry.context.BarContext`, with the same backwards-only
    ``lookback`` counting and the same treatment of a lookback that runs off the
    start of the series (``None``, meaning "no such bar").
    """

    __slots__ = ("_bars", "_reverse_signal_side", "_ticks")

    def __init__(
        self,
        bars: BarContext,
        ticks: Sequence[Tick] = (),
        reverse_signal_side: Side | None = None,
    ) -> None:
        """Wrap a bar context, optionally with the ticks of that same bar.

        Args:
            bars: The P06 context for bar ``t``.
            ticks: Ticks that traded inside bar ``t``, ascending by time. Empty
                means the run has no tick data, which is the normal case.
            reverse_signal_side: Direction of an opposing entry signal
                recognised on this bar's close, if any. ``None`` is the default
                for almost every bar.

        Raises:
            ValueError: If a tick is naive, falls outside ``[open(t), close(t))``,
                or the ticks are not ascending. Each of those is a way for a
                future price to reach a rule, so none of them is tolerated.
        """
        self._bars = bars
        self._reverse_signal_side = reverse_signal_side
        close_ts = bars.bar_close_ts
        open_ts = close_ts - bars.timeframe.duration

        normalised: list[Tick] = []
        previous: datetime | None = None
        for tick in ticks:
            ts = ensure_utc(tick.ts)
            if not open_ts <= ts < close_ts:
                raise ValueError(
                    f"tick at {ts:%F %T%z} is outside bar [{open_ts:%F %T%z}, "
                    f"{close_ts:%F %T%z}); a context carries the ticks of its own bar only"
                )
            if previous is not None and ts < previous:
                raise ValueError(
                    f"ticks must be ascending; {ts:%F %T%z} follows {previous:%F %T%z}"
                )
            previous = ts
            normalised.append(Tick(ts=ts, price=tick.price))
        self._ticks = tuple(normalised)

    def __repr__(self) -> str:
        """Compact description naming the bar and how many ticks it carries."""
        return f"ExitContext({self._bars!r}, ticks={len(self._ticks)})"

    @property
    def bars(self) -> BarContext:
        """The wrapped bar context.

        Exposed rather than hidden: it is the same history-only surface, so a
        rule reaching for it gains no reach it did not already have, and a rule
        needing something not delegated here is not stuck.
        """
        return self._bars

    @property
    def index(self) -> int:
        """Position of bar ``t`` in the series."""
        return self._bars.index

    @property
    def symbol(self) -> str:
        """Instrument the bars belong to."""
        return self._bars.symbol

    @property
    def timeframe(self) -> Timeframe:
        """Bar size."""
        return self._bars.timeframe

    @property
    def bar_open_ts(self) -> datetime:
        """Open time of bar ``t``, UTC."""
        return self._bars.bar_close_ts - self._bars.timeframe.duration

    @property
    def bar_close_ts(self) -> datetime:
        """Close time of bar ``t``, UTC.

        The instant a decision taken on this bar's close may first be filled.
        """
        return self._bars.bar_close_ts

    @property
    def has_ticks(self) -> bool:
        """Whether this bar carries tick detail."""
        return bool(self._ticks)

    @property
    def ticks(self) -> tuple[Tick, ...]:
        """Ticks of bar ``t``, ascending. Empty when the run has no tick data."""
        return self._ticks

    @property
    def reverse_signal_side(self) -> Side | None:
        """Direction of an opposing entry signal recognised on this bar's close.

        ``None`` on almost every bar. Filled in by whatever composed this
        context — normally a backtest orchestrator that runs both engines —
        never derived here.
        """
        return self._reverse_signal_side

    def price(self, field: str, lookback: int = 0) -> float | None:
        """A price of the bar ``lookback`` bars before ``t``.

        Args:
            field: One of :data:`~trading_system.entry.context.PRICE_FIELDS`.
            lookback: Bars back from ``t``. Zero is the current bar.

        Returns:
            The price, or ``None`` if that bar precedes the series.
        """
        return self._bars.price(field, lookback)

    def feature(self, key: str, lookback: int = 0) -> float | None:
        """A feature value of the bar ``lookback`` bars before ``t``.

        Args:
            key: Feature column name.
            lookback: Bars back from ``t``. Zero is the current bar.

        Returns:
            The value, or ``None`` if that bar precedes the series or the
            indicator had not warmed up on it.
        """
        return self._bars.feature(key, lookback)

    def labels(self, category: str, lookback: int = 0) -> frozenset[str] | None:
        """The categorical labels of the bar ``lookback`` bars before ``t``.

        Args:
            category: A :class:`~trading_system.entry.labels.LabelCategory` value.
            lookback: Bars back from ``t``. Zero is the current bar.

        Returns:
            The label set, or ``None`` if that bar precedes the series or cannot
            be classified.
        """
        return self._bars.labels(category, lookback)

    def feature_snapshot(self) -> dict[str, float | None]:
        """Every feature's value on bar ``t``, for the exit's audit trail.

        Returns:
            Feature key to value, reading bar ``t`` only.
        """
        return self._bars.feature_snapshot()


def exit_contexts(
    series: BarSeries,
    ticks: Mapping[int, Sequence[Tick]] | None = None,
    reverse_signals: Mapping[int, Side] | None = None,
) -> Iterator[ExitContext]:
    """Yield one exit context per bar of a series, oldest first.

    Args:
        series: Bars, features and labels to walk.
        ticks: Ticks per bar index. Bars absent from the mapping carry none,
            which is what a run without tick data looks like.
        reverse_signals: Opposing entry signal side per bar index. Bars absent
            from the mapping carry ``None``, which is almost every bar.

    Yields:
        A context per bar, in the order a live loop would see them.

    Raises:
        ValueError: If any tick falls outside the bar it is filed under.
    """
    per_bar_ticks = ticks or {}
    per_bar_reversal = reverse_signals or {}
    for bar in series.contexts():
        yield ExitContext(bar, per_bar_ticks.get(bar.index, ()), per_bar_reversal.get(bar.index))
