"""Indicator contracts: one definition, two evaluation modes.

Every indicator here exists twice over. The **vectorised** path runs a polars
expression across a whole frame and feeds backtests and research. The
**incremental** path is a state machine that consumes one closed bar at a time
and feeds live trading. They are required to agree to within ``1e-9``: if they
do not, the backtest is measuring something the live system will never produce,
and every number downstream of it is fiction. :meth:`BaseIndicator.verify_parity`
turns that requirement into an executable check, and the test suite applies it
to every indicator in the registry.

Conventions that hold for all indicators:

* **Missing means missing.** Values that are not yet computable are polars
  *nulls*, never NaN and never forward-filled. ``compute_frame`` normalises any
  NaN produced by arithmetic into a null, so ``is_null()`` is the single test a
  caller needs.
* **Warmup is uniform across outputs.** An indicator with several channels
  publishes all of them on the same bar. MACD's line is arithmetically available
  before its signal, but exposing it early would let a strategy trade a row that
  the live state machine would have reported as "not ready".
* **No lookahead.** The value at bar ``t`` is a function of bars ``≤ t`` only.
  Appending bars ``t+1..t+n`` must not change it — the property the incremental
  path enforces structurally and the vectorised path is tested for.
"""

import math
from abc import ABC, abstractmethod
from collections.abc import Iterator
from datetime import UTC, datetime
from typing import Any, ClassVar, Protocol, runtime_checkable

import polars as pl

from trading_system.core.exceptions import ValidationError
from trading_system.core.types import Bar, Price, Volume
from trading_system.data.models import OHLCV_COLUMNS, TIMESTAMP_COLUMN, OHLCVFrame

#: What one bar of an indicator yields: a bare float, or a NamedTuple of floats
#: for multi-channel indicators such as MACD.
type IndicatorValue = float | tuple[float, ...]

#: Stand-in passed to state machines that declared no interest in bar times.
#: Materialising a million ``datetime`` objects costs ~0.4 s; skipping it when
#: nothing reads them keeps the recursive vector path cheap.
_NO_TIMESTAMP = datetime(1970, 1, 1, tzinfo=UTC)

DEFAULT_ATOL = 1e-9


@runtime_checkable
class Indicator(Protocol):
    """A vectorised, single-valued indicator.

    This is the shape most of the library takes. Multi-channel indicators
    implement :class:`MultiOutputIndicator` instead.
    """

    @property
    def name(self) -> str:
        """Stable identifier encoding the indicator kind and its parameters."""

    @property
    def warmup(self) -> int:
        """Number of leading bars that cannot carry a value."""

    def compute(self, frame: OHLCVFrame) -> pl.Series:
        """Evaluate across the whole frame, one value per bar."""


@runtime_checkable
class MultiOutputIndicator(Protocol):
    """A vectorised indicator emitting several aligned channels."""

    @property
    def name(self) -> str:
        """Stable identifier encoding the indicator kind and its parameters."""

    @property
    def warmup(self) -> int:
        """Number of leading bars that cannot carry a value."""

    @property
    def outputs(self) -> tuple[str, ...]:
        """Channel names, in column order."""

    def compute_frame(self, frame: OHLCVFrame) -> pl.DataFrame:
        """Evaluate across the whole frame, one row per bar."""


@runtime_checkable
class StreamingIndicator(Protocol):
    """An incremental, single-valued indicator.

    ``update`` must be O(1) in the number of bars seen so far — a live process
    runs for months and cannot re-scan its history on every tick.
    """

    @property
    def warmup(self) -> int:
        """Number of leading bars that return ``None``."""

    def update(self, bar: Bar) -> float | None:
        """Consume the next *closed* bar, returning its value or ``None``."""

    def reset(self) -> None:
        """Discard all state, returning to the pre-warmup condition."""


class BaseStreaming[IndicatorT: BaseIndicator[Any], ValueT: IndicatorValue](ABC):
    """State machine shared by every incremental indicator.

    Parameters live on the owning indicator rather than being copied here, so a
    warmup formula or a period is defined exactly once. Subclasses put all their
    mutable state in :meth:`reset`, which the constructor calls.
    """

    def __init__(self, indicator: IndicatorT) -> None:
        """Bind the state machine to the indicator holding its parameters.

        Args:
            indicator: Indicator whose parameters and warmup this instance uses.
        """
        self._indicator = indicator
        self.reset()

    @property
    def indicator(self) -> IndicatorT:
        """The indicator this state machine evaluates."""
        return self._indicator

    @property
    def warmup(self) -> int:
        """Number of leading bars that return ``None``."""
        return int(self._indicator.warmup)

    @property
    def outputs(self) -> tuple[str, ...]:
        """Channel names, in the order :meth:`step` returns them."""
        return tuple(self._indicator.outputs)

    def update(self, bar: Bar) -> ValueT | None:
        """Consume the next closed bar.

        Args:
            bar: The bar that has just closed.

        Returns:
            The indicator value, or ``None`` while still warming up.
        """
        return self.step(bar.timestamp, bar.open, bar.high, bar.low, bar.close, bar.volume)

    @abstractmethod
    def step(
        self,
        timestamp: datetime,
        open_price: float,
        high: float,
        low: float,
        close: float,
        volume: float,
        /,
    ) -> ValueT | None:
        """Advance the state machine by one bar, given its raw fields.

        The scalar signature exists so the vectorised path of a recursive
        indicator can drive the same code without allocating a :class:`Bar` per
        row, which would dominate the runtime on million-bar frames. Arguments
        are positional-only so implementations may underscore-prefix the fields
        they ignore.

        Args:
            timestamp: Bar OPEN time, UTC. Meaningful only when the indicator
                sets ``requires_timestamps``.
            open_price: Bar open.
            high: Bar high.
            low: Bar low.
            close: Bar close.
            volume: Bar volume.

        Returns:
            The indicator value, or ``None`` while still warming up.
        """

    @abstractmethod
    def reset(self) -> None:
        """(Re-)initialise every piece of mutable state."""


class BaseIndicator[ValueT: IndicatorValue](ABC):
    """Shared machinery: identity, output shape, warmup masking, parity check."""

    #: Channel names, in column order. Single-valued indicators keep the
    #: inherited ``("value",)`` and are published under their own name.
    outputs: ClassVar[tuple[str, ...]] = ("value",)

    #: Whether :meth:`BaseStreaming.step` reads its ``timestamp`` argument.
    requires_timestamps: ClassVar[bool] = False

    @property
    @abstractmethod
    def name(self) -> str:
        """Stable identifier encoding the indicator kind and its parameters."""

    @property
    @abstractmethod
    def warmup(self) -> int:
        """Number of leading bars that cannot carry a value.

        Equivalently: the index of the first bar at which the indicator becomes
        valid. ``SMA(20).warmup == 19``.
        """

    @abstractmethod
    def streaming(self) -> BaseStreaming[Any, ValueT]:
        """Build a fresh state machine for the incremental path."""

    @abstractmethod
    def _evaluate(self, frame: OHLCVFrame, /) -> pl.DataFrame:
        """Compute the raw channels, before warmup masking.

        Args:
            frame: Bars to evaluate over.

        Returns:
            One column per entry of :attr:`outputs`, same height as ``frame``.
        """

    def compute_frame(self, frame: OHLCVFrame) -> pl.DataFrame:
        """Evaluate the indicator over a whole frame.

        Rows before :attr:`warmup` are nulled out in every channel, and NaN is
        normalised to null, so the result carries exactly one representation of
        "no value here".

        Args:
            frame: Bars to evaluate over.

        Returns:
            A DataFrame with one column per channel and one row per bar.

        Raises:
            ValidationError: If the implementation returned the wrong columns or
                the wrong number of rows.
        """
        raw = self._evaluate(frame)
        if tuple(raw.columns) != self.outputs:
            raise ValidationError(
                f"{self.name}: _evaluate returned columns {raw.columns}, "
                f"expected {list(self.outputs)}"
            )
        if raw.height != len(frame):
            raise ValidationError(
                f"{self.name}: _evaluate returned {raw.height} rows for {len(frame)} bars"
            )
        valid = pl.int_range(0, pl.len(), dtype=pl.Int64) >= self.warmup
        return raw.select(
            pl.when(valid)
            .then(pl.col(channel).cast(pl.Float64).fill_nan(None))
            .otherwise(pl.lit(None, dtype=pl.Float64))
            .alias(channel)
            for channel in self.outputs
        )

    @property
    def parity_rtol(self) -> float:
        """Relative slack this indicator's own arithmetic requires in :meth:`verify_parity`.

        **Zero, and it stays zero unless an indicator can say why it cannot be.**
        The parity contract's promise — both paths run the same formula in IEEE
        doubles, so genuine agreement lands near machine epsilon — holds for
        every expression whose intermediate values stay the same size as its
        inputs. It does **not** hold for an expression with a cancellation
        stage: a subtraction of two nearly equal numbers throws away the leading
        digits they share, and everything computed from the remainder inherits a
        relative error scaled up by however many digits went. Two paths that
        round one intermediate differently — by a single bit — then disagree in
        the output by that bit times the amplification.

        An indicator that overrides this owes the reader the amplification and
        where it comes from, not merely a number. Overriding it to silence a
        failure whose cause has not been traced to cancellation is how a real
        formula divergence gets buried; the default of zero is what makes such
        an override a visible, argued exception rather than an inherited one.
        """
        return 0.0

    def verify_parity(self, frame: OHLCVFrame, *, atol: float = DEFAULT_ATOL) -> None:
        """Assert the vectorised and incremental paths agree on ``frame``.

        Args:
            frame: Bars to evaluate both paths over.
            atol: Absolute tolerance. For an indicator whose arithmetic never
                cancels, this is the whole test: both paths run the same
                formula in IEEE doubles, so genuine agreement lands near
                machine epsilon and anything approaching this bound is a
                formula difference rather than rounding. An indicator that
                cancels by construction relaxes the comparison through
                :attr:`parity_rtol`, which it must declare and justify itself.

        Raises:
            ValidationError: On the first bar and channel where the two paths
                disagree, including the case where one is null and the other is
                not.
        """
        vector = self.compute_frame(frame)
        stream = run_streaming(self, frame)
        rtol = self.parity_rtol
        for channel in self.outputs:
            expected = vector[channel].to_list()
            actual = stream[channel].to_list()
            for index, (left, right) in enumerate(zip(expected, actual, strict=True)):
                if left is None and right is None:
                    continue
                if left is None or right is None:
                    raise ValidationError(
                        f"{self.name}.{channel}[{index}]: vector={left!r} stream={right!r}; "
                        "one path reports a value the other calls unavailable"
                    )
                if not math.isclose(left, right, rel_tol=rtol, abs_tol=atol):
                    bound = (
                        f"atol={atol:.1e}" if rtol == 0.0 else f"atol={atol:.1e}, rtol={rtol:.1e}"
                    )
                    raise ValidationError(
                        f"{self.name}.{channel}[{index}]: vector={left!r} stream={right!r}, "
                        f"|diff|={abs(left - right):.3e} exceeds {bound}"
                    )


class ScalarIndicator(BaseIndicator[float]):
    """An indicator producing exactly one value per bar.

    Subclasses supply a single polars expression over the OHLCV columns. An
    indicator whose value depends on more than an expression can — Supertrend's
    ratcheting bands — overrides :meth:`_evaluate` instead, or derives from
    :class:`RecursiveIndicator`.
    """

    outputs: ClassVar[tuple[str, ...]] = ("value",)

    @abstractmethod
    def _expression(self) -> pl.Expr:
        """Return the expression computing this indicator over the OHLCV columns."""

    def _evaluate(self, frame: OHLCVFrame, /) -> pl.DataFrame:
        """Select the single expression into the canonical ``value`` column."""
        return _select(frame, [self._expression().alias("value")])

    def compute(self, frame: OHLCVFrame) -> pl.Series:
        """Evaluate across the whole frame.

        Args:
            frame: Bars to evaluate over.

        Returns:
            One value per bar, named after the indicator, null through warmup.
        """
        return self.compute_frame(frame)["value"].rename(self.name)


class MultiExpressionIndicator[ValueT: IndicatorValue](BaseIndicator[ValueT]):
    """A multi-channel indicator whose channels are all polars expressions."""

    @abstractmethod
    def _expressions(self) -> dict[str, pl.Expr]:
        """Return one expression per channel, keyed by channel name."""

    def _evaluate(self, frame: OHLCVFrame, /) -> pl.DataFrame:
        """Select every channel expression, in :attr:`outputs` order."""
        expressions = self._expressions()
        return _select(frame, [expressions[channel].alias(channel) for channel in self.outputs])


class RecursiveIndicator[ValueT: IndicatorValue](BaseIndicator[ValueT]):
    """An indicator whose recursion has no polars window equivalent.

    Supertrend's bands, and any rule that flips state based on its own previous
    output, cannot be written as a rolling expression. Rather than maintain a
    second implementation that would drift from the first, the vectorised path
    *is* the state machine, driven over raw columns. Parity is therefore exact
    by construction, and the cost is roughly 0.3 s per million bars.
    """

    def _evaluate(self, frame: OHLCVFrame, /) -> pl.DataFrame:
        """Run the state machine over the frame's columns, row by row."""
        height = len(frame)
        state = self.streaming()
        opens, highs, lows, closes, volumes = (
            frame.df[column].to_list() for column in OHLCV_COLUMNS
        )
        timestamps = (
            frame.df[TIMESTAMP_COLUMN].to_list()
            if self.requires_timestamps
            else [_NO_TIMESTAMP] * height
        )

        buffers: list[list[float | None]] = [[] for _ in self.outputs]
        for index in range(height):
            value = state.step(
                timestamps[index],
                opens[index],
                highs[index],
                lows[index],
                closes[index],
                volumes[index],
            )
            row = (None,) * len(self.outputs) if value is None else as_row(value)
            for buffer, item in zip(buffers, row, strict=True):
                buffer.append(item)

        return pl.DataFrame(
            {
                channel: pl.Series(channel, buffer, dtype=pl.Float64)
                for channel, buffer in zip(self.outputs, buffers, strict=True)
            }
        )


def _select(frame: OHLCVFrame, expressions: list[pl.Expr]) -> pl.DataFrame:
    """Evaluate expressions over a frame's columns, through the lazy engine.

    Indicator expressions routinely name the same sub-expression several times —
    MACD's line feeds both its signal and its histogram, RSI's smoothed loss
    appears in three branches. Eager ``select`` evaluates each reference
    independently; the lazy optimiser eliminates the duplicates, which is worth
    roughly a factor of two on the worst offenders and costs nothing on the rest.

    Args:
        frame: Bars supplying the columns.
        expressions: Aliased expressions to evaluate.

    Returns:
        One column per expression.
    """
    return frame.df.lazy().select(expressions).collect()


def as_row(value: float | tuple[float, ...]) -> tuple[float, ...]:
    """Normalise a streaming value into a tuple of channel values.

    Args:
        value: A bare float, or a NamedTuple of floats for a multi-channel
            indicator. A NamedTuple is returned as-is, being already a tuple.

    Returns:
        The value as a positional tuple.
    """
    if isinstance(value, tuple):
        return value
    return (value,)


def iter_bars(frame: OHLCVFrame) -> Iterator[Bar]:
    """Yield the frame's rows as :class:`Bar` objects, oldest first.

    Args:
        frame: Bars to iterate.

    Yields:
        One validated bar per row.
    """
    for timestamp, open_price, high, low, close, volume in frame.df.iter_rows():
        yield Bar(
            timestamp=timestamp,
            open=Price(open_price),
            high=Price(high),
            low=Price(low),
            close=Price(close),
            volume=Volume(volume),
        )


def run_streaming(indicator: BaseIndicator[Any], frame: OHLCVFrame) -> pl.DataFrame:
    """Drive an indicator's state machine over a frame, bar by bar.

    Goes through the public :meth:`BaseStreaming.update` API — including
    :class:`Bar` construction — so what is compared against the vectorised path
    is exactly what a live loop would call.

    Args:
        indicator: Indicator to run incrementally.
        frame: Bars to feed, in order.

    Returns:
        A DataFrame shaped like :meth:`BaseIndicator.compute_frame`'s result.
    """
    state = indicator.streaming()
    buffers: list[list[float | None]] = [[] for _ in indicator.outputs]
    for bar in iter_bars(frame):
        value = state.update(bar)
        row = (None,) * len(indicator.outputs) if value is None else as_row(value)
        for buffer, item in zip(buffers, row, strict=True):
            buffer.append(item)
    return pl.DataFrame(
        {
            channel: pl.Series(channel, buffer, dtype=pl.Float64)
            for channel, buffer in zip(indicator.outputs, buffers, strict=True)
        }
    )
