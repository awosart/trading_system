"""Momentum indicators: RSI, Stochastic, CCI, Williams %R, ROC, MFI.

Every oscillator here has a degenerate case — a window with no losses, no range,
or no volume flow — where the textbook ratio divides by zero. Each is resolved
explicitly and identically in both evaluation paths rather than left to produce
an infinity, because an infinity propagated into a feature matrix is discovered
much later and much more expensively than a documented convention.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import ClassVar, NamedTuple

import polars as pl

from trading_system.features.base import (
    BaseStreaming,
    MultiExpressionIndicator,
    ScalarIndicator,
)
from trading_system.features.expressions import (
    rolling_mad,
    safe_divide,
    source_expression,
    source_suffix,
    source_value,
    typical_price,
    validate_source,
    wilder_rma,
)
from trading_system.features.rolling import (
    RollingExtreme,
    RollingWindow,
    SeededEma,
    wilder_alpha,
)

#: What an oscillator reports when its window is perfectly flat: neither side
#: dominates, so the neutral midpoint is the only defensible answer.
_NEUTRAL_RSI = 50.0
_NEUTRAL_STOCHASTIC = 50.0
_NEUTRAL_WILLIAMS = -50.0

#: Relative size below which a derived spread is indistinguishable from zero.
#: Ten orders of magnitude above double-precision noise and ten below the
#: smallest spread any real quote produces, so it can only ever catch the former.
_FLAT_TOLERANCE = 1e-12


@dataclass(frozen=True)
class RSI(ScalarIndicator):
    """Relative strength index, Wilder-smoothed.

    A window with no losses conventionally reports 100. A window with neither
    gains nor losses — a perfectly flat stretch — reports 50 instead: calling a
    flat market maximally overbought is an artefact of the ``0/0``, not a
    reading.

    Attributes:
        period: Smoothing period.
        source: Price series to measure.
    """

    period: int = 14
    source: str = "close"

    def __post_init__(self) -> None:
        """Reject non-positive periods and unknown sources."""
        if self.period < 1:
            raise ValueError(f"period must be positive, got {self.period}")
        validate_source(self.source)

    @property
    def name(self) -> str:
        """Identifier, e.g. ``rsi_14``."""
        return f"rsi_{self.period}{source_suffix(self.source)}"

    @property
    def warmup(self) -> int:
        """Changes start on bar 1, so the Wilder seed completes on bar ``period``."""
        return self.period

    def _expression(self) -> pl.Expr:
        """Ratio of Wilder-smoothed gains to losses, mapped onto 0-100."""
        change = source_expression(self.source).diff()
        # The first bar has no change at all, which is not the same as a change
        # of zero: letting it through would seed the average one bar early, over
        # a window the incremental path never sees.
        gain = pl.when(change.is_null()).then(None).when(change > 0).then(change).otherwise(0.0)
        loss = pl.when(change.is_null()).then(None).when(change < 0).then(-change).otherwise(0.0)
        average_gain = wilder_rma(gain, self.period)
        average_loss = wilder_rma(loss, self.period)
        return (
            pl.when(average_loss == 0)
            .then(pl.when(average_gain == 0).then(_NEUTRAL_RSI).otherwise(100.0))
            .otherwise(100 - 100 / (1 + average_gain / average_loss))
        )

    def streaming(self) -> "RSIStream":
        """Build the incremental counterpart."""
        return RSIStream(self)


class RSIStream(BaseStreaming[RSI, float]):
    """Incremental :class:`RSI`."""

    def reset(self) -> None:
        """Clear both smoothed averages and the previous input."""
        period = self._indicator.period
        self._gains = SeededEma(period, wilder_alpha(period))
        self._losses = SeededEma(period, wilder_alpha(period))
        self._previous: float | None = None

    def step(
        self,
        _timestamp: datetime,
        open_price: float,
        high: float,
        low: float,
        close: float,
        volume: float,
        /,
    ) -> float | None:
        """Split one bar's change into gain and loss and smooth each."""
        value = source_value(self._indicator.source, open_price, high, low, close, volume)
        previous = self._previous
        self._previous = value
        if previous is None:
            return None
        change = value - previous
        average_gain = self._gains.push(change if change > 0 else 0.0)
        average_loss = self._losses.push(-change if change < 0 else 0.0)
        if average_gain is None or average_loss is None:
            return None
        if average_loss == 0:
            return _NEUTRAL_RSI if average_gain == 0 else 100.0
        return 100 - 100 / (1 + average_gain / average_loss)


class StochasticValue(NamedTuple):
    """One bar of Stochastic output.

    Attributes:
        k: Smoothed position of the close within the recent range.
        d: Moving average of ``k``.
    """

    k: float
    d: float


@dataclass(frozen=True)
class Stochastic(MultiExpressionIndicator[StochasticValue]):
    """Stochastic oscillator, %K and %D.

    A window whose high equals its low places the close nowhere in particular;
    that case reports the neutral 50.

    Attributes:
        k_period: Lookback for the high/low range.
        k_smooth: Smoothing applied to raw %K. ``1`` gives the fast stochastic.
        d_period: Smoothing applied to %K to produce %D.
    """

    outputs: ClassVar[tuple[str, ...]] = ("k", "d")

    k_period: int = 14
    k_smooth: int = 3
    d_period: int = 3

    def __post_init__(self) -> None:
        """Reject non-positive periods."""
        for label, period in (
            ("k_period", self.k_period),
            ("k_smooth", self.k_smooth),
            ("d_period", self.d_period),
        ):
            if period < 1:
                raise ValueError(f"{label} must be positive, got {period}")

    @property
    def name(self) -> str:
        """Identifier, e.g. ``stoch_14_3_3``."""
        return f"stoch_{self.k_period}_{self.k_smooth}_{self.d_period}"

    @property
    def warmup(self) -> int:
        """The range window plus both smoothing windows."""
        return (self.k_period - 1) + (self.k_smooth - 1) + (self.d_period - 1)

    def _expressions(self) -> dict[str, pl.Expr]:
        """Position within the range, smoothed once into %K and again into %D."""
        highest = pl.col("high").rolling_max(self.k_period)
        lowest = pl.col("low").rolling_min(self.k_period)
        raw = safe_divide(100 * (pl.col("close") - lowest), highest - lowest, _NEUTRAL_STOCHASTIC)
        percent_k = raw.rolling_mean(self.k_smooth)
        return {"k": percent_k, "d": percent_k.rolling_mean(self.d_period)}

    def streaming(self) -> "StochasticStream":
        """Build the incremental counterpart."""
        return StochasticStream(self)


class StochasticStream(BaseStreaming[Stochastic, StochasticValue]):
    """Incremental :class:`Stochastic`."""

    def reset(self) -> None:
        """Clear the range extremes and both smoothing windows."""
        indicator = self._indicator
        self._high = RollingExtreme(indicator.k_period, largest=True)
        self._low = RollingExtreme(indicator.k_period, largest=False)
        self._k_window = RollingWindow(indicator.k_smooth)
        self._d_window = RollingWindow(indicator.d_period)

    def step(
        self,
        _timestamp: datetime,
        _open_price: float,
        high: float,
        low: float,
        close: float,
        _volume: float,
        /,
    ) -> StochasticValue | None:
        """Locate the close in the range, then smooth twice."""
        highest = self._high.push(high)
        lowest = self._low.push(low)
        if highest is None or lowest is None:
            return None
        span = highest - lowest
        raw = _NEUTRAL_STOCHASTIC if span == 0 else 100 * (close - lowest) / span
        self._k_window.push(raw)
        if not self._k_window.full:
            return None
        percent_k = self._k_window.mean()
        self._d_window.push(percent_k)
        if not self._d_window.full:
            return None
        return StochasticValue(k=percent_k, d=self._d_window.mean())


@dataclass(frozen=True)
class CCI(ScalarIndicator):
    """Commodity channel index.

    Measures the typical price's distance from its own mean in units of mean
    absolute deviation. A window with no deviation — every bar identical — has no
    scale to measure against and reports 0.

    "No deviation" is judged relative to the price level rather than against
    exact zero. On a flat window the deviation is not zero but a few ULPs of
    rounding noise, and the numerator is the same noise, so the ratio is a
    perfectly finite number made entirely of floating-point error — and the two
    evaluation paths disagree on it, because they sum the window in different
    orders. Anything below :data:`_FLAT_TOLERANCE` of the price level is flat to
    the limit of double precision, and both paths call it flat.

    Attributes:
        period: Window length.
        constant: Lambert's scaling factor; ``0.015`` puts roughly 70-80 % of
            readings inside ±100.
    """

    period: int = 20
    constant: float = 0.015

    def __post_init__(self) -> None:
        """Reject non-positive periods and constants."""
        if self.period < 1:
            raise ValueError(f"period must be positive, got {self.period}")
        if self.constant <= 0:
            raise ValueError(f"constant must be positive, got {self.constant}")

    @property
    def name(self) -> str:
        """Identifier, e.g. ``cci_20``."""
        return f"cci_{self.period}"

    @property
    def warmup(self) -> int:
        """Bars consumed before the window is full."""
        return self.period - 1

    def _expression(self) -> pl.Expr:
        """Deviation from the mean typical price, scaled by mean absolute deviation."""
        typical = source_expression("hlc3")
        mean = typical.rolling_mean(self.period)
        spread = rolling_mad(typical, self.period)
        return (
            pl.when(spread <= _FLAT_TOLERANCE * mean.abs())
            .then(0.0)
            .otherwise((typical - mean) / (self.constant * spread))
        )

    def streaming(self) -> "CCIStream":
        """Build the incremental counterpart."""
        return CCIStream(self)


class CCIStream(BaseStreaming[CCI, float]):
    """Incremental :class:`CCI`."""

    def reset(self) -> None:
        """Empty the typical-price window."""
        self._window = RollingWindow(self._indicator.period)

    def step(
        self,
        _timestamp: datetime,
        _open_price: float,
        high: float,
        low: float,
        close: float,
        _volume: float,
        /,
    ) -> float | None:
        """Push the typical price and scale its distance from the window mean."""
        typical = typical_price(high, low, close)
        self._window.push(typical)
        if not self._window.full:
            return None
        mean = self._window.mean()
        spread = self._window.mean_absolute_deviation()
        if spread <= _FLAT_TOLERANCE * abs(mean):
            return 0.0
        return (typical - mean) / (self._indicator.constant * spread)


@dataclass(frozen=True)
class WilliamsR(ScalarIndicator):
    """Williams %R: the close's position in the recent range, on ``[-100, 0]``.

    A flat window reports the neutral -50.

    Attributes:
        period: Lookback for the high/low range.
    """

    period: int = 14

    def __post_init__(self) -> None:
        """Reject non-positive periods."""
        if self.period < 1:
            raise ValueError(f"period must be positive, got {self.period}")

    @property
    def name(self) -> str:
        """Identifier, e.g. ``willr_14``."""
        return f"willr_{self.period}"

    @property
    def warmup(self) -> int:
        """Bars consumed before the window is full."""
        return self.period - 1

    def _expression(self) -> pl.Expr:
        """Distance from the window high, normalised by the window range."""
        highest = pl.col("high").rolling_max(self.period)
        lowest = pl.col("low").rolling_min(self.period)
        return safe_divide(-100 * (highest - pl.col("close")), highest - lowest, _NEUTRAL_WILLIAMS)

    def streaming(self) -> "WilliamsRStream":
        """Build the incremental counterpart."""
        return WilliamsRStream(self)


class WilliamsRStream(BaseStreaming[WilliamsR, float]):
    """Incremental :class:`WilliamsR`."""

    def reset(self) -> None:
        """Clear both range extremes."""
        period = self._indicator.period
        self._high = RollingExtreme(period, largest=True)
        self._low = RollingExtreme(period, largest=False)

    def step(
        self,
        _timestamp: datetime,
        _open_price: float,
        high: float,
        low: float,
        close: float,
        _volume: float,
        /,
    ) -> float | None:
        """Push one bar and locate the close below the window high."""
        highest = self._high.push(high)
        lowest = self._low.push(low)
        if highest is None or lowest is None:
            return None
        span = highest - lowest
        if span == 0:
            return _NEUTRAL_WILLIAMS
        return -100 * (highest - close) / span


@dataclass(frozen=True)
class ROC(ScalarIndicator):
    """Rate of change, in percent, over ``period`` bars.

    A zero reference price has no percentage relative to it; that case reports 0.

    Attributes:
        period: Lookback.
        source: Price series to compare.
    """

    period: int = 12
    source: str = "close"

    def __post_init__(self) -> None:
        """Reject non-positive periods and unknown sources."""
        if self.period < 1:
            raise ValueError(f"period must be positive, got {self.period}")
        validate_source(self.source)

    @property
    def name(self) -> str:
        """Identifier, e.g. ``roc_12``."""
        return f"roc_{self.period}{source_suffix(self.source)}"

    @property
    def warmup(self) -> int:
        """The reference bar sits ``period`` bars back."""
        return self.period

    def _expression(self) -> pl.Expr:
        """Percentage change against the bar ``period`` back."""
        values = source_expression(self.source)
        reference = values.shift(self.period)
        return safe_divide(100 * (values - reference), reference, 0.0)

    def streaming(self) -> "ROCStream":
        """Build the incremental counterpart."""
        return ROCStream(self)


class ROCStream(BaseStreaming[ROC, float]):
    """Incremental :class:`ROC`."""

    def reset(self) -> None:
        """Empty the lookback window."""
        self._window = RollingWindow(self._indicator.period + 1)

    def step(
        self,
        _timestamp: datetime,
        open_price: float,
        high: float,
        low: float,
        close: float,
        volume: float,
        /,
    ) -> float | None:
        """Compare the newest value against the oldest one retained."""
        value = source_value(self._indicator.source, open_price, high, low, close, volume)
        self._window.push(value)
        if not self._window.full:
            return None
        reference = self._window.oldest
        if reference == 0:
            return 0.0
        return 100 * (value - reference) / reference


@dataclass(frozen=True)
class MFI(ScalarIndicator):
    """Money flow index: RSI applied to volume-weighted typical price.

    Bars where the typical price is unchanged contribute to neither side, as
    Chaikin defined it. A window with no negative flow reports 100, and one with
    no flow at all reports the neutral 50.

    Attributes:
        period: Window length.
    """

    period: int = 14

    def __post_init__(self) -> None:
        """Reject non-positive periods."""
        if self.period < 1:
            raise ValueError(f"period must be positive, got {self.period}")

    @property
    def name(self) -> str:
        """Identifier, e.g. ``mfi_14``."""
        return f"mfi_{self.period}"

    @property
    def warmup(self) -> int:
        """Flow starts on bar 1, so a full window of it ends on bar ``period``."""
        return self.period

    def _expression(self) -> pl.Expr:
        """Ratio of up-flow to down-flow over the window, mapped onto 0-100."""
        typical = source_expression("hlc3")
        flow = typical * pl.col("volume")
        previous = typical.shift(1)
        positive = (
            pl.when(previous.is_null())
            .then(None)
            .when(typical > previous)
            .then(flow)
            .otherwise(0.0)
        )
        negative = (
            pl.when(previous.is_null())
            .then(None)
            .when(typical < previous)
            .then(flow)
            .otherwise(0.0)
        )
        up = positive.rolling_sum(self.period)
        down = negative.rolling_sum(self.period)
        return (
            pl.when(down == 0)
            .then(pl.when(up == 0).then(_NEUTRAL_RSI).otherwise(100.0))
            .otherwise(100 - 100 / (1 + up / down))
        )

    def streaming(self) -> "MFIStream":
        """Build the incremental counterpart."""
        return MFIStream(self)


class MFIStream(BaseStreaming[MFI, float]):
    """Incremental :class:`MFI`."""

    def reset(self) -> None:
        """Empty both flow windows and forget the previous typical price."""
        period = self._indicator.period
        self._up = RollingWindow(period)
        self._down = RollingWindow(period)
        self._previous: float | None = None

    def step(
        self,
        _timestamp: datetime,
        _open_price: float,
        high: float,
        low: float,
        close: float,
        volume: float,
        /,
    ) -> float | None:
        """Route one bar's money flow to the up or down window."""
        typical = typical_price(high, low, close)
        previous = self._previous
        self._previous = typical
        if previous is None:
            return None
        flow = typical * volume
        self._up.push(flow if typical > previous else 0.0)
        self._down.push(flow if typical < previous else 0.0)
        if not self._up.full:
            return None
        down = self._down.sum()
        up = self._up.sum()
        if down == 0:
            return _NEUTRAL_RSI if up == 0 else 100.0
        return 100 - 100 / (1 + up / down)
