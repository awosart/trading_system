"""Trend indicators: moving averages, MACD, ADX, Supertrend, Ichimoku, Donchian.

Every exponential average here is *SMA-seeded*: it starts from the simple mean
of its first ``period`` inputs rather than from a single value. That is the
convention Wilder described and the one charting packages implement, and — more
useful here — it makes the warmup a fact rather than an approximation. A
first-value seed emits a number on bar 0 that is still visibly wrong hundreds of
bars later, and a strategy would happily trade on it.
"""

import math
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from typing import ClassVar, NamedTuple

import polars as pl

from trading_system.data.models import OHLCVFrame
from trading_system.features.base import (
    BaseIndicator,
    BaseStreaming,
    MultiExpressionIndicator,
    RecursiveIndicator,
    ScalarIndicator,
)
from trading_system.features.expressions import (
    seeded_ema,
    source_expression,
    source_suffix,
    source_value,
    true_range,
    validate_source,
    weighted_rolling_mean,
    wilder_rma,
)
from trading_system.features.rolling import (
    RollingExtreme,
    RollingWindow,
    SeededEma,
    linear_weights,
    wilder_alpha,
)


@dataclass(frozen=True)
class SMA(ScalarIndicator):
    """Simple moving average.

    Attributes:
        period: Number of bars averaged.
        source: Price series to average.
    """

    period: int = 20
    source: str = "close"

    def __post_init__(self) -> None:
        """Reject non-positive periods and unknown sources."""
        if self.period < 1:
            raise ValueError(f"period must be positive, got {self.period}")
        validate_source(self.source)

    @property
    def name(self) -> str:
        """Identifier, e.g. ``sma_20``."""
        return f"sma_{self.period}{source_suffix(self.source)}"

    @property
    def warmup(self) -> int:
        """Bars consumed before the first average is complete."""
        return self.period - 1

    def _expression(self) -> pl.Expr:
        """Rolling arithmetic mean of the configured source."""
        return source_expression(self.source).rolling_mean(self.period)

    def streaming(self) -> "SMAStream":
        """Build the incremental counterpart."""
        return SMAStream(self)


class SMAStream(BaseStreaming[SMA, float]):
    """Incremental :class:`SMA`."""

    def reset(self) -> None:
        """Empty the window."""
        self._window = RollingWindow(self._indicator.period)
        self._source = self._indicator.source

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
        """Push one bar and return the mean once the window is full."""
        self._window.push(source_value(self._source, open_price, high, low, close, volume))
        return self._window.mean() if self._window.full else None


@dataclass(frozen=True)
class EMA(ScalarIndicator):
    """Exponential moving average, ``alpha = 2 / (period + 1)``.

    Attributes:
        period: Smoothing period, also the length of the SMA seed.
        source: Price series to average.
    """

    period: int = 20
    source: str = "close"

    def __post_init__(self) -> None:
        """Reject non-positive periods and unknown sources."""
        if self.period < 1:
            raise ValueError(f"period must be positive, got {self.period}")
        validate_source(self.source)

    @property
    def name(self) -> str:
        """Identifier, e.g. ``ema_20``."""
        return f"ema_{self.period}{source_suffix(self.source)}"

    @property
    def warmup(self) -> int:
        """Bars consumed before the SMA seed is complete."""
        return self.period - 1

    @property
    def alpha(self) -> float:
        """Smoothing factor."""
        return 2.0 / (self.period + 1)

    def _expression(self) -> pl.Expr:
        """SMA-seeded exponential average of the configured source."""
        return seeded_ema(source_expression(self.source), self.period, self.alpha)

    def streaming(self) -> "EMAStream":
        """Build the incremental counterpart."""
        return EMAStream(self)


class EMAStream(BaseStreaming[EMA, float]):
    """Incremental :class:`EMA`."""

    def reset(self) -> None:
        """Return the average to its pre-seed state."""
        self._ema = SeededEma(self._indicator.period, self._indicator.alpha)
        self._source = self._indicator.source

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
        """Fold one bar into the average."""
        return self._ema.push(source_value(self._source, open_price, high, low, close, volume))


@dataclass(frozen=True)
class WMA(ScalarIndicator):
    """Linearly weighted moving average; the newest bar carries weight ``period``.

    Attributes:
        period: Number of bars averaged.
        source: Price series to average.
    """

    period: int = 20
    source: str = "close"

    def __post_init__(self) -> None:
        """Reject non-positive periods and unknown sources."""
        if self.period < 1:
            raise ValueError(f"period must be positive, got {self.period}")
        validate_source(self.source)

    @property
    def name(self) -> str:
        """Identifier, e.g. ``wma_20``."""
        return f"wma_{self.period}{source_suffix(self.source)}"

    @property
    def warmup(self) -> int:
        """Bars consumed before the first average is complete."""
        return self.period - 1

    def _expression(self) -> pl.Expr:
        """Rolling mean weighted 1..period, newest heaviest."""
        return weighted_rolling_mean(source_expression(self.source), linear_weights(self.period))

    def streaming(self) -> "WMAStream":
        """Build the incremental counterpart."""
        return WMAStream(self)


class WMAStream(BaseStreaming[WMA, float]):
    """Incremental :class:`WMA`."""

    def reset(self) -> None:
        """Empty the window and re-derive the weights."""
        self._window = RollingWindow(self._indicator.period)
        self._weights = linear_weights(self._indicator.period)
        self._source = self._indicator.source

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
        """Push one bar and return the weighted mean once the window is full."""
        self._window.push(source_value(self._source, open_price, high, low, close, volume))
        return self._window.weighted_mean(self._weights) if self._window.full else None


@dataclass(frozen=True)
class HMA(ScalarIndicator):
    """Hull moving average: ``WMA(2·WMA(n/2) − WMA(n), √n)``.

    Attributes:
        period: Base period ``n``. At least 4, so both the half period and the
            square-root period are usable window lengths.
        source: Price series to average.
    """

    period: int = 20
    source: str = "close"

    def __post_init__(self) -> None:
        """Reject periods below 4 and unknown sources."""
        if self.period < 4:
            raise ValueError(f"period must be at least 4, got {self.period}")
        validate_source(self.source)

    @property
    def name(self) -> str:
        """Identifier, e.g. ``hma_20``."""
        return f"hma_{self.period}{source_suffix(self.source)}"

    @property
    def half_period(self) -> int:
        """Length of the fast inner average."""
        return self.period // 2

    @property
    def smoothing_period(self) -> int:
        """Length of the outer average, ``round(√period)``."""
        return round(math.sqrt(self.period))

    @property
    def warmup(self) -> int:
        """Inner warmup plus outer warmup."""
        return (self.period - 1) + (self.smoothing_period - 1)

    def _expression(self) -> pl.Expr:
        """Weighted average of the doubled-fast-minus-slow difference."""
        values = source_expression(self.source)
        fast = weighted_rolling_mean(values, linear_weights(self.half_period))
        slow = weighted_rolling_mean(values, linear_weights(self.period))
        return weighted_rolling_mean(2 * fast - slow, linear_weights(self.smoothing_period))

    def streaming(self) -> "HMAStream":
        """Build the incremental counterpart."""
        return HMAStream(self)


class HMAStream(BaseStreaming[HMA, float]):
    """Incremental :class:`HMA`, three chained weighted windows."""

    def reset(self) -> None:
        """Empty all three windows."""
        indicator = self._indicator
        self._fast = RollingWindow(indicator.half_period)
        self._fast_weights = linear_weights(indicator.half_period)
        self._slow = RollingWindow(indicator.period)
        self._slow_weights = linear_weights(indicator.period)
        self._outer = RollingWindow(indicator.smoothing_period)
        self._outer_weights = linear_weights(indicator.smoothing_period)
        self._source = indicator.source

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
        """Push one bar through the fast, slow and outer averages."""
        value = source_value(self._source, open_price, high, low, close, volume)
        self._fast.push(value)
        self._slow.push(value)
        if not self._slow.full:
            return None
        raw = 2 * self._fast.weighted_mean(self._fast_weights) - self._slow.weighted_mean(
            self._slow_weights
        )
        self._outer.push(raw)
        return self._outer.weighted_mean(self._outer_weights) if self._outer.full else None


@dataclass(frozen=True)
class VWMA(ScalarIndicator):
    """Volume-weighted moving average of the close.

    A window whose volume is entirely zero carries no weighting information, so
    it falls back to the unweighted mean rather than emitting a NaN. FX feeds
    quoting tick volume do produce such bars in thin sessions.

    Attributes:
        period: Number of bars averaged.
    """

    period: int = 20

    def __post_init__(self) -> None:
        """Reject non-positive periods."""
        if self.period < 1:
            raise ValueError(f"period must be positive, got {self.period}")

    @property
    def name(self) -> str:
        """Identifier, e.g. ``vwma_20``."""
        return f"vwma_{self.period}"

    @property
    def warmup(self) -> int:
        """Bars consumed before the first average is complete."""
        return self.period - 1

    def _expression(self) -> pl.Expr:
        """Volume-weighted mean close, degrading to the plain mean at zero volume."""
        weighted = (pl.col("close") * pl.col("volume")).rolling_sum(self.period)
        weight = pl.col("volume").rolling_sum(self.period)
        return (
            pl.when(weight == 0)
            .then(pl.col("close").rolling_mean(self.period))
            .otherwise(weighted / weight)
        )

    def streaming(self) -> "VWMAStream":
        """Build the incremental counterpart."""
        return VWMAStream(self)


class VWMAStream(BaseStreaming[VWMA, float]):
    """Incremental :class:`VWMA`."""

    def reset(self) -> None:
        """Empty the price, weighted-price and volume windows."""
        period = self._indicator.period
        self._closes = RollingWindow(period)
        self._weighted = RollingWindow(period)
        self._volumes = RollingWindow(period)

    def step(
        self,
        _timestamp: datetime,
        _open_price: float,
        _high: float,
        _low: float,
        close: float,
        volume: float,
        /,
    ) -> float | None:
        """Push one bar and return the volume-weighted mean once full."""
        self._closes.push(close)
        self._weighted.push(close * volume)
        self._volumes.push(volume)
        if not self._volumes.full:
            return None
        weight = self._volumes.sum()
        if weight == 0:
            return self._closes.mean()
        return self._weighted.sum() / weight


class MacdValue(NamedTuple):
    """One bar of MACD output.

    Attributes:
        macd: Fast EMA minus slow EMA.
        signal: EMA of the MACD line.
        histogram: MACD line minus signal line.
    """

    macd: float
    signal: float
    histogram: float


@dataclass(frozen=True)
class MACD(MultiExpressionIndicator[MacdValue]):
    """Moving average convergence/divergence.

    The MACD line is arithmetically available ``signal_period - 1`` bars before
    the signal line is. All three channels are nonetheless published together,
    because a strategy reading the line while the histogram is still null would
    be reading a bar the live state machine declines to emit.

    Attributes:
        fast_period: Period of the fast EMA.
        slow_period: Period of the slow EMA.
        signal_period: Period of the EMA applied to the MACD line.
        source: Price series the EMAs run on.
    """

    outputs: ClassVar[tuple[str, ...]] = ("macd", "signal", "histogram")

    fast_period: int = 12
    slow_period: int = 26
    signal_period: int = 9
    source: str = "close"

    def __post_init__(self) -> None:
        """Reject bad periods, a fast period at or above the slow one, and unknown sources."""
        for label, period in (
            ("fast_period", self.fast_period),
            ("slow_period", self.slow_period),
            ("signal_period", self.signal_period),
        ):
            if period < 1:
                raise ValueError(f"{label} must be positive, got {period}")
        if self.fast_period >= self.slow_period:
            raise ValueError(
                f"fast_period {self.fast_period} must be below slow_period {self.slow_period}"
            )
        validate_source(self.source)

    @property
    def name(self) -> str:
        """Identifier, e.g. ``macd_12_26_9``."""
        suffix = source_suffix(self.source)
        return f"macd_{self.fast_period}_{self.slow_period}_{self.signal_period}{suffix}"

    @property
    def warmup(self) -> int:
        """Slow EMA warmup plus the signal EMA's own seed."""
        return (self.slow_period - 1) + (self.signal_period - 1)

    def _expressions(self) -> dict[str, pl.Expr]:
        """Fast and slow EMAs, their difference, and the signal EMA of it."""
        values = source_expression(self.source)
        fast = seeded_ema(values, self.fast_period, 2.0 / (self.fast_period + 1))
        slow = seeded_ema(values, self.slow_period, 2.0 / (self.slow_period + 1))
        macd = fast - slow
        signal = seeded_ema(macd, self.signal_period, 2.0 / (self.signal_period + 1))
        return {"macd": macd, "signal": signal, "histogram": macd - signal}

    def streaming(self) -> "MACDStream":
        """Build the incremental counterpart."""
        return MACDStream(self)


class MACDStream(BaseStreaming[MACD, MacdValue]):
    """Incremental :class:`MACD`."""

    def reset(self) -> None:
        """Return all three averages to their pre-seed state."""
        indicator = self._indicator
        self._fast = SeededEma(indicator.fast_period)
        self._slow = SeededEma(indicator.slow_period)
        self._signal = SeededEma(indicator.signal_period)
        self._source = indicator.source

    def step(
        self,
        _timestamp: datetime,
        open_price: float,
        high: float,
        low: float,
        close: float,
        volume: float,
        /,
    ) -> MacdValue | None:
        """Fold one bar through the fast, slow and signal averages."""
        value = source_value(self._source, open_price, high, low, close, volume)
        fast = self._fast.push(value)
        slow = self._slow.push(value)
        if fast is None or slow is None:
            return None
        macd = fast - slow
        signal = self._signal.push(macd)
        if signal is None:
            return None
        return MacdValue(macd=macd, signal=signal, histogram=macd - signal)


class AdxValue(NamedTuple):
    """One bar of ADX output.

    Attributes:
        adx: Trend strength, 0-100, direction-agnostic.
        plus_di: Positive directional indicator.
        minus_di: Negative directional indicator.
    """

    adx: float
    plus_di: float
    minus_di: float


@dataclass(frozen=True)
class ADX(BaseIndicator[AdxValue]):
    """Average directional index with its two directional indicators.

    Wilder smoothing is applied twice — once to the directional movement and the
    true range, once to the resulting DX — which is why the warmup spans roughly
    two periods rather than one.

    The vectorised path materialises each stage instead of composing one
    expression, because the composed form names the smoothed true range in four
    places and the smoothed DX depends on all of them. polars evaluates each
    reference independently, so a single expression re-runs the recursive smoother
    around a dozen times: measured at 9.4 s per million bars against 0.15 s for
    the staged form below.

    Attributes:
        period: Smoothing period for both stages.
    """

    outputs: ClassVar[tuple[str, ...]] = ("adx", "plus_di", "minus_di")

    period: int = 14

    def __post_init__(self) -> None:
        """Reject non-positive periods."""
        if self.period < 1:
            raise ValueError(f"period must be positive, got {self.period}")

    @property
    def name(self) -> str:
        """Identifier, e.g. ``adx_14``."""
        return f"adx_{self.period}"

    @property
    def warmup(self) -> int:
        """First DX lands at ``period``; its own seed takes ``period`` more bars."""
        return 2 * self.period - 1

    def _evaluate(self, frame: OHLCVFrame, /) -> pl.DataFrame:
        """Stage directional movement, smooth it, derive DX, then smooth again."""
        previous_high = pl.col("high").shift(1)
        previous_low = pl.col("low").shift(1)
        up_move = pl.col("high") - previous_high
        down_move = previous_low - pl.col("low")
        period = self.period

        staged = (
            frame.df.select(
                true_range().alias("range"),
                pl.when(previous_high.is_null())
                .then(None)
                .when((up_move > down_move) & (up_move > 0))
                .then(up_move)
                .otherwise(0.0)
                .alias("plus_dm"),
                pl.when(previous_low.is_null())
                .then(None)
                .when((down_move > up_move) & (down_move > 0))
                .then(down_move)
                .otherwise(0.0)
                .alias("minus_dm"),
            )
            .select(
                wilder_rma(pl.col("range"), period).alias("range"),
                wilder_rma(pl.col("plus_dm"), period).alias("plus_dm"),
                wilder_rma(pl.col("minus_dm"), period).alias("minus_dm"),
            )
            .select(
                pl.when(pl.col("range") == 0)
                .then(0.0)
                .otherwise(100 * pl.col("plus_dm") / pl.col("range"))
                .alias("plus_di"),
                pl.when(pl.col("range") == 0)
                .then(0.0)
                .otherwise(100 * pl.col("minus_dm") / pl.col("range"))
                .alias("minus_di"),
            )
            .with_columns(
                pl.when(pl.col("plus_di") + pl.col("minus_di") == 0)
                .then(0.0)
                .otherwise(
                    100
                    * (pl.col("plus_di") - pl.col("minus_di")).abs()
                    / (pl.col("plus_di") + pl.col("minus_di"))
                )
                .alias("dx")
            )
        )
        return staged.select(
            wilder_rma(pl.col("dx"), period).alias("adx"),
            pl.col("plus_di"),
            pl.col("minus_di"),
        )

    def streaming(self) -> "ADXStream":
        """Build the incremental counterpart."""
        return ADXStream(self)


class ADXStream(BaseStreaming[ADX, AdxValue]):
    """Incremental :class:`ADX`."""

    def reset(self) -> None:
        """Clear both smoothing stages and the previous bar."""
        period = self._indicator.period
        self._range = SeededEma(period, wilder_alpha(period))
        self._plus = SeededEma(period, wilder_alpha(period))
        self._minus = SeededEma(period, wilder_alpha(period))
        self._dx = SeededEma(period, wilder_alpha(period))
        self._previous: tuple[float, float, float] | None = None

    def step(
        self,
        _timestamp: datetime,
        _open_price: float,
        high: float,
        low: float,
        close: float,
        _volume: float,
        /,
    ) -> AdxValue | None:
        """Fold one bar through directional movement, true range and DX."""
        previous = self._previous
        self._previous = (high, low, close)
        if previous is None:
            return None
        previous_high, previous_low, previous_close = previous

        up_move = high - previous_high
        down_move = previous_low - low
        plus_dm = up_move if (up_move > down_move and up_move > 0) else 0.0
        minus_dm = down_move if (down_move > up_move and down_move > 0) else 0.0
        span = max(high - low, abs(high - previous_close), abs(low - previous_close))

        smoothed_range = self._range.push(span)
        smoothed_plus = self._plus.push(plus_dm)
        smoothed_minus = self._minus.push(minus_dm)
        if smoothed_range is None or smoothed_plus is None or smoothed_minus is None:
            return None

        if smoothed_range == 0:
            plus_di = 0.0
            minus_di = 0.0
        else:
            plus_di = 100 * smoothed_plus / smoothed_range
            minus_di = 100 * smoothed_minus / smoothed_range
        directional_sum = plus_di + minus_di
        dx = 0.0 if directional_sum == 0 else 100 * abs(plus_di - minus_di) / directional_sum

        adx = self._dx.push(dx)
        if adx is None:
            return None
        return AdxValue(adx=adx, plus_di=plus_di, minus_di=minus_di)


class SupertrendValue(NamedTuple):
    """One bar of Supertrend output.

    Attributes:
        line: The active band — the lower one while the trend is up.
        direction: ``1.0`` while the trend is up, ``-1.0`` while it is down.
        upper: Current upper band.
        lower: Current lower band.
    """

    line: float
    direction: float
    upper: float
    lower: float


@dataclass(frozen=True)
class Supertrend(RecursiveIndicator[SupertrendValue]):
    """ATR-banded trend follower with ratcheting bands.

    Each band may only move in the direction that tightens it until price closes
    through it. That rule reads the band's own previous output, so it has no
    rolling-window form; the vectorised path here *is* the state machine (see
    :class:`~trading_system.features.base.RecursiveIndicator`).

    On the first bar with a valid ATR the trend is taken to be up. No prior close
    exists to infer it from, and the choice is displaced by the first close
    through a band.

    Attributes:
        period: ATR period.
        multiplier: Band width in ATRs.
    """

    outputs: ClassVar[tuple[str, ...]] = ("line", "direction", "upper", "lower")

    period: int = 10
    multiplier: float = 3.0

    def __post_init__(self) -> None:
        """Reject non-positive periods and multipliers."""
        if self.period < 1:
            raise ValueError(f"period must be positive, got {self.period}")
        if self.multiplier <= 0:
            raise ValueError(f"multiplier must be positive, got {self.multiplier}")

    @property
    def name(self) -> str:
        """Identifier, e.g. ``supertrend_10_3.0``."""
        return f"supertrend_{self.period}_{self.multiplier}"

    @property
    def warmup(self) -> int:
        """True range starts on bar 1, so its Wilder seed completes on bar ``period``."""
        return self.period

    def streaming(self) -> "SupertrendStream":
        """Build the incremental counterpart."""
        return SupertrendStream(self)


class SupertrendStream(BaseStreaming[Supertrend, SupertrendValue]):
    """Incremental :class:`Supertrend`."""

    def reset(self) -> None:
        """Clear the ATR, the bands and the trend direction."""
        indicator = self._indicator
        self._atr = SeededEma(indicator.period, wilder_alpha(indicator.period))
        self._multiplier = indicator.multiplier
        self._previous_close: float | None = None
        self._upper: float | None = None
        self._lower: float | None = None
        self._direction = 1.0

    def step(
        self,
        _timestamp: datetime,
        _open_price: float,
        high: float,
        low: float,
        close: float,
        _volume: float,
        /,
    ) -> SupertrendValue | None:
        """Advance the bands and, if price closed through one, flip the trend."""
        previous_close = self._previous_close
        self._previous_close = close
        if previous_close is None:
            return None

        span = max(high - low, abs(high - previous_close), abs(low - previous_close))
        atr = self._atr.push(span)
        if atr is None:
            return None

        midpoint = (high + low) / 2
        basic_upper = midpoint + self._multiplier * atr
        basic_lower = midpoint - self._multiplier * atr

        if self._upper is None or self._lower is None:
            upper, lower = basic_upper, basic_lower
        else:
            upper = (
                basic_upper
                if (basic_upper < self._upper or previous_close > self._upper)
                else self._upper
            )
            lower = (
                basic_lower
                if (basic_lower > self._lower or previous_close < self._lower)
                else self._lower
            )
        self._upper, self._lower = upper, lower

        if self._direction > 0 and close < lower:
            self._direction = -1.0
        elif self._direction < 0 and close > upper:
            self._direction = 1.0

        line = lower if self._direction > 0 else upper
        return SupertrendValue(line=line, direction=self._direction, upper=upper, lower=lower)


class IchimokuValue(NamedTuple):
    """One bar of Ichimoku output.

    Attributes:
        tenkan: Conversion line.
        kijun: Base line.
        senkou_a: Leading span A, as plotted at the current bar.
        senkou_b: Leading span B, as plotted at the current bar.
    """

    tenkan: float
    kijun: float
    senkou_a: float
    senkou_b: float


def _channel_midpoint(period: int) -> pl.Expr:
    """Midpoint of the highest high and lowest low over ``period`` bars."""
    return (pl.col("high").rolling_max(period) + pl.col("low").rolling_min(period)) / 2


def _midpoint(high: float | None, low: float | None) -> float | None:
    """Average two optional extremes, propagating absence."""
    if high is None or low is None:
        return None
    return (high + low) / 2


@dataclass(frozen=True)
class Ichimoku(MultiExpressionIndicator[IchimokuValue]):
    """Ichimoku cloud, restricted to the spans that can be known causally.

    The leading spans are published *at the bar they are plotted at*, computed
    from data ``displacement`` bars earlier. That is the cloud a strategy
    compares the current price against, and it uses no future data.

    Chikou is deliberately absent. At the position it is drawn, its value is the
    close ``displacement`` bars in the future, so there is no causal form of it.
    A strategy wanting the same information should compare ``close[t]`` against
    ``close[t - displacement]`` directly, which is honest about the lag.

    Attributes:
        tenkan_period: Conversion line window.
        kijun_period: Base line window.
        senkou_b_period: Leading span B window.
        displacement: Bars the cloud is shifted forward on a chart.
    """

    outputs: ClassVar[tuple[str, ...]] = ("tenkan", "kijun", "senkou_a", "senkou_b")

    tenkan_period: int = 9
    kijun_period: int = 26
    senkou_b_period: int = 52
    displacement: int = 26

    def __post_init__(self) -> None:
        """Reject non-positive periods and negative displacement."""
        for label, period in (
            ("tenkan_period", self.tenkan_period),
            ("kijun_period", self.kijun_period),
            ("senkou_b_period", self.senkou_b_period),
        ):
            if period < 1:
                raise ValueError(f"{label} must be positive, got {period}")
        if self.displacement < 0:
            raise ValueError(f"displacement must be non-negative, got {self.displacement}")

    @property
    def name(self) -> str:
        """Identifier, e.g. ``ichimoku_9_26_52_26``."""
        return (
            f"ichimoku_{self.tenkan_period}_{self.kijun_period}"
            f"_{self.senkou_b_period}_{self.displacement}"
        )

    @property
    def warmup(self) -> int:
        """Longest span window, plus the displacement it is read through."""
        longest = max(self.kijun_period, self.senkou_b_period)
        return longest - 1 + self.displacement

    def _expressions(self) -> dict[str, pl.Expr]:
        """Conversion and base lines, plus both leading spans read back in time."""
        tenkan = _channel_midpoint(self.tenkan_period)
        kijun = _channel_midpoint(self.kijun_period)
        return {
            "tenkan": tenkan,
            "kijun": kijun,
            "senkou_a": ((tenkan + kijun) / 2).shift(self.displacement),
            "senkou_b": _channel_midpoint(self.senkou_b_period).shift(self.displacement),
        }

    def streaming(self) -> "IchimokuStream":
        """Build the incremental counterpart."""
        return IchimokuStream(self)


class IchimokuStream(BaseStreaming[Ichimoku, IchimokuValue]):
    """Incremental :class:`Ichimoku`."""

    def reset(self) -> None:
        """Clear the three midpoint windows and the displacement queues."""
        indicator = self._indicator
        self._tenkan_high = RollingExtreme(indicator.tenkan_period, largest=True)
        self._tenkan_low = RollingExtreme(indicator.tenkan_period, largest=False)
        self._kijun_high = RollingExtreme(indicator.kijun_period, largest=True)
        self._kijun_low = RollingExtreme(indicator.kijun_period, largest=False)
        self._senkou_high = RollingExtreme(indicator.senkou_b_period, largest=True)
        self._senkou_low = RollingExtreme(indicator.senkou_b_period, largest=False)
        self._displacement = indicator.displacement
        self._span_a: deque[float | None] = deque()
        self._span_b: deque[float | None] = deque()

    def step(
        self,
        _timestamp: datetime,
        _open_price: float,
        high: float,
        low: float,
        _close: float,
        _volume: float,
        /,
    ) -> IchimokuValue | None:
        """Advance every window and read the spans out of the delay queues."""
        tenkan = _midpoint(self._tenkan_high.push(high), self._tenkan_low.push(low))
        kijun = _midpoint(self._kijun_high.push(high), self._kijun_low.push(low))
        span_b_now = _midpoint(self._senkou_high.push(high), self._senkou_low.push(low))
        span_a_now = None if tenkan is None or kijun is None else (tenkan + kijun) / 2

        self._span_a.append(span_a_now)
        self._span_b.append(span_b_now)
        if len(self._span_a) <= self._displacement:
            return None
        senkou_a = self._span_a.popleft()
        senkou_b = self._span_b.popleft()

        if tenkan is None or kijun is None or senkou_a is None or senkou_b is None:
            return None
        return IchimokuValue(tenkan=tenkan, kijun=kijun, senkou_a=senkou_a, senkou_b=senkou_b)


class DonchianValue(NamedTuple):
    """One bar of Donchian output.

    Attributes:
        upper: Highest high of the window, current bar included.
        lower: Lowest low of the window, current bar included.
        middle: Midpoint of the channel.
    """

    upper: float
    lower: float
    middle: float


@dataclass(frozen=True)
class Donchian(MultiExpressionIndicator[DonchianValue]):
    """Donchian channel over the last ``period`` bars.

    The current bar is inside the window, which is what a channel *display*
    shows. A breakout rule asking "did price exceed the prior range" must compare
    against the previous bar's channel, since the current bar's own high sits
    inside the current upper band by construction.

    Attributes:
        period: Channel window.
    """

    outputs: ClassVar[tuple[str, ...]] = ("upper", "lower", "middle")

    period: int = 20

    def __post_init__(self) -> None:
        """Reject non-positive periods."""
        if self.period < 1:
            raise ValueError(f"period must be positive, got {self.period}")

    @property
    def name(self) -> str:
        """Identifier, e.g. ``donchian_20``."""
        return f"donchian_{self.period}"

    @property
    def warmup(self) -> int:
        """Bars consumed before the window is full."""
        return self.period - 1

    def _expressions(self) -> dict[str, pl.Expr]:
        """Rolling extremes of high and low, and their midpoint."""
        upper = pl.col("high").rolling_max(self.period)
        lower = pl.col("low").rolling_min(self.period)
        return {"upper": upper, "lower": lower, "middle": (upper + lower) / 2}

    def streaming(self) -> "DonchianStream":
        """Build the incremental counterpart."""
        return DonchianStream(self)


class DonchianStream(BaseStreaming[Donchian, DonchianValue]):
    """Incremental :class:`Donchian`."""

    def reset(self) -> None:
        """Clear both monotonic deques."""
        period = self._indicator.period
        self._high = RollingExtreme(period, largest=True)
        self._low = RollingExtreme(period, largest=False)

    def step(
        self,
        _timestamp: datetime,
        _open_price: float,
        high: float,
        low: float,
        _close: float,
        _volume: float,
        /,
    ) -> DonchianValue | None:
        """Push one bar and read both extremes."""
        upper = self._high.push(high)
        lower = self._low.push(low)
        if upper is None or lower is None:
            return None
        return DonchianValue(upper=upper, lower=lower, middle=(upper + lower) / 2)
