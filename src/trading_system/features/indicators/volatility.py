"""Volatility indicators: ATR, standard deviation, Bollinger, Keltner, Choppiness.

True range is undefined on the first bar of a frame — it needs a previous close
— so every indicator built on it warms up one bar later than its period alone
would suggest. That off-by-one is deliberate and is what the ``warmup``
properties here report.
"""

import math
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
    safe_divide,
    seeded_ema,
    source_expression,
    source_suffix,
    source_value,
    true_range,
    validate_source,
    wilder_rma,
)
from trading_system.features.rolling import (
    RollingExtreme,
    RollingWindow,
    SeededEma,
    wilder_alpha,
)


@dataclass(frozen=True)
class ATR(ScalarIndicator):
    """Average true range, Wilder-smoothed.

    Attributes:
        period: Smoothing period.
    """

    period: int = 14

    def __post_init__(self) -> None:
        """Reject non-positive periods."""
        if self.period < 1:
            raise ValueError(f"period must be positive, got {self.period}")

    @property
    def name(self) -> str:
        """Identifier, e.g. ``atr_14``."""
        return f"atr_{self.period}"

    @property
    def warmup(self) -> int:
        """True range starts on bar 1, so the Wilder seed completes on bar ``period``."""
        return self.period

    def _expression(self) -> pl.Expr:
        """Wilder average of the true range."""
        return wilder_rma(true_range(), self.period)

    def streaming(self) -> "ATRStream":
        """Build the incremental counterpart."""
        return ATRStream(self)


class ATRStream(BaseStreaming[ATR, float]):
    """Incremental :class:`ATR`."""

    def reset(self) -> None:
        """Clear the average and the previous close."""
        period = self._indicator.period
        self._average = SeededEma(period, wilder_alpha(period))
        self._previous_close: float | None = None

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
        """Fold one bar's true range into the average."""
        previous_close = self._previous_close
        self._previous_close = close
        if previous_close is None:
            return None
        span = max(high - low, abs(high - previous_close), abs(low - previous_close))
        return self._average.push(span)


@dataclass(frozen=True)
class StdDev(ScalarIndicator):
    """Rolling standard deviation.

    Attributes:
        period: Window length.
        source: Price series to measure.
        ddof: Delta degrees of freedom. ``0`` — the population form — is what
            Bollinger Bands are drawn with; pass ``1`` for the sample form.
    """

    period: int = 20
    source: str = "close"
    ddof: int = 0

    def __post_init__(self) -> None:
        """Reject windows too short for the requested degrees of freedom."""
        if self.period < 2:
            raise ValueError(f"period must be at least 2, got {self.period}")
        if self.ddof < 0 or self.ddof >= self.period:
            raise ValueError(f"ddof must be in [0, {self.period}), got {self.ddof}")
        validate_source(self.source)

    @property
    def name(self) -> str:
        """Identifier, e.g. ``stddev_20``."""
        return f"stddev_{self.period}{source_suffix(self.source)}"

    @property
    def warmup(self) -> int:
        """Bars consumed before the window is full."""
        return self.period - 1

    def _expression(self) -> pl.Expr:
        """Rolling standard deviation of the configured source."""
        return source_expression(self.source).rolling_std(self.period, ddof=self.ddof)

    def streaming(self) -> "StdDevStream":
        """Build the incremental counterpart."""
        return StdDevStream(self)


class StdDevStream(BaseStreaming[StdDev, float]):
    """Incremental :class:`StdDev`."""

    def reset(self) -> None:
        """Empty the window."""
        self._window = RollingWindow(self._indicator.period)

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
        """Push one bar and return the deviation once the window is full."""
        indicator = self._indicator
        self._window.push(source_value(indicator.source, open_price, high, low, close, volume))
        return self._window.std(indicator.ddof) if self._window.full else None


class BollingerValue(NamedTuple):
    """One bar of Bollinger output.

    Attributes:
        upper: Middle band plus ``num_std`` deviations.
        middle: Simple moving average.
        lower: Middle band minus ``num_std`` deviations.
        bandwidth: Channel width as a fraction of the middle band.
    """

    upper: float
    middle: float
    lower: float
    bandwidth: float


@dataclass(frozen=True)
class BollingerBands(MultiExpressionIndicator[BollingerValue]):
    """Bollinger bands and their normalised width.

    ``bandwidth`` is the squeeze measure — channel width divided by the middle
    band — which is comparable across instruments in a way the raw width is not.

    Attributes:
        period: Window length for both the average and the deviation.
        num_std: Band distance in standard deviations.
        source: Price series to measure.
    """

    outputs: ClassVar[tuple[str, ...]] = ("upper", "middle", "lower", "bandwidth")

    period: int = 20
    num_std: float = 2.0
    source: str = "close"

    def __post_init__(self) -> None:
        """Reject short windows, non-positive band distances and unknown sources."""
        if self.period < 2:
            raise ValueError(f"period must be at least 2, got {self.period}")
        if self.num_std <= 0:
            raise ValueError(f"num_std must be positive, got {self.num_std}")
        validate_source(self.source)

    @property
    def name(self) -> str:
        """Identifier, e.g. ``bbands_20_2.0``."""
        return f"bbands_{self.period}_{self.num_std}{source_suffix(self.source)}"

    @property
    def warmup(self) -> int:
        """Bars consumed before the window is full."""
        return self.period - 1

    def _expressions(self) -> dict[str, pl.Expr]:
        """Mean, mean ± k·σ, and the normalised channel width."""
        values = source_expression(self.source)
        middle = values.rolling_mean(self.period)
        deviation = values.rolling_std(self.period, ddof=0)
        upper = middle + self.num_std * deviation
        lower = middle - self.num_std * deviation
        return {
            "upper": upper,
            "middle": middle,
            "lower": lower,
            "bandwidth": safe_divide(upper - lower, middle, 0.0),
        }

    def streaming(self) -> "BollingerStream":
        """Build the incremental counterpart."""
        return BollingerStream(self)


class BollingerStream(BaseStreaming[BollingerBands, BollingerValue]):
    """Incremental :class:`BollingerBands`."""

    def reset(self) -> None:
        """Empty the window."""
        self._window = RollingWindow(self._indicator.period)

    def step(
        self,
        _timestamp: datetime,
        open_price: float,
        high: float,
        low: float,
        close: float,
        volume: float,
        /,
    ) -> BollingerValue | None:
        """Push one bar and derive the three bands plus the width."""
        indicator = self._indicator
        self._window.push(source_value(indicator.source, open_price, high, low, close, volume))
        if not self._window.full:
            return None
        middle = self._window.mean()
        offset = indicator.num_std * self._window.std(0)
        upper = middle + offset
        lower = middle - offset
        bandwidth = 0.0 if middle == 0 else (upper - lower) / middle
        return BollingerValue(upper=upper, middle=middle, lower=lower, bandwidth=bandwidth)


class KeltnerValue(NamedTuple):
    """One bar of Keltner output.

    Attributes:
        upper: Middle line plus ``multiplier`` ATRs.
        middle: Exponential moving average of the close.
        lower: Middle line minus ``multiplier`` ATRs.
    """

    upper: float
    middle: float
    lower: float


@dataclass(frozen=True)
class Keltner(MultiExpressionIndicator[KeltnerValue]):
    """Keltner channel: an EMA with ATR-scaled bands.

    Attributes:
        ema_period: Period of the centre line.
        atr_period: Period of the band width.
        multiplier: Band width in ATRs.
    """

    outputs: ClassVar[tuple[str, ...]] = ("upper", "middle", "lower")

    ema_period: int = 20
    atr_period: int = 10
    multiplier: float = 2.0

    def __post_init__(self) -> None:
        """Reject non-positive periods and multipliers."""
        if self.ema_period < 1:
            raise ValueError(f"ema_period must be positive, got {self.ema_period}")
        if self.atr_period < 1:
            raise ValueError(f"atr_period must be positive, got {self.atr_period}")
        if self.multiplier <= 0:
            raise ValueError(f"multiplier must be positive, got {self.multiplier}")

    @property
    def name(self) -> str:
        """Identifier, e.g. ``keltner_20_10_2.0``."""
        return f"keltner_{self.ema_period}_{self.atr_period}_{self.multiplier}"

    @property
    def warmup(self) -> int:
        """Whichever of the centre line and the band width is slower."""
        return max(self.ema_period - 1, self.atr_period)

    def _expressions(self) -> dict[str, pl.Expr]:
        """EMA centre line with ATR-scaled bands either side."""
        middle = seeded_ema(pl.col("close"), self.ema_period, 2.0 / (self.ema_period + 1))
        width = self.multiplier * wilder_rma(true_range(), self.atr_period)
        return {"upper": middle + width, "middle": middle, "lower": middle - width}

    def streaming(self) -> "KeltnerStream":
        """Build the incremental counterpart."""
        return KeltnerStream(self)


class KeltnerStream(BaseStreaming[Keltner, KeltnerValue]):
    """Incremental :class:`Keltner`."""

    def reset(self) -> None:
        """Clear the centre average, the ATR and the previous close."""
        indicator = self._indicator
        self._middle = SeededEma(indicator.ema_period)
        self._atr = SeededEma(indicator.atr_period, wilder_alpha(indicator.atr_period))
        self._previous_close: float | None = None

    def step(
        self,
        _timestamp: datetime,
        _open_price: float,
        high: float,
        low: float,
        close: float,
        _volume: float,
        /,
    ) -> KeltnerValue | None:
        """Advance both averages and place the bands."""
        previous_close = self._previous_close
        self._previous_close = close
        middle = self._middle.push(close)
        if previous_close is None:
            return None
        span = max(high - low, abs(high - previous_close), abs(low - previous_close))
        atr = self._atr.push(span)
        if middle is None or atr is None:
            return None
        width = self._indicator.multiplier * atr
        return KeltnerValue(upper=middle + width, middle=middle, lower=middle - width)


@dataclass(frozen=True)
class Choppiness(ScalarIndicator):
    """Choppiness index: how much ground price covered versus how far it went.

    Near 100 the bar-by-bar travel dwarfs the net range and the market is
    ranging; near 0 the travel went almost entirely in one direction. A window
    with no range at all — every bar identical — is maximally choppy by
    definition and reports 100 rather than dividing by zero.

    Attributes:
        period: Window length.
    """

    period: int = 14

    def __post_init__(self) -> None:
        """Reject periods below 2, for which the logarithm has no scale."""
        if self.period < 2:
            raise ValueError(f"period must be at least 2, got {self.period}")

    @property
    def name(self) -> str:
        """Identifier, e.g. ``chop_14``."""
        return f"chop_{self.period}"

    @property
    def warmup(self) -> int:
        """True range starts on bar 1, so a full window of it ends on bar ``period``."""
        return self.period

    def _expression(self) -> pl.Expr:
        """Log ratio of summed true range to the window's net range."""
        travelled = true_range().rolling_sum(self.period)
        span = pl.col("high").rolling_max(self.period) - pl.col("low").rolling_min(self.period)
        ratio = safe_divide(travelled, span, 1.0)
        return (
            pl.when(span == 0).then(100.0).otherwise(100 * ratio.log10() / math.log10(self.period))
        )

    def streaming(self) -> "ChoppinessStream":
        """Build the incremental counterpart."""
        return ChoppinessStream(self)


class ChoppinessStream(BaseStreaming[Choppiness, float]):
    """Incremental :class:`Choppiness`."""

    def reset(self) -> None:
        """Clear the true-range window, both extremes and the previous close."""
        period = self._indicator.period
        self._ranges = RollingWindow(period)
        self._high = RollingExtreme(period, largest=True)
        self._low = RollingExtreme(period, largest=False)
        self._scale = math.log10(period)
        self._previous_close: float | None = None

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
        """Push one bar and compare travel against net range."""
        previous_close = self._previous_close
        self._previous_close = close
        highest = self._high.push(high)
        lowest = self._low.push(low)
        if previous_close is None:
            return None
        self._ranges.push(max(high - low, abs(high - previous_close), abs(low - previous_close)))
        if not self._ranges.full or highest is None or lowest is None:
            return None
        span = highest - lowest
        if span == 0:
            return 100.0
        return 100 * math.log10(self._ranges.sum() / span) / self._scale
