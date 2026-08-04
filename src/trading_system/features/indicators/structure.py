"""Market structure: swing pivots, pivot points, HH/HL/LH/LL, ranges, S/R levels.

Everything here turns on one rule. **A pivot is reported at the bar that
confirms it, not at the bar it occurred on.** A fractal high at bar ``c`` is only
knowable once ``lookback`` further bars have closed lower, so it becomes visible
at bar ``c + lookback`` and no earlier. Charting packages draw the marker back at
``c``, which is fine for a human reading history and catastrophic for a backtest:
a strategy reading the marker at ``c`` is trading on ``lookback`` bars it has not
seen. Every swing here carries both indices, and the value published at bar ``t``
depends only on bars ``≤ t``.

Support and resistance levels are a function rather than an indicator. Their
output is a variable-length set of price levels, not one number per bar, and
forcing that into the per-bar contract would misrepresent it.
"""

from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from enum import IntEnum, StrEnum
from typing import ClassVar, NamedTuple

import polars as pl

from trading_system.data.models import TIMESTAMP_COLUMN, OHLCVFrame
from trading_system.features.base import (
    BaseStreaming,
    MultiExpressionIndicator,
    RecursiveIndicator,
)
from trading_system.features.expressions import safe_divide, true_range, wilder_rma
from trading_system.features.rolling import RollingExtreme, SeededEma, wilder_alpha


class SwingKind(StrEnum):
    """Whether a pivot is a local high or a local low."""

    HIGH = "HIGH"
    LOW = "LOW"


@dataclass(frozen=True)
class Swing:
    """A confirmed pivot.

    Attributes:
        index: Bar the pivot occurred on.
        kind: Whether it is a high or a low.
        price: The pivot's extreme price.
        confirmed_at: Bar at which the pivot became knowable, ``index +
            lookback``. Nothing may act on the pivot before this bar.
    """

    index: int
    kind: SwingKind
    price: float
    confirmed_at: int


class _FractalDetector:
    """Confirms fractal pivots ``lookback`` bars after they occur.

    A pivot must be *strictly* more extreme than every bar within ``lookback``
    on both sides. Plateaus therefore produce no pivot at all, which is the
    conservative reading: two bars sharing the same high do not identify which
    of them the market turned on.

    Also tracks the extreme of the confirmed prefix, which the indicators use as
    a stand-in until the first genuine pivot forms — on a monotonic run there
    may never be one.
    """

    __slots__ = ("_highs", "_index", "_lookback", "_lows", "confirmed_high", "confirmed_low")

    def __init__(self, lookback: int) -> None:
        """Create a detector over a ``2 * lookback + 1`` bar window.

        Args:
            lookback: Bars required on each side of a pivot.

        Raises:
            ValueError: If ``lookback`` is not positive.
        """
        if lookback < 1:
            raise ValueError(f"lookback must be positive, got {lookback}")
        self._lookback = lookback
        span = 2 * lookback + 1
        self._highs: deque[float] = deque(maxlen=span)
        self._lows: deque[float] = deque(maxlen=span)
        self._index = 0
        self.confirmed_high: tuple[int, float] | None = None
        self.confirmed_low: tuple[int, float] | None = None

    @property
    def lookback(self) -> int:
        """Bars required on each side of a pivot."""
        return self._lookback

    @property
    def warmup(self) -> int:
        """Bar index at which the first pivot can be confirmed."""
        return 2 * self._lookback

    @property
    def seen(self) -> int:
        """Number of bars pushed so far."""
        return self._index

    def clear(self) -> None:
        """Forget every bar seen."""
        self._highs.clear()
        self._lows.clear()
        self._index = 0
        self.confirmed_high = None
        self.confirmed_low = None

    def push(self, high: float, low: float) -> tuple[Swing | None, Swing | None]:
        """Consume one bar and report any pivot it confirms.

        Args:
            high: Bar high.
            low: Bar low.

        Returns:
            The swing high and swing low confirmed by this bar, each ``None`` if
            this bar confirmed none.
        """
        current = self._index
        self._index += 1
        self._highs.append(high)
        self._lows.append(low)

        lookback = self._lookback
        centre = current - lookback
        if centre >= 0:
            offset = len(self._highs) - 1 - lookback
            centre_high = self._highs[offset]
            centre_low = self._lows[offset]
            if self.confirmed_high is None or centre_high > self.confirmed_high[1]:
                self.confirmed_high = (centre, centre_high)
            if self.confirmed_low is None or centre_low < self.confirmed_low[1]:
                self.confirmed_low = (centre, centre_low)

        if current < self.warmup:
            return None, None

        # "Strictly the largest in the window" is equivalent to "equal to the
        # window maximum, and unique". Phrasing it that way keeps the per-bar
        # work inside two C-level deque scans instead of a Python comprehension,
        # which halves the cost of these indicators on a million-bar frame.
        swing_high = None
        swing_low = None
        highs = self._highs
        lows = self._lows
        pivot_high = highs[lookback]
        pivot_low = lows[lookback]
        if pivot_high == max(highs) and highs.count(pivot_high) == 1:
            swing_high = Swing(
                index=centre, kind=SwingKind.HIGH, price=pivot_high, confirmed_at=current
            )
        if pivot_low == min(lows) and lows.count(pivot_low) == 1:
            swing_low = Swing(
                index=centre, kind=SwingKind.LOW, price=pivot_low, confirmed_at=current
            )
        return swing_high, swing_low


def find_swings(highs: Sequence[float], lows: Sequence[float], lookback: int) -> list[Swing]:
    """Find every confirmed fractal pivot in a series.

    Args:
        highs: Bar highs, oldest first.
        lows: Bar lows, aligned with ``highs``.
        lookback: Bars required on each side of a pivot.

    Returns:
        Pivots ordered by the bar that confirmed them. A pivot whose
        ``confirmed_at`` exceeds the last index is not returned at all, so the
        result is exactly what was knowable by the end of the series.

    Raises:
        ValueError: If the two series differ in length.
    """
    if len(highs) != len(lows):
        raise ValueError(f"highs and lows differ in length: {len(highs)} vs {len(lows)}")
    detector = _FractalDetector(lookback)
    swings: list[Swing] = []
    for high, low in zip(highs, lows, strict=True):
        swing_high, swing_low = detector.push(high, low)
        if swing_high is not None:
            swings.append(swing_high)
        if swing_low is not None:
            swings.append(swing_low)
    return swings


class SwingValue(NamedTuple):
    """One bar of swing-point output.

    Attributes:
        swing_high: Price of the most recent confirmed swing high.
        swing_high_age: Bars since that high occurred, at least ``lookback``.
        swing_low: Price of the most recent confirmed swing low.
        swing_low_age: Bars since that low occurred, at least ``lookback``.
    """

    swing_high: float
    swing_high_age: float
    swing_low: float
    swing_low_age: float


@dataclass(frozen=True)
class SwingPoints(RecursiveIndicator[SwingValue]):
    """The most recent confirmed swing high and low, carried forward.

    Until a fractal pivot has formed, the extreme of the *confirmed* prefix
    stands in for it. On a monotonic run no fractal high ever forms, and the
    highest bar seen so far is the honest answer to "where is resistance"; the
    stand-in is displaced as soon as a real pivot appears.

    Attributes:
        lookback: Bars required on each side of a pivot.
    """

    outputs: ClassVar[tuple[str, ...]] = (
        "swing_high",
        "swing_high_age",
        "swing_low",
        "swing_low_age",
    )

    lookback: int = 5

    def __post_init__(self) -> None:
        """Reject non-positive lookbacks."""
        if self.lookback < 1:
            raise ValueError(f"lookback must be positive, got {self.lookback}")

    @property
    def name(self) -> str:
        """Identifier, e.g. ``swing_5``."""
        return f"swing_{self.lookback}"

    @property
    def warmup(self) -> int:
        """A pivot needs ``lookback`` bars either side before it can be confirmed."""
        return 2 * self.lookback

    def streaming(self) -> "SwingPointsStream":
        """Build the incremental counterpart."""
        return SwingPointsStream(self)


class SwingPointsStream(BaseStreaming[SwingPoints, SwingValue]):
    """Incremental :class:`SwingPoints`."""

    def reset(self) -> None:
        """Clear the detector and the remembered pivots."""
        self._detector = _FractalDetector(self._indicator.lookback)
        self._high: tuple[int, float] | None = None
        self._low: tuple[int, float] | None = None

    def step(
        self,
        _timestamp: datetime,
        _open_price: float,
        high: float,
        low: float,
        _close: float,
        _volume: float,
        /,
    ) -> SwingValue | None:
        """Push one bar and report the pivots standing after it."""
        current = self._detector.seen
        swing_high, swing_low = self._detector.push(high, low)
        if swing_high is not None:
            self._high = (swing_high.index, swing_high.price)
        if swing_low is not None:
            self._low = (swing_low.index, swing_low.price)

        latest_high = self._high or self._detector.confirmed_high
        latest_low = self._low or self._detector.confirmed_low
        # The prefix extremes exist from bar `lookback` onwards, but publishing
        # them before the first pivot *could* have been confirmed would make the
        # indicator valid earlier than its stated warmup.
        if current < self._detector.warmup or latest_high is None or latest_low is None:
            return None
        return SwingValue(
            swing_high=latest_high[1],
            swing_high_age=float(current - latest_high[0]),
            swing_low=latest_low[1],
            swing_low_age=float(current - latest_low[0]),
        )


class PivotMethod(StrEnum):
    """Formula used to place pivot support and resistance levels."""

    CLASSIC = "classic"
    FIBONACCI = "fibonacci"


class PivotLevels(NamedTuple):
    """A pivot point and its three support and resistance levels.

    Attributes:
        pivot: The central pivot.
        r1: First resistance.
        r2: Second resistance.
        r3: Third resistance.
        s1: First support.
        s2: Second support.
        s3: Third support.
    """

    pivot: float
    r1: float
    r2: float
    r3: float
    s1: float
    s2: float
    s3: float


def pivot_levels(high: float, low: float, close: float, method: PivotMethod) -> PivotLevels:
    """Derive pivot levels from one completed period's high, low and close.

    Args:
        high: Period high.
        low: Period low.
        close: Period close.
        method: Which formula to apply.

    Returns:
        The pivot and its six derived levels.

    Raises:
        ValueError: If ``method`` is not a known pivot method.
    """
    pivot = (high + low + close) / 3
    span = high - low
    if method is PivotMethod.CLASSIC:
        return PivotLevels(
            pivot=pivot,
            r1=2 * pivot - low,
            r2=pivot + span,
            r3=high + 2 * (pivot - low),
            s1=2 * pivot - high,
            s2=pivot - span,
            s3=low - 2 * (high - pivot),
        )
    if method is PivotMethod.FIBONACCI:
        return PivotLevels(
            pivot=pivot,
            r1=pivot + 0.382 * span,
            r2=pivot + 0.618 * span,
            r3=pivot + span,
            s1=pivot - 0.382 * span,
            s2=pivot - 0.618 * span,
            s3=pivot - span,
        )
    raise ValueError(f"unknown pivot method {method!r}")


@dataclass(frozen=True)
class PivotPoints(MultiExpressionIndicator[PivotLevels]):
    """Pivot levels derived from the *previous* bar.

    Levels are computed from the last completed bar and published on the bar
    they apply to, which is the only causal reading of a pivot: the day's levels
    are known at the day's open because yesterday has closed. Run this on a D1
    frame for classic daily pivots, or on any other timeframe for its own.

    Attributes:
        method: Which formula to apply.
    """

    outputs: ClassVar[tuple[str, ...]] = ("pivot", "r1", "r2", "r3", "s1", "s2", "s3")

    method: PivotMethod = PivotMethod.CLASSIC

    @property
    def name(self) -> str:
        """Identifier, e.g. ``pivots_classic``."""
        return f"pivots_{self.method.value}"

    @property
    def warmup(self) -> int:
        """The first bar has no completed predecessor to measure."""
        return 1

    def _expressions(self) -> dict[str, pl.Expr]:
        """Apply the pivot formula to the previous bar's high, low and close."""
        high = pl.col("high").shift(1)
        low = pl.col("low").shift(1)
        close = pl.col("close").shift(1)
        pivot = (high + low + close) / 3
        span = high - low
        if self.method is PivotMethod.CLASSIC:
            return {
                "pivot": pivot,
                "r1": 2 * pivot - low,
                "r2": pivot + span,
                "r3": high + 2 * (pivot - low),
                "s1": 2 * pivot - high,
                "s2": pivot - span,
                "s3": low - 2 * (high - pivot),
            }
        return {
            "pivot": pivot,
            "r1": pivot + 0.382 * span,
            "r2": pivot + 0.618 * span,
            "r3": pivot + span,
            "s1": pivot - 0.382 * span,
            "s2": pivot - 0.618 * span,
            "s3": pivot - span,
        }

    def streaming(self) -> "PivotPointsStream":
        """Build the incremental counterpart."""
        return PivotPointsStream(self)


class PivotPointsStream(BaseStreaming[PivotPoints, PivotLevels]):
    """Incremental :class:`PivotPoints`."""

    def reset(self) -> None:
        """Forget the previous bar."""
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
    ) -> PivotLevels | None:
        """Publish levels from the previous bar, then remember this one."""
        previous = self._previous
        self._previous = (high, low, close)
        if previous is None:
            return None
        return pivot_levels(*previous, self._indicator.method)


class StructureLabel(IntEnum):
    """Classification of the most recent confirmed pivot.

    Encoded as an integer so it can travel in a float feature column; read it
    back with ``StructureLabel(int(value))``.
    """

    NONE = 0
    HIGHER_HIGH = 1
    HIGHER_LOW = 2
    LOWER_HIGH = 3
    LOWER_LOW = 4


class StructureValue(NamedTuple):
    """One bar of market-structure output.

    Attributes:
        label: :class:`StructureLabel` code of the most recent pivot event.
        trend: ``1`` while highs and lows are both rising, ``-1`` while both are
            falling, ``0`` when the two disagree or nothing is established yet.
    """

    label: float
    trend: float


@dataclass(frozen=True)
class MarketStructure(RecursiveIndicator[StructureValue]):
    """Higher-high / higher-low / lower-high / lower-low classification.

    Each confirmed pivot is compared against the previous pivot of the same
    kind. The trend reads as up only when the most recent high was higher *and*
    the most recent low was higher — the classical Dow reading — and reverts to
    0 the moment the two disagree, rather than clinging to a stale direction.

    Both channels are 0 until enough pivots exist to classify, which is a value
    rather than a gap: "no structure established" is a real state a strategy
    should be able to see.

    Attributes:
        lookback: Bars required on each side of a pivot.
    """

    outputs: ClassVar[tuple[str, ...]] = ("label", "trend")

    lookback: int = 5

    def __post_init__(self) -> None:
        """Reject non-positive lookbacks."""
        if self.lookback < 1:
            raise ValueError(f"lookback must be positive, got {self.lookback}")

    @property
    def name(self) -> str:
        """Identifier, e.g. ``structure_5``."""
        return f"structure_{self.lookback}"

    @property
    def warmup(self) -> int:
        """A pivot needs ``lookback`` bars either side before it can be confirmed."""
        return 2 * self.lookback

    def streaming(self) -> "MarketStructureStream":
        """Build the incremental counterpart."""
        return MarketStructureStream(self)


class MarketStructureStream(BaseStreaming[MarketStructure, StructureValue]):
    """Incremental :class:`MarketStructure`."""

    def reset(self) -> None:
        """Clear the detector, the last pivot prices and the classifications."""
        self._detector = _FractalDetector(self._indicator.lookback)
        self._last_high: float | None = None
        self._last_low: float | None = None
        self._high_label = StructureLabel.NONE
        self._low_label = StructureLabel.NONE
        self._label = StructureLabel.NONE

    def step(
        self,
        _timestamp: datetime,
        _open_price: float,
        high: float,
        low: float,
        _close: float,
        _volume: float,
        /,
    ) -> StructureValue | None:
        """Classify any pivot this bar confirms, then report the standing structure."""
        ready = self._detector.seen >= self._detector.warmup
        swing_high, swing_low = self._detector.push(high, low)

        if swing_high is not None:
            if self._last_high is not None:
                self._high_label = (
                    StructureLabel.HIGHER_HIGH
                    if swing_high.price > self._last_high
                    else StructureLabel.LOWER_HIGH
                )
                self._label = self._high_label
            self._last_high = swing_high.price
        if swing_low is not None:
            if self._last_low is not None:
                self._low_label = (
                    StructureLabel.HIGHER_LOW
                    if swing_low.price > self._last_low
                    else StructureLabel.LOWER_LOW
                )
                self._label = self._low_label
            self._last_low = swing_low.price

        if not ready:
            return None

        rising = (
            self._high_label is StructureLabel.HIGHER_HIGH
            and self._low_label is StructureLabel.HIGHER_LOW
        )
        falling = (
            self._high_label is StructureLabel.LOWER_HIGH
            and self._low_label is StructureLabel.LOWER_LOW
        )
        trend = 1.0 if rising else (-1.0 if falling else 0.0)
        return StructureValue(label=float(self._label), trend=trend)


class RangeValue(NamedTuple):
    """One bar of range-detector output.

    Attributes:
        in_range: ``1.0`` when the window is consolidating, ``0.0`` otherwise.
        upper: Top of the window's range.
        lower: Bottom of the window's range.
        width_atr: Range height measured in ATRs.
    """

    in_range: float
    upper: float
    lower: float
    width_atr: float


@dataclass(frozen=True)
class RangeState(MultiExpressionIndicator[RangeValue]):
    """Consolidation detector: is the window's range small for its volatility?

    Measuring the range in ATRs rather than in price makes the threshold
    portable across instruments, but not across timeframes or window lengths — a
    20-bar window spans more ATRs than a 10-bar one whatever the market is
    doing. ``threshold`` therefore has no universal value and belongs in the
    config, not in a strategy.

    Attributes:
        period: Window whose range is measured.
        atr_period: ATR period the range is scaled by.
        threshold: Width in ATRs at or below which the window counts as a range.
    """

    outputs: ClassVar[tuple[str, ...]] = ("in_range", "upper", "lower", "width_atr")

    period: int = 20
    atr_period: int = 14
    threshold: float = 5.0

    def __post_init__(self) -> None:
        """Reject non-positive periods and thresholds."""
        if self.period < 2:
            raise ValueError(f"period must be at least 2, got {self.period}")
        if self.atr_period < 1:
            raise ValueError(f"atr_period must be positive, got {self.atr_period}")
        if self.threshold <= 0:
            raise ValueError(f"threshold must be positive, got {self.threshold}")

    @property
    def name(self) -> str:
        """Identifier, e.g. ``range_20_14_5.0``."""
        return f"range_{self.period}_{self.atr_period}_{self.threshold}"

    @property
    def warmup(self) -> int:
        """Whichever of the range window and the ATR is slower."""
        return max(self.period - 1, self.atr_period)

    def _expressions(self) -> dict[str, pl.Expr]:
        """Window extremes, and their separation measured in ATRs."""
        upper = pl.col("high").rolling_max(self.period)
        lower = pl.col("low").rolling_min(self.period)
        atr = wilder_rma(true_range(), self.atr_period)
        width = safe_divide(upper - lower, atr, 0.0)
        return {
            "in_range": pl.when(width <= self.threshold).then(1.0).otherwise(0.0),
            "upper": upper,
            "lower": lower,
            "width_atr": width,
        }

    def streaming(self) -> "RangeStateStream":
        """Build the incremental counterpart."""
        return RangeStateStream(self)


class RangeStateStream(BaseStreaming[RangeState, RangeValue]):
    """Incremental :class:`RangeState`."""

    def reset(self) -> None:
        """Clear the extremes, the ATR and the previous close."""
        indicator = self._indicator
        self._high = RollingExtreme(indicator.period, largest=True)
        self._low = RollingExtreme(indicator.period, largest=False)
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
    ) -> RangeValue | None:
        """Push one bar and scale the window's range by the current ATR."""
        previous_close = self._previous_close
        self._previous_close = close
        upper = self._high.push(high)
        lower = self._low.push(low)
        if previous_close is None:
            return None
        span = max(high - low, abs(high - previous_close), abs(low - previous_close))
        atr = self._atr.push(span)
        if upper is None or lower is None or atr is None:
            return None
        width = 0.0 if atr == 0 else (upper - lower) / atr
        in_range = 1.0 if width <= self._indicator.threshold else 0.0
        return RangeValue(in_range=in_range, upper=upper, lower=lower, width_atr=width)


@dataclass(frozen=True)
class Level:
    """A price level several confirmed pivots agree on.

    Attributes:
        price: Mean price of the clustered pivots.
        kind: Whether the pivots were highs (resistance) or lows (support).
        touches: How many pivots fell into the cluster.
        first_seen: Open time of the earliest pivot in the cluster.
        last_seen: Open time of the latest pivot in the cluster.
    """

    price: float
    kind: SwingKind
    touches: int
    first_seen: datetime
    last_seen: datetime


def support_resistance_levels(
    frame: OHLCVFrame,
    *,
    lookback: int = 5,
    tolerance: float = 0.001,
    min_touches: int = 2,
) -> list[Level]:
    """Cluster confirmed pivots into support and resistance levels.

    Only pivots the frame actually confirms are considered, so a level cannot
    appear on the strength of a pivot the market had not yet revealed. Clustering
    is a single pass over sorted pivot prices: a pivot joins the current cluster
    while it stays within ``tolerance`` of that cluster's first price, and starts
    a new one otherwise. Highs and lows are clustered separately — a level that
    has only ever been rejected from above is a different object from one only
    ever held from below.

    Args:
        frame: Bars to search.
        lookback: Bars required on each side of a pivot.
        tolerance: Relative price distance within which pivots are the same
            level; ``0.001`` is ten pips on a 1.0000 quote.
        min_touches: Minimum pivots for a cluster to count as a level.

    Returns:
        Levels sorted by price, ascending.

    Raises:
        ValueError: If ``tolerance`` is not positive or ``min_touches`` is below 1.
    """
    if tolerance <= 0:
        raise ValueError(f"tolerance must be positive, got {tolerance}")
    if min_touches < 1:
        raise ValueError(f"min_touches must be at least 1, got {min_touches}")

    highs = frame.df["high"].to_list()
    lows = frame.df["low"].to_list()
    timestamps = frame.df[TIMESTAMP_COLUMN].to_list()
    swings = find_swings(highs, lows, lookback)

    levels: list[Level] = []
    for kind in (SwingKind.HIGH, SwingKind.LOW):
        selected = sorted(
            (swing for swing in swings if swing.kind is kind), key=lambda swing: swing.price
        )
        levels.extend(_cluster(selected, kind, tolerance, min_touches, timestamps))
    return sorted(levels, key=lambda level: level.price)


def _cluster(
    swings: Sequence[Swing],
    kind: SwingKind,
    tolerance: float,
    min_touches: int,
    timestamps: Sequence[datetime],
) -> list[Level]:
    """Group price-sorted pivots of one kind into levels."""
    levels: list[Level] = []
    current: list[Swing] = []

    def flush() -> None:
        if len(current) < min_touches:
            return
        indices = [swing.index for swing in current]
        levels.append(
            Level(
                price=sum(swing.price for swing in current) / len(current),
                kind=kind,
                touches=len(current),
                first_seen=timestamps[min(indices)],
                last_seen=timestamps[max(indices)],
            )
        )

    for swing in swings:
        if current and abs(swing.price - current[0].price) > tolerance * abs(current[0].price):
            flush()
            current = []
        current.append(swing)
    flush()
    return levels
