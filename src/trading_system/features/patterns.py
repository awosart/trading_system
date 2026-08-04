"""Candlestick patterns and divergence, as pure functions over a window of bars.

The predicates take :class:`~trading_system.core.types.Bar` objects and return
booleans. They hold no state, read no configuration, and look at nothing outside
the window handed to them, which makes each one directly testable against a
hand-written candle and impossible to accidentally feed future data.

Thresholds are arguments with documented defaults rather than constants. "Small
body" means something different on a 1-minute FX bar and a daily index bar, and
a pattern library that hard-codes the answer is silently wrong on one of them.

:func:`detect_patterns` vectorises nothing: it walks the frame calling the same
predicates. One definition of each pattern is worth more than a fast second one
that disagrees with it at the edges.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum

import polars as pl

from trading_system.core.types import Bar, Side
from trading_system.data.models import OHLCVFrame
from trading_system.features.base import iter_bars
from trading_system.features.indicators.structure import Swing, SwingKind, find_swings


class Pattern(StrEnum):
    """A recognised candlestick formation."""

    DOJI = "DOJI"
    INSIDE_BAR = "INSIDE_BAR"
    OUTSIDE_BAR = "OUTSIDE_BAR"
    BULLISH_ENGULFING = "BULLISH_ENGULFING"
    BEARISH_ENGULFING = "BEARISH_ENGULFING"
    HAMMER = "HAMMER"
    SHOOTING_STAR = "SHOOTING_STAR"
    MORNING_STAR = "MORNING_STAR"
    EVENING_STAR = "EVENING_STAR"


#: How many bars each pattern needs, counting the bar it is reported on.
PATTERN_WINDOW: dict[Pattern, int] = {
    Pattern.DOJI: 1,
    Pattern.INSIDE_BAR: 2,
    Pattern.OUTSIDE_BAR: 2,
    Pattern.BULLISH_ENGULFING: 2,
    Pattern.BEARISH_ENGULFING: 2,
    Pattern.HAMMER: 1,
    Pattern.SHOOTING_STAR: 1,
    Pattern.MORNING_STAR: 3,
    Pattern.EVENING_STAR: 3,
}


def body(bar: Bar) -> float:
    """Absolute distance between open and close."""
    return abs(bar.close - bar.open)


def bar_range(bar: Bar) -> float:
    """Distance between high and low."""
    return bar.high - bar.low


def upper_wick(bar: Bar) -> float:
    """Distance from the high to the top of the body."""
    return bar.high - max(bar.open, bar.close)


def lower_wick(bar: Bar) -> float:
    """Distance from the bottom of the body to the low."""
    return min(bar.open, bar.close) - bar.low


def is_bullish(bar: Bar) -> bool:
    """Whether the bar closed above its open."""
    return bar.close > bar.open


def is_bearish(bar: Bar) -> bool:
    """Whether the bar closed below its open."""
    return bar.close < bar.open


def is_doji(bar: Bar, *, max_body_ratio: float = 0.1) -> bool:
    """Whether the bar has almost no body.

    Args:
        bar: Bar to test.
        max_body_ratio: Largest body, as a fraction of the bar's range, still
            counted as a doji.

    Returns:
        ``True`` for an indecision candle. A bar with no range at all is not a
        doji — it is a bar with no information.
    """
    span = bar_range(bar)
    return span > 0 and body(bar) <= max_body_ratio * span


def is_inside_bar(previous: Bar, current: Bar) -> bool:
    """Whether ``current`` sits strictly inside ``previous``.

    Strict on both sides: a bar matching the previous high exactly has tested it
    rather than failed to reach it, and treating that as compression is what
    turns an inside-bar breakout system into a coin flip.

    Args:
        previous: The prior bar.
        current: The bar being classified.

    Returns:
        ``True`` when the current range is contained in the previous one.
    """
    return current.high < previous.high and current.low > previous.low


def is_outside_bar(previous: Bar, current: Bar) -> bool:
    """Whether ``current`` strictly contains ``previous``.

    Args:
        previous: The prior bar.
        current: The bar being classified.

    Returns:
        ``True`` when the current range covers the previous one on both sides.
    """
    return current.high > previous.high and current.low < previous.low


def is_bullish_engulfing(previous: Bar, current: Bar) -> bool:
    """Whether a down bar is engulfed by the up bar that follows it.

    Bodies are compared, not ranges: the wicks record where price was rejected,
    the bodies record where it settled.

    Containment is non-strict on each side but must be strict on at least one.
    Spot FX is effectively gapless — a bar opens where the last one closed — so
    demanding ``open < previous.close`` would mean this pattern never fires
    outside of weekend gaps, which is a definition that quietly disables itself
    on the instrument class this system trades.

    Args:
        previous: The prior bar, which must be bearish.
        current: The bar being classified, which must be bullish.

    Returns:
        ``True`` for a bullish engulfing pair.
    """
    if not (is_bearish(previous) and is_bullish(current)):
        return False
    covers = current.open <= previous.close and current.close >= previous.open
    return covers and (current.open < previous.close or current.close > previous.open)


def is_bearish_engulfing(previous: Bar, current: Bar) -> bool:
    """Whether an up bar is engulfed by the down bar that follows it.

    Mirror of :func:`is_bullish_engulfing`, including its gapless-market
    tolerance.

    Args:
        previous: The prior bar, which must be bullish.
        current: The bar being classified, which must be bearish.

    Returns:
        ``True`` for a bearish engulfing pair.
    """
    if not (is_bullish(previous) and is_bearish(current)):
        return False
    covers = current.open >= previous.close and current.close <= previous.open
    return covers and (current.open > previous.close or current.close < previous.open)


def is_hammer(
    bar: Bar,
    *,
    min_wick_to_body: float = 2.0,
    min_wick_to_range: float = 0.5,
    max_opposite_wick_to_range: float = 0.3,
) -> bool:
    """Whether the bar is a bullish pin bar: a long lower wick, little else.

    Args:
        bar: Bar to test.
        min_wick_to_body: Smallest lower-wick-to-body ratio accepted.
        min_wick_to_range: Smallest share of the bar's range the lower wick must
            occupy.
        max_opposite_wick_to_range: Largest share the upper wick may occupy.

    Returns:
        ``True`` for a hammer. A zero-body bar passes the ratio test trivially,
        so the share-of-range tests carry the decision there.
    """
    span = bar_range(bar)
    if span <= 0:
        return False
    lower = lower_wick(bar)
    return (
        lower >= min_wick_to_body * body(bar)
        and lower >= min_wick_to_range * span
        and upper_wick(bar) <= max_opposite_wick_to_range * span
    )


def is_shooting_star(
    bar: Bar,
    *,
    min_wick_to_body: float = 2.0,
    min_wick_to_range: float = 0.5,
    max_opposite_wick_to_range: float = 0.3,
) -> bool:
    """Whether the bar is a bearish pin bar: a long upper wick, little else.

    Args:
        bar: Bar to test.
        min_wick_to_body: Smallest upper-wick-to-body ratio accepted.
        min_wick_to_range: Smallest share of the bar's range the upper wick must
            occupy.
        max_opposite_wick_to_range: Largest share the lower wick may occupy.

    Returns:
        ``True`` for a shooting star.
    """
    span = bar_range(bar)
    if span <= 0:
        return False
    upper = upper_wick(bar)
    return (
        upper >= min_wick_to_body * body(bar)
        and upper >= min_wick_to_range * span
        and lower_wick(bar) <= max_opposite_wick_to_range * span
    )


def is_morning_star(
    first: Bar, second: Bar, third: Bar, *, max_star_body_ratio: float = 0.5
) -> bool:
    """Whether three bars form a bullish reversal: down, pause, up.

    Args:
        first: The bar that established the downtrend leg; must be bearish.
        second: The star — a small body at or below the first bar's close. The
            bound is non-strict for the same gapless-market reason as
            :func:`is_bullish_engulfing`.
        third: The reversal bar; must be bullish and recover past the midpoint
            of the first bar's body.
        max_star_body_ratio: Largest star body, relative to the first bar's
            body, still counted as a pause.

    Returns:
        ``True`` for a morning star.
    """
    first_body = body(first)
    if first_body <= 0 or not is_bearish(first) or not is_bullish(third):
        return False
    star_top = max(second.open, second.close)
    midpoint = (first.open + first.close) / 2
    return (
        body(second) <= max_star_body_ratio * first_body
        and star_top <= first.close
        and third.close > midpoint
    )


def is_evening_star(
    first: Bar, second: Bar, third: Bar, *, max_star_body_ratio: float = 0.5
) -> bool:
    """Whether three bars form a bearish reversal: up, pause, down.

    Args:
        first: The bar that established the uptrend leg; must be bullish.
        second: The star — a small body at or above the first bar's close.
        third: The reversal bar; must be bearish and give back past the midpoint
            of the first bar's body.
        max_star_body_ratio: Largest star body, relative to the first bar's
            body, still counted as a pause.

    Returns:
        ``True`` for an evening star.
    """
    first_body = body(first)
    if first_body <= 0 or not is_bullish(first) or not is_bearish(third):
        return False
    star_bottom = min(second.open, second.close)
    midpoint = (first.open + first.close) / 2
    return (
        body(second) <= max_star_body_ratio * first_body
        and star_bottom >= first.close
        and third.close < midpoint
    )


def patterns_at(window: Sequence[Bar]) -> frozenset[Pattern]:
    """Every pattern completing on the last bar of ``window``.

    Args:
        window: Bars in chronological order. Patterns needing more bars than are
            supplied are simply not tested.

    Returns:
        The patterns that fire on ``window[-1]``.

    Raises:
        ValueError: If ``window`` is empty.
    """
    if not window:
        raise ValueError("window must contain at least one bar")

    found: set[Pattern] = set()
    current = window[-1]
    if is_doji(current):
        found.add(Pattern.DOJI)
    if is_hammer(current):
        found.add(Pattern.HAMMER)
    if is_shooting_star(current):
        found.add(Pattern.SHOOTING_STAR)

    if len(window) >= 2:
        previous = window[-2]
        if is_inside_bar(previous, current):
            found.add(Pattern.INSIDE_BAR)
        if is_outside_bar(previous, current):
            found.add(Pattern.OUTSIDE_BAR)
        if is_bullish_engulfing(previous, current):
            found.add(Pattern.BULLISH_ENGULFING)
        if is_bearish_engulfing(previous, current):
            found.add(Pattern.BEARISH_ENGULFING)

    if len(window) >= 3:
        first, second = window[-3], window[-2]
        if is_morning_star(first, second, current):
            found.add(Pattern.MORNING_STAR)
        if is_evening_star(first, second, current):
            found.add(Pattern.EVENING_STAR)

    return frozenset(found)


def detect_patterns(frame: OHLCVFrame) -> pl.DataFrame:
    """Evaluate every pattern on every bar of a frame.

    Walks the frame calling the predicates above rather than re-expressing them
    as polars expressions: one definition per pattern cannot drift from itself.
    The cost is roughly a microsecond per bar, which is irrelevant next to the
    indicator pipeline and would not be worth a second implementation.

    Args:
        frame: Bars to classify.

    Returns:
        One Boolean column per :class:`Pattern`, in enum order, with one row per
        bar. Bars too early in the frame to have the pattern's full window are
        null rather than ``False`` — "could not look" is not "looked and found
        nothing".
    """
    columns: dict[str, list[bool | None]] = {pattern.value: [] for pattern in Pattern}
    window: list[Bar] = []
    for index, bar in enumerate(iter_bars(frame)):
        window.append(bar)
        if len(window) > 3:
            window.pop(0)
        found = patterns_at(window)
        for pattern in Pattern:
            available = index + 1 >= PATTERN_WINDOW[pattern]
            columns[pattern.value].append(pattern in found if available else None)
    return pl.DataFrame(
        {name: pl.Series(name, values, dtype=pl.Boolean) for name, values in columns.items()}
    )


class DivergenceKind(StrEnum):
    """Whether a divergence argues for reversal or for continuation."""

    REGULAR = "REGULAR"
    HIDDEN = "HIDDEN"


@dataclass(frozen=True)
class Divergence:
    """A disagreement between price pivots and oscillator pivots.

    Attributes:
        kind: Regular divergence warns of reversal; hidden divergence argues the
            existing trend resumes.
        direction: ``BUY`` for a bullish reading, ``SELL`` for a bearish one.
        left: The earlier pivot.
        right: The later pivot.
        oscillator_left: Oscillator value at the earlier pivot.
        oscillator_right: Oscillator value at the later pivot.
        confirmed_at: Bar index at which the later pivot became knowable, and so
            the earliest bar this divergence may be acted on.
    """

    kind: DivergenceKind
    direction: Side
    left: Swing
    right: Swing
    oscillator_left: float
    oscillator_right: float
    confirmed_at: int


def divergence(
    price: Sequence[float],
    oscillator: Sequence[float],
    lookback: int,
    *,
    max_pivot_distance: int | None = None,
) -> Divergence | None:
    """Find the most recently confirmed divergence between price and an oscillator.

    Both pivots are fractal pivots of ``price``, and the later one is only
    considered once ``lookback`` further bars have closed — the same confirmation
    rule the structure indicators use. The result is therefore stable: running
    this over a prefix of the data returns what was knowable at the end of that
    prefix, and appending bars never rewrites it.

    The four readings, in terms of consecutive pivots of the same kind:

    * Regular bullish — price makes a lower low, the oscillator a higher low.
    * Hidden bullish — price makes a higher low, the oscillator a lower low.
    * Regular bearish — price makes a higher high, the oscillator a lower high.
    * Hidden bearish — price makes a lower high, the oscillator a higher high.

    Args:
        price: Price series to find pivots in, oldest first.
        oscillator: Oscillator sampled on the same bars.
        lookback: Bars required on each side of a pivot.
        max_pivot_distance: Reject pairs whose pivots are further apart than
            this many bars. ``None`` accepts any separation, which on a long
            frame will happily pair pivots months apart.

    Returns:
        The divergence confirmed most recently, or ``None`` if the last two
        pivots of each kind agree with the oscillator.

    Raises:
        ValueError: If the two series differ in length, or
            ``max_pivot_distance`` is not positive.
    """
    if len(price) != len(oscillator):
        raise ValueError(
            f"price and oscillator differ in length: {len(price)} vs {len(oscillator)}"
        )
    if max_pivot_distance is not None and max_pivot_distance < 1:
        raise ValueError(f"max_pivot_distance must be positive, got {max_pivot_distance}")

    swings = find_swings(price, price, lookback)
    candidates = [
        found
        for kind in (SwingKind.LOW, SwingKind.HIGH)
        if (found := _divergence_for(kind, swings, oscillator, max_pivot_distance)) is not None
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda item: item.confirmed_at)


def _divergence_for(
    kind: SwingKind,
    swings: Sequence[Swing],
    oscillator: Sequence[float],
    max_pivot_distance: int | None,
) -> Divergence | None:
    """Compare the last two pivots of one kind against the oscillator."""
    same_kind = [swing for swing in swings if swing.kind is kind]
    if len(same_kind) < 2:
        return None
    left, right = same_kind[-2], same_kind[-1]
    if max_pivot_distance is not None and right.index - left.index > max_pivot_distance:
        return None

    left_value = oscillator[left.index]
    right_value = oscillator[right.index]
    price_rose = right.price > left.price
    oscillator_rose = right_value > left_value
    if price_rose == oscillator_rose:
        return None

    if kind is SwingKind.LOW:
        direction = Side.BUY
        kind_of = DivergenceKind.REGULAR if not price_rose else DivergenceKind.HIDDEN
    else:
        direction = Side.SELL
        kind_of = DivergenceKind.REGULAR if price_rose else DivergenceKind.HIDDEN

    return Divergence(
        kind=kind_of,
        direction=direction,
        left=left,
        right=right,
        oscillator_left=left_value,
        oscillator_right=right_value,
        confirmed_at=right.confirmed_at,
    )
