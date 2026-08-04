"""Candlestick predicates and divergence, on candles written by hand."""

from datetime import UTC, datetime, timedelta

import pytest

from trading_system.core.types import Bar, Price, Side, Volume
from trading_system.features.patterns import (
    Divergence,
    DivergenceKind,
    Pattern,
    detect_patterns,
    divergence,
    is_bearish_engulfing,
    is_bullish_engulfing,
    is_doji,
    is_evening_star,
    is_hammer,
    is_inside_bar,
    is_morning_star,
    is_outside_bar,
    is_shooting_star,
    patterns_at,
)

from .conftest import frame_from_bars, frame_from_closes

_START = datetime(2024, 1, 2, 21, 0, tzinfo=UTC)


def bar(open_price: float, high: float, low: float, close: float, *, minute: int = 0) -> Bar:
    """Build one bar, with the OHLC relationships the domain type enforces."""
    return Bar(
        timestamp=_START + timedelta(minutes=minute),
        open=Price(open_price),
        high=Price(high),
        low=Price(low),
        close=Price(close),
        volume=Volume(100.0),
    )


def test_doji_needs_a_body_small_against_its_own_range() -> None:
    assert is_doji(bar(10.0, 11.0, 9.0, 10.05))
    assert not is_doji(bar(10.0, 11.0, 9.0, 10.9))


def test_a_bar_with_no_range_is_not_a_doji() -> None:
    """Zero range is a bar with no information, not a bar expressing indecision."""
    assert not is_doji(bar(10.0, 10.0, 10.0, 10.0))


def test_inside_bar_requires_strict_containment_on_both_sides() -> None:
    previous = bar(10.0, 12.0, 8.0, 11.0)
    assert is_inside_bar(previous, bar(10.5, 11.5, 9.0, 10.0, minute=1))
    # Matching the previous high is a test of it, not compression away from it.
    assert not is_inside_bar(previous, bar(10.5, 12.0, 9.0, 10.0, minute=1))


def test_outside_bar_requires_covering_both_extremes() -> None:
    previous = bar(10.0, 12.0, 8.0, 11.0)
    assert is_outside_bar(previous, bar(10.5, 13.0, 7.0, 12.0, minute=1))
    assert not is_outside_bar(previous, bar(10.5, 13.0, 9.0, 12.0, minute=1))


def test_bullish_engulfing_fires_without_a_gap() -> None:
    """Spot FX opens where it closed; requiring a gap would disable the pattern."""
    previous = bar(12.0, 12.5, 9.5, 10.0)
    current = bar(10.0, 13.5, 9.8, 13.0, minute=1)
    assert is_bullish_engulfing(previous, current)


def test_bullish_engulfing_needs_the_bodies_to_actually_cover() -> None:
    previous = bar(12.0, 12.5, 9.5, 10.0)
    assert not is_bullish_engulfing(previous, bar(10.0, 11.5, 9.8, 11.0, minute=1))
    # An up bar following an up bar is not an engulfing of anything.
    assert not is_bullish_engulfing(bar(10.0, 12.5, 9.5, 12.0), current_up())


def current_up() -> Bar:
    """A bullish bar for the negative engulfing cases."""
    return bar(12.0, 14.0, 11.5, 13.5, minute=1)


def test_bearish_engulfing_mirrors_the_bullish_case() -> None:
    previous = bar(10.0, 12.5, 9.5, 12.0)
    current = bar(12.0, 12.5, 8.5, 9.0, minute=1)
    assert is_bearish_engulfing(previous, current)
    assert not is_bearish_engulfing(previous, bar(12.0, 12.5, 10.5, 11.0, minute=1))


def test_hammer_needs_a_dominant_lower_wick() -> None:
    assert is_hammer(bar(10.0, 10.2, 7.0, 10.1))
    assert not is_hammer(bar(10.0, 13.0, 9.8, 10.1))


def test_shooting_star_needs_a_dominant_upper_wick() -> None:
    assert is_shooting_star(bar(10.0, 13.0, 9.9, 10.1))
    assert not is_shooting_star(bar(10.0, 10.2, 7.0, 10.1))


def test_pin_bar_thresholds_are_arguments_not_constants() -> None:
    """A "long wick" means different things per instrument, so it is configurable."""
    candle = bar(10.0, 10.4, 9.0, 10.2)
    assert is_hammer(candle, min_wick_to_range=0.5)
    assert not is_hammer(candle, min_wick_to_range=0.9)


def test_morning_star_is_a_down_bar_a_pause_and_a_recovery() -> None:
    first = bar(12.0, 12.2, 9.8, 10.0)
    second = bar(10.0, 10.1, 9.5, 9.8, minute=1)
    third = bar(9.8, 11.6, 9.7, 11.5, minute=2)
    assert is_morning_star(first, second, third)


def test_morning_star_needs_the_third_bar_past_the_midpoint() -> None:
    first = bar(12.0, 12.2, 9.8, 10.0)
    second = bar(10.0, 10.1, 9.5, 9.8, minute=1)
    weak = bar(9.8, 10.6, 9.7, 10.5, minute=2)
    assert not is_morning_star(first, second, weak)


def test_evening_star_mirrors_the_morning_star() -> None:
    first = bar(10.0, 12.2, 9.8, 12.0)
    second = bar(12.0, 12.5, 11.9, 12.2, minute=1)
    third = bar(12.2, 12.3, 10.4, 10.5, minute=2)
    assert is_evening_star(first, second, third)


def test_a_bodyless_first_bar_forms_no_star() -> None:
    flat = bar(10.0, 10.0, 10.0, 10.0)
    assert not is_morning_star(flat, flat, bar(10.0, 11.0, 10.0, 11.0, minute=2))


def test_patterns_at_reports_everything_completing_on_the_last_bar() -> None:
    window = [
        bar(12.0, 12.2, 9.8, 10.0),
        bar(10.0, 10.1, 9.5, 9.8, minute=1),
        bar(9.8, 11.6, 9.7, 11.5, minute=2),
    ]
    found = patterns_at(window)
    assert Pattern.MORNING_STAR in found
    assert Pattern.BULLISH_ENGULFING in found


def test_patterns_at_rejects_an_empty_window() -> None:
    with pytest.raises(ValueError, match="at least one bar"):
        patterns_at([])


def test_detect_patterns_leaves_unevaluable_bars_null() -> None:
    """A null means "could not look", which is not "looked and found nothing"."""
    frame = frame_from_closes([1.0, 1.1, 1.2, 1.3])
    detected = detect_patterns(frame)
    assert detected[Pattern.DOJI.value].item(0) is not None
    assert detected[Pattern.INSIDE_BAR.value].item(0) is None
    assert detected[Pattern.INSIDE_BAR.value].item(1) is not None
    assert detected[Pattern.MORNING_STAR.value].to_list()[:2] == [None, None]
    assert detected[Pattern.MORNING_STAR.value].item(2) is not None


def test_detect_patterns_agrees_with_the_predicates_it_calls() -> None:
    rows = [
        (12.0, 12.2, 9.8, 10.0, 1.0),
        (10.0, 10.1, 9.5, 9.8, 1.0),
        (9.8, 11.6, 9.7, 11.5, 1.0),
    ]
    detected = detect_patterns(frame_from_bars(rows))
    assert detected[Pattern.MORNING_STAR.value].item(2)
    assert detected.height == 3
    assert list(detected.columns) == [pattern.value for pattern in Pattern]


def sawtooth(peaks: list[float], troughs: list[float]) -> list[float]:
    """Weave peaks and troughs into a series with clean fractal pivots."""
    series: list[float] = [troughs[0] - 1.0, troughs[0]]
    for peak, trough in zip(peaks, troughs[1:] + [troughs[-1]], strict=True):
        series.extend([peak - 1.0, peak, peak - 1.0, trough + 1.0, trough, trough + 1.0])
    return series


def test_regular_bearish_divergence_is_higher_price_lower_oscillator() -> None:
    price = sawtooth([10.0, 12.0], [4.0, 4.0])
    oscillator = sawtooth([10.0, 8.0], [4.0, 4.0])
    found = divergence(price, oscillator, 2)
    assert isinstance(found, Divergence)
    assert found.kind is DivergenceKind.REGULAR
    assert found.direction is Side.SELL
    assert found.right.price > found.left.price
    assert found.oscillator_right < found.oscillator_left


def test_hidden_bearish_divergence_is_lower_price_higher_oscillator() -> None:
    price = sawtooth([12.0, 10.0], [4.0, 4.0])
    oscillator = sawtooth([8.0, 10.0], [4.0, 4.0])
    found = divergence(price, oscillator, 2)
    assert isinstance(found, Divergence)
    assert found.kind is DivergenceKind.HIDDEN
    assert found.direction is Side.SELL


def test_no_divergence_when_price_and_oscillator_agree() -> None:
    price = sawtooth([10.0, 12.0], [4.0, 4.0])
    assert divergence(price, price, 2) is None


def test_divergence_is_only_reported_once_its_pivot_is_confirmed() -> None:
    """The honest confirmation rule: nothing before the pivot's own lookback closes."""
    price = sawtooth([10.0, 12.0], [4.0, 4.0])
    oscillator = sawtooth([10.0, 8.0], [4.0, 4.0])
    found = divergence(price, oscillator, 2)
    assert found is not None
    confirmed_at = found.confirmed_at
    assert divergence(price[:confirmed_at], oscillator[:confirmed_at], 2) != found
    assert divergence(price[: confirmed_at + 1], oscillator[: confirmed_at + 1], 2) == found


def test_divergence_can_reject_pivots_too_far_apart() -> None:
    price = sawtooth([10.0, 12.0], [4.0, 4.0])
    oscillator = sawtooth([10.0, 8.0], [4.0, 4.0])
    assert divergence(price, oscillator, 2, max_pivot_distance=2) is None


def test_divergence_validates_its_inputs() -> None:
    with pytest.raises(ValueError, match="length"):
        divergence([1.0, 2.0], [1.0], 2)
    with pytest.raises(ValueError, match="max_pivot_distance"):
        divergence([1.0, 2.0], [1.0, 2.0], 2, max_pivot_distance=0)
