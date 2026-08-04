"""Market structure: pivots reported when confirmed, never when they happened."""

import pytest

from trading_system.data.models import OHLCVFrame
from trading_system.features.indicators.structure import (
    MarketStructure,
    PivotMethod,
    PivotPoints,
    RangeState,
    StructureLabel,
    SwingKind,
    SwingPoints,
    find_swings,
    pivot_levels,
    support_resistance_levels,
)

from .conftest import frame_from_bars, frame_from_closes

#: A clean zigzag: one pivot high at index 4, one pivot low at index 8.
ZIGZAG = [1.0, 2.0, 3.0, 4.0, 5.0, 4.0, 3.0, 2.0, 1.0, 2.0, 3.0, 4.0, 5.0]

#: Rising peaks and rising troughs, for the higher-high / higher-low reading.
RISING_ZIGZAG = [1.0, 3.0, 5.0, 3.0, 2.0, 4.0, 6.0, 8.0, 6.0, 4.0, 6.0, 8.0, 10.0]


def test_find_swings_locates_both_pivots() -> None:
    swings = find_swings(ZIGZAG, ZIGZAG, 2)
    assert [(swing.index, swing.kind, swing.price) for swing in swings] == [
        (4, SwingKind.HIGH, 5.0),
        (8, SwingKind.LOW, 1.0),
    ]


def test_a_pivot_is_confirmed_lookback_bars_after_it_occurred() -> None:
    """The whole no-lookahead story for structure, in one assertion."""
    swings = find_swings(ZIGZAG, ZIGZAG, 2)
    assert swings[0].index == 4
    assert swings[0].confirmed_at == 6


def test_a_pivot_is_invisible_until_its_confirming_bar_closes() -> None:
    """Six bars of history cannot know about the peak at bar 4; seven can."""
    assert find_swings(ZIGZAG[:6], ZIGZAG[:6], 2) == []
    assert len(find_swings(ZIGZAG[:7], ZIGZAG[:7], 2)) == 1


def test_a_plateau_is_not_a_pivot() -> None:
    """Two bars sharing the high do not say which one the market turned on."""
    plateau = [1.0, 2.0, 5.0, 5.0, 2.0, 1.0, 0.0]
    assert find_swings(plateau, plateau, 2) == []


def test_find_swings_rejects_mismatched_series() -> None:
    with pytest.raises(ValueError, match="length"):
        find_swings([1.0, 2.0], [1.0], 1)


def test_swing_points_carry_the_confirmed_pivot_forward() -> None:
    frame = frame_from_closes(ZIGZAG)
    computed = SwingPoints(lookback=2).compute_frame(frame)
    # The pivot high at bar 4 is published from its confirming bar 6 onwards.
    assert computed["swing_high"].item(6) == pytest.approx(5.0)
    assert computed["swing_high_age"].item(6) == pytest.approx(2.0)
    assert computed["swing_high_age"].item(7) == pytest.approx(3.0)


def test_swing_points_age_is_never_below_the_lookback() -> None:
    frame = frame_from_closes(ZIGZAG)
    computed = SwingPoints(lookback=2).compute_frame(frame).drop_nulls()
    assert min(computed["swing_high_age"].to_list()) >= 2.0
    assert min(computed["swing_low_age"].to_list()) >= 2.0


def test_swing_points_fall_back_to_the_confirmed_extreme_on_a_monotonic_run() -> None:
    """A straight line never forms a fractal, but resistance is still knowable."""
    closes = [float(i) for i in range(30)]
    frame = frame_from_closes(closes)
    indicator = SwingPoints(lookback=3)
    computed = indicator.compute_frame(frame)
    index = 20
    # The confirmed prefix ends `lookback` bars back.
    assert computed["swing_high"].item(index) == pytest.approx(closes[index - 3])
    assert computed["swing_low"].item(index) == pytest.approx(closes[0])


def test_pivot_levels_classic_formula() -> None:
    levels = pivot_levels(110.0, 90.0, 100.0, PivotMethod.CLASSIC)
    assert levels.pivot == pytest.approx(100.0)
    assert levels.r1 == pytest.approx(110.0)
    assert levels.s1 == pytest.approx(90.0)
    assert levels.r2 == pytest.approx(120.0)
    assert levels.s2 == pytest.approx(80.0)
    assert levels.r3 == pytest.approx(130.0)
    assert levels.s3 == pytest.approx(70.0)


def test_pivot_levels_fibonacci_formula() -> None:
    levels = pivot_levels(110.0, 90.0, 100.0, PivotMethod.FIBONACCI)
    assert levels.pivot == pytest.approx(100.0)
    assert levels.r1 == pytest.approx(107.64)
    assert levels.r2 == pytest.approx(112.36)
    assert levels.r3 == pytest.approx(120.0)
    assert levels.s1 == pytest.approx(92.36)
    assert levels.s2 == pytest.approx(87.64)
    assert levels.s3 == pytest.approx(80.0)


def test_pivot_points_are_derived_from_the_previous_bar() -> None:
    """Today's levels are known at today's open because yesterday has closed."""
    rows = [(100.0, 110.0, 90.0, 100.0, 1.0), (100.0, 101.0, 99.0, 100.0, 1.0)]
    computed = PivotPoints().compute_frame(frame_from_bars(rows))
    assert computed["pivot"].item(0) is None
    assert computed["pivot"].item(1) == pytest.approx(100.0)
    assert computed["r1"].item(1) == pytest.approx(110.0)


def test_market_structure_reads_rising_peaks_and_troughs_as_an_uptrend() -> None:
    frame = frame_from_closes(RISING_ZIGZAG)
    computed = MarketStructure(lookback=2).compute_frame(frame)
    # The higher low at bar 9 is confirmed at bar 11 and completes the reading.
    assert computed["label"].item(11) == pytest.approx(float(StructureLabel.HIGHER_LOW))
    assert computed["trend"].item(11) == pytest.approx(1.0)


def test_market_structure_reports_no_structure_before_it_has_pivots() -> None:
    """A zero label is a state, not a gap: "nothing established yet" is real."""
    frame = frame_from_closes(RISING_ZIGZAG)
    indicator = MarketStructure(lookback=2)
    computed = indicator.compute_frame(frame)
    assert computed["label"].item(indicator.warmup) == pytest.approx(float(StructureLabel.NONE))
    assert computed["trend"].item(indicator.warmup) == pytest.approx(0.0)


def test_market_structure_labels_are_readable_as_the_enum(
    reference_frame: OHLCVFrame,
) -> None:
    computed = MarketStructure().compute_frame(reference_frame).drop_nulls()
    labels = {StructureLabel(int(value)) for value in computed["label"].to_list()}
    assert labels <= set(StructureLabel)


def test_range_state_calls_a_flat_market_a_range() -> None:
    frame = frame_from_closes([5.0] * 40)
    computed = RangeState(period=20, atr_period=14).compute_frame(frame).drop_nulls()
    assert computed["in_range"].to_list() == pytest.approx([1.0] * computed.height)
    assert computed["width_atr"].to_list() == pytest.approx([0.0] * computed.height)


def test_range_state_rejects_a_trend_at_a_tight_threshold() -> None:
    """A clean ramp covers many ATRs of ground in a window, so it is not a range."""
    closes = [float(i) for i in range(60)]
    frame = frame_from_closes(closes)
    computed = RangeState(period=20, atr_period=14, threshold=5.0).compute_frame(frame).drop_nulls()
    assert computed["in_range"].to_list() == pytest.approx([0.0] * computed.height)
    assert min(computed["width_atr"].to_list()) > 5.0


def test_range_state_bounds_come_from_the_window() -> None:
    frame = frame_from_closes([float(i % 7) + 1 for i in range(60)])
    indicator = RangeState(period=20, atr_period=14)
    computed = indicator.compute_frame(frame).drop_nulls()
    assert max(computed["upper"].to_list()) == pytest.approx(7.0)
    assert min(computed["lower"].to_list()) == pytest.approx(1.0)


def test_support_resistance_clusters_repeated_pivots() -> None:
    """Three rejections from the same shelf are one level, not three."""
    shelf = []
    for _ in range(3):
        shelf.extend([100.0, 101.0, 105.0, 101.0, 100.0])
    frame = frame_from_closes(shelf)
    levels = support_resistance_levels(frame, lookback=2, tolerance=0.01, min_touches=2)
    resistance = [level for level in levels if level.kind is SwingKind.HIGH]
    assert len(resistance) == 1
    assert resistance[0].price == pytest.approx(105.0)
    assert resistance[0].touches == 3
    assert resistance[0].first_seen < resistance[0].last_seen


def test_support_resistance_drops_clusters_below_the_touch_threshold() -> None:
    frame = frame_from_closes(ZIGZAG)
    assert support_resistance_levels(frame, lookback=2, min_touches=2) == []


@pytest.mark.parametrize(("tolerance", "min_touches"), [(0.0, 2), (-0.1, 2), (0.001, 0)])
def test_support_resistance_rejects_impossible_settings(
    tolerance: float, min_touches: int, reference_frame: OHLCVFrame
) -> None:
    with pytest.raises(ValueError):  # noqa: PT011 - each case has its own message
        support_resistance_levels(reference_frame, tolerance=tolerance, min_touches=min_touches)


def test_structure_indicators_reject_a_non_positive_lookback() -> None:
    with pytest.raises(ValueError, match="positive"):
        SwingPoints(lookback=0)
    with pytest.raises(ValueError, match="positive"):
        MarketStructure(lookback=0)
