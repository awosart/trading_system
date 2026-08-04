"""Volatility indicators, including the true-range off-by-one that sets their warmup."""

import math

import pytest

from trading_system.data.models import OHLCVFrame
from trading_system.features.indicators.volatility import (
    ATR,
    BollingerBands,
    Choppiness,
    Keltner,
    StdDev,
)

from .conftest import frame_from_bars, frame_from_closes


def test_atr_of_a_constant_range_is_that_range() -> None:
    """Twenty identical bars: every true range is 2.0, so the average is too."""
    rows = [(10.0, 11.0, 9.0, 10.0, 1.0)] * 20
    values = ATR(5).compute(frame_from_bars(rows)).to_list()
    assert values[:5] == [None] * 5
    assert values[5] == pytest.approx(2.0)
    assert values[-1] == pytest.approx(2.0)


def test_atr_warmup_accounts_for_the_missing_first_true_range() -> None:
    """Bar 0 has no previous close, so the Wilder seed lands on bar ``period``."""
    assert ATR(14).warmup == 14


def test_atr_folds_in_the_gap_to_the_previous_close() -> None:
    """A bar that gaps away has a true range larger than its own high-low."""
    rows = [(10.0, 10.5, 9.5, 10.0, 1.0), (20.0, 20.5, 19.5, 20.0, 1.0)]
    values = ATR(1).compute(frame_from_bars(rows)).to_list()
    assert values[1] == pytest.approx(10.5)  # |20.5 - 10.0|, not 20.5 - 19.5


def test_stddev_matches_the_population_formula() -> None:
    closes = [1.0, 2.0, 3.0, 4.0]
    values = StdDev(4).compute(frame_from_closes(closes)).to_list()
    assert values[3] == pytest.approx(math.sqrt(1.25))


def test_stddev_honours_the_sample_convention() -> None:
    closes = [1.0, 2.0, 3.0, 4.0]
    values = StdDev(4, ddof=1).compute(frame_from_closes(closes)).to_list()
    assert values[3] == pytest.approx(math.sqrt(5 / 3))


def test_stddev_rejects_degrees_of_freedom_it_cannot_afford() -> None:
    with pytest.raises(ValueError, match="ddof"):
        StdDev(3, ddof=3)


def test_bollinger_bands_are_symmetric_about_the_mean(reference_frame: OHLCVFrame) -> None:
    computed = BollingerBands().compute_frame(reference_frame).drop_nulls()
    above = (computed["upper"] - computed["middle"]).to_list()
    below = (computed["middle"] - computed["lower"]).to_list()
    assert above == pytest.approx(below)


def test_bollinger_bandwidth_normalises_by_the_middle_band(
    reference_frame: OHLCVFrame,
) -> None:
    computed = BollingerBands().compute_frame(reference_frame).drop_nulls()
    expected = ((computed["upper"] - computed["lower"]) / computed["middle"]).to_list()
    assert computed["bandwidth"].to_list() == pytest.approx(expected)


def test_bollinger_collapses_on_a_flat_market() -> None:
    frame = frame_from_closes([5.0] * 30)
    computed = BollingerBands(20).compute_frame(frame).drop_nulls()
    assert computed["upper"].to_list() == pytest.approx(computed["lower"].to_list())
    assert computed["bandwidth"].to_list() == pytest.approx([0.0] * computed.height)


def test_keltner_bands_sit_a_fixed_number_of_atrs_from_the_centre(
    reference_frame: OHLCVFrame,
) -> None:
    indicator = Keltner(ema_period=20, atr_period=10, multiplier=2.0)
    computed = indicator.compute_frame(reference_frame).drop_nulls()
    atr = ATR(10).compute(reference_frame).to_list()
    offset = len(reference_frame) - computed.height
    widths = (computed["upper"] - computed["middle"]).to_list()
    expected = [2.0 * atr[offset + i] for i in range(computed.height)]
    assert widths == pytest.approx(expected)


def test_choppiness_stays_within_its_scale(reference_frame: OHLCVFrame) -> None:
    values = Choppiness(14).compute(reference_frame).drop_nulls().to_list()
    assert min(values) >= 0.0
    assert max(values) <= 100.0 + 1e-9


def test_choppiness_reports_a_flat_window_as_maximally_choppy() -> None:
    """No net range means nothing to divide by; a market going nowhere is a range."""
    frame = frame_from_closes([3.0] * 40)
    values = Choppiness(14).compute(frame).drop_nulls().to_list()
    assert values == pytest.approx([100.0] * len(values))


def test_choppiness_is_near_zero_on_a_clean_trend() -> None:
    """When every bar's travel is net progress, travel equals range and the log is 0."""
    closes = [float(i) for i in range(1, 41)]
    values = Choppiness(14).compute(frame_from_closes(closes)).drop_nulls().to_list()
    assert max(values) < 5.0


def test_choppiness_rejects_a_period_with_no_logarithmic_scale() -> None:
    with pytest.raises(ValueError, match="at least 2"):
        Choppiness(1)
