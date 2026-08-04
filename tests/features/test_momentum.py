"""Momentum oscillators, with particular attention to their degenerate cases."""

import pytest

from trading_system.data.models import OHLCVFrame
from trading_system.features.indicators.momentum import (
    CCI,
    MFI,
    ROC,
    RSI,
    Stochastic,
    WilliamsR,
)

from .conftest import frame_from_bars, frame_from_closes


def test_rsi_saturates_on_an_unbroken_advance() -> None:
    closes = [float(i) for i in range(1, 40)]
    values = RSI(14).compute(frame_from_closes(closes)).drop_nulls().to_list()
    assert values == pytest.approx([100.0] * len(values))


def test_rsi_bottoms_out_on_an_unbroken_decline() -> None:
    closes = [float(i) for i in range(40, 1, -1)]
    values = RSI(14).compute(frame_from_closes(closes)).drop_nulls().to_list()
    assert values == pytest.approx([0.0] * len(values))


def test_rsi_calls_a_flat_market_neutral_rather_than_overbought() -> None:
    """With no gains and no losses the ratio is 0/0; 100 would be a lie about it."""
    values = RSI(14).compute(frame_from_closes([2.0] * 40)).drop_nulls().to_list()
    assert values == pytest.approx([50.0] * len(values))


def test_rsi_stays_inside_its_range(reference_frame: OHLCVFrame) -> None:
    values = RSI(14).compute(reference_frame).drop_nulls().to_list()
    assert min(values) >= 0.0
    assert max(values) <= 100.0


def test_stochastic_places_the_close_within_the_range() -> None:
    """Close at the top of a 10-wide window reads 100; at the bottom, 0."""
    rows = [(5.0, 10.0, 0.0, 5.0, 1.0)] * 4 + [(5.0, 10.0, 0.0, 10.0, 1.0)]
    computed = Stochastic(k_period=3, k_smooth=1, d_period=1).compute_frame(frame_from_bars(rows))
    assert computed["k"].item(4) == pytest.approx(100.0)


def test_stochastic_reports_a_flat_window_as_neutral() -> None:
    computed = Stochastic(k_period=3, k_smooth=1, d_period=1).compute_frame(
        frame_from_closes([7.0] * 10)
    )
    assert computed["k"].drop_nulls().to_list() == pytest.approx([50.0] * 8)


def test_stochastic_d_smooths_k(reference_frame: OHLCVFrame) -> None:
    computed = Stochastic(14, 3, 3).compute_frame(reference_frame).drop_nulls()
    k = computed["k"].to_list()
    d = computed["d"].to_list()
    index = 20
    assert d[index] == pytest.approx(sum(k[index - 2 : index + 1]) / 3)


def test_stochastic_stays_inside_its_range(reference_frame: OHLCVFrame) -> None:
    computed = Stochastic().compute_frame(reference_frame).drop_nulls()
    for channel in ("k", "d"):
        values = computed[channel].to_list()
        assert min(values) >= 0.0
        assert max(values) <= 100.0 + 1e-9


def test_cci_is_zero_when_the_window_has_no_spread() -> None:
    values = CCI(20).compute(frame_from_closes([4.0] * 30)).drop_nulls().to_list()
    assert values == pytest.approx([0.0] * len(values))


def test_cci_matches_the_definition_at_one_bar(reference_frame: OHLCVFrame) -> None:
    indicator = CCI(20)
    values = indicator.compute(reference_frame).to_list()
    highs = reference_frame.df["high"].to_list()
    lows = reference_frame.df["low"].to_list()
    closes = reference_frame.df["close"].to_list()
    typical = [(h + low + c) / 3 for h, low, c in zip(highs, lows, closes, strict=True)]

    index = 100
    window = typical[index - 19 : index + 1]
    mean = sum(window) / 20
    spread = sum(abs(value - mean) for value in window) / 20
    assert values[index] == pytest.approx((typical[index] - mean) / (0.015 * spread))


def test_williams_r_is_zero_at_the_window_high() -> None:
    rows = [(5.0, 10.0, 0.0, 5.0, 1.0)] * 4 + [(5.0, 10.0, 0.0, 10.0, 1.0)]
    values = WilliamsR(3).compute(frame_from_bars(rows)).to_list()
    assert values[4] == pytest.approx(0.0)


def test_williams_r_is_minus_one_hundred_at_the_window_low() -> None:
    rows = [(5.0, 10.0, 0.0, 5.0, 1.0)] * 4 + [(5.0, 10.0, 0.0, 0.0, 1.0)]
    values = WilliamsR(3).compute(frame_from_bars(rows)).to_list()
    assert values[4] == pytest.approx(-100.0)


def test_williams_r_reports_a_flat_window_as_neutral() -> None:
    values = WilliamsR(3).compute(frame_from_closes([7.0] * 10)).drop_nulls().to_list()
    assert values == pytest.approx([-50.0] * len(values))


def test_roc_measures_percentage_change_over_the_lookback() -> None:
    closes = [100.0, 101.0, 102.0, 110.0]
    values = ROC(3).compute(frame_from_closes(closes)).to_list()
    assert values[:3] == [None, None, None]
    assert values[3] == pytest.approx(10.0)


def test_roc_warmup_reaches_back_a_full_period() -> None:
    assert ROC(12).warmup == 12


def test_mfi_saturates_when_every_bar_flows_up() -> None:
    rows = [(float(i), float(i) + 1, float(i) - 1, float(i), 100.0) for i in range(1, 30)]
    values = MFI(14).compute(frame_from_bars(rows)).drop_nulls().to_list()
    assert values == pytest.approx([100.0] * len(values))


def test_mfi_reports_no_flow_as_neutral() -> None:
    """An unchanged typical price contributes to neither side, as Chaikin defined it."""
    values = MFI(14).compute(frame_from_closes([3.0] * 40)).drop_nulls().to_list()
    assert values == pytest.approx([50.0] * len(values))


def test_mfi_stays_inside_its_range(reference_frame: OHLCVFrame) -> None:
    values = MFI(14).compute(reference_frame).drop_nulls().to_list()
    assert min(values) >= 0.0
    assert max(values) <= 100.0


@pytest.mark.parametrize(
    "build",
    [
        lambda: RSI(0),
        lambda: Stochastic(k_period=0),
        lambda: CCI(period=0),
        lambda: CCI(constant=0.0),
        lambda: WilliamsR(0),
        lambda: ROC(0),
        lambda: MFI(0),
    ],
)
def test_invalid_parameters_fail_at_construction(build: object) -> None:
    with pytest.raises(ValueError):  # noqa: PT011 - each case has its own message
        build()  # type: ignore[operator]
