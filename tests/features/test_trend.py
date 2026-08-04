"""Trend indicators, checked against values computed by hand."""

import pytest

from trading_system.data.models import OHLCVFrame
from trading_system.features.indicators.trend import (
    ADX,
    EMA,
    HMA,
    MACD,
    SMA,
    VWMA,
    WMA,
    Donchian,
    Ichimoku,
    Supertrend,
)

from .conftest import frame_from_closes


def test_sma_averages_the_window() -> None:
    values = SMA(3).compute(frame_from_closes([1.0, 2.0, 3.0, 4.0, 5.0])).to_list()
    assert values == [None, None, 2.0, 3.0, 4.0]


def test_sma_can_average_a_derived_source() -> None:
    frame = frame_from_closes([1.0, 3.0, 5.0])
    assert SMA(2, "hl2").compute(frame).to_list() == [None, 2.0, 4.0]


def test_ema_is_seeded_with_the_simple_mean() -> None:
    """EMA(3) seeds at bar 2 with mean(1,2,3)=2, then folds bar 3 in at alpha=0.5."""
    values = EMA(3).compute(frame_from_closes([1.0, 2.0, 3.0, 4.0])).to_list()
    assert values[:2] == [None, None]
    assert values[2] == pytest.approx(2.0)
    assert values[3] == pytest.approx(2.0 + 0.5 * (4.0 - 2.0))


def test_wma_weights_the_newest_bar_most() -> None:
    values = WMA(3).compute(frame_from_closes([1.0, 2.0, 3.0])).to_list()
    assert values[2] == pytest.approx((1 * 1 + 2 * 2 + 3 * 3) / 6)


def test_hma_warmup_covers_both_stages() -> None:
    indicator = HMA(16)
    assert indicator.half_period == 8
    assert indicator.smoothing_period == 4
    assert indicator.warmup == 15 + 3


def test_hma_lags_a_ramp_far_less_than_the_plain_weighted_average() -> None:
    """The point of the Hull construction, stated as arithmetic.

    On a straight line a ``WMA(n)`` sits ``(n - 1) / 3`` bars behind. Subtracting
    the slow average from twice the fast one cancels most of that, leaving
    ``(√n - 2) / 3`` — a third of a bar at ``n = 16`` against five.
    """
    closes = [float(i) for i in range(1, 41)]
    frame = frame_from_closes(closes)
    assert WMA(16).compute(frame).to_list()[-1] == pytest.approx(40.0 - 15 / 3)
    assert HMA(16).compute(frame).to_list()[-1] == pytest.approx(40.0 - 2 / 3)


def test_hma_rejects_a_period_it_cannot_decompose() -> None:
    with pytest.raises(ValueError, match="at least 4"):
        HMA(3)


def test_vwma_weights_by_volume() -> None:
    frame = frame_from_closes([1.0, 2.0, 3.0], volumes=[1.0, 1.0, 8.0])
    values = VWMA(3).compute(frame).to_list()
    assert values[2] == pytest.approx((1 * 1 + 2 * 1 + 3 * 8) / 10)


def test_vwma_falls_back_to_the_plain_mean_without_volume() -> None:
    frame = frame_from_closes([1.0, 2.0, 3.0], volumes=[0.0, 0.0, 0.0])
    assert VWMA(3).compute(frame).to_list()[2] == pytest.approx(2.0)


def test_macd_histogram_is_the_gap_between_line_and_signal(
    reference_frame: OHLCVFrame,
) -> None:
    computed = MACD().compute_frame(reference_frame)
    valid = computed.drop_nulls()
    difference = (valid["macd"] - valid["signal"]).to_list()
    assert difference == pytest.approx(valid["histogram"].to_list())


def test_macd_publishes_every_channel_on_the_same_bar(reference_frame: OHLCVFrame) -> None:
    computed = MACD().compute_frame(reference_frame)
    null_counts = {channel: computed[channel].null_count() for channel in computed.columns}
    assert len(set(null_counts.values())) == 1


def test_macd_rejects_a_fast_period_at_or_above_the_slow_one() -> None:
    with pytest.raises(ValueError, match="below slow_period"):
        MACD(fast_period=26, slow_period=26)


def test_adx_stays_inside_its_range(reference_frame: OHLCVFrame) -> None:
    computed = ADX(14).compute_frame(reference_frame).drop_nulls()
    for channel in ("adx", "plus_di", "minus_di"):
        values = computed[channel].to_list()
        assert min(values) >= 0.0
        assert max(values) <= 100.0 + 1e-9


def test_adx_is_zero_on_a_perfectly_flat_market() -> None:
    """No range and no directional movement leaves nothing for the index to measure."""
    frame = frame_from_closes([1.0] * 60)
    computed = ADX(14).compute_frame(frame).drop_nulls()
    assert computed["adx"].to_list() == pytest.approx([0.0] * computed.height)


def test_supertrend_line_follows_the_active_band(reference_frame: OHLCVFrame) -> None:
    computed = Supertrend().compute_frame(reference_frame).drop_nulls()
    for row in computed.rows(named=True):
        expected = row["lower"] if row["direction"] > 0 else row["upper"]
        assert row["line"] == pytest.approx(expected)


def test_supertrend_flips_direction_when_price_closes_through_a_band() -> None:
    """A sharp reversal after a steady climb must turn the trend down."""
    closes = [100.0 + i for i in range(40)] + [139.0 - 4 * i for i in range(1, 25)]
    computed = Supertrend(period=10, multiplier=2.0).compute_frame(frame_from_closes(closes))
    directions = [value for value in computed["direction"].to_list() if value is not None]
    assert directions[0] == 1.0
    assert -1.0 in directions


def test_ichimoku_spans_are_read_back_through_the_displacement(
    reference_frame: OHLCVFrame,
) -> None:
    """Senkou at bar t is the midpoint computed at bar t - displacement, never later."""
    indicator = Ichimoku()
    computed = indicator.compute_frame(reference_frame)
    highs = reference_frame.df["high"].to_list()
    lows = reference_frame.df["low"].to_list()

    index = indicator.warmup + 20
    source = index - indicator.displacement
    window = slice(source - indicator.senkou_b_period + 1, source + 1)
    expected = (max(highs[window]) + min(lows[window])) / 2
    assert computed["senkou_b"].item(index) == pytest.approx(expected)


def test_ichimoku_omits_chikou() -> None:
    """The lagging span has no causal form, so it is not offered at all."""
    assert "chikou" not in Ichimoku().outputs


def test_donchian_reports_the_window_extremes(reference_frame: OHLCVFrame) -> None:
    indicator = Donchian(20)
    computed = indicator.compute_frame(reference_frame)
    highs = reference_frame.df["high"].to_list()
    lows = reference_frame.df["low"].to_list()
    index = 100
    assert computed["upper"].item(index) == pytest.approx(max(highs[index - 19 : index + 1]))
    assert computed["lower"].item(index) == pytest.approx(min(lows[index - 19 : index + 1]))
    assert computed["middle"].item(index) == pytest.approx(
        (computed["upper"].item(index) + computed["lower"].item(index)) / 2
    )


@pytest.mark.parametrize(
    "build",
    [
        lambda: SMA(0),
        lambda: EMA(-1),
        lambda: WMA(0),
        lambda: VWMA(0),
        lambda: ADX(0),
        lambda: Donchian(0),
        lambda: Supertrend(period=0),
        lambda: Supertrend(multiplier=0.0),
        lambda: Ichimoku(tenkan_period=0),
        lambda: Ichimoku(displacement=-1),
        lambda: SMA(5, "nonsense"),
    ],
)
def test_invalid_parameters_fail_at_construction(build: object) -> None:
    with pytest.raises(ValueError):  # noqa: PT011 - each case has its own message
        build()  # type: ignore[operator]
