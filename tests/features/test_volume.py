"""Volume indicators, including the session boundary both VWAPs turn on."""

from datetime import UTC, datetime

import pytest

from trading_system.data.models import OHLCVFrame
from trading_system.data.resample import FX_DAY_ORIGIN, DayOrigin
from trading_system.features.indicators.volume import (
    OBV,
    AnchoredVwap,
    RelativeVolume,
    SessionVwap,
    VolumeMA,
)

from .conftest import frame_from_bars, frame_from_closes

# 17:00 New York in January is 22:00 UTC, so a frame starting at 21:58 UTC
# crosses the FX rollover two bars in.
BEFORE_ROLLOVER = datetime(2024, 1, 2, 21, 58, tzinfo=UTC)


def test_obv_adds_volume_on_up_closes_and_subtracts_on_down() -> None:
    frame = frame_from_closes([10.0, 11.0, 10.0, 10.0, 12.0], volumes=[5.0, 3.0, 4.0, 7.0, 2.0])
    values = OBV().compute(frame).to_list()
    assert values == [None, 3.0, -1.0, -1.0, 1.0]


def test_obv_has_no_value_on_the_first_bar() -> None:
    """Direction needs a previous close; bar 0 has none, so it carries nothing."""
    assert OBV().warmup == 1


def test_session_vwap_restarts_at_the_trading_day_boundary() -> None:
    rows = [
        (1.0, 1.0, 1.0, 1.0, 1.0),
        (3.0, 3.0, 3.0, 3.0, 1.0),
        (5.0, 5.0, 5.0, 5.0, 1.0),
        (7.0, 7.0, 7.0, 7.0, 1.0),
    ]
    frame = frame_from_bars(rows, start=BEFORE_ROLLOVER)
    values = SessionVwap().compute(frame).to_list()
    assert values[0] == pytest.approx(1.0)
    assert values[1] == pytest.approx(2.0)  # (1 + 3) / 2, still yesterday
    assert values[2] == pytest.approx(5.0)  # new session: only this bar
    assert values[3] == pytest.approx(6.0)  # (5 + 7) / 2


def test_session_vwap_weights_by_volume() -> None:
    rows = [(1.0, 1.0, 1.0, 1.0, 1.0), (3.0, 3.0, 3.0, 3.0, 3.0)]
    frame = frame_from_bars(rows, start=BEFORE_ROLLOVER)
    values = SessionVwap().compute(frame).to_list()
    assert values[1] == pytest.approx((1 * 1 + 3 * 3) / 4)


def test_session_vwap_falls_back_to_the_plain_mean_without_volume() -> None:
    rows = [(1.0, 1.0, 1.0, 1.0, 0.0), (3.0, 3.0, 3.0, 3.0, 0.0)]
    frame = frame_from_bars(rows, start=BEFORE_ROLLOVER)
    assert SessionVwap().compute(frame).to_list()[1] == pytest.approx(2.0)


def test_session_vwap_honours_a_custom_day_origin() -> None:
    """A UTC-midnight origin puts all four bars in one session, unlike the FX default."""
    rows = [(float(i), float(i), float(i), float(i), 1.0) for i in (1, 3, 5, 7)]
    frame = frame_from_bars(rows, start=BEFORE_ROLLOVER)
    values = SessionVwap(origin=DayOrigin(tz="UTC")).compute(frame).to_list()
    assert values[3] == pytest.approx(4.0)
    assert SessionVwap().origin == FX_DAY_ORIGIN


def test_anchored_vwap_without_an_anchor_starts_at_the_first_bar() -> None:
    frame = frame_from_closes([1.0, 3.0, 5.0], volumes=[1.0, 1.0, 1.0])
    values = AnchoredVwap().compute(frame).to_list()
    assert values == pytest.approx([1.0, 2.0, 3.0])


def test_anchored_vwap_is_null_before_its_anchor() -> None:
    """Validity here is a property of the data, not a bar count — hence warmup 0."""
    start = datetime(2024, 1, 2, 21, 0, tzinfo=UTC)
    frame = frame_from_closes([1.0, 3.0, 5.0, 7.0], volumes=[1.0] * 4, start=start)
    indicator = AnchoredVwap(anchor=datetime(2024, 1, 2, 21, 2, tzinfo=UTC))
    values = indicator.compute(frame).to_list()
    assert indicator.warmup == 0
    assert values[0] is None
    assert values[1] is None
    assert values[2] == pytest.approx(5.0)
    assert values[3] == pytest.approx(6.0)


def test_anchored_vwap_rejects_a_naive_anchor() -> None:
    with pytest.raises(ValueError, match="tz-aware"):
        AnchoredVwap(anchor=datetime(2024, 1, 2, 21, 0))  # noqa: DTZ001 - the point of the test


def test_volume_ma_averages_volume() -> None:
    frame = frame_from_closes([1.0] * 4, volumes=[1.0, 2.0, 3.0, 4.0])
    assert VolumeMA(3).compute(frame).to_list() == [None, None, 2.0, 3.0]


def test_relative_volume_excludes_the_bar_it_measures() -> None:
    """Including the current bar in its own baseline mutes the spikes being looked for."""
    frame = frame_from_closes([1.0] * 4, volumes=[10.0, 20.0, 30.0, 40.0])
    values = RelativeVolume(2).compute(frame).to_list()
    assert values[:2] == [None, None]
    assert values[2] == pytest.approx(30 / 15)
    assert values[3] == pytest.approx(40 / 25)


def test_relative_volume_treats_a_dead_baseline_as_average() -> None:
    frame = frame_from_closes([1.0] * 4, volumes=[0.0, 0.0, 30.0, 40.0])
    assert RelativeVolume(2).compute(frame).to_list()[2] == pytest.approx(1.0)


def test_volume_indicators_reject_impossible_periods() -> None:
    with pytest.raises(ValueError, match="positive"):
        VolumeMA(0)
    with pytest.raises(ValueError, match="positive"):
        RelativeVolume(0)


def test_session_vwap_sits_inside_the_price_range(reference_frame: OHLCVFrame) -> None:
    values = SessionVwap().compute(reference_frame).to_list()
    lows = reference_frame.df["low"].to_list()
    highs = reference_frame.df["high"].to_list()
    assert min(values) >= min(lows) - 1e-9
    assert max(values) <= max(highs) + 1e-9
