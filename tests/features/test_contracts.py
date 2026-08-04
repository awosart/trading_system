"""The contracts every registered indicator must satisfy.

These are the P03 definition-of-done checks, applied to the whole registry
rather than to a hand-maintained list. Adding an indicator to
:data:`~trading_system.features.registry.INDICATOR_TYPES` automatically subjects
it to all of them, which is the point: the failure mode being guarded against is
someone adding indicator number 31 and quietly not testing it.
"""

from typing import Any

import pytest
from hypothesis import HealthCheck, given, settings

from trading_system.data.models import OHLCVFrame
from trading_system.features.base import (
    BaseIndicator,
    Indicator,
    MultiOutputIndicator,
    StreamingIndicator,
    iter_bars,
    run_streaming,
)
from trading_system.features.registry import INDICATOR_TYPES, default_indicators

from .conftest import assert_series_close, ohlcv_frames

INDICATORS = default_indicators()
IDS = [indicator.name for indicator in INDICATORS]


def test_every_registered_kind_is_instantiable() -> None:
    assert len(INDICATORS) == len(INDICATOR_TYPES)
    assert len({indicator.name for indicator in INDICATORS}) == len(INDICATORS)


@pytest.mark.parametrize("indicator", INDICATORS, ids=IDS)
def test_vector_matches_streaming_on_reference_data(
    indicator: BaseIndicator[Any], reference_frame: OHLCVFrame
) -> None:
    """The headline invariant: both evaluation paths agree to 1e-9."""
    indicator.verify_parity(reference_frame)


@given(frame=ohlcv_frames())
@settings(max_examples=25, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_vector_matches_streaming_on_arbitrary_data(frame: OHLCVFrame) -> None:
    """Parity holds on generated data too, including flat bars and zero volume.

    Every indicator is checked inside one example rather than the test being
    parametrised over them. Generating a frame is far more expensive than
    evaluating an indicator on it, so sharing each drawn frame across the whole
    registry buys thirty times the coverage for the same budget.
    """
    for indicator in INDICATORS:
        indicator.verify_parity(frame)


@pytest.mark.parametrize("indicator", INDICATORS, ids=IDS)
def test_values_do_not_change_when_later_bars_arrive(
    indicator: BaseIndicator[Any], reference_frame: OHLCVFrame
) -> None:
    """No lookahead: bar t's value is fixed once bar t has closed.

    Evaluating a prefix must reproduce the full frame's values exactly over the
    overlap. An indicator that peeked forward — a centred window, an unshifted
    Ichimoku span, a pivot marked at the bar it occurred on rather than the bar
    that confirmed it — changes its answer here.
    """
    full = indicator.compute_frame(reference_frame)
    total = len(reference_frame)
    for length in (indicator.warmup + 5, total // 3, total // 2, total - 1):
        prefix = reference_frame.with_df(reference_frame.df.head(length))
        partial = indicator.compute_frame(prefix)
        for channel in indicator.outputs:
            assert_series_close(
                partial[channel],
                full[channel].head(length),
                context=f"{indicator.name}.{channel} over {length} bars",
            )


@given(frame=ohlcv_frames())
@settings(max_examples=15, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_prefix_stability_on_arbitrary_data(frame: OHLCVFrame) -> None:
    """Dropping the last quarter of the data leaves the earlier values untouched."""
    length = (len(frame) * 3) // 4
    prefix = frame.with_df(frame.df.head(length))
    for indicator in INDICATORS:
        full = indicator.compute_frame(frame)
        partial = indicator.compute_frame(prefix)
        for channel in indicator.outputs:
            assert_series_close(
                partial[channel],
                full[channel].head(length),
                context=f"{indicator.name}.{channel}",
            )


@pytest.mark.parametrize("indicator", INDICATORS, ids=IDS)
def test_warmup_is_exactly_the_null_prefix(
    indicator: BaseIndicator[Any], reference_frame: OHLCVFrame
) -> None:
    """The first ``warmup`` values are null and nothing after them is.

    Both halves matter. A warmup that overstates leaves usable rows on the
    floor; one that understates hands a strategy a value computed from a
    half-filled window.
    """
    computed = indicator.compute_frame(reference_frame)
    for channel in indicator.outputs:
        values = computed[channel].to_list()
        assert all(value is None for value in values[: indicator.warmup]), (
            f"{indicator.name}.{channel} has a value inside its {indicator.warmup}-bar warmup"
        )
        assert values[indicator.warmup] is not None, (
            f"{indicator.name}.{channel} is still null at bar {indicator.warmup}"
        )
        assert all(value is not None for value in values[indicator.warmup :]), (
            f"{indicator.name}.{channel} goes null again after its warmup"
        )


@pytest.mark.parametrize("indicator", INDICATORS, ids=IDS)
def test_streaming_withholds_values_through_warmup(
    indicator: BaseIndicator[Any], reference_frame: OHLCVFrame
) -> None:
    """The incremental path returns ``None`` for exactly the warmup bars."""
    state = indicator.streaming()
    assert state.warmup == indicator.warmup
    for index, bar in enumerate(iter_bars(reference_frame)):
        value = state.update(bar)
        assert (value is None) == (index < indicator.warmup), (
            f"{indicator.name} at bar {index}: got {value!r} with warmup {indicator.warmup}"
        )


@pytest.mark.parametrize("indicator", INDICATORS, ids=IDS)
def test_reset_returns_the_state_machine_to_bar_zero(
    indicator: BaseIndicator[Any], reference_frame: OHLCVFrame
) -> None:
    """A reset state machine reproduces its first run bit for bit."""
    state = indicator.streaming()
    bars = list(iter_bars(reference_frame))
    first = [state.update(bar) for bar in bars]
    state.reset()
    second = [state.update(bar) for bar in bars]
    assert first == second


@pytest.mark.parametrize("indicator", INDICATORS, ids=IDS)
def test_short_frames_yield_all_nulls_rather_than_failing(
    indicator: BaseIndicator[Any], short_frame: OHLCVFrame
) -> None:
    """A frame shorter than the warmup produces nulls, not an exception."""
    computed = indicator.compute_frame(short_frame)
    assert computed.height == len(short_frame)
    if indicator.warmup >= len(short_frame):
        for channel in indicator.outputs:
            assert computed[channel].null_count() == len(short_frame)


@pytest.mark.parametrize("indicator", INDICATORS, ids=IDS)
def test_empty_frames_produce_empty_output(
    indicator: BaseIndicator[Any], reference_frame: OHLCVFrame
) -> None:
    empty = OHLCVFrame.empty(reference_frame.symbol, reference_frame.timeframe)
    computed = indicator.compute_frame(empty)
    assert computed.height == 0
    assert tuple(computed.columns) == indicator.outputs


@pytest.mark.parametrize("indicator", INDICATORS, ids=IDS)
def test_indicators_satisfy_the_declared_protocols(indicator: BaseIndicator[Any]) -> None:
    """Single-valued indicators expose ``compute``; multi-channel ones expose ``compute_frame``."""
    if len(indicator.outputs) == 1:
        assert isinstance(indicator, Indicator)
        assert isinstance(indicator.streaming(), StreamingIndicator)
    else:
        assert isinstance(indicator, MultiOutputIndicator)
        assert not isinstance(indicator, Indicator)


@pytest.mark.parametrize("indicator", INDICATORS, ids=IDS)
def test_run_streaming_matches_compute_frame_shape(
    indicator: BaseIndicator[Any], reference_frame: OHLCVFrame
) -> None:
    streamed = run_streaming(indicator, reference_frame)
    computed = indicator.compute_frame(reference_frame)
    assert streamed.columns == computed.columns
    assert streamed.height == computed.height
