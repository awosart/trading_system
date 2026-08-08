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

    **"Agreement lands near machine epsilon" is false for an expression that
    cancels, and you will meet this before you meet the exception.** The promise
    holds while every intermediate stays the size of its inputs: two paths over
    one formula in IEEE doubles then differ in the last bit or two. It stops
    holding the moment the formula subtracts two nearly equal numbers. That
    subtraction discards the leading digits they share, and everything computed
    from the remainder carries a relative error multiplied by however many
    digits went — so a single differently-rounded intermediate, one ULP, comes
    out of the far end amplified.

    This was not reasoned about in advance; it was found. ``cci_20`` failed here
    at 1.169e-07 on a value of 666.67, and the diagnosis was that the two paths'
    window means differed by exactly one ULP — the exact answer, computed in
    rational arithmetic on the same inputs, sat between them with each path
    wrong by 5.84e-08 in opposite directions.

    **If you are adding an indicator that subtracts close values — a deviation
    from a mean, a difference of two long averages, anything divided by a
    spread — and it fails here, this is the first thing to check, not the last.**
    Diagnose it: compute the exact answer in :mod:`fractions`, and perturb the
    suspect intermediate by one ULP to see whether that reproduces the gap. If
    it does, the fix is a declared, justified
    :attr:`~trading_system.features.base.BaseIndicator.parity_rtol` on that
    indicator — see :class:`~trading_system.features.indicators.momentum.CCI`
    for the shape of the argument. If it does not, you have a real formula
    divergence and relaxing the tolerance would bury it.
    """
    for indicator in INDICATORS:
        indicator.verify_parity(frame)


def test_relaxed_parity_is_declared_per_indicator_and_stays_rare() -> None:
    """Relative slack must be an argued exception, never something inherited.

    The default is zero, so an indicator gets slack only by overriding
    ``parity_rtol`` and saying why. This test pins the set of indicators that
    do: adding one is a deliberate act that fails here until the list is
    updated, which is the moment to check that a cancellation argument was
    actually made rather than a failing test silenced.
    """
    relaxed = {
        indicator.name: indicator.parity_rtol
        for indicator in INDICATORS
        if indicator.parity_rtol != 0.0
    }
    assert relaxed == {"cci_20": 1e-7}
    assert all(0.0 < value < 1e-5 for value in relaxed.values())


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
