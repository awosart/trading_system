"""Reading an operand ``n`` bars back — the lag that made channel breakouts inexpressible."""

import pytest
from pydantic import TypeAdapter
from pydantic import ValidationError as PydanticValidationError

from tests.entry.conftest import frame_from_closes, leaf, ref, strategy_spec
from trading_system.entry.compiler import compile_entry, compile_operand
from trading_system.entry.context import BarSeries
from trading_system.entry.features import FeatureRegistry
from trading_system.strategies.schema import (
    FeatureRef,
    Invalidation,
    PriceRef,
    StrategySpec,
    operand_price_field,
    operand_shift,
)


def price_ref(text: str) -> str:
    """Validate a ``price:`` string the way a loaded spec would."""
    return str(TypeAdapter(PriceRef).validate_python(text))


def series_with_real_features(closes: list[float], spec: StrategySpec) -> BarSeries:
    """Run the real feature pipeline for ``spec`` over ``closes``.

    Hand-written feature columns will not do here: the defect under test is a
    property of what :class:`~trading_system.features.indicators.Donchian`
    actually computes — that its window contains the bar being tested — and a
    column written by hand could simply be given different values.
    """
    frame = frame_from_closes(closes)
    pipeline = FeatureRegistry.from_strategy(spec).pipeline()
    return BarSeries.from_frame(frame, pipeline.compute(frame))


class TestShiftIsARearwardLookback:
    def test_the_default_is_the_current_bar(self) -> None:
        # Every spec written before this field existed must keep its meaning.
        assert FeatureRef(indicator="ema").shift == 0
        assert operand_shift("price:close") == 0

    @pytest.mark.parametrize("shift", [0, 1, 5, 200])
    def test_a_rearward_shift_is_accepted_in_both_operand_forms(self, shift: int) -> None:
        assert FeatureRef(indicator="ema", shift=shift).shift == shift
        assert operand_shift(price_ref(f"price:close@{shift}")) == shift

    def test_a_negative_feature_shift_is_refused_and_says_why(self) -> None:
        with pytest.raises(PydanticValidationError, match="not closed yet"):
            FeatureRef(indicator="ema", shift=-1)

    def test_a_negative_price_shift_is_refused_and_says_why(self) -> None:
        with pytest.raises(PydanticValidationError, match="not closed yet"):
            price_ref("price:close@-1")

    def test_an_unshifted_price_ref_still_parses(self) -> None:
        assert price_ref("price:close") == "price:close"

    @pytest.mark.parametrize(
        "bad", ["price:close@", "price:close@x", "price:close@1.5", "price:close@@1"]
    )
    def test_a_malformed_shift_is_refused(self, bad: str) -> None:
        with pytest.raises(PydanticValidationError):
            price_ref(bad)

    def test_the_price_field_is_unchanged_by_a_shift(self) -> None:
        assert operand_price_field("price:close@3") == "close"


class TestShiftComposesWithTheOperatorsOwnLookback:
    @pytest.fixture
    def registry(self) -> FeatureRegistry:
        spec = strategy_spec(trigger=leaf("gt", "price:close", ref("sma", period=1)))
        return FeatureRegistry.from_strategy(spec)

    @pytest.fixture
    def series(self, registry: FeatureRegistry) -> BarSeries:
        frame = frame_from_closes([float(index) for index in range(1, 11)])
        return BarSeries.from_frame(frame, registry.pipeline().compute(frame))

    def test_a_shifted_operand_reads_the_earlier_bar(
        self, registry: FeatureRegistry, series: BarSeries
    ) -> None:
        # sma(period=1) is the close itself, so a shift of 2 must read exactly
        # the close two bars back.
        unshifted = compile_operand(ref("sma", period=1), registry)
        shifted = compile_operand(
            FeatureRef(indicator="sma", params={"period": 1}, shift=2), registry
        )
        context = series.context(6)
        assert unshifted(context, 0) == 7.0
        assert shifted(context, 0) == 5.0

    def test_shift_adds_to_the_lookback_rather_than_replacing_it(
        self, registry: FeatureRegistry, series: BarSeries
    ) -> None:
        # A crossing reads both bar t and bar t-1. If a shift replaced the
        # operator's own lookback instead of adding to it, both reads would land
        # on one bar and every crossing would collapse into a level test.
        shifted = compile_operand(
            FeatureRef(indicator="sma", params={"period": 1}, shift=1), registry
        )
        context = series.context(6)
        assert shifted(context, 0) == 6.0
        assert shifted(context, 1) == 5.0, "the operator's own lookback was discarded"

    def test_a_shifted_price_operand_reads_the_earlier_bar(
        self, registry: FeatureRegistry, series: BarSeries
    ) -> None:
        shifted = compile_operand(price_ref("price:close@3"), registry)
        assert shifted(series.context(7), 0) == 5.0

    def test_a_shift_past_the_start_of_the_series_is_unknown_not_an_error(
        self, registry: FeatureRegistry, series: BarSeries
    ) -> None:
        # Same rule the rest of the layer follows: not yet knowable is None, and
        # None keeps the leaf unknown rather than deciding it False.
        shifted = compile_operand(price_ref("price:close@5"), registry)
        assert shifted(series.context(2), 0) is None


def donchian_breakout_spec(shift: int) -> StrategySpec:
    """The channel breakout as first written, with the channel read ``shift`` bars back.

    This is the spec that took zero trades on real EURUSD H4: a Donchian channel
    contains the bar being tested, so ``upper >= high >= close`` holds by
    construction and ``close cross_above upper`` cannot fire. Reading the
    channel one bar back is the rule that was meant all along.
    """
    return strategy_spec(
        trigger=leaf(
            "cross_above",
            "price:close",
            FeatureRef(indicator="donchian", params={"period": 20}, channel="upper", shift=shift),
        ),
        invalidation=Invalidation(
            price_level=FeatureRef(indicator="donchian", params={"period": 20}, channel="lower")
        ),
    )


class TestTheChannelBreakoutThatCouldNotFire:
    """The regression this field exists for, on the spec that exposed it."""

    @pytest.fixture
    def closes(self) -> list[float]:
        # Four pushes, each preceded by a pullback deep enough to drop back
        # inside the 20-bar channel, so there are four genuine breakouts rather
        # than one long ride. A monotone ramp would cross once and prove less:
        # the count would then rest on a single bar.
        level = 1.0
        closes = [level] * 25
        for _ in range(4):
            closes += [level - 0.001 * step for step in range(1, 16)]
            level = closes[-1]
            closes += [level + 0.004 * step for step in range(1, 26)]
            level = closes[-1]
        return closes

    def signals(self, shift: int, closes: list[float]) -> int:
        spec = donchian_breakout_spec(shift)
        series = series_with_real_features(closes, spec)
        return len(compile_entry(spec, FeatureRegistry.from_strategy(spec)).run(series).signals)

    def test_the_unshifted_channel_can_never_fire(self, closes: list[float]) -> None:
        # Not "fires rarely" — cannot fire, because upper >= high >= close is an
        # identity of the indicator rather than a property of this data.
        assert self.signals(0, closes) == 0

    def test_the_shifted_channel_fires_once_per_genuine_breakout(self, closes: list[float]) -> None:
        # Four pushes were built, each from inside the channel: four crossings.
        assert self.signals(1, closes) == 4

    def test_the_shift_is_the_only_difference_between_the_two_specs(self) -> None:
        without = donchian_breakout_spec(0).model_dump()
        with_shift = donchian_breakout_spec(1).model_dump()
        trigger = ("entries", 0, "trigger")
        assert without["entries"][0]["invalidation"] == with_shift["entries"][0]["invalidation"]
        without["entries"][0]["trigger"]["right"]["shift"] = 1
        assert without == with_shift, f"specs differ somewhere other than {trigger}"

    def test_both_specs_share_one_feature_column(self) -> None:
        # A shift is a read offset, not a different series: two references
        # differing only in shift must not make the pipeline compute twice.
        keys = [
            sorted(FeatureRegistry.from_strategy(donchian_breakout_spec(n)).keys) for n in (0, 1)
        ]
        assert keys[0] == keys[1]
