"""Two-legged strategies, and the drop counts a run carries out with it."""

import pytest

from trading_system.core.types import Side
from trading_system.entry.compiler import DropReason, EntryRun, compile_entry
from trading_system.entry.context import BarSeries
from trading_system.entry.features import FeatureRegistry, feature_key
from trading_system.strategies.schema import Invalidation, StrategySpec

from .conftest import entry_spec, labels, leaf, ref, series_from_features, strategy_spec

LOWER = ref("donchian", channel="lower", period=20)
UPPER = ref("donchian", channel="upper", period=20)
LOWER_KEY = feature_key(LOWER)
UPPER_KEY = feature_key(UPPER)
RSI = ref("rsi", period=14)
RSI_KEY = feature_key(RSI)
SMA5 = ref("sma", period=5)
SMA5_KEY = feature_key(SMA5)


def range_fade_spec() -> StrategySpec:
    """A range fade: buy a pierce of the lower edge, sell a pierce of the upper.

    The case that made ``direction: BOTH`` unworkable. The legs are not mirror
    images — each is invalidated on its own side of the range — which is exactly
    what one shared ``Invalidation`` could not express.
    """
    return strategy_spec(
        entries=[
            entry_spec(
                direction="LONG",
                trigger=leaf("lt", "price:low", LOWER),
                invalidation=Invalidation(price_level=1.00),
            ),
            entry_spec(
                direction="SHORT",
                trigger=leaf("gt", "price:high", UPPER),
                invalidation=Invalidation(price_level=2.00),
            ),
        ]
    )


def range_fade_series() -> BarSeries:
    """Two bars: the first pierces only the low, the second pierces both edges."""
    return series_from_features(
        [1.10, 1.50],
        {LOWER_KEY: [1.20, 1.20], UPPER_KEY: [1.80, 1.80]},
        lows=[1.10, 1.05],
        highs=[1.10, 1.95],
    )


class TestTwoLegs:
    """A strategy that trades both ways is two entries, not a flipped one."""

    def test_the_engine_compiles_one_evaluator_per_direction(self) -> None:
        engine = compile_entry(range_fade_spec(), FeatureRegistry.from_strategy(range_fade_spec()))
        assert engine.sides == (Side.BUY, Side.SELL)
        assert len(engine.evaluators) == 2

    def test_legs_share_the_strategy_id(self) -> None:
        spec = range_fade_spec()
        engine = compile_entry(spec, FeatureRegistry.from_strategy(spec))
        assert {evaluator.strategy_id for evaluator in engine.evaluators} == {spec.id}

    def test_each_leg_fires_on_its_own_edge(self) -> None:
        spec = range_fade_spec()
        signals = (
            compile_entry(spec, FeatureRegistry.from_strategy(spec))
            .run(range_fade_series())
            .signals
        )
        assert [(s.context["signal_bar_index"], s.side) for s in signals] == [
            (0, Side.BUY),
            (1, Side.BUY),
            (1, Side.SELL),
        ]

    def test_the_legs_keep_their_own_invalidation_levels(self) -> None:
        spec = range_fade_spec()
        signals = (
            compile_entry(spec, FeatureRegistry.from_strategy(spec))
            .run(range_fade_series())
            .signals
        )
        long_signal, short_signal = signals[1], signals[2]
        assert long_signal.invalidation_price == 1.00
        assert short_signal.invalidation_price == 2.00

    def test_a_pending_leg_does_not_block_the_other(self) -> None:
        spec = strategy_spec(
            entries=[
                entry_spec(
                    direction="LONG",
                    trigger=leaf("lt", RSI, 30.0),
                    confirmation=[leaf("gt", SMA5, 99.0)],
                    confirmation_window_bars=5,
                    invalidation=Invalidation(price_level=1.00),
                ),
                entry_spec(
                    direction="SHORT",
                    trigger=leaf("gt", RSI, 70.0),
                    invalidation=Invalidation(price_level=2.00),
                ),
            ]
        )
        series = series_from_features(
            [1.50, 1.50, 1.50],
            {RSI_KEY: [20.0, 80.0, 80.0], SMA5_KEY: [1.0, 1.0, 1.0]},
        )
        result = compile_entry(spec, FeatureRegistry.from_strategy(spec)).run(series)
        # The long leg is stuck waiting for a confirmation that never comes; the
        # short leg is unaffected and fires twice.
        assert [(s.context["signal_bar_index"], s.side) for s in result.signals] == [
            (1, Side.SELL),
            (2, Side.SELL),
        ]


class TestConcurrentSides:
    """The fact of simultaneity is stated, never resolved."""

    def test_both_signals_are_emitted_when_both_legs_fire(self) -> None:
        # Entry has no portfolio state, so choosing between them would be a
        # guess dressed as a decision. Both go out; the Risk Engine decides.
        spec = range_fade_spec()
        signals = (
            compile_entry(spec, FeatureRegistry.from_strategy(spec))
            .run(range_fade_series())
            .signals
        )
        same_bar = [s for s in signals if s.context["signal_bar_index"] == 1]
        assert {s.side for s in same_bar} == {Side.BUY, Side.SELL}

    def test_every_signal_records_which_sides_fired_on_its_bar(self) -> None:
        spec = range_fade_spec()
        signals = (
            compile_entry(spec, FeatureRegistry.from_strategy(spec))
            .run(range_fade_series())
            .signals
        )
        assert [s.context["concurrent_sides"] for s in signals] == [
            ("BUY",),
            ("BUY", "SELL"),
            ("BUY", "SELL"),
        ]

    def test_the_key_is_present_even_without_a_conflict(self) -> None:
        # An absent key would have to mean both "no conflict" and "written before
        # this existed"; the Risk Engine must never have to guess which.
        spec = strategy_spec(trigger=leaf("gt", RSI, 50.0))
        series = series_from_features([1.10, 1.20], {RSI_KEY: [40.0, 60.0]})
        signals = compile_entry(spec, FeatureRegistry.from_strategy(spec)).run(series).signals
        assert signals[0].context["concurrent_sides"] == ("BUY",)

    def test_stamping_leaves_the_rest_of_the_context_intact(self) -> None:
        spec = range_fade_spec()
        signals = (
            compile_entry(spec, FeatureRegistry.from_strategy(spec))
            .run(range_fade_series())
            .signals
        )
        context = signals[0].context
        assert context["trigger_bar_index"] == 0
        assert context["base_quality"] == 0.5
        assert context["entry_order"] == "MARKET"


class TestDropCounts:
    """A warning on a 200k-bar run is unread; a count reaches the report."""

    def _wrong_sided_run(self) -> EntryRun:
        spec = strategy_spec(
            trigger=leaf("gt", RSI, 50.0), invalidation=Invalidation(price_level=SMA5)
        )
        series = series_from_features(
            [1.10, 1.20, 1.30, 1.40],
            # Bars 1 and 3 both trigger; only bar 3's level is usable.
            {RSI_KEY: [40.0, 60.0, 40.0, 60.0], SMA5_KEY: [1.0, 1.5, 1.0, 1.0]},
        )
        return compile_entry(spec, FeatureRegistry.from_strategy(spec)).run(series)

    def test_a_wrong_sided_invalidation_is_counted(self) -> None:
        result = self._wrong_sided_run()
        assert result.drops[DropReason.WRONG_SIDED_INVALIDATION] == 1
        assert result.dropped == 1

    def test_the_run_still_reports_the_signals_it_did_produce(self) -> None:
        result = self._wrong_sided_run()
        assert len(result.signals) == 1
        assert result.bars == 4

    def test_an_unavailable_invalidation_level_is_counted_separately(self) -> None:
        spec = strategy_spec(
            trigger=leaf("gt", RSI, 50.0), invalidation=Invalidation(price_level=SMA5)
        )
        series = series_from_features([1.10, 1.20], {RSI_KEY: [40.0, 60.0], SMA5_KEY: [None, None]})
        result = compile_entry(spec, FeatureRegistry.from_strategy(spec)).run(series)
        assert result.drops[DropReason.INVALIDATION_UNAVAILABLE] == 1
        assert result.drops[DropReason.WRONG_SIDED_INVALIDATION] == 0

    def test_every_reason_is_reported_even_at_zero(self) -> None:
        # "No wrong-sided invalidations" is a recorded fact, not a missing key
        # for a reader downstream to interpret.
        spec = strategy_spec(trigger=leaf("gt", RSI, 50.0))
        series = series_from_features([1.10, 1.20], {RSI_KEY: [40.0, 60.0]})
        result = compile_entry(spec, FeatureRegistry.from_strategy(spec)).run(series)
        assert set(result.drops) == set(DropReason)
        assert result.dropped == 0

    def test_counts_are_zeroed_by_a_reset_so_folds_do_not_accumulate(self) -> None:
        spec = strategy_spec(
            trigger=leaf("gt", RSI, 50.0), invalidation=Invalidation(price_level=SMA5)
        )
        series = series_from_features([1.10, 1.20], {RSI_KEY: [40.0, 60.0], SMA5_KEY: [1.0, 1.5]})
        engine = compile_entry(spec, FeatureRegistry.from_strategy(spec))
        first = engine.run(series)
        second = engine.run(series)
        assert first.drops[DropReason.WRONG_SIDED_INVALIDATION] == 1
        assert second.drops[DropReason.WRONG_SIDED_INVALIDATION] == 1

    def test_counts_are_summed_across_legs(self) -> None:
        spec = strategy_spec(
            entries=[
                entry_spec(
                    direction="LONG",
                    trigger=leaf("gt", RSI, 50.0),
                    invalidation=Invalidation(price_level=SMA5),
                ),
                entry_spec(
                    direction="SHORT",
                    trigger=leaf("gt", RSI, 50.0),
                    invalidation=Invalidation(price_level=SMA5),
                ),
            ]
        )
        # A level of 1.15 against a 1.20 close is above it for the long leg's
        # purposes and below it for the short's, so both legs drop their signal.
        series = series_from_features([1.10, 1.20], {RSI_KEY: [40.0, 60.0], SMA5_KEY: [1.0, 1.25]})
        result = compile_entry(spec, FeatureRegistry.from_strategy(spec)).run(series)
        assert result.drops[DropReason.WRONG_SIDED_INVALIDATION] == 1
        assert result.signals[0].side is Side.SELL

    def test_the_drop_mapping_is_read_only(self) -> None:
        spec = strategy_spec(trigger=leaf("gt", RSI, 50.0))
        series = series_from_features([1.10, 1.20], {RSI_KEY: [40.0, 60.0]})
        result = compile_entry(spec, FeatureRegistry.from_strategy(spec)).run(series)
        with pytest.raises(TypeError):
            result.drops[DropReason.WRONG_SIDED_INVALIDATION] = 99  # type: ignore[index]


class TestSchemaGuarantees:
    """What P04 now refuses, so the compiler no longer has to."""

    def test_a_second_entry_in_the_same_direction_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="distinct directions"):
            strategy_spec(
                entries=[
                    entry_spec(direction="LONG", trigger=leaf("gt", RSI, 50.0)),
                    entry_spec(direction="LONG", trigger=leaf("gt", RSI, 60.0)),
                ]
            )

    def test_three_entries_are_rejected(self) -> None:
        with pytest.raises(ValueError, match="at most 2"):
            strategy_spec(
                entries=[
                    entry_spec(direction="LONG", trigger=leaf("gt", RSI, 50.0)),
                    entry_spec(direction="SHORT", trigger=leaf("gt", RSI, 60.0)),
                    entry_spec(direction="LONG", trigger=leaf("gt", RSI, 70.0)),
                ]
            )

    def test_direction_both_no_longer_exists(self) -> None:
        with pytest.raises(ValueError, match="BOTH"):
            entry_spec(direction="BOTH", trigger=leaf("gt", RSI, 50.0))

    def test_entry_for_finds_the_leg_by_direction(self) -> None:
        from trading_system.strategies.schema import Direction

        spec = range_fade_spec()
        assert spec.entry_for(Direction.LONG) is not None
        assert spec.entry_for(Direction.SHORT) is not None
        assert strategy_spec(trigger=leaf("gt", RSI, 50.0)).entry_for(Direction.SHORT) is None

    def test_labels_are_rejected_where_a_number_belongs(self) -> None:
        spec = strategy_spec(trigger=leaf("gt", RSI, labels("DOJI")))
        with pytest.raises(Exception, match="compares numbers"):
            compile_entry(spec, FeatureRegistry.from_strategy(spec))
