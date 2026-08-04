"""Compilation, nesting, the confirmation state machine, and no-lookahead."""

import math
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from typing import Any

import pytest
from structlog.testing import capture_logs

from trading_system.core.exceptions import ValidationError
from trading_system.core.types import Price, Side
from trading_system.entry.compiler import compile_condition, compile_entry
from trading_system.entry.context import BarSeries
from trading_system.entry.features import FeatureRegistry, feature_key
from trading_system.entry.operators import Truth
from trading_system.entry.signal import EntrySignal
from trading_system.strategies.schema import (
    AllOf,
    AnyOf,
    Condition,
    FeatureRef,
    Invalidation,
    LimitOrder,
    Not,
    QualityModifier,
    StopOrder,
)

from .conftest import (
    frame_from_closes,
    labels,
    leaf,
    ref,
    series_from_features,
    strategy_spec,
)

RSI = ref("rsi", period=14)
RSI_KEY = feature_key(RSI)
SMA5 = ref("sma", period=5)
SMA5_KEY = feature_key(SMA5)
MACD_SIGNAL = ref("macd", channel="signal")
MACD_SIGNAL_KEY = feature_key(MACD_SIGNAL)


def evaluate_condition(
    condition: Condition,
    columns: Mapping[str, Sequence[float | None]],
    refs: list[FeatureRef],
) -> list[Truth]:
    """Compile a condition and evaluate it on every bar of a hand-fed series."""
    registry = FeatureRegistry(refs)
    compiled = compile_condition(condition, registry)
    length = len(next(iter(columns.values())))
    series = series_from_features([1.10] * length, columns)
    return [compiled(ctx) for ctx in series.contexts()]


class TestConditionCompilation:
    """Leaves, and the operand kinds they can be built from."""

    def test_feature_against_a_constant(self) -> None:
        outcomes = evaluate_condition(leaf("lt", RSI, 30.0), {RSI_KEY: [25.0, 35.0, None]}, [RSI])
        assert outcomes == [True, False, None]

    def test_price_against_a_feature(self) -> None:
        series = series_from_features([1.10, 1.20, 1.30], {SMA5_KEY: [1.15, 1.15, 1.15]})
        compiled = compile_condition(leaf("gt", "price:close", SMA5), FeatureRegistry([SMA5]))
        assert [compiled(ctx) for ctx in series.contexts()] == [False, True, True]

    def test_a_multi_output_channel_resolves_to_its_own_column(self) -> None:
        outcomes = evaluate_condition(
            leaf("gt", MACD_SIGNAL, 0.0), {MACD_SIGNAL_KEY: [-1.0, 1.0]}, [MACD_SIGNAL]
        )
        assert outcomes == [False, True]

    def test_crossing_reads_the_previous_closed_bar(self) -> None:
        outcomes = evaluate_condition(
            leaf("cross_above", RSI, 50.0),
            {RSI_KEY: [40.0, 45.0, 55.0, 60.0, 45.0]},
            [RSI],
        )
        # Bar 0 has no previous bar, so the crossing is unknown, not false.
        assert outcomes == [None, False, True, False, False]

    def test_crossing_is_unknown_while_the_indicator_is_warming_up(self) -> None:
        outcomes = evaluate_condition(
            leaf("cross_above", RSI, 50.0), {RSI_KEY: [None, None, 55.0, 60.0]}, [RSI]
        )
        assert outcomes == [None, None, None, False]

    def test_range_operators(self) -> None:
        columns = {RSI_KEY: [29.0, 30.0, 50.0, 70.0, 71.0]}
        assert evaluate_condition(leaf("between", RSI, (30.0, 70.0)), columns, [RSI]) == [
            False,
            True,
            True,
            True,
            False,
        ]
        assert evaluate_condition(leaf("inside_range", RSI, (30.0, 70.0)), columns, [RSI]) == [
            False,
            False,
            True,
            False,
            False,
        ]

    def test_slope_defaults_to_one_bar(self) -> None:
        outcomes = evaluate_condition(
            leaf("rising", RSI), {RSI_KEY: [40.0, 45.0, 45.0, 40.0]}, [RSI]
        )
        assert outcomes == [None, True, False, False]

    def test_slope_honours_an_explicit_lookback(self) -> None:
        outcomes = evaluate_condition(
            leaf("falling", RSI, 3.0), {RSI_KEY: [60.0, 10.0, 10.0, 50.0, 55.0]}, [RSI]
        )
        # Bar 3 compares against bar 0: 50 < 60. Bar 4 against bar 1: 55 > 10.
        assert outcomes == [None, None, None, True, False]


class TestNesting:
    """Arbitrarily nested AllOf / AnyOf / Not, including unknown propagation."""

    def test_all_of_requires_every_child(self) -> None:
        condition = AllOf(conditions=[leaf("gt", RSI, 50.0), leaf("lt", RSI, 70.0)])
        outcomes = evaluate_condition(condition, {RSI_KEY: [40.0, 60.0, 80.0]}, [RSI])
        assert outcomes == [False, True, False]

    def test_any_of_needs_only_one(self) -> None:
        condition = AnyOf(conditions=[leaf("lt", RSI, 30.0), leaf("gt", RSI, 70.0)])
        outcomes = evaluate_condition(condition, {RSI_KEY: [20.0, 50.0, 80.0]}, [RSI])
        assert outcomes == [True, False, True]

    def test_not_inverts(self) -> None:
        condition = Not(condition=leaf("gt", RSI, 50.0))
        outcomes = evaluate_condition(condition, {RSI_KEY: [40.0, 60.0]}, [RSI])
        assert outcomes == [True, False]

    def test_not_over_unknown_stays_unknown(self) -> None:
        # The failure this guards: a warmup null becoming a fired condition.
        condition = Not(condition=leaf("gt", RSI, 50.0))
        assert evaluate_condition(condition, {RSI_KEY: [None]}, [RSI]) == [None]

    def test_four_levels_of_nesting(self) -> None:
        # all_of( any_of( rsi<30, not(rsi>70) ), all_of( rsi != None via >0 ) )
        condition = AllOf(
            conditions=[
                AnyOf(
                    conditions=[
                        leaf("lt", RSI, 30.0),
                        Not(condition=leaf("gt", RSI, 70.0)),
                    ]
                ),
                AllOf(conditions=[leaf("gt", RSI, 0.0)]),
            ]
        )
        outcomes = evaluate_condition(condition, {RSI_KEY: [20.0, 50.0, 80.0]}, [RSI])
        assert outcomes == [True, True, False]

    def test_nesting_short_circuits_to_false_before_unknown(self) -> None:
        # all_of(false, unknown) is false: no amount of missing data can rescue a
        # child that has already been decided against.
        condition = AllOf(conditions=[leaf("gt", RSI, 90.0), leaf("gt", SMA5, 1.0)])
        outcomes = evaluate_condition(condition, {RSI_KEY: [50.0], SMA5_KEY: [None]}, [RSI, SMA5])
        assert outcomes == [False]

    def test_nesting_reports_unknown_when_it_would_otherwise_be_true(self) -> None:
        condition = AllOf(conditions=[leaf("gt", RSI, 10.0), leaf("gt", SMA5, 1.0)])
        outcomes = evaluate_condition(condition, {RSI_KEY: [50.0], SMA5_KEY: [None]}, [RSI, SMA5])
        assert outcomes == [None]


class TestRejectedSpecs:
    """Defects that must surface at compile time, never mid-backtest."""

    def test_regime_is_is_rejected_until_a_regime_module_exists(self) -> None:
        # The schema accepts it so specs can be written against the eventual
        # contract; the compiler will not pretend it can classify a bar.
        spec = strategy_spec(trigger=leaf("regime_is", None, labels("RANGE")))
        with pytest.raises(ValidationError, match="no Regime module"):
            compile_entry(spec, FeatureRegistry.from_strategy(spec))

    @pytest.mark.parametrize("op", ["pattern_is", "session_is"])
    def test_a_categorical_operator_without_labels_is_rejected(self, op: str) -> None:
        spec = strategy_spec(trigger=leaf(op, None, 1.0))
        with pytest.raises(ValidationError, match="requires a label set"):
            compile_entry(spec, FeatureRegistry.from_strategy(spec))

    def test_a_categorical_operator_with_a_left_operand_is_rejected(self) -> None:
        spec = strategy_spec(trigger=leaf("pattern_is", RSI, labels("DOJI")))
        with pytest.raises(ValidationError, match="left must be omitted"):
            compile_entry(spec, FeatureRegistry.from_strategy(spec))

    def test_labels_handed_to_a_numeric_operator_are_rejected(self) -> None:
        spec = strategy_spec(trigger=leaf("gt", RSI, labels("DOJI")))
        with pytest.raises(ValidationError, match="compares numbers"):
            compile_entry(spec, FeatureRegistry.from_strategy(spec))

    def test_a_feature_the_registry_does_not_provide_is_rejected(self) -> None:
        spec = strategy_spec(trigger=leaf("gt", RSI, 50.0))
        with pytest.raises(ValidationError, match="not provided by this registry"):
            compile_entry(spec, FeatureRegistry([SMA5]))

    def test_inverted_range_bounds_are_rejected(self) -> None:
        with pytest.raises(ValidationError, match="inverted bounds"):
            compile_condition(leaf("between", RSI, (70.0, 30.0)), FeatureRegistry([RSI]))

    def test_a_range_where_a_scalar_belongs_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="single right operand"):
            compile_condition(leaf("gt", RSI, (30.0, 70.0)), FeatureRegistry([RSI]))

    def test_a_scalar_where_a_range_belongs_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match=r"\[low, high\] pair"):
            compile_condition(leaf("between", RSI, 30.0), FeatureRegistry([RSI]))

    def test_a_missing_left_operand_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="requires a left operand"):
            compile_condition(leaf("gt", None, 30.0), FeatureRegistry([]))

    @pytest.mark.parametrize("lookback", [0.0, -1.0, 1.5])
    def test_a_nonsensical_slope_lookback_is_rejected(self, lookback: float) -> None:
        with pytest.raises(ValidationError, match="whole lookback"):
            compile_condition(leaf("rising", RSI, lookback), FeatureRegistry([RSI]))


class TestEvaluation:
    """The trigger / confirmation / invalidation state machine."""

    def test_a_bare_trigger_fires_on_its_own_bar(self) -> None:
        spec = strategy_spec(trigger=leaf("gt", RSI, 50.0))
        series = series_from_features([1.10] * 4, {RSI_KEY: [40.0, 60.0, 40.0, 70.0]})
        signals = compile_entry(spec, FeatureRegistry.from_strategy(spec)).run(series).signals
        assert [signal.context["signal_bar_index"] for signal in signals] == [1, 3]

    def test_confirmation_may_land_on_the_trigger_bar_itself(self) -> None:
        spec = strategy_spec(
            trigger=leaf("gt", RSI, 50.0),
            confirmation=[leaf("gt", RSI, 55.0)],
            confirmation_window_bars=1,
        )
        series = series_from_features([1.10] * 3, {RSI_KEY: [40.0, 60.0, 40.0]})
        signals = compile_entry(spec, FeatureRegistry.from_strategy(spec)).run(series).signals
        assert [signal.context["signal_bar_index"] for signal in signals] == [1]

    def test_confirmation_arriving_later_in_the_window_fires_then(self) -> None:
        spec = strategy_spec(
            trigger=leaf("gt", RSI, 50.0),
            confirmation=[leaf("gt", RSI, 70.0)],
            confirmation_window_bars=3,
        )
        series = series_from_features([1.10] * 5, {RSI_KEY: [40.0, 60.0, 60.0, 80.0, 40.0]})
        signals = compile_entry(spec, FeatureRegistry.from_strategy(spec)).run(series).signals
        assert len(signals) == 1
        assert signals[0].context["trigger_bar_index"] == 1
        assert signals[0].context["signal_bar_index"] == 3
        assert signals[0].context["bars_to_confirm"] == 2

    def test_a_window_of_n_gives_n_bars_including_the_trigger_bar(self) -> None:
        # An edge trigger, so the window is the only thing under test: a level
        # trigger would still hold at bar 3 and open a second setup there.
        rsi = [40.0, 60.0, 60.0, 80.0, 40.0]
        series = series_from_features([1.10] * 5, {RSI_KEY: rsi})

        def spec_with(window: int) -> Any:
            return strategy_spec(
                trigger=leaf("cross_above", RSI, 50.0),
                confirmation=[leaf("gt", RSI, 70.0)],
                confirmation_window_bars=window,
            )

        # Trigger at bar 1. A window of 2 covers bars 1 and 2; confirmation only
        # arrives at bar 3, one bar too late. A window of 3 reaches it.
        narrow = spec_with(2)
        assert (
            compile_entry(narrow, FeatureRegistry.from_strategy(narrow)).run(series).signals == ()
        )

        wide = spec_with(3)
        signals = compile_entry(wide, FeatureRegistry.from_strategy(wide)).run(series).signals
        assert [signal.context["signal_bar_index"] for signal in signals] == [3]

    def test_a_dead_setup_does_not_block_a_new_trigger_on_the_same_bar(self) -> None:
        # A setup that expired and a trigger firing again are two different
        # setups: killing the first says nothing about the second, which gets its
        # own window and its own confirmations. Only the dead setup's latched
        # confirmations are discarded. Blocking the bar instead would silently
        # lose an edge trigger that happened to land on it.
        spec = strategy_spec(
            trigger=leaf("gt", RSI, 50.0),
            confirmation=[leaf("gt", RSI, 70.0)],
            confirmation_window_bars=2,
        )
        series = series_from_features([1.10] * 5, {RSI_KEY: [40.0, 60.0, 60.0, 80.0, 40.0]})
        signals = compile_entry(spec, FeatureRegistry.from_strategy(spec)).run(series).signals
        assert [signal.context["trigger_bar_index"] for signal in signals] == [3]

    def test_several_confirmations_latch_independently(self) -> None:
        # "All hold within n bars of the trigger" is read as each becoming true
        # somewhere in the window, not all on one bar simultaneously.
        spec = strategy_spec(
            trigger=leaf("gt", RSI, 50.0),
            confirmation=[leaf("gt", SMA5, 1.2), leaf("gt", RSI, 70.0)],
            confirmation_window_bars=4,
        )
        series = series_from_features(
            [1.10] * 5,
            {RSI_KEY: [40.0, 60.0, 60.0, 80.0, 40.0], SMA5_KEY: [1.0, 1.0, 1.5, 1.0, 1.0]},
        )
        signals = compile_entry(spec, FeatureRegistry.from_strategy(spec)).run(series).signals
        assert len(signals) == 1
        assert signals[0].context["confirmation_bars"] == (2, 3)

    def test_a_fresh_trigger_is_ignored_while_a_setup_is_pending(self) -> None:
        spec = strategy_spec(
            trigger=leaf("gt", RSI, 50.0),
            confirmation=[leaf("gt", RSI, 70.0)],
            confirmation_window_bars=4,
        )
        series = series_from_features([1.10] * 5, {RSI_KEY: [40.0, 60.0, 65.0, 80.0, 40.0]})
        signals = compile_entry(spec, FeatureRegistry.from_strategy(spec)).run(series).signals
        assert len(signals) == 1
        assert signals[0].context["trigger_bar_index"] == 1

    def test_a_pending_setup_is_killed_when_price_pierces_the_invalidation_level(self) -> None:
        spec = strategy_spec(
            trigger=leaf("cross_above", RSI, 50.0),
            confirmation=[leaf("gt", RSI, 70.0)],
            confirmation_window_bars=4,
            invalidation=Invalidation(price_level=1.05),
        )
        closes = [1.10] * 5
        rsi = [40.0, 60.0, 60.0, 80.0, 40.0]
        registry = FeatureRegistry.from_strategy(spec)

        survived = (
            compile_entry(spec, registry).run(series_from_features(closes, {RSI_KEY: rsi})).signals
        )
        assert len(survived) == 1

        pierced = (
            compile_entry(spec, registry)
            .run(series_from_features(closes, {RSI_KEY: rsi}, lows=[1.10, 1.10, 1.04, 1.10, 1.10]))
            .signals
        )
        assert pierced == ()

    def test_the_trigger_bar_itself_is_not_subject_to_invalidation(self) -> None:
        # A level drawn from a feature the trigger references would otherwise
        # kill nearly every setup on the bar it was born, since the bar's low
        # routinely pierces a level its close sits above.
        spec = strategy_spec(
            trigger=leaf("gt", RSI, 50.0),
            confirmation=[leaf("gt", RSI, 70.0)],
            confirmation_window_bars=4,
            invalidation=Invalidation(price_level=1.05),
        )
        series = series_from_features(
            [1.10] * 4, {RSI_KEY: [40.0, 80.0, 60.0, 60.0]}, lows=[1.10, 1.04, 1.10, 1.10]
        )
        signals = compile_entry(spec, FeatureRegistry.from_strategy(spec)).run(series).signals
        assert [signal.context["signal_bar_index"] for signal in signals] == [1]

    def test_an_invalidation_condition_also_kills_a_pending_setup(self) -> None:
        spec = strategy_spec(
            trigger=leaf("cross_above", RSI, 50.0),
            confirmation=[leaf("gt", RSI, 70.0)],
            confirmation_window_bars=4,
            invalidation=Invalidation(price_level=1.05, condition=leaf("lt", SMA5, 1.0)),
        )
        series = series_from_features(
            [1.10] * 5,
            {RSI_KEY: [40.0, 60.0, 60.0, 80.0, 40.0], SMA5_KEY: [1.5, 1.5, 0.5, 1.5, 1.5]},
        )
        assert compile_entry(spec, FeatureRegistry.from_strategy(spec)).run(series).signals == ()

    def test_reset_discards_a_pending_setup(self) -> None:
        spec = strategy_spec(
            trigger=leaf("gt", RSI, 50.0),
            confirmation=[leaf("gt", RSI, 70.0)],
            confirmation_window_bars=4,
        )
        series = series_from_features([1.10] * 3, {RSI_KEY: [40.0, 60.0, 60.0]})
        engine = compile_entry(spec, FeatureRegistry.from_strategy(spec))
        (leg,) = engine.evaluators
        engine.evaluate(series.context(0))
        engine.evaluate(series.context(1))
        assert leg.has_pending_setup
        engine.reset()
        assert not leg.has_pending_setup


class TestSignalContents:
    """What ends up on the signal, and what deliberately does not."""

    def _one_signal(self, **kwargs: Any) -> EntrySignal:
        spec = strategy_spec(trigger=leaf("gt", RSI, 50.0), **kwargs)
        series = series_from_features([1.10, 1.20], {RSI_KEY: [40.0, 60.0]})
        signals = compile_entry(spec, FeatureRegistry.from_strategy(spec)).run(series).signals
        assert len(signals) == 1
        return signals[0]

    def test_market_order_anchors_to_the_signal_bar_close(self) -> None:
        assert self._one_signal().reference_price == pytest.approx(1.20)

    def test_limit_order_anchors_below_the_close_for_a_long(self) -> None:
        signal = self._one_signal(order=LimitOrder(offset=0.0005))
        assert signal.reference_price == pytest.approx(1.1995)

    def test_stop_order_anchors_above_the_close_for_a_long(self) -> None:
        signal = self._one_signal(order=StopOrder(offset=0.0005))
        assert signal.reference_price == pytest.approx(1.2005)

    def test_limit_order_anchors_above_the_close_for_a_short(self) -> None:
        spec = strategy_spec(
            trigger=leaf("gt", RSI, 50.0),
            direction="SHORT",
            invalidation=Invalidation(price_level=1.5),
            order=LimitOrder(offset=0.0005),
        )
        series = series_from_features([1.10, 1.20], {RSI_KEY: [40.0, 60.0]})
        signals = compile_entry(spec, FeatureRegistry.from_strategy(spec)).run(series).signals
        assert signals[0].side is Side.SELL
        assert signals[0].reference_price == pytest.approx(1.2005)

    def test_the_signal_bar_close_time_is_the_bar_after_its_open(self) -> None:
        signal = self._one_signal()
        series = series_from_features([1.10, 1.20], {RSI_KEY: [40.0, 60.0]})
        assert signal.bar_close_ts == series.context(1).bar_close_ts

    def test_quality_is_the_base_when_no_modifier_holds(self) -> None:
        assert self._one_signal(base_quality=0.4).quality == pytest.approx(0.4)

    def test_a_holding_modifier_adds_its_delta(self) -> None:
        signal = self._one_signal(
            base_quality=0.4,
            quality_modifiers=[
                QualityModifier(condition=leaf("gt", RSI, 55.0), delta=0.2, reason="strong"),
                QualityModifier(condition=leaf("gt", RSI, 95.0), delta=0.3, reason="extreme"),
            ],
        )
        assert signal.quality == pytest.approx(0.6)
        assert signal.context["quality_modifiers"] == (("strong", 0.2),)

    def test_quality_is_clamped_and_the_raw_sum_kept_for_forensics(self) -> None:
        signal = self._one_signal(
            base_quality=0.9,
            quality_modifiers=[
                QualityModifier(condition=leaf("gt", RSI, 55.0), delta=0.5, reason="strong")
            ],
        )
        assert signal.quality == 1.0
        assert signal.context["quality_before_clamp"] == pytest.approx(1.4)

    def test_the_context_records_what_fired(self) -> None:
        context = self._one_signal().context
        assert context["trigger_bar_index"] == 1
        assert context["signal_bar_index"] == 1
        assert context["entry_order"] == "MARKET"
        assert context["features"] == {RSI_KEY: 60.0}

    def test_a_signal_whose_invalidation_lands_on_the_wrong_side_is_dropped(self) -> None:
        # Long, but the invalidation level sits above the entry: a defective
        # setup definition. Dropped here rather than handed on to be abs()-ed
        # into a plausible position size.
        spec = strategy_spec(
            trigger=leaf("gt", RSI, 50.0), invalidation=Invalidation(price_level=1.5)
        )
        series = series_from_features([1.10, 1.20], {RSI_KEY: [40.0, 60.0]})
        assert compile_entry(spec, FeatureRegistry.from_strategy(spec)).run(series).signals == ()

    def test_a_wrong_sided_bar_is_dropped_but_the_run_carries_on(self) -> None:
        # The reason this is a per-bar drop and not a raise: the invalidation
        # level is a feature value, so whether it lands on the right side is a
        # property of the bar, not of the spec. One defective bar must not end a
        # backtest, and must not be quietly repaired either.
        spec = strategy_spec(
            trigger=leaf("gt", RSI, 50.0), invalidation=Invalidation(price_level=SMA5)
        )
        series = series_from_features(
            [1.10, 1.20, 1.30, 1.40],
            # Bar 1: level 1.50 sits above the 1.20 entry — defective, dropped.
            # Bar 3: level 1.00 sits below the 1.40 entry — a usable signal.
            {RSI_KEY: [40.0, 60.0, 40.0, 60.0], SMA5_KEY: [1.0, 1.5, 1.0, 1.0]},
        )
        with capture_logs() as logs:
            signals = compile_entry(spec, FeatureRegistry.from_strategy(spec)).run(series).signals

        assert [signal.context["signal_bar_index"] for signal in signals] == [3]
        dropped = [entry for entry in logs if entry["event"] == "entry.signal_dropped"]
        assert len(dropped) == 1
        assert dropped[0]["log_level"] == "warning"
        assert dropped[0]["bar_index"] == 1
        assert dropped[0]["reference_price"] == pytest.approx(1.20)
        assert dropped[0]["invalidation_price"] == pytest.approx(1.50)

    def test_the_signal_type_itself_refuses_to_hold_a_wrong_sided_stop(self) -> None:
        # The second layer: even if a future caller assembled a signal by hand,
        # bypassing the evaluator, the object cannot exist.
        with pytest.raises(ValueError, match="strictly below"):
            EntrySignal(
                strategy_id="test-entry",
                symbol="TESTFX",
                bar_close_ts=datetime(2024, 1, 2, 9, 15, tzinfo=UTC),
                side=Side.BUY,
                reference_price=Price(1.20),
                invalidation_price=Price(1.50),
                quality=0.5,
            )

    def test_a_signal_is_dropped_when_the_invalidation_feature_has_no_value(self) -> None:
        spec = strategy_spec(
            trigger=leaf("gt", RSI, 50.0), invalidation=Invalidation(price_level=SMA5)
        )
        series = series_from_features([1.10, 1.20], {RSI_KEY: [40.0, 60.0], SMA5_KEY: [None, None]})
        assert compile_entry(spec, FeatureRegistry.from_strategy(spec)).run(series).signals == ()


class TestNoLookahead:
    """A longer history must not change what was decided on an earlier bar."""

    @staticmethod
    def _oscillating_closes(count: int) -> list[float]:
        """A deterministic wave that crosses its own moving average repeatedly."""
        return [1.1000 + 0.0040 * math.sin(index / 3.0) + 0.00002 * index for index in range(count)]

    def _run(self, spec: Any, registry: FeatureRegistry, closes: list[float]) -> list[EntrySignal]:
        """Compute features over exactly these bars and run the entry over them."""
        frame = frame_from_closes(
            closes,
            lows=[close - 0.0020 for close in closes],
            highs=[close + 0.0020 for close in closes],
        )
        features = registry.pipeline().compute(frame)
        return list(
            compile_entry(spec, registry).run(BarSeries.from_frame(frame, features)).signals
        )

    @pytest.mark.parametrize("prefix_length", [90, 110, 130])
    def test_a_prefix_run_matches_the_full_run_on_the_overlap(self, prefix_length: int) -> None:
        spec = strategy_spec(
            trigger=leaf("cross_above", "price:close", SMA5),
            confirmation=[leaf("gt", "price:close", SMA5)],
            confirmation_window_bars=3,
            invalidation=Invalidation(price_level=1.0900),
        )
        registry = FeatureRegistry.from_strategy(spec)
        closes = self._oscillating_closes(160)

        full = self._run(spec, registry, closes)
        prefix = self._run(spec, registry, closes[:prefix_length])

        assert prefix, "the fixture must produce signals or the test proves nothing"
        cutoff = prefix[-1].bar_close_ts
        assert prefix == [signal for signal in full if signal.bar_close_ts <= cutoff]

    def test_the_overlap_holds_for_a_multi_output_indicator_too(self) -> None:
        spec = strategy_spec(
            trigger=leaf("cross_above", ref("macd", channel="macd"), MACD_SIGNAL),
            invalidation=Invalidation(price_level=1.0900),
        )
        registry = FeatureRegistry.from_strategy(spec)
        closes = self._oscillating_closes(200)

        full = self._run(spec, registry, closes)
        prefix = self._run(spec, registry, closes[:150])

        assert prefix
        cutoff = prefix[-1].bar_close_ts
        assert prefix == [signal for signal in full if signal.bar_close_ts <= cutoff]

    def test_a_setup_pending_at_the_cut_is_not_resolved_by_absent_future_bars(self) -> None:
        spec = strategy_spec(
            trigger=leaf("gt", RSI, 50.0),
            confirmation=[leaf("gt", RSI, 70.0)],
            confirmation_window_bars=4,
        )
        registry = FeatureRegistry.from_strategy(spec)
        rsi = [40.0, 60.0, 60.0, 80.0]
        closes = [1.10] * 4

        evaluator = compile_entry(spec, registry)
        cut = evaluator.run(series_from_features(closes[:3], {RSI_KEY: rsi[:3]})).signals
        assert cut == ()
        assert evaluator.evaluators[0].has_pending_setup, (
            "the setup must still be waiting, not resolved by bars that do not exist yet"
        )

        whole = (
            compile_entry(spec, registry).run(series_from_features(closes, {RSI_KEY: rsi})).signals
        )
        assert [signal.context["signal_bar_index"] for signal in whole] == [3]
