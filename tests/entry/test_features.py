"""FeatureRef binding: one definition of which column answers which reference."""

import pytest

from trading_system.core.exceptions import ValidationError
from trading_system.entry.context import BarSeries
from trading_system.entry.features import FeatureRegistry, feature_key, iter_feature_refs
from trading_system.strategies.schema import Invalidation, QualityModifier, VolatilityRangeFilter

from .conftest import frame_from_closes, leaf, ref, strategy_spec


class TestFeatureKey:
    """The key is the resolved indicator's identity, plus a channel if it has one."""

    def test_single_output_key_is_the_indicator_name(self) -> None:
        assert feature_key(ref("rsi", period=14)) == "rsi_14"

    def test_multi_output_key_carries_the_channel(self) -> None:
        assert feature_key(ref("macd", channel="signal")) == "macd_12_26_9_signal"

    def test_defaults_resolve_to_the_same_key_as_writing_them_out(self) -> None:
        # Two spellings of the same feature must compute once, not twice.
        assert feature_key(ref("rsi")) == feature_key(ref("rsi", period=14))

    def test_parameter_order_does_not_change_the_key(self) -> None:
        assert feature_key(ref("macd", channel="macd", fast_period=12, slow_period=26)) == (
            feature_key(ref("macd", channel="macd", slow_period=26, fast_period=12))
        )

    def test_an_unknown_indicator_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="not registered"):
            feature_key(ref("emma", period=20))

    def test_a_missing_channel_on_a_multi_output_indicator_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="channel is required"):
            feature_key(ref("macd"))

    def test_an_unknown_channel_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="no channel"):
            feature_key(ref("macd", channel="trend"))

    def test_a_channel_on_a_single_output_indicator_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="single-output"):
            feature_key(ref("rsi", channel="value", period=14))

    def test_invalid_parameters_are_rejected(self) -> None:
        with pytest.raises(ValidationError, match="cannot build indicator"):
            feature_key(ref("rsi", period=0))


class TestCollection:
    """Every place in a spec a feature can hide."""

    def test_every_reference_in_a_spec_is_found(self) -> None:
        spec = strategy_spec(
            trigger=leaf("gt", ref("rsi", period=14), 50.0),
            confirmation=[leaf("gt", "price:close", ref("sma", period=5))],
            confirmation_window_bars=2,
            invalidation=Invalidation(
                price_level=ref("ema", period=20),
                condition=leaf("lt", ref("atr", period=14), 0.001),
            ),
            quality_modifiers=[
                QualityModifier(
                    condition=leaf("gt", ref("adx", channel="adx", period=14), 25.0), delta=0.1
                )
            ],
        )
        spec = spec.model_copy(
            update={
                "filters": [VolatilityRangeFilter(feature=ref("atr", period=20), min_value=0.0001)]
            }
        )
        keys = FeatureRegistry.from_strategy(spec).keys
        assert keys == ("adx_14_adx", "atr_14", "atr_20", "ema_20", "rsi_14", "sma_5")

    def test_duplicate_references_collapse(self) -> None:
        spec = strategy_spec(
            trigger=leaf("gt", ref("rsi", period=14), 50.0),
            confirmation=[leaf("lt", ref("rsi", period=14), 70.0)],
            confirmation_window_bars=2,
        )
        assert list(iter_feature_refs(spec)).count(ref("rsi", period=14)) == 2
        assert FeatureRegistry.from_strategy(spec).keys == ("rsi_14",)


class TestRegistry:
    """Resolution, and the pipeline that produces what it promises."""

    def test_resolve_returns_the_column_name(self) -> None:
        registry = FeatureRegistry([ref("rsi", period=14)])
        assert registry.resolve(ref("rsi", period=14)) == "rsi_14"

    def test_resolving_a_feature_it_does_not_provide_lists_what_it_does(self) -> None:
        registry = FeatureRegistry([ref("rsi", period=14)])
        with pytest.raises(ValidationError, match=r"available: \['rsi_14'\]"):
            registry.resolve(ref("ema", period=20))

    def test_the_pipeline_produces_every_key(self) -> None:
        registry = FeatureRegistry(
            [ref("rsi", period=14), ref("macd", channel="signal"), ref("sma", period=5)]
        )
        frame = frame_from_closes([1.10 + 0.001 * index for index in range(120)])
        columns = registry.pipeline().compute(frame).feature_columns
        assert set(registry.keys) <= set(columns)

    def test_a_multi_output_indicator_is_requested_once(self) -> None:
        registry = FeatureRegistry(
            [ref("macd", channel="signal"), ref("macd", channel="histogram")]
        )
        assert len(registry.specs) == 1
        assert registry.keys == ("macd_12_26_9_histogram", "macd_12_26_9_signal")

    def test_the_pipeline_output_drops_straight_into_a_bar_series(self) -> None:
        registry = FeatureRegistry([ref("sma", period=5), ref("macd", channel="signal")])
        frame = frame_from_closes([1.10 + 0.001 * index for index in range(120)])
        series = BarSeries.from_frame(frame, registry.pipeline().compute(frame))
        for key in registry.keys:
            assert series.context(119).feature(key) is not None

    def test_an_empty_registry_provides_nothing(self) -> None:
        assert FeatureRegistry().keys == ()
        assert FeatureRegistry().specs == ()
