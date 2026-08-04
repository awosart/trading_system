"""The feature pipeline: naming, warmup, validation and caching."""

import polars as pl
import pytest

from trading_system.core.exceptions import ValidationError
from trading_system.data.models import TIMESTAMP_COLUMN, OHLCVFrame
from trading_system.features.pipeline import (
    FeaturePipeline,
    FeaturePipelineConfig,
    FeatureSpec,
    InMemoryFeatureCache,
    pipeline_from_specs,
)

SPECS = [
    FeatureSpec(name="rsi", kind="rsi", params={"period": 14}),
    FeatureSpec(name="trend", kind="macd"),
    FeatureSpec(name="vol", kind="atr", params={"period": 20}),
]


class CountingCache:
    """A cache that records how often it was consulted."""

    def __init__(self) -> None:
        """Start empty with both counters at zero."""
        self.entries: dict[str, pl.DataFrame] = {}
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> pl.DataFrame | None:
        """Look a key up, counting the hit or miss."""
        value = self.entries.get(key)
        if value is None:
            self.misses += 1
        else:
            self.hits += 1
        return value

    def put(self, key: str, value: pl.DataFrame) -> None:
        """Store a frame."""
        self.entries[key] = value


def test_single_valued_indicators_take_the_spec_name() -> None:
    pipeline = FeaturePipeline([FeatureSpec(name="fast", kind="ema", params={"period": 8})])
    assert pipeline.feature_columns == ("fast",)


def test_multi_channel_indicators_are_prefixed_by_the_spec_name() -> None:
    pipeline = FeaturePipeline([FeatureSpec(name="trend", kind="macd")])
    assert pipeline.feature_columns == ("trend_macd", "trend_signal", "trend_histogram")


def test_warmup_is_the_slowest_indicator_in_the_pipeline() -> None:
    pipeline = FeaturePipeline(SPECS)
    assert pipeline.warmup_bars == max(indicator.warmup for indicator in pipeline.indicators)


def test_compute_aligns_features_to_the_frames_bars(reference_frame: OHLCVFrame) -> None:
    features = FeaturePipeline(SPECS).compute(reference_frame)
    assert len(features) == len(reference_frame)
    assert features.timestamps.to_list() == reference_frame.timestamps.to_list()
    assert tuple(features.df.columns) == (TIMESTAMP_COLUMN, *features.feature_columns)
    assert features.symbol == reference_frame.symbol
    assert features.timeframe is reference_frame.timeframe


def test_each_column_keeps_its_own_indicators_warmup(
    reference_frame: OHLCVFrame,
) -> None:
    """No value at all through the warmup — not a filled one — column by column.

    A fast RSI is honestly computable before a slow MACD is, so it is published
    there. The pipeline-wide boundary is applied by :meth:`FeatureSet.valid`, not
    by truncating every series to the slowest member.
    """
    pipeline = FeaturePipeline(SPECS)
    features = pipeline.compute(reference_frame)
    for spec, indicator in zip(pipeline.specs, pipeline.indicators, strict=True):
        columns = [
            column
            for column in features.feature_columns
            if column == spec.name or column.startswith(f"{spec.name}_")
        ]
        for column in columns:
            values = features.df[column].to_list()
            assert values[: indicator.warmup] == [None] * indicator.warmup
            assert values[indicator.warmup] is not None


def test_valid_drops_the_warmup_instead_of_imputing_it(
    reference_frame: OHLCVFrame,
) -> None:
    features = FeaturePipeline(SPECS).compute(reference_frame)
    valid = features.valid()
    assert valid.height == len(reference_frame) - features.warmup_bars
    assert valid[TIMESTAMP_COLUMN].item(0) == reference_frame.timestamps.item(features.warmup_bars)
    for column in features.feature_columns:
        assert valid[column].null_count() == 0


def test_valid_mask_also_excludes_data_dependent_gaps(
    reference_frame: OHLCVFrame,
) -> None:
    """An anchored VWAP is null before its anchor, whatever the bar-count warmup says."""
    anchor = reference_frame.timestamps.item(100)
    pipeline = FeaturePipeline(
        [FeatureSpec(name="avwap", kind="vwap_anchored", params={"anchor": anchor})]
    )
    features = pipeline.compute(reference_frame)
    assert features.warmup_bars == 0
    assert features.valid_mask().to_list()[:100] == [False] * 100
    assert features.valid().height == len(reference_frame) - 100


def test_an_empty_pipeline_still_produces_aligned_timestamps(
    reference_frame: OHLCVFrame,
) -> None:
    features = FeaturePipeline([]).compute(reference_frame)
    assert features.feature_columns == ()
    assert features.warmup_bars == 0
    assert features.valid().height == len(reference_frame)


def test_duplicate_feature_names_are_rejected() -> None:
    with pytest.raises(ValidationError, match="duplicate feature names"):
        FeaturePipeline([FeatureSpec(name="x", kind="sma"), FeatureSpec(name="x", kind="ema")])


def test_colliding_generated_columns_are_rejected() -> None:
    """``trend`` + MACD emits ``trend_macd``, which a second spec must not also claim."""
    with pytest.raises(ValidationError, match="colliding columns"):
        FeaturePipeline(
            [
                FeatureSpec(name="trend", kind="macd"),
                FeatureSpec(name="trend_macd", kind="sma"),
            ]
        )


def test_an_unknown_indicator_fails_at_construction() -> None:
    with pytest.raises(ValidationError, match="unknown indicator"):
        FeaturePipeline([FeatureSpec(name="x", kind="not_an_indicator")])


def test_invalid_indicator_parameters_fail_at_construction() -> None:
    with pytest.raises(ValidationError, match="cannot build indicator"):
        FeaturePipeline([FeatureSpec(name="x", kind="sma", params={"period": 0})])
    with pytest.raises(ValidationError, match="cannot build indicator"):
        FeaturePipeline([FeatureSpec(name="x", kind="sma", params={"nonsense": 1})])


def test_specs_reject_unknown_keys() -> None:
    with pytest.raises(Exception, match="extra_forbidden|Extra inputs"):
        FeatureSpec.model_validate({"name": "x", "kind": "sma", "typo": 1})


def test_pipeline_from_specs_accepts_plain_mappings(reference_frame: OHLCVFrame) -> None:
    pipeline = pipeline_from_specs(
        [
            {"name": "rsi", "kind": "rsi", "params": {"period": 7}},
            {"name": "bb", "kind": "bbands"},
        ]
    )
    assert pipeline.feature_columns[0] == "rsi"
    assert pipeline.compute(reference_frame).df.height == len(reference_frame)


def test_pipeline_from_specs_reports_malformed_input() -> None:
    with pytest.raises(ValidationError, match="invalid feature specification"):
        pipeline_from_specs([{"kind": "sma"}])


def test_a_config_object_is_accepted_directly() -> None:
    config = FeaturePipelineConfig(features=SPECS)
    assert FeaturePipeline(config).feature_columns == FeaturePipeline(SPECS).feature_columns


def test_a_repeated_computation_is_served_from_the_cache(
    reference_frame: OHLCVFrame,
) -> None:
    cache = CountingCache()
    pipeline = FeaturePipeline(SPECS, cache=cache)
    first = pipeline.compute(reference_frame)
    second = pipeline.compute(reference_frame)
    assert (cache.misses, cache.hits) == (1, 1)
    assert first.df.equals(second.df)


def test_the_cache_key_covers_the_bar_values(reference_frame: OHLCVFrame) -> None:
    """Correcting one bar in the middle of stored history must invalidate the entry."""
    pipeline = FeaturePipeline(SPECS, cache=CountingCache())
    original = pipeline.cache_key(reference_frame)

    edited = reference_frame.with_df(
        reference_frame.df.with_columns(
            pl.when(pl.int_range(0, pl.len()) == 42)
            .then(pl.col("close") * 1.01)
            .otherwise(pl.col("close"))
            .alias("close")
        )
    )
    assert pipeline.cache_key(edited) != original


def test_the_cache_key_covers_the_instrument(reference_frame: OHLCVFrame) -> None:
    pipeline = FeaturePipeline(SPECS)
    other = OHLCVFrame(reference_frame.df, "OTHER", reference_frame.timeframe)
    assert pipeline.cache_key(other) != pipeline.cache_key(reference_frame)


def test_the_cache_key_covers_the_pipeline_definition(
    reference_frame: OHLCVFrame,
) -> None:
    slow = FeaturePipeline([FeatureSpec(name="x", kind="sma", params={"period": 50})])
    fast = FeaturePipeline([FeatureSpec(name="x", kind="sma", params={"period": 10})])
    assert slow.cache_key(reference_frame) != fast.cache_key(reference_frame)


def test_pipelines_without_a_cache_recompute(reference_frame: OHLCVFrame) -> None:
    pipeline = FeaturePipeline(SPECS)
    first = pipeline.compute(reference_frame)
    second = pipeline.compute(reference_frame)
    assert first.df.equals(second.df)


def test_in_memory_cache_evicts_the_least_recently_used() -> None:
    cache = InMemoryFeatureCache(max_entries=2)
    frames = {name: pl.DataFrame({"v": [1.0]}) for name in ("a", "b", "c")}
    cache.put("a", frames["a"])
    cache.put("b", frames["b"])
    assert cache.get("a") is not None  # "a" is now the most recent
    cache.put("c", frames["c"])
    assert len(cache) == 2
    assert cache.get("b") is None
    assert cache.get("a") is not None
    assert cache.get("c") is not None


def test_in_memory_cache_rejects_a_zero_bound() -> None:
    with pytest.raises(ValueError, match="positive"):
        InMemoryFeatureCache(max_entries=0)


def test_signature_names_the_resolved_indicators() -> None:
    pipeline = FeaturePipeline([FeatureSpec(name="fast", kind="ema", params={"period": 8})])
    assert pipeline.signature() == "fast=ema_8"
