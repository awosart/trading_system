"""Feature engineering: indicators, patterns, and the pipeline that runs them.

Indicators are defined once and evaluated two ways — vectorised for research,
incrementally for live — and the two are held to agreement by
:meth:`~trading_system.features.base.BaseIndicator.verify_parity`. See
:mod:`trading_system.features.base` for the contracts every indicator obeys.
"""

from trading_system.features.base import (
    BaseIndicator,
    BaseStreaming,
    Indicator,
    MultiOutputIndicator,
    StreamingIndicator,
    iter_bars,
    run_streaming,
)
from trading_system.features.pipeline import (
    FeatureCache,
    FeaturePipeline,
    FeaturePipelineConfig,
    FeatureSet,
    FeatureSpec,
    InMemoryFeatureCache,
    pipeline_from_specs,
)
from trading_system.features.registry import INDICATOR_TYPES, build_indicator, default_indicators

__all__ = [
    "INDICATOR_TYPES",
    "BaseIndicator",
    "BaseStreaming",
    "FeatureCache",
    "FeaturePipeline",
    "FeaturePipelineConfig",
    "FeatureSet",
    "FeatureSpec",
    "InMemoryFeatureCache",
    "Indicator",
    "MultiOutputIndicator",
    "StreamingIndicator",
    "build_indicator",
    "default_indicators",
    "iter_bars",
    "pipeline_from_specs",
    "run_streaming",
]
