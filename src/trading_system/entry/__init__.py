"""Entry Engine: recognising setups on closed bars and scoring them.

The layer between a :class:`~trading_system.strategies.schema.StrategySpec` and
the Risk Engine. It answers "is there a setup here, which way, and how good is
it" and refuses to answer "how much" — an :class:`~.signal.EntrySignal` carries
no size and no money, by construction.

A run has four steps::

    registry   = FeatureRegistry.from_strategy(spec)
    features   = registry.pipeline().compute(frame)
    categories = required_label_categories(spec)
    series     = BarSeries.from_frame(frame, features, categories)
    result     = compile_entry(spec, registry).run(series)   # .signals and .drops

The Exit Engine is a separate library and is not imported here; a spec names its
exit by ``exit_ref`` so the two combine N×M.
"""

from trading_system.entry.compiler import (
    ConditionFn,
    DropReason,
    EntryEngine,
    EntryEvaluator,
    EntryRun,
    OperandFn,
    compile_condition,
    compile_entry,
    compile_operand,
)
from trading_system.entry.context import PRICE_FIELDS, BarContext, BarSeries
from trading_system.entry.features import (
    FeatureRegistry,
    feature_key,
    iter_feature_refs,
    required_label_categories,
)
from trading_system.entry.labels import LabelCategory, label_columns
from trading_system.entry.operators import Truth
from trading_system.entry.signal import EntrySignal

__all__ = [
    "PRICE_FIELDS",
    "BarContext",
    "BarSeries",
    "ConditionFn",
    "DropReason",
    "EntryEngine",
    "EntryEvaluator",
    "EntryRun",
    "EntrySignal",
    "FeatureRegistry",
    "LabelCategory",
    "OperandFn",
    "Truth",
    "compile_condition",
    "compile_entry",
    "compile_operand",
    "feature_key",
    "iter_feature_refs",
    "label_columns",
    "required_label_categories",
]
