"""Binding P04's :class:`FeatureRef` to the columns a P03 pipeline produces.

A :class:`~trading_system.strategies.schema.FeatureRef` is structural — an
indicator name, its parameters, and a channel. A
:class:`~trading_system.features.pipeline.FeatureSet` is a table of named
columns. Something has to be the single definition of which column answers which
reference, and it is :func:`feature_key` here. Both the pipeline that computes
the features and the compiled conditions that read them derive their names from
it, so they cannot disagree; a registry built from a spec can then build exactly
the pipeline that spec needs, with no hand-maintained list in between.

The key is the resolved indicator's own ``name``, which already encodes its
parameters (``ema_20``, ``macd_12_26_9``), suffixed with the channel for
multi-output indicators (``macd_12_26_9_signal``). Two references written
differently but meaning the same thing — parameters in another order, a period
given explicitly that happens to be the default — collapse onto one key and are
computed once.

Validation goes through
:func:`~trading_system.strategies.validator.check_feature_reference`, the same
path ``ts strategy validate`` uses, rather than a second implementation that
would eventually disagree with it about what a valid reference is.
"""

from collections.abc import Iterable, Iterator

from trading_system.core.exceptions import ValidationError
from trading_system.features.pipeline import FeaturePipeline, FeatureSpec
from trading_system.features.registry import build_indicator
from trading_system.strategies.schema import (
    FeatureRef,
    StrategySpec,
    VolatilityRangeFilter,
    iter_leaf_conditions,
)
from trading_system.strategies.validator import Severity, check_feature_reference


def feature_key(ref: FeatureRef) -> str:
    """The column name that carries ``ref``.

    Args:
        ref: Structural feature reference from a strategy spec.

    Returns:
        The indicator's identifier for a single-output indicator, or that
        identifier suffixed with the channel for a multi-output one.

    Raises:
        ValidationError: If the indicator is unregistered, its parameters are
            invalid, or the channel is missing, unknown, or given where the
            indicator has only one output.
    """
    issues = [issue for issue in check_feature_reference(ref) if issue.severity is Severity.ERROR]
    if issues:
        detail = "; ".join(issue.message for issue in issues)
        raise ValidationError(f"invalid feature reference {ref!r}: {detail}")
    indicator = build_indicator(ref.indicator, ref.params)
    if len(indicator.outputs) == 1:
        return indicator.name
    return f"{indicator.name}_{ref.channel}"


def iter_feature_refs(spec: StrategySpec) -> Iterator[FeatureRef]:
    """Every feature reference a strategy needs in order to be evaluated.

    Covers the trigger, the confirmations, the invalidation condition and price
    level, the quality modifiers, and the volatility filter — everything the
    Entry Engine reads on a bar. Filters that consult state the Entry Engine does
    not own (news, spread, correlation) contribute nothing here.

    Args:
        spec: Strategy to walk.

    Yields:
        Each reference found, with duplicates.
    """
    conditions = [
        spec.entry.trigger,
        *spec.entry.confirmation,
        *(modifier.condition for modifier in spec.risk_profile.quality_modifiers),
    ]
    invalidation = spec.entry.invalidation
    if invalidation is not None and invalidation.condition is not None:
        conditions.append(invalidation.condition)

    for condition in conditions:
        for leaf in iter_leaf_conditions(condition):
            for operand in (leaf.left, leaf.right):
                if isinstance(operand, FeatureRef):
                    yield operand

    if invalidation is not None and isinstance(invalidation.price_level, FeatureRef):
        yield invalidation.price_level

    for filter_spec in spec.filters:
        if isinstance(filter_spec, VolatilityRangeFilter):
            yield filter_spec.feature


class FeatureRegistry:
    """The features available to a compiled entry, and their column names.

    A compiler resolves every :class:`FeatureRef` through a registry, so a
    reference to something the pipeline was never asked to compute fails at
    compile time with a list of what *is* available, rather than at bar 40 000
    with a ``KeyError``.
    """

    __slots__ = ("_keys", "_specs")

    def __init__(self, refs: Iterable[FeatureRef] = ()) -> None:
        """Resolve and de-duplicate a set of references.

        Args:
            refs: References the compiled entry will read. Duplicates and
                equivalent spellings collapse.

        Raises:
            ValidationError: If any reference is invalid.
        """
        keys: set[str] = set()
        specs: dict[str, FeatureSpec] = {}
        for ref in refs:
            keys.add(feature_key(ref))
            indicator = build_indicator(ref.indicator, ref.params)
            specs.setdefault(
                indicator.name,
                FeatureSpec(name=indicator.name, kind=ref.indicator, params=dict(ref.params)),
            )
        self._keys = tuple(sorted(keys))
        self._specs = tuple(specs[name] for name in sorted(specs))

    @classmethod
    def from_strategy(cls, spec: StrategySpec) -> "FeatureRegistry":
        """Build the registry a strategy needs.

        Args:
            spec: Strategy whose references to collect.

        Returns:
            A registry covering every feature the Entry Engine will read.

        Raises:
            ValidationError: If any reference in the spec is invalid.
        """
        return cls(iter_feature_refs(spec))

    @property
    def keys(self) -> tuple[str, ...]:
        """Every feature column this registry provides, sorted."""
        return self._keys

    @property
    def specs(self) -> tuple[FeatureSpec, ...]:
        """Pipeline specifications producing exactly :attr:`keys`."""
        return self._specs

    def resolve(self, ref: FeatureRef) -> str:
        """The column name carrying ``ref``, if this registry provides it.

        Args:
            ref: Reference to resolve.

        Returns:
            The column name.

        Raises:
            ValidationError: If the reference is invalid, or valid but not
                provided by this registry.
        """
        key = feature_key(ref)
        if key not in self._keys:
            raise ValidationError(
                f"feature {key!r} is not provided by this registry; available: {list(self._keys)}"
            )
        return key

    def pipeline(self) -> FeaturePipeline:
        """A pipeline computing every column in :attr:`keys`.

        Each indicator is requested once, under its own identifier, so a
        multi-output one publishes ``{indicator}_{channel}`` — which is what
        :func:`feature_key` returns for it. The pipeline therefore emits a
        superset of :attr:`keys`: an indicator's unused channels come along for
        free, and cost nothing extra since the warmup masking is shared. Only
        :attr:`keys` is resolvable, so a condition still cannot read a channel
        its spec never named.

        Returns:
            The pipeline. Caching is left to the caller, since a pipeline that
            silently owned a cache could not be shared between symbols.
        """
        return FeaturePipeline(self._specs)
