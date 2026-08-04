"""Semantic validation of :class:`StrategySpec`, layered above pydantic's syntax check.

Pydantic already rejects a structurally malformed spec — wrong types, missing
fields, an out-of-range quality score. What it cannot check is whether the spec
*means something coherent*: whether a referenced feature exists, whether the
named exit is registered, whether the timeframes are ordered sensibly, whether
two strategies claim the same id. Those checks live here and report as
:class:`ValidationIssue`, not exceptions, so a CLI invocation can show every
problem in one pass instead of stopping at the first.

Feature references are checked directly against
:data:`~trading_system.features.registry.INDICATOR_TYPES`: a :class:`.FeatureRef`
names an indicator and its constructor parameters structurally, so
:func:`~trading_system.features.registry.build_indicator` — the same function a
real :class:`~trading_system.features.pipeline.FeaturePipeline` uses — already
does the work of confirming the indicator exists, its parameters match the
constructor, and their values pass that indicator's own bounds checks. This
registry is complete today, unlike the Exit DB: ``known_exit_ids`` is still
supplied by the caller and defaults to skipping that check, because no Exit
Engine exists yet to consult.

Categorical labels are checked the same way, against the real enumerations
rather than a copy of them: ``pattern_is`` against
:class:`~trading_system.features.patterns.Pattern`, ``session_is`` against
:class:`~trading_system.data.sessions.Session`, ``regime_is`` against
:class:`~trading_system.strategies.schema.Regime`. A mistyped label is otherwise
indistinguishable from a deliberate one until a backtest reports a strategy that
never traded.
"""

from collections.abc import Collection, Iterator, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import pydantic

from trading_system.core.exceptions import ValidationError as IndicatorValidationError
from trading_system.data.sessions import Session
from trading_system.features.patterns import Pattern
from trading_system.features.registry import INDICATOR_TYPES, build_indicator
from trading_system.strategies.schema import (
    Condition,
    ConditionOp,
    FeatureRef,
    LabelSet,
    LeafCondition,
    Regime,
    SessionFilter,
    StrategySpec,
    VolatilityRangeFilter,
    iter_leaf_conditions,
)

#: Operators whose operand is a :class:`LabelSet`, and the vocabulary each one
#: draws from. Checked against the real enumerations for the same reason a
#: :class:`FeatureRef` is checked against the indicator registry: a typo in a
#: label is indistinguishable from a deliberate choice until something looks it
#: up, and by then it is a condition that silently never fires.
LABEL_VOCABULARIES: dict[ConditionOp, frozenset[str]] = {
    ConditionOp.PATTERN_IS: frozenset(member.value for member in Pattern),
    ConditionOp.REGIME_IS: frozenset(member.value for member in Regime),
    ConditionOp.SESSION_IS: frozenset(member.value for member in Session),
}


class Severity(StrEnum):
    """How serious a semantic validation finding is."""

    ERROR = "ERROR"
    WARNING = "WARNING"


@dataclass(frozen=True)
class ValidationIssue:
    """One semantic finding against a strategy spec.

    Attributes:
        severity: Whether the spec is unusable (``ERROR``) or merely
            suspicious (``WARNING``).
        code: Stable, machine-readable identifier.
        message: Human-readable explanation.
        path: Dotted location of the offending field, best-effort.
    """

    severity: Severity
    code: str
    message: str
    path: str = ""


@dataclass(frozen=True)
class ParsedStrategy:
    """The outcome of loading one strategy file.

    Attributes:
        path: File the spec was read from.
        spec: The parsed spec, or ``None`` if it failed to load.
        issues: Syntactic issues found while loading. Empty when ``spec`` is
            not ``None``.
    """

    path: Path
    spec: StrategySpec | None
    issues: tuple[ValidationIssue, ...]


def _iter_spec_conditions(spec: StrategySpec) -> Iterator[Condition]:
    """Every condition tree embedded in a spec: trigger, confirmation, and so on."""
    for entry in spec.entries:
        yield entry.trigger
        yield from entry.confirmation
        if entry.invalidation.condition is not None:
            yield entry.invalidation.condition
    for modifier in spec.risk_profile.quality_modifiers:
        yield modifier.condition


def check_feature_reference(ref: FeatureRef) -> list[ValidationIssue]:
    """Validate one structural feature reference against the indicator registry.

    Checks, in order: the indicator is registered; its channel is required and
    known for a multi-output indicator, or absent for a single-output one; its
    parameters match the indicator's constructor and pass that indicator's own
    bounds checks (via :func:`~trading_system.features.registry.build_indicator`,
    the same path a real pipeline uses).

    Args:
        ref: Feature reference to check.

    Returns:
        Every issue found. Empty means the reference is usable as-is.
    """
    indicator_type = INDICATOR_TYPES.get(ref.indicator)
    if indicator_type is None:
        return [
            ValidationIssue(
                severity=Severity.ERROR,
                code="unknown_indicator",
                message=(
                    f"indicator {ref.indicator!r} is not registered; "
                    f"known: {sorted(INDICATOR_TYPES)}"
                ),
                path="entry",
            )
        ]

    issues: list[ValidationIssue] = []
    outputs = indicator_type.outputs
    if len(outputs) > 1 and ref.channel is None:
        issues.append(
            ValidationIssue(
                severity=Severity.ERROR,
                code="missing_channel",
                message=(
                    f"indicator {ref.indicator!r} is multi-output; channel is required, "
                    f"one of {outputs}"
                ),
                path="entry",
            )
        )
    elif len(outputs) > 1 and ref.channel not in outputs:
        issues.append(
            ValidationIssue(
                severity=Severity.ERROR,
                code="unknown_channel",
                message=(
                    f"indicator {ref.indicator!r} has no channel {ref.channel!r}; known: {outputs}"
                ),
                path="entry",
            )
        )
    elif len(outputs) == 1 and ref.channel is not None:
        issues.append(
            ValidationIssue(
                severity=Severity.ERROR,
                code="unexpected_channel",
                message=f"indicator {ref.indicator!r} is single-output; channel must be omitted",
                path="entry",
            )
        )

    try:
        build_indicator(ref.indicator, ref.params)
    except IndicatorValidationError as error:
        issues.append(
            ValidationIssue(
                severity=Severity.ERROR,
                code="invalid_indicator_params",
                message=str(error),
                path="entry",
            )
        )
    return issues


def check_feature_references(spec: StrategySpec) -> list[ValidationIssue]:
    """Validate every feature reference embedded in a spec.

    Args:
        spec: Strategy to check.

    Returns:
        Every issue found across every :class:`FeatureRef` in the spec's
        conditions and filters.
    """
    issues: list[ValidationIssue] = []
    for condition in _iter_spec_conditions(spec):
        for leaf in iter_leaf_conditions(condition):
            for operand in (leaf.left, leaf.right):
                if isinstance(operand, FeatureRef):
                    issues.extend(check_feature_reference(operand))
    for entry in spec.entries:
        # The invalidation level is an operand like any other and is routinely a
        # feature (an EMA, a swing low). It is not inside any condition tree, so
        # walking conditions alone would leave the one reference the Risk Engine
        # depends on unchecked.
        if isinstance(entry.invalidation.price_level, FeatureRef):
            issues.extend(check_feature_reference(entry.invalidation.price_level))
    for filter_spec in spec.filters:
        if isinstance(filter_spec, VolatilityRangeFilter):
            issues.extend(check_feature_reference(filter_spec.feature))
    return issues


def check_leaf_labels(leaf: LeafCondition) -> list[ValidationIssue]:
    """Validate one leaf's use of categorical labels.

    Three things, all of which pydantic lets through because they are about the
    *pairing* of an operator with its operands rather than about either alone:
    a categorical operator must take a :class:`LabelSet`, its labels must exist
    in that operator's vocabulary, and a numeric operator must not be handed a
    :class:`LabelSet`.

    Args:
        leaf: Condition leaf to check.

    Returns:
        Every issue found. Empty means the leaf's operands fit its operator.
    """
    vocabulary = LABEL_VOCABULARIES.get(leaf.op)
    if vocabulary is None:
        if isinstance(leaf.left, LabelSet) or isinstance(leaf.right, LabelSet):
            return [
                ValidationIssue(
                    severity=Severity.ERROR,
                    code="unexpected_labels",
                    message=(
                        f"operator {leaf.op.value!r} compares numbers; a label set is only "
                        f"meaningful to {sorted(op.value for op in LABEL_VOCABULARIES)}"
                    ),
                    path="entries",
                )
            ]
        return []

    if not isinstance(leaf.right, LabelSet):
        return [
            ValidationIssue(
                severity=Severity.ERROR,
                code="missing_labels",
                message=(
                    f"operator {leaf.op.value!r} requires a label set as its right operand, "
                    f"got {leaf.right!r}; known labels: {sorted(vocabulary)}"
                ),
                path="entries",
            )
        ]

    issues: list[ValidationIssue] = []
    if leaf.left is not None:
        issues.append(
            ValidationIssue(
                severity=Severity.ERROR,
                code="unexpected_left_operand",
                message=(
                    f"operator {leaf.op.value!r} tests the bar itself, not a series; "
                    "left must be omitted"
                ),
                path="entries",
            )
        )
    unknown = sorted(set(leaf.right.labels) - vocabulary)
    if unknown:
        issues.append(
            ValidationIssue(
                severity=Severity.ERROR,
                code="unknown_label",
                message=(
                    f"operator {leaf.op.value!r} has no label(s) {unknown}; "
                    f"known: {sorted(vocabulary)}"
                ),
                path="entries",
            )
        )
    return issues


def check_condition_labels(spec: StrategySpec) -> list[ValidationIssue]:
    """Validate every categorical operand in a spec.

    Args:
        spec: Strategy to check.

    Returns:
        Every issue found across every leaf in the spec's conditions.
    """
    return [
        issue
        for condition in _iter_spec_conditions(spec)
        for leaf in iter_leaf_conditions(condition)
        for issue in check_leaf_labels(leaf)
    ]


def check_session_filters(spec: StrategySpec) -> list[ValidationIssue]:
    """Flag session names a trading calendar would not recognise.

    The same check as :func:`check_leaf_labels` applies to sessions, but the
    names arrive by a second route — :class:`SessionFilter` carries bare strings
    rather than a :class:`LabelSet`. An unrecognised name there does not fail
    loudly; it produces a filter that blocks every bar, which reads downstream as
    a strategy that simply never trades.

    Args:
        spec: Strategy to check.

    Returns:
        One ``ERROR`` per unrecognised session name.
    """
    known = frozenset(member.value for member in Session)
    issues: list[ValidationIssue] = []
    for filter_spec in spec.filters:
        if not isinstance(filter_spec, SessionFilter):
            continue
        unknown = sorted(set(filter_spec.sessions) - known)
        if unknown:
            issues.append(
                ValidationIssue(
                    severity=Severity.ERROR,
                    code="unknown_session",
                    message=f"unknown session(s) {unknown}; known: {sorted(known)}",
                    path="filters",
                )
            )
    return issues


def check_exit_ref(
    spec: StrategySpec, known_exit_ids: Collection[str] | None
) -> list[ValidationIssue]:
    """Flag an ``exit_ref`` absent from the Exit DB.

    Args:
        spec: Strategy to check.
        known_exit_ids: Registered exit ids. ``None`` skips the check
            entirely — used until the Exit DB exists to consult.

    Returns:
        One ``ERROR`` if ``known_exit_ids`` is given and doesn't contain
        ``spec.exit_ref``; otherwise empty.
    """
    if known_exit_ids is None or spec.exit_ref in known_exit_ids:
        return []
    return [
        ValidationIssue(
            severity=Severity.ERROR,
            code="unknown_exit_ref",
            message=f"exit_ref {spec.exit_ref!r} is not registered in the Exit DB",
            path="exit_ref",
        )
    ]


def check_timeframe_order(spec: StrategySpec) -> list[ValidationIssue]:
    """Flag timeframes that violate ``htf_filter_tf > signal_tf >= entry_tf``.

    Args:
        spec: Strategy to check.

    Returns:
        One ``ERROR`` per violated ordering constraint.
    """
    issues: list[ValidationIssue] = []
    timeframes = spec.timeframes
    if timeframes.entry_tf.duration > timeframes.signal_tf.duration:
        issues.append(
            ValidationIssue(
                severity=Severity.ERROR,
                code="timeframe_order",
                message=(
                    f"entry_tf {timeframes.entry_tf} is coarser than signal_tf "
                    f"{timeframes.signal_tf}; entry_tf must be <= signal_tf"
                ),
                path="timeframes",
            )
        )
    if (
        timeframes.htf_filter_tf is not None
        and timeframes.htf_filter_tf.duration <= timeframes.signal_tf.duration
    ):
        issues.append(
            ValidationIssue(
                severity=Severity.ERROR,
                code="timeframe_order",
                message=(
                    f"htf_filter_tf {timeframes.htf_filter_tf} must be strictly coarser than "
                    f"signal_tf {timeframes.signal_tf}"
                ),
                path="timeframes",
            )
        )
    return issues


_BREAKOUT_OPS = frozenset({ConditionOp.CROSS_ABOVE, ConditionOp.CROSS_BELOW})


def check_regime_trigger_contradiction(spec: StrategySpec) -> list[ValidationIssue]:
    """Warn when a RANGE regime is paired with a breakout-shaped trigger.

    A ``cross_above``/``cross_below`` trigger is the idiomatic shape for a
    breakout entry, which is in tension with restricting the strategy to
    ``RANGE``. This is a heuristic, not a proof of contradiction — reported as
    a warning to be confirmed by a human, not rejected outright.

    Args:
        spec: Strategy to check.

    Returns:
        One ``WARNING`` if the pattern is present; otherwise empty.
    """
    if Regime.RANGE not in spec.market_regimes:
        return []
    has_breakout_shape = any(
        leaf.op in _BREAKOUT_OPS
        for entry in spec.entries
        for leaf in iter_leaf_conditions(entry.trigger)
    )
    if not has_breakout_shape:
        return []
    return [
        ValidationIssue(
            severity=Severity.WARNING,
            code="range_breakout_conflict",
            message=(
                "market_regimes includes RANGE but the trigger uses a breakout-shaped "
                "cross_above/cross_below condition; confirm this is intentional"
            ),
            path="market_regimes",
        )
    ]


def validate_spec(
    spec: StrategySpec,
    *,
    known_exit_ids: Collection[str] | None = None,
) -> list[ValidationIssue]:
    """Run every semantic check against one spec.

    Args:
        spec: Strategy to check.
        known_exit_ids: Registered exit ids. ``None`` skips the ``exit_ref``
            existence check.

    Returns:
        Every issue found, in check order. Empty means the spec is clean.
    """
    return [
        *check_feature_references(spec),
        *check_condition_labels(spec),
        *check_session_filters(spec),
        *check_exit_ref(spec, known_exit_ids),
        *check_timeframe_order(spec),
        *check_regime_trigger_contradiction(spec),
    ]


def check_unique_ids(specs: Sequence[StrategySpec]) -> list[ValidationIssue]:
    """Flag ids shared by more than one spec in ``specs``.

    Args:
        specs: Specs considered together, e.g. a whole strategy database.

    Returns:
        One ``ERROR`` per id used more than once.
    """
    counts: dict[str, int] = {}
    for spec in specs:
        counts[spec.id] = counts.get(spec.id, 0) + 1
    return [
        ValidationIssue(
            severity=Severity.ERROR,
            code="duplicate_id",
            message=f"strategy id {strategy_id!r} is used by {count} specs",
            path="id",
        )
        for strategy_id, count in counts.items()
        if count > 1
    ]


def load_spec(path: Path) -> ParsedStrategy:
    """Read and syntactically parse one strategy file.

    Args:
        path: JSON file to load.

    Returns:
        A :class:`ParsedStrategy` carrying either the parsed spec or a list of
        readable syntactic issues, one per pydantic error.
    """
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as error:
        return ParsedStrategy(
            path=path,
            spec=None,
            issues=(
                ValidationIssue(severity=Severity.ERROR, code="read_error", message=str(error)),
            ),
        )
    try:
        spec = StrategySpec.model_validate_json(raw)
    except pydantic.ValidationError as error:
        issues = tuple(
            ValidationIssue(
                severity=Severity.ERROR,
                code="schema_error",
                message=detail["msg"],
                path=".".join(str(part) for part in detail["loc"]),
            )
            for detail in error.errors()
        )
        return ParsedStrategy(path=path, spec=None, issues=issues)
    return ParsedStrategy(path=path, spec=spec, issues=())


def validate_paths(
    paths: Sequence[Path],
    *,
    known_exit_ids: Collection[str] | None = None,
) -> dict[Path, list[ValidationIssue]]:
    """Load and fully validate a batch of strategy files together.

    Ids are checked for uniqueness across the whole batch, in addition to each
    file's own syntactic and semantic issues — the point of validating several
    files in one invocation rather than one at a time.

    Args:
        paths: Strategy JSON files to validate.
        known_exit_ids: Forwarded to :func:`validate_spec`.

    Returns:
        Issues found for each path, in the order ``paths`` was given. A path
        mapping to an empty list is clean.
    """
    parsed = [load_spec(path) for path in paths]
    results: dict[Path, list[ValidationIssue]] = {item.path: list(item.issues) for item in parsed}
    for item in parsed:
        if item.spec is not None:
            results[item.path].extend(validate_spec(item.spec, known_exit_ids=known_exit_ids))

    by_id: dict[str, list[Path]] = {}
    for item in parsed:
        if item.spec is not None:
            by_id.setdefault(item.spec.id, []).append(item.path)
    for strategy_id, owners in by_id.items():
        if len(owners) <= 1:
            continue
        for owner in owners:
            others = [str(p) for p in owners if p != owner]
            results[owner].append(
                ValidationIssue(
                    severity=Severity.ERROR,
                    code="duplicate_id",
                    message=f"strategy id {strategy_id!r} also used by {others}",
                    path="id",
                )
            )
    return results
