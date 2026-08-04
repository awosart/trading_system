"""Semantic validation of :class:`StrategySpec`, layered above pydantic's syntax check.

Pydantic already rejects a structurally malformed spec — wrong types, missing
fields, an out-of-range quality score. What it cannot check is whether the spec
*means something coherent*: whether a referenced feature exists, whether the
named exit is registered, whether the timeframes are ordered sensibly, whether
two strategies claim the same id. Those checks live here and report as
:class:`ValidationIssue`, not exceptions, so a CLI invocation can show every
problem in one pass instead of stopping at the first.

``known_feature_kinds`` and ``known_exit_ids`` are supplied by the caller rather
than hardcoded, because neither registry is reachable from this module without
creating a dependency cycle: the Exit DB doesn't exist yet (built in a later
phase), and feature specs only become concrete *column* names once assembled
into a :class:`~trading_system.features.pipeline.FeaturePipeline`, which a bare
:class:`~trading_system.strategies.schema.StrategySpec` does not carry. Absent an
explicit collection, feature references are checked against indicator *kinds*
in :data:`~trading_system.features.registry.INDICATOR_TYPES` by prefix — e.g.
``feature:rsi_14`` matches kind ``"rsi"`` — which catches typos and unknown
indicators without requiring the caller to enumerate every parameterised
column name up front.
"""

from collections.abc import Collection, Iterator, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import pydantic

from trading_system.features.registry import INDICATOR_TYPES
from trading_system.strategies.schema import (
    Condition,
    ConditionOp,
    Regime,
    StrategySpec,
    iter_leaf_conditions,
    operand_feature_name,
)


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
    yield spec.entry.trigger
    yield from spec.entry.confirmation
    if spec.entry.invalidation is not None and spec.entry.invalidation.condition is not None:
        yield spec.entry.invalidation.condition
    for modifier in spec.risk_profile.quality_modifiers:
        yield modifier.condition


def _feature_kind_known(feature_name: str, known_kinds: Collection[str]) -> bool:
    """Whether ``feature_name`` plausibly comes from one of ``known_kinds``.

    A feature column is usually named ``{kind}_{params}`` (e.g. ``rsi_14``,
    matching indicator kind ``"rsi"``) but may also be a bare kind name with no
    parameters (e.g. ``obv``), so both forms are accepted.
    """
    return any(feature_name == kind or feature_name.startswith(f"{kind}_") for kind in known_kinds)


def check_feature_references(
    spec: StrategySpec, known_feature_kinds: Collection[str]
) -> list[ValidationIssue]:
    """Flag ``feature:*`` references that match no registered indicator kind.

    Args:
        spec: Strategy to check.
        known_feature_kinds: Registry keys (e.g. ``INDICATOR_TYPES``) that a
            feature reference's prefix must match.

    Returns:
        One ``ERROR`` per distinct unresolved feature name.
    """
    unresolved: dict[str, None] = {}
    for condition in _iter_spec_conditions(spec):
        for leaf in iter_leaf_conditions(condition):
            for operand in (leaf.left, leaf.right):
                if operand is None or isinstance(operand, tuple):
                    continue
                name = operand_feature_name(operand)
                if name is not None and not _feature_kind_known(name, known_feature_kinds):
                    unresolved[name] = None
    for filter_spec in spec.filters:
        if filter_spec.kind == "volatility_range":
            name = operand_feature_name(filter_spec.feature)
            if name is not None and not _feature_kind_known(name, known_feature_kinds):
                unresolved[name] = None
    return [
        ValidationIssue(
            severity=Severity.ERROR,
            code="unknown_feature",
            message=f"feature:{name} matches no registered indicator kind",
            path="entry",
        )
        for name in unresolved
    ]


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
        leaf.op in _BREAKOUT_OPS for leaf in iter_leaf_conditions(spec.entry.trigger)
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
    known_feature_kinds: Collection[str] | None = None,
    known_exit_ids: Collection[str] | None = None,
) -> list[ValidationIssue]:
    """Run every semantic check against one spec.

    Args:
        spec: Strategy to check.
        known_feature_kinds: Registry keys feature references are matched
            against. Defaults to every kind in
            :data:`~trading_system.features.registry.INDICATOR_TYPES`.
        known_exit_ids: Registered exit ids. ``None`` skips the ``exit_ref``
            existence check.

    Returns:
        Every issue found, in check order. Empty means the spec is clean.
    """
    kinds = known_feature_kinds if known_feature_kinds is not None else INDICATOR_TYPES.keys()
    return [
        *check_feature_references(spec, kinds),
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
    known_feature_kinds: Collection[str] | None = None,
    known_exit_ids: Collection[str] | None = None,
) -> dict[Path, list[ValidationIssue]]:
    """Load and fully validate a batch of strategy files together.

    Ids are checked for uniqueness across the whole batch, in addition to each
    file's own syntactic and semantic issues — the point of validating several
    files in one invocation rather than one at a time.

    Args:
        paths: Strategy JSON files to validate.
        known_feature_kinds: Forwarded to :func:`validate_spec`.
        known_exit_ids: Forwarded to :func:`validate_spec`.

    Returns:
        Issues found for each path, in the order ``paths`` was given. A path
        mapping to an empty list is clean.
    """
    parsed = [load_spec(path) for path in paths]
    results: dict[Path, list[ValidationIssue]] = {item.path: list(item.issues) for item in parsed}
    for item in parsed:
        if item.spec is not None:
            results[item.path].extend(
                validate_spec(
                    item.spec,
                    known_feature_kinds=known_feature_kinds,
                    known_exit_ids=known_exit_ids,
                )
            )

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
