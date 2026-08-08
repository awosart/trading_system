"""Deriving a search space from a strategy spec, so nobody writes a JSON pointer.

**The inversion is the point.** The obvious design — the author declares an axis
and the tool finds its occurrences — fails on the first real spec. ``ema_pullback``
carries two axes on one indicator: ``ema.period=50`` (the trend filter, and the
invalidation level derived from it) and ``ema.period=20`` (the confirmation).
"The ema period axis" would collect all three and fuse two parameters that were
deliberately separate — which is the exact error a multi-pointer axis exists to
prevent. Separating them again means naming a value or a path, i.e. writing
pointers after all.

So the tool enumerates instead: every tunable position in the spec, grouped, with
pointers already filled in. The author edits the draft — deleting axes that should
not vary and adjusting ranges — and both of those are safe. A pointer is never
typed, so a typo in one is not a thing that can happen.

**Two positions are one axis when their role signatures match**, where a role
signature is ``(kind, owner, parameter, current value)``. Including the current
value is not a heuristic; it is how a person reads the file — "the fifty is the
slow EMA". It splits ``ema_slow`` from ``ema_fast`` for free, and for free it
collapses the two legs of a symmetric strategy, whose long and short sides carry
the same value by construction.

**Ranges are proposals and say so.** A single-point axis is not a space: the
manual work would simply move from pointers to numbers, and numbers are where an
author makes fewer mistakes. So a ladder is proposed per parameter kind, the rule
is printed alongside it, and a kind with no rule gets one point and an explicit
"no ladder proposed" rather than a bad guess.

**Bounds come from ``build_indicator``, never from a table here.** An indicator's
legal parameter range is enforced by its own constructor, which is the authority
:mod:`trading_system.strategies.validator` already uses. A declared min/max in
this module would be a second authority, free to drift from the first. Every
proposed value for an indicator parameter is therefore *built* and dropped if it
raises. The one range this module does declare is the range of an indicator's
**output**, for clamping a threshold — nothing anywhere else states it.
"""

import json
import math
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from trading_system.core.exceptions import ValidationError
from trading_system.features.registry import build_indicator
from trading_system.strategies.schema import StrategySpec

#: Parameter names that mean "a window in bars": tuned multiplicatively, integers.
PERIOD_PARAMS = frozenset(
    {
        "period",
        "lookback",
        "fast",
        "slow",
        "signal",
        "fast_period",
        "slow_period",
        "signal_period",
        "k_period",
        "d_period",
        "smooth",
        "conversion",
        "base",
        "span_b",
        "displacement",
        "atr_period",
    }
)

#: Spec fields that mean "a count of bars": same ladder, no registry to consult.
COUNT_FIELDS = frozenset(
    {
        "confirmation_window_bars",
        "expire_after_bars",
        "cooldown_bars_after_loss",
        "max_bars_held",
        "lookback_bars",
        "max_concurrent_positions",
    }
)

#: Names that mean "a coefficient": tuned additively around the current value.
MULTIPLE_NAMES = frozenset({"multiple", "buffer_atr_multiple", "num_std", "r_multiple"})

#: Where an indicator's *output* lives, for clamping a threshold to it. Declared
#: here because nothing else in the system states it — unlike a parameter's
#: range, which its own constructor already enforces. Only facts belong in this
#: table: an RSI cannot exceed 100 and a relative volume cannot be negative.
#: An indicator whose range is a matter of opinion is left out, and its
#: threshold ladder then says the range is unknown rather than inventing one.
OUTPUT_RANGES: dict[str, tuple[float, float]] = {
    # Bounded oscillators.
    "rsi": (0.0, 100.0),
    "adx": (0.0, 100.0),
    "stoch": (0.0, 100.0),
    "mfi": (0.0, 100.0),
    "willr": (-100.0, 0.0),
    "chop": (0.0, 100.0),
    # Non-negative by construction: a ratio, a range, a volume.
    "rvol": (0.0, math.inf),
    "atr": (0.0, math.inf),
    "stddev": (0.0, math.inf),
    "volume_ma": (0.0, math.inf),
}

#: How many points a proposed ladder carries.
LADDER_POINTS = 5

#: Said out loud when a parameter kind has no rule. Better than a bad guess.
NO_LADDER = "no ladder proposed for this parameter kind — widen it by hand"


@dataclass(frozen=True)
class RoleSignature:
    """What makes two positions in a spec the same tunable parameter.

    Attributes:
        kind: ``"feature"``, ``"shift"``, ``"field"`` or ``"threshold"``.
        owner: Indicator name for a feature, the field's own name for a field,
            the compared indicator for a threshold.
        param: Parameter name, or the operator for a threshold.
        value: What the spec currently holds there. Part of the signature
            because it is what distinguishes a slow EMA from a fast one, and
            what makes the two legs of a symmetric strategy collapse into one.
    """

    kind: str
    owner: str
    param: str
    value: float

    def label(self) -> str:
        """A readable description of the role, for the draft's own output."""
        if self.kind == "field":
            return f"{self.owner}={_plain(self.value)}"
        return f"{self.owner}.{self.param}={_plain(self.value)}"


@dataclass(frozen=True)
class CandidateAxis:
    """One axis the tool proposes, with its pointers already resolved.

    Attributes:
        name: Suggested axis name.
        signature: What grouped these positions together.
        pointers: Every place in the spec this axis writes to.
        values: The proposed ladder, current value always included.
        rule: How the ladder was built, in words, or :data:`NO_LADDER`.
    """

    name: str
    signature: RoleSignature
    pointers: tuple[str, ...]
    values: tuple[float | int, ...]
    rule: str

    def to_axis(self) -> dict[str, Any]:
        """This axis as a :class:`~trading_system.validation.optimization.Axis` document."""
        return {
            "name": self.name,
            "paths": list(self.pointers),
            "values": [_plain(value) for value in self.values],
        }


def _plain(value: float) -> float | int:
    """A float that is really an integer, as an integer.

    Args:
        value: The number.

    Returns:
        ``int`` when the value is whole, else the float unchanged. Periods must
        reach the spec as integers or the indicator will refuse them.
    """
    return int(value) if isinstance(value, float) and value.is_integer() else value


def _escape(token: str) -> str:
    """Escape one JSON Pointer token per RFC 6901.

    Args:
        token: The raw key.

    Returns:
        The token with ``~`` and ``/`` escaped.
    """
    return token.replace("~", "~0").replace("/", "~1")


# ---------------------------------------------------------------------------
# Ladders
# ---------------------------------------------------------------------------


def period_ladder(current: int) -> tuple[tuple[int, ...], str]:
    """A multiplicative ladder of whole bars around ``current``.

    Multiplicative because a window's effect is roughly scale-free: the step
    from 10 to 15 changes an indicator about as much as 100 to 150 does, and an
    additive ladder around 200 would explore almost nothing.

    Args:
        current: The value the spec holds.

    Returns:
        The ladder and the rule that produced it. A current value below one has
        no multiplicative neighbourhood — every factor collapses onto the same
        floor — so it gets one point and says so rather than a ladder of one
        repeated number.
    """
    if current < 1:
        return (current,), NO_LADDER
    rule = f"period: 0.5x..2x of {current}, {LADDER_POINTS} points, whole bars"
    factors = [0.5, 0.7, 1.0, 1.4, 2.0]
    values = sorted({max(1, round(current * factor)) for factor in factors})
    return tuple(values), rule


def multiple_ladder(current: float) -> tuple[tuple[float, ...], str]:
    """An additive ladder around a coefficient.

    Additive because a coefficient is already a ratio: 2.0 and 2.5 ATR are a
    meaningful step apart, and doubling would jump straight past the range
    anyone tunes.

    Args:
        current: The value the spec holds.

    Returns:
        The ladder and the rule that produced it.
    """
    step = max(0.5, round(abs(current) * 0.25, 1)) if current else 0.5
    rule = f"multiple: {current} +/- {step} additive, {LADDER_POINTS} points, above zero"
    offsets = range(-(LADDER_POINTS // 2), LADDER_POINTS // 2 + 1)
    values = sorted(
        {round(current + step * offset, 4) for offset in offsets if current + step * offset > 0}
    )
    return tuple(values), rule


def threshold_ladder(current: float, indicator: str) -> tuple[tuple[float, ...], str]:
    """An additive ladder around a comparison threshold, clamped to the output range.

    Args:
        current: The threshold the spec holds.
        indicator: What the threshold is compared against, for its output range.

    Returns:
        The ladder and the rule that produced it.
    """
    bounds = OUTPUT_RANGES.get(indicator)
    step = max(1.0, round(abs(current) * 0.2, 1)) if current else 1.0
    offsets = range(-(LADDER_POINTS // 2), LADDER_POINTS // 2 + 1)
    values = sorted({round(current + step * offset, 4) for offset in offsets})
    if bounds is None:
        rule = (
            f"threshold: {current} +/- {step} additive, {LADDER_POINTS} points, "
            f"range of {indicator} unknown"
        )
        return tuple(values), rule
    low, high = bounds
    values = sorted({value for value in values if low <= value <= high})
    span = f"[{_plain(low)}, {'inf' if math.isinf(high) else _plain(high)}]"
    rule = f"threshold: {current} +/- {step} additive, clamped to {indicator} output {span}"
    return tuple(values), rule


def shift_ladder(current: int) -> tuple[tuple[int, ...], str]:
    """Small whole bars from zero.

    A shift larger than a couple of bars is a different rule, not a tuning of
    this one: reading a channel five bars back is not a variant of reading it
    one bar back.

    Args:
        current: The shift the spec holds.

    Returns:
        The ladder and the rule that produced it.
    """
    values = sorted({0, 1, 2, current})
    return tuple(values), f"shift: 0..2 plus the current {current}, whole bars"


def _bounded_by_registry(
    values: Sequence[float], *, indicator: str, params: Mapping[str, Any], param: str
) -> tuple[float, ...]:
    """Drop proposed values the indicator itself refuses.

    Uses :func:`~trading_system.features.registry.build_indicator` rather than a
    declared range, so the bound honoured here is the same object that
    :mod:`trading_system.strategies.validator` enforces and cannot drift from it.

    Args:
        values: The proposed ladder.
        indicator: Registry key.
        params: The reference's other parameters, held fixed.
        param: Which parameter is varying.

    Returns:
        Only the values that build.
    """
    kept = []
    for value in values:
        try:
            build_indicator(indicator, {**params, param: _plain(value)})
        except (ValidationError, ValueError):
            continue
        kept.append(value)
    return tuple(kept)


# ---------------------------------------------------------------------------
# Walking the spec
# ---------------------------------------------------------------------------


def _is_number(value: Any) -> bool:
    """Whether ``value`` is a tunable number rather than a bool."""
    return isinstance(value, int | float) and not isinstance(value, bool)


def _walk(node: Any, pointer: str) -> Iterator[tuple[RoleSignature, str, dict[str, Any]]]:
    """Yield every tunable position under ``node``.

    Args:
        node: A fragment of the dumped spec.
        pointer: JSON Pointer to ``node``.

    Yields:
        ``(signature, pointer, context)``, where context carries whatever the
        ladder needs — an indicator's other parameters, for instance.
    """
    if isinstance(node, dict):
        # An indicator's parameters are claimed by the feature branch below.
        # Without this, a name that is both an indicator parameter and a spec
        # field — ``num_std`` is both — yields two axes over one position, and
        # a search would then move the same number along two independent dials.
        inside_indicator_params = "indicator" in node and isinstance(node.get("params"), dict)
        if inside_indicator_params:
            indicator = str(node["indicator"])
            for name, value in node["params"].items():
                if _is_number(value):
                    yield (
                        RoleSignature("feature", indicator, str(name), float(value)),
                        f"{pointer}/params/{_escape(str(name))}",
                        {"indicator": indicator, "params": node["params"], "param": name},
                    )
            # Only when the author already set one. A shift of zero is the
            # default on every reference in every spec, so proposing an axis for
            # each would triple the draft with noise; and moving a reference off
            # zero changes *which rule* is expressed rather than tuning it — a
            # channel read one bar back is a different sentence from a channel
            # read on the bar itself. A reference that already carries a shift
            # has had that decision made, and its size is then a real dial.
            if _is_number(node.get("shift")) and node["shift"]:
                yield (
                    RoleSignature("shift", indicator, "shift", float(node["shift"])),
                    f"{pointer}/shift",
                    {},
                )
        if node.get("type") == "leaf" and _is_number(node.get("right")):
            left = node.get("left")
            owner = (
                left["indicator"] if isinstance(left, dict) and "indicator" in left else str(left)
            )
            yield (
                RoleSignature("threshold", str(owner), str(node.get("op")), float(node["right"])),
                f"{pointer}/right",
                {"indicator": owner},
            )
        for name, value in node.items():
            if inside_indicator_params and name == "params":
                continue
            if (name in COUNT_FIELDS or name in MULTIPLE_NAMES) and _is_number(value):
                yield (
                    RoleSignature("field", str(name), str(name), float(value)),
                    f"{pointer}/{_escape(str(name))}",
                    {},
                )
            yield from _walk(value, f"{pointer}/{_escape(str(name))}")
    elif isinstance(node, list):
        for index, value in enumerate(node):
            yield from _walk(value, f"{pointer}/{index}")


def _ladder_for(
    signature: RoleSignature, context: Mapping[str, Any]
) -> tuple[tuple[float | int, ...], str]:
    """Propose a ladder for one role, or one point and a reason.

    Args:
        signature: What the role is.
        context: Extra information the ladder needs.

    Returns:
        The ladder and its rule.
    """
    if signature.kind == "shift":
        return shift_ladder(int(signature.value))

    if signature.kind == "threshold":
        return threshold_ladder(signature.value, str(context.get("indicator", "")))

    if signature.kind == "feature":
        if signature.param in PERIOD_PARAMS:
            values, rule = period_ladder(int(signature.value))
            bounded = _bounded_by_registry(
                values,
                indicator=str(context["indicator"]),
                params=context["params"],
                param=str(context["param"]),
            )
            return tuple(int(value) for value in bounded), rule
        if signature.param in MULTIPLE_NAMES:
            values_f, rule = multiple_ladder(signature.value)
            bounded = _bounded_by_registry(
                values_f,
                indicator=str(context["indicator"]),
                params=context["params"],
                param=str(context["param"]),
            )
            return bounded, rule
        return (_plain(signature.value),), NO_LADDER

    if signature.param in COUNT_FIELDS:
        return period_ladder(int(signature.value))
    if signature.param in MULTIPLE_NAMES:
        return multiple_ladder(signature.value)
    return (_plain(signature.value),), NO_LADDER


def _name_axes(groups: Mapping[RoleSignature, list[str]]) -> dict[RoleSignature, str]:
    """Name each axis, disambiguating by value only where a name repeats.

    Args:
        groups: Signature to its pointers.

    Returns:
        Signature to axis name.
    """
    base: dict[RoleSignature, str] = {}
    for signature in groups:
        if signature.kind == "field":
            base[signature] = signature.owner
        elif signature.kind == "threshold":
            base[signature] = f"{signature.owner}_{signature.param}"
        else:
            base[signature] = f"{signature.owner}_{signature.param}"

    counts: dict[str, int] = {}
    for name in base.values():
        counts[name] = counts.get(name, 0) + 1
    return {
        signature: (name if counts[name] == 1 else f"{name}_{_plain(signature.value)}")
        for signature, name in base.items()
    }


def build_candidates(spec: StrategySpec) -> list[CandidateAxis]:
    """Every axis the tool would propose for ``spec``.

    Args:
        spec: The strategy.

    Returns:
        Candidate axes, ordered by name. Pointers within an axis keep the order
        they were found in, which is document order.
    """
    document = spec.model_dump(mode="json", exclude_none=True)
    groups: dict[RoleSignature, list[str]] = {}
    contexts: dict[RoleSignature, dict[str, Any]] = {}
    for signature, pointer, context in _walk(document, ""):
        groups.setdefault(signature, [])
        if pointer not in groups[signature]:
            groups[signature].append(pointer)
        contexts.setdefault(signature, dict(context))

    names = _name_axes(groups)
    axes = [
        CandidateAxis(
            name=names[signature],
            signature=signature,
            pointers=tuple(pointers),
            values=_ladder_for(signature, contexts[signature])[0],
            rule=_ladder_for(signature, contexts[signature])[1],
        )
        for signature, pointers in groups.items()
    ]
    return sorted(axes, key=lambda axis: axis.name)


def infer_constraints(axes: Sequence[CandidateAxis]) -> list[dict[str, str]]:
    """Order constraints implied by two axes on the same parameter.

    Two axes over the same ``(indicator, parameter)`` whose current values are
    ordered are almost always a fast/slow pair, and the ordering is meant to
    survive tuning: a search that puts the slow EMA below the fast one is
    exploring a strategy nobody wrote. Emitted for the author to delete when
    wrong, rather than left out for the author to remember.

    Args:
        axes: The candidate axes.

    Returns:
        Constraint documents, ``{"less": ..., "greater": ...}``.
    """
    by_role: dict[tuple[str, str, str], list[CandidateAxis]] = {}
    for axis in axes:
        if axis.signature.kind != "feature":
            continue
        key = (axis.signature.kind, axis.signature.owner, axis.signature.param)
        by_role.setdefault(key, []).append(axis)

    constraints = []
    for members in by_role.values():
        if len(members) != 2:
            continue
        low, high = sorted(members, key=lambda axis: axis.signature.value)
        if low.signature.value < high.signature.value:
            constraints.append({"less": low.name, "greater": high.name})
    return constraints


def build_space_document(
    spec: StrategySpec, *, keep: Sequence[str] | None = None
) -> dict[str, Any]:
    """The draft search space for ``spec``.

    Args:
        spec: The strategy.
        keep: Axis names to include. ``None`` keeps every candidate — the draft
            is meant to be edited down, not guessed at.

    Returns:
        A document a :class:`~trading_system.validation.optimization.SearchSpace`
        parses.
    """
    axes = build_candidates(spec)
    if keep is not None:
        wanted = set(keep)
        axes = [axis for axis in axes if axis.name in wanted]
    document: dict[str, Any] = {"axes": [axis.to_axis() for axis in axes]}
    constraints = infer_constraints(axes)
    if constraints:
        document["constraints"] = constraints
    return document


def _applies(
    spec: StrategySpec, document: Mapping[str, Any], axis_index: int, value_index: int
) -> bool:
    """Whether setting one axis to one value leaves a spec that still validates.

    Args:
        spec: The strategy.
        document: The space document.
        axis_index: Which axis to move.
        value_index: Which of its values to use.

    Returns:
        True when the resulting spec is valid.
    """
    from trading_system.validation.optimization import SearchSpace

    space = SearchSpace.model_validate(document)
    coords = [0] * len(space.axes)
    coords[axis_index] = value_index
    try:
        space.apply(spec, space.point(coords))
    except Exception:  # noqa: BLE001 - the question asked is exactly "did it fail"
        return False
    return True


def prune(spec: StrategySpec, document: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Drop proposed values the spec itself refuses, and axes left with nothing to vary.

    A value that resolves but yields an invalid spec is not a bug in the walker;
    it is a coupling the schema enforces — ``confirmation_window_bars`` means
    nothing on an entry with no confirmation, so the field is present, tunable
    in appearance and pinned at zero in fact. Pruning is the honest response, in
    the same spirit as :meth:`SearchSpace.feasible_size` excluding points a
    constraint forbids. A pointer that does not *resolve* is a different matter
    and stays a hard error in :func:`verify`.

    Args:
        spec: The strategy the space was built from.
        document: The generated space document, modified into the return value.

    Returns:
        The pruned document and one note per drop, for the caller to print.
    """
    notes: list[str] = []
    kept_axes = []
    for axis_index, axis in enumerate(document["axes"]):
        good = [
            value
            for value_index, value in enumerate(axis["values"])
            if _applies(spec, document, axis_index, value_index)
        ]
        dropped = [value for value in axis["values"] if value not in good]
        if dropped:
            notes.append(f"{axis['name']}: dropped {dropped} — the spec rejects them")
        if len(good) < 2:
            notes.append(f"{axis['name']}: removed — nothing left to vary")
            continue
        kept_axes.append({**axis, "values": good})

    names = {axis["name"] for axis in kept_axes}
    document["axes"] = kept_axes
    if "constraints" in document:
        document["constraints"] = [
            item
            for item in document["constraints"]
            if item["less"] in names and item["greater"] in names
        ]
        if not document["constraints"]:
            del document["constraints"]
    return document, notes


def verify(spec: StrategySpec, document: Mapping[str, Any]) -> None:
    """Check every generated pointer resolves and every point leaves a valid spec.

    This is what makes "a typo in a pointer cannot happen" a fact rather than a
    claim: the pointers were not typed, and they are proven to land.

    Args:
        spec: The strategy the space was built from.
        document: The generated space document, already pruned.

    Raises:
        ValidationError: If a pointer does not resolve, or applying a point
            produces something that is no longer a valid spec.
    """
    from trading_system.validation.optimization import SearchSpace

    space = SearchSpace.model_validate(document)
    for index, axis in enumerate(space.axes):
        for value_index, value in enumerate(axis.values):
            coords = [0] * len(space.axes)
            coords[index] = value_index
            try:
                applied = space.apply(spec, space.point(coords))
                StrategySpec.model_validate(applied.model_dump(mode="json"))
            except Exception as error:  # noqa: BLE001 - re-raised with the axis named
                raise ValidationError(
                    f"axis {axis.name!r} value {value!r} does not apply: {error}"
                ) from error


def render(spec: StrategySpec, axes: Sequence[CandidateAxis]) -> str:
    """A human-readable summary of the draft, rules included.

    The rule is printed rather than left in the source because an author who
    disagrees with a proposed range needs to know what they are disagreeing
    with.

    Args:
        spec: The strategy.
        axes: The candidates.

    Returns:
        The text.
    """
    lines = [
        f"{spec.id}: {len(axes)} candidate axes, "
        f"{sum(len(axis.pointers) for axis in axes)} pointers, none written by hand.",
        "Ranges below are PROPOSALS. Delete axes that should not vary; edit values freely.",
        "",
    ]
    for axis in axes:
        legs = len(axis.pointers)
        lines.append(
            f"  {axis.name:<28} {axis.signature.label():<26} "
            f"{legs} pointer{'s' if legs != 1 else ''}"
        )
        lines.append(f"      values  {[_plain(value) for value in axis.values]}")
        lines.append(f"      rule    {axis.rule}")
    return "\n".join(lines)


def write_space(document: Mapping[str, Any], path: Any) -> None:
    """Write a space document as JSON.

    Args:
        document: What :func:`build_space_document` produced.
        path: Where to write it.
    """
    path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
