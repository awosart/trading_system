"""Searching a parameter space on in-sample data, and nothing but in-sample data.

**Every axis is discrete, and that is what lets one description serve all three
search methods.** A continuous range would force :class:`GridSearch` to invent
its own discretisation rule — how many points, spaced how — which is precisely
a method-specific section of the space description, the thing this stage is
required not to have. Declaring every axis as an ordered list of values (which
``low``/``high``/``step`` merely expands into, so a config file can still be
terse) makes all three methods read the *same* object and differ only in which
points they visit and in what order. It also gives every evaluated point an
integer coordinate per axis, without which
:func:`~trading_system.validation.objective.analyse_plateau` would need an
arbitrary distance metric over mixed units to say what a "neighbourhood" is.

**One logical parameter may live at several places in a strategy spec, so an
axis carries a list of pointers rather than one.** In ``ema_pullback`` the slow
EMA period appears twice — once in the entry trigger and once in
``invalidation.price_level`` — and an axis that wrote only the first would
produce a strategy filtering on EMA(65) while being invalidated by EMA(50): a
perfectly plausible spec that no one intended and nothing would flag. All of an
axis's pointers are written with the same value, and a pointer that does not
resolve is a :class:`ValueError`, never a silently created key.

**The search cannot reach an out-of-sample bar, and this is enforced by two
types refusing to exist rather than by a rule anyone has to follow.**
:class:`ISWindowView` validates in its own constructor that no stream it holds
contains a bar at or after the in-sample window's ``trade_end``;
:class:`TrialRunner` validates that the run template it was handed carries
exactly the view's streams and no others. So the only object a
:class:`ParameterSearch` can evaluate through is one whose bars provably stop
at the in-sample boundary — the same discipline
:class:`~trading_system.backtest.engine.BarStore` applies inside a run, applied
here across a fold. What remains visible to the selector is the *datetime* of
the OOS window, which the runner needs anyway and which does not let anyone
read a price.

**Trials run sequentially, in-process, and are never stored as runs.** Two
measurements decide this, both from P13. A trial's
:class:`~trading_system.backtest.spec.RunInputs` pickles to about 2.9 MB, so
shipping 240 of them to a worker pool would push ~700 MB through a pipe to buy
a few seconds of compute — the exact bottleneck P13 documented and stage 1
resolved by moving the *write* into the worker. And a trial's
:class:`~trading_system.backtest.orchestrator.BacktestResult` holds one
``Decimal``-bearing row per bar, so keeping 240 of them alive at once would cost
hundreds of megabytes; each is therefore reduced immediately to a score, a
vector of daily returns and a few counters, and dropped. Parallelism stays where
stage 1 put it: across folds, where the unit of work is large and what travels
back is a 1.4 KB :class:`~trading_system.backtest.parallel.StoredRun`.

**The trial ledger is written to disk per fold, so a crashed search resumes.**
``n_trials`` is required by CLAUDE.md "for the Deflated Sharpe", and
:func:`~trading_system.analytics.statistical.deflated_sharpe_ratio` does exist
and does take it — but it is not called here, and that is deliberate. That
function's own docstring states that ``n_trials`` is taken as given and that "a
caller who knows their trials are correlated is responsible for discounting
``n_trials`` accordingly". A 240-point grid over four axes is emphatically
correlated: EMA(50) and EMA(52) trade almost identically. Feeding 240 into a
correction designed for 240 independent variants would overstate the penalty —
conservative-looking, and wrong. So this stage measures and stores the trial
count honestly, stores the per-trial daily returns that a correlation-aware
discount would need, and leaves the discount itself to the stage that renders a
verdict.
"""

import json
import math
import statistics
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, Protocol

import polars as pl
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, model_validator

from trading_system.analytics.metrics import daily_curve, simple_returns
from trading_system.backtest.clock import StreamKey
from trading_system.backtest.orchestrator import StrategyBinding
from trading_system.backtest.spec import RunInputs
from trading_system.core.logging import get_logger
from trading_system.data.models import OHLCVFrame
from trading_system.exit.library import ExitPresetSpec, build_plan
from trading_system.risk.sizing.config import SizingConfig, build_sizing_method
from trading_system.strategies.schema import StrategySpec
from trading_system.validation.objective import (
    Objective,
    PlateauAnalysis,
    ScoredPoint,
    analyse_plateau,
)
from trading_system.validation.splitting import FoldWindow

logger = get_logger(__name__)

#: Filenames written per fold under the optimiser's own store directory.
TRIALS_FILE = "trials.parquet"
RETURNS_FILE = "returns.parquet"
SELECTION_FILE = "selection.json"

#: Value type an axis can carry. ``int`` stays ``int`` through pydantic's smart
#: union, which matters: an indicator ``period`` written back as ``50.0`` would
#: be a different JSON document and, for a stricter field, a validation error.
#: ``str`` is what makes an axis *categorical* — see :class:`ParameterAxis`.
AxisValue = int | float | str


class AxisTarget(StrEnum):
    """Which document an axis's pointers resolve in.

    Three documents, because the three things worth varying live in three
    places and no amount of pointer syntax merges them — and each is reachable
    either wholesale, by naming it, or field by field, by pointing into it. The
    strategy is a
    :class:`~trading_system.strategies.schema.StrategySpec`, the exit is an
    :class:`~trading_system.exit.library.ExitPresetSpec` reached through
    :class:`~trading_system.backtest.orchestrator.StrategyBinding`, and the
    sizing is a :class:`~trading_system.risk.sizing.config.SizingConfig` that
    :class:`~trading_system.backtest.spec.RunInputs` has already turned into a
    built method by the time a run exists.

    Members:
        SPEC: Pointer into the serialised strategy spec. The default, and what
            every axis written before categorical axes existed means.
        EXIT_PRESET: The *choice* of exit, by preset id. Carries no pointer:
            it replaces the bound preset wholesale and writes the matching
            ``exit_ref`` back into the spec, so the recorded strategy can never
            name one exit while the run executed another.
        EXIT_PARAM: Pointer into the serialised exit preset — a fixed target's
            reward multiple, a ladder's fractions, a trail's ``k``.
        SIZING_METHOD: The *choice* of sizing method, by name. Carries no
            pointer, for the same reason :attr:`EXIT_PRESET` does not: the four
            methods take different parameters, so a name is the only thing they
            have in common to vary.
        RUN: Pointer into the run-knobs document, ``{"sizing": ...}``. The one
            place a trial can vary something that is not part of the strategy
            at all — ``risk_pct`` within a fixed method, for instance.
    """

    SPEC = "spec"
    EXIT_PRESET = "exit_preset"
    EXIT_PARAM = "exit_param"
    SIZING_METHOD = "sizing_method"
    RUN = "run"


#: Targets that replace a whole named document, paired with the target that
#: points *inside* the document they replace. The two cannot be searched
#: together: a pointer only resolves in the document that declares the field,
#: so their product contains points describing no configuration at all.
_INCOMPATIBLE_TARGETS: tuple[tuple[AxisTarget, AxisTarget], ...] = (
    (AxisTarget.EXIT_PRESET, AxisTarget.EXIT_PARAM),
    (AxisTarget.SIZING_METHOD, AxisTarget.RUN),
)

#: Targets naming a document by id rather than pointing into one.
_CHOICE_TARGETS = frozenset({AxisTarget.EXIT_PRESET, AxisTarget.SIZING_METHOD})


# ---------------------------------------------------------------------------
# The space
# ---------------------------------------------------------------------------


def _pointer_parts(pointer: str) -> list[str]:
    """Split an RFC 6901 JSON Pointer into its unescaped tokens.

    Args:
        pointer: The pointer, e.g. ``/entries/0/confirmation_window_bars``.

    Returns:
        The tokens.

    Raises:
        ValueError: If the pointer is empty or does not start with ``/`` — the
            whole-document pointer ``""`` is meaningless as a parameter
            location and is rejected rather than silently replacing the spec.
    """
    if not pointer.startswith("/"):
        raise ValueError(f"JSON pointer must start with '/', got {pointer!r}")
    return [token.replace("~1", "/").replace("~0", "~") for token in pointer[1:].split("/")]


def _write_pointer(document: Any, pointer: str, value: AxisValue) -> None:
    """Set ``pointer`` in ``document`` to ``value``, in place.

    Args:
        document: A JSON-shaped structure of dicts and lists.
        pointer: Where to write.
        value: What to write.

    Raises:
        ValueError: If any step of the pointer does not already exist. Creating
            a missing key would turn a typo in a search space into a strategy
            field nothing reads, which is a silent no-op search axis — the
            failure mode this whole module is built to make impossible.
    """
    parts = _pointer_parts(pointer)
    cursor: Any = document
    for depth, token in enumerate(parts[:-1]):
        walked = "/" + "/".join(parts[: depth + 1])
        if isinstance(cursor, list):
            try:
                cursor = cursor[int(token)]
            except (ValueError, IndexError) as error:
                raise ValueError(f"pointer {pointer!r} does not resolve at {walked!r}") from error
        elif isinstance(cursor, dict) and token in cursor:
            cursor = cursor[token]
        else:
            raise ValueError(f"pointer {pointer!r} does not resolve at {walked!r}")
    leaf = parts[-1]
    if isinstance(cursor, list):
        try:
            cursor[int(leaf)] = value
        except (ValueError, IndexError) as error:
            raise ValueError(f"pointer {pointer!r} does not resolve at its leaf") from error
    elif isinstance(cursor, dict) and leaf in cursor:
        cursor[leaf] = value
    else:
        raise ValueError(f"pointer {pointer!r} does not resolve at its leaf")


class ParameterAxis(BaseModel):
    """One tunable parameter: where it lives in a strategy spec, and its values.

    **An axis is categorical when its values are names rather than numbers, and
    that distinction is load-bearing well beyond typing.** A numeric axis has an
    order and a step, so "one index away" means something and the plateau
    machinery in :mod:`trading_system.validation.objective` can measure a
    neighbourhood along it. A categorical axis has neither: the only order its
    values have is the order somebody typed them into a JSON file, so treating
    index adjacency as adjacency would make a measured plateau width a function
    of that typing. Categoricality is therefore derived from the values (all
    strings, or all numbers — never a mix, which would leave the axis ordered
    for some pairs and not others) rather than declared in a separate field that
    could disagree with them.

    Attributes:
        name: How the axis is referred to in constraints and reports.
        target: Which document :attr:`paths` resolve in. Defaults to
            :attr:`AxisTarget.SPEC`, so every axis written before this field
            existed keeps its meaning.
        paths: Every JSON Pointer into the target document this parameter
            occupies. More than one is normal, not exceptional — see the module
            docstring on the slow EMA appearing twice in ``ema_pullback``. Empty
            for :attr:`AxisTarget.EXIT_PRESET`, which replaces a whole document
            rather than a field inside one.
        values: The ordered, discrete domain. Given directly, or expanded from
            ``low``/``high``/``step`` before validation.
        low: Inclusive lower bound, when expanding a range.
        high: Inclusive upper bound, when expanding a range.
        step: Spacing, when expanding a range.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(min_length=1)
    target: AxisTarget = AxisTarget.SPEC
    paths: tuple[str, ...] = ()
    values: tuple[AxisValue, ...] = Field(min_length=1)
    low: AxisValue | None = None
    high: AxisValue | None = None
    step: AxisValue | None = None

    @model_validator(mode="before")
    @classmethod
    def _expand_range(cls, data: Any) -> Any:
        """Turn ``low``/``high``/``step`` into an explicit ``values`` list.

        Expansion happens before validation so that the constructed object is
        always discrete, whichever form the config used: nothing downstream
        needs a branch for "range axis" versus "list axis", because after this
        there is only one kind.

        Raises:
            ValueError: If both forms or neither are given, if the range is
                incomplete, or if ``step`` is not positive.
        """
        if not isinstance(data, dict):
            return data
        has_values = data.get("values") is not None
        bounds = [data.get(key) for key in ("low", "high", "step")]
        has_range = any(bound is not None for bound in bounds)
        if has_values and has_range:
            raise ValueError("ParameterAxis takes either 'values' or 'low'/'high'/'step', not both")
        if not has_values and not has_range:
            raise ValueError("ParameterAxis needs 'values' or 'low'/'high'/'step'")
        if has_values:
            return data
        low, high, step = bounds
        if low is None or high is None or step is None:
            raise ValueError("ParameterAxis range needs all of 'low', 'high' and 'step'")
        if any(isinstance(bound, str) for bound in (low, high, step)):
            raise ValueError(
                "ParameterAxis range bounds must be numeric; a categorical axis has no step "
                "to walk and must list its values explicitly"
            )
        if step <= 0:
            raise ValueError(f"ParameterAxis.step must be positive, got {step}")
        if high < low:
            raise ValueError(f"ParameterAxis range is inverted: low={low}, high={high}")
        count = int(math.floor((high - low) / step + 1e-9)) + 1
        integral = all(isinstance(bound, int) for bound in (low, high, step))
        expanded: list[AxisValue] = []
        for index in range(count):
            raw = low + index * step
            expanded.append(int(raw) if integral else round(float(raw), 12))
        return {**data, "values": tuple(expanded)}

    @model_validator(mode="after")
    def _check_values(self) -> "ParameterAxis":
        """Reject a domain with repeats, or one mixing names with numbers.

        Raises:
            ValueError: If ``values`` contains a duplicate — two grid indices
                naming the same parameter value would make a plateau's measured
                width depend on how many times a value was written down — or if
                it mixes strings with numbers, which would leave the axis
                ordered for some pairs of values and unordered for others.
        """
        if len(set(self.values)) != len(self.values):
            raise ValueError(f"ParameterAxis {self.name!r} has duplicate values: {self.values}")
        named = sum(1 for value in self.values if isinstance(value, str))
        if named not in (0, len(self.values)):
            raise ValueError(
                f"ParameterAxis {self.name!r} mixes names with numbers: {self.values}. An axis is "
                "either categorical (all names, no order) or numeric (all numbers, ordered); "
                "half of each has no neighbourhood."
            )
        return self

    @model_validator(mode="after")
    def _check_paths_match_target(self) -> "ParameterAxis":
        """Require pointers where they mean something and forbid them where they do not.

        Raises:
            ValueError: If a pointer-bearing target carries no pointer, or if
                :attr:`AxisTarget.EXIT_PRESET` carries one. The latter would be
                a pointer nothing reads, which is the same defect class as a
                condition that silently never fires.
        """
        if self.target in _CHOICE_TARGETS:
            if self.paths:
                raise ValueError(
                    f"ParameterAxis {self.name!r} targets {self.target.value} and must not carry "
                    f"paths: it replaces the whole document, so a pointer would go unread. "
                    f"Got {self.paths}"
                )
            if any(not isinstance(value, str) for value in self.values):
                raise ValueError(
                    f"ParameterAxis {self.name!r} targets {self.target.value}, so its values must "
                    f"be ids naming entries of a library; got {self.values}"
                )
        elif not self.paths:
            raise ValueError(
                f"ParameterAxis {self.name!r} targets {self.target.value} and needs at least one "
                "JSON Pointer saying where in that document the value goes"
            )
        return self

    @property
    def categorical(self) -> bool:
        """Whether this axis names its values instead of ordering them."""
        return self.target in _CHOICE_TARGETS or isinstance(self.values[0], str)


class AxisOrdering(BaseModel):
    """A constraint that one axis's value must stay strictly below another's.

    Declarative rather than an expression string: a search space is a config
    file, and a config file that carries Python to be evaluated is a config file
    that can do anything. The one relation this stage actually needs is
    ``fast period < slow period``.

    Numeric axes only. On a categorical axis ``<`` would be a lexicographic
    comparison of names, which is an answer to a question nobody asked;
    :class:`SearchSpace` refuses such a constraint rather than evaluating it.

    Attributes:
        less: Name of the axis whose value must be smaller.
        greater: Name of the axis whose value must be larger.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    less: str
    greater: str


@dataclass(frozen=True)
class ParamSet:
    """One point in the space: a value per axis, and its grid coordinates.

    Attributes:
        values: ``(axis name, value)`` in the space's own axis order.
        coords: The grid index of each value on its axis, same order — what
            :mod:`trading_system.validation.objective` measures neighbourhoods
            in.
    """

    values: tuple[tuple[str, AxisValue], ...]
    coords: tuple[int, ...]

    def as_dict(self) -> dict[str, AxisValue]:
        """The values as a plain mapping."""
        return dict(self.values)

    def __str__(self) -> str:
        """``name=value`` pairs, comma separated, for logs and reports."""
        return ", ".join(f"{name}={value}" for name, value in self.values)


#: Validates and dumps a :data:`~trading_system.risk.sizing.config.SizingConfig`,
#: which is an annotated union rather than a class and so has no ``.model_validate``.
_SIZING_ADAPTER: TypeAdapter[SizingConfig] = TypeAdapter(SizingConfig)


@dataclass(frozen=True)
class VariationTargets:
    """The three documents a trial varies, plus the libraries choices resolve against.

    One object rather than three arguments because
    :meth:`SearchSpace.apply_to` returns all of them and every caller needs
    them together; splitting them would invite writing the varied spec while
    keeping the template's exit, which is exactly the inconsistency
    :attr:`AxisTarget.EXIT_PRESET` exists to prevent.

    Attributes:
        spec: The strategy.
        exit_preset: The exit it is bound to.
        sizing: The description the run's sizing method was built from, or
            ``None`` when nothing varies it. Required as a *description*
            because :class:`~trading_system.backtest.spec.RunInputs` holds the
            built method, from which the description cannot be recovered.
        exit_library: Presets by id, for an :attr:`AxisTarget.EXIT_PRESET`
            axis to resolve against.
        sizing_library: Sizing configs by name, for an
            :attr:`AxisTarget.SIZING_METHOD` axis to resolve against.
    """

    spec: StrategySpec
    exit_preset: ExitPresetSpec
    sizing: SizingConfig | None = None
    exit_library: Mapping[str, ExitPresetSpec] | None = None
    sizing_library: Mapping[str, SizingConfig] | None = None

    def binding(self, keys: tuple[StreamKey, ...]) -> StrategyBinding:
        """The binding these targets describe, for the streams given."""
        return StrategyBinding(spec=self.spec, exit_preset=self.exit_preset, keys=keys)


class SearchSpace(BaseModel):
    """The parameters a search may vary, discrete on every axis.

    Attributes:
        axes: The tunable parameters, in the order coordinates are reported in.
        constraints: Orderings that must hold between axes. A point violating
            one is not part of the space at all — it is never suggested,
            rather than suggested and then scored as a failure.
        disabled_axes: Axes carried by the file but not searched. This is how
            the generator proposes a categorical axis without switching it on:
            an exit axis of eight values multiplies every numeric trial by
            eight, so it is offered where the author will see it and turned on
            by *moving the entry into* :attr:`axes` — a visible edit in a diff,
            not a flag whose default someone has to remember. Nothing else
            reads them: they are excluded from :attr:`grid_size`, from
            coordinates, and from every applied value.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    axes: tuple[ParameterAxis, ...] = Field(min_length=1)
    constraints: tuple[AxisOrdering, ...] = ()
    disabled_axes: tuple[ParameterAxis, ...] = ()

    @model_validator(mode="after")
    def _check_names(self) -> "SearchSpace":
        """Reject duplicate axis names and constraints naming unknown or unordered axes.

        Raises:
            ValueError: On a duplicate name (including one shared with a
                disabled axis, which would make "enable it" ambiguous), on a
                constraint naming a misspelled axis — which would otherwise be
                a constraint that silently never binds — or on a constraint
                naming a categorical axis, whose values have no order to
                constrain.
        """
        names = [axis.name for axis in self.axes]
        if len(set(names)) != len(names):
            raise ValueError(f"SearchSpace has duplicate axis names: {names}")
        clash = sorted({axis.name for axis in self.disabled_axes} & set(names))
        if clash:
            raise ValueError(
                f"SearchSpace has axes that are both enabled and disabled: {clash}. "
                "Enabling an axis means moving it, not copying it."
            )
        by_name = {axis.name: axis for axis in self.axes}
        for constraint in self.constraints:
            for role, name in (("less", constraint.less), ("greater", constraint.greater)):
                axis = by_name.get(name)
                if axis is None:
                    raise ValueError(
                        f"constraint {role}={name!r} names no axis; "
                        f"known axes are {sorted(by_name)}"
                    )
                if axis.categorical:
                    raise ValueError(
                        f"constraint {role}={name!r} names a categorical axis; its values are "
                        "names, and ordering names would compare them lexicographically — an "
                        "answer to a question nobody asked"
                    )
        return self

    @model_validator(mode="after")
    def _check_choice_and_parameter_targets(self) -> "SearchSpace":
        """Reject varying which document is used and that document's own fields at once.

        Raises:
            ValueError: If a choice axis and the pointer axis that reaches
                inside what it chooses are both enabled. "Reward multiple 2.5"
                is undefined when the chosen preset has no fixed target at all,
                and ``risk_pct`` is undefined when the chosen sizing method is
                ``FIXED_AMOUNT``, so the product of the two axes contains
                points that are not configurations of anything. Pruning them
                would make the trial count stop being the product of the axis
                lengths and the plateau stop being a rectangle; refusing the
                combination keeps both honest, and the same ground is covered
                by one search per chosen document.
        """
        targets = {axis.target for axis in self.axes}
        for choice, parameter in _INCOMPATIBLE_TARGETS:
            if choice in targets and parameter in targets:
                raise ValueError(
                    f"a SearchSpace may vary {choice.value} or {parameter.value}, not both: a "
                    "pointer only resolves inside the document that declares the field, so the "
                    "product of the two axes contains points that describe no configuration. "
                    "Run one search per choice instead."
                )
        return self

    @property
    def categorical_mask(self) -> tuple[bool, ...]:
        """Per axis, in coordinate order, whether it is categorical.

        What :func:`~trading_system.validation.objective.analyse_plateau` needs
        in order not to treat "the next preset in the list" as a neighbour.
        """
        return tuple(axis.categorical for axis in self.axes)

    @property
    def grid_size(self) -> int:
        """How many points the full cartesian product holds, before constraints."""
        return math.prod(len(axis.values) for axis in self.axes)

    def point(self, coords: Sequence[int]) -> ParamSet:
        """Build the :class:`ParamSet` at the given grid coordinates.

        Args:
            coords: One index per axis, in axis order.

        Returns:
            The point.

        Raises:
            ValueError: If the coordinate count is wrong.
            IndexError: If any index is out of its axis's range.
        """
        if len(coords) != len(self.axes):
            raise ValueError(f"expected {len(self.axes)} coordinates, got {len(coords)}")
        return ParamSet(
            values=tuple(
                (axis.name, axis.values[index])
                for axis, index in zip(self.axes, coords, strict=True)
            ),
            coords=tuple(coords),
        )

    def satisfies(self, params: ParamSet) -> bool:
        """Whether a point respects every :attr:`constraints` ordering."""
        values = params.as_dict()
        for constraint in self.constraints:
            less, greater = values[constraint.less], values[constraint.greater]
            # Unreachable: _check_names rejects a constraint on a categorical
            # axis at construction. Kept as a narrowing guard rather than a
            # cast so the impossible case cannot become a silent comparison.
            if isinstance(less, str) or isinstance(greater, str):
                raise TypeError(f"constraint {constraint} compares a categorical axis")
            if not less < greater:
                return False
        return True

    def enumerate(self) -> Iterator[ParamSet]:
        """Every feasible point, in a fixed order: last axis varying fastest.

        The order is a property of the space, not of any search, which is what
        lets :class:`GridSearch` be deterministic without holding state.

        Yields:
            Feasible points.
        """
        widths = [len(axis.values) for axis in self.axes]
        total = math.prod(widths)
        for flat in range(total):
            coords: list[int] = []
            remainder = flat
            for width in reversed(widths):
                coords.append(remainder % width)
                remainder //= width
            candidate = self.point(tuple(reversed(coords)))
            if self.satisfies(candidate):
                yield candidate

    def feasible_size(self) -> int:
        """How many points survive :attr:`constraints`."""
        return sum(1 for _ in self.enumerate())

    def apply(self, spec: StrategySpec, params: ParamSet) -> StrategySpec:
        """A copy of ``spec`` with every spec-target axis written to this point's values.

        Round-tripped through the pydantic model rather than mutated in place,
        so a parameter combination that produces an invalid strategy fails here,
        at the point of construction, rather than somewhere inside a backtest.

        Args:
            spec: The strategy to vary.
            params: The point to write.

        Returns:
            The varied strategy.

        Raises:
            ValueError: If any pointer does not resolve, if the result is not a
                valid :class:`~trading_system.strategies.schema.StrategySpec`,
                or if this space carries an axis that does not target the spec.
                Refused rather than ignored: a space varying an exit through
                this method would return a strategy the caller would reasonably
                read as fully varied, and the exit axis would have done nothing.
        """
        foreign = sorted({a.name for a in self.axes if a.target is not AxisTarget.SPEC})
        if foreign:
            raise ValueError(
                f"apply() varies a strategy spec, but axes {foreign} target another document; "
                "use apply_to(), which varies all three"
            )
        return self._apply_spec(spec, params)

    def _apply_spec(self, spec: StrategySpec, params: ParamSet) -> StrategySpec:
        """Write the spec-target axes into ``spec``."""
        document = spec.model_dump(mode="json")
        values = params.as_dict()
        for axis in self.axes:
            if axis.target is not AxisTarget.SPEC:
                continue
            for pointer in axis.paths:
                _write_pointer(document, pointer, values[axis.name])
        return StrategySpec.model_validate(document)

    def apply_to(self, targets: "VariationTargets", params: ParamSet) -> "VariationTargets":
        """Everything one point varies, resolved together.

        Each document is round-tripped through its own validator, and the exit
        additionally through
        :func:`~trading_system.exit.library.build_plan` — the same two-layer
        split P07 stage 3 already uses, so a preset whose fields are each legal
        but whose ladder sums past 100% fails here rather than at the first
        position the backtest opens.

        Args:
            targets: What to vary, and the exit library to resolve preset ids
                against.
            params: The point to write.

        Returns:
            The varied targets. ``sizing`` is ``None`` in and ``None`` out when
            no run-target axis exists.

        Raises:
            ValueError: If a pointer does not resolve, a document fails its own
                validation, a named preset is not in the library, or a
                run-target axis was declared without the sizing config it
                varies.
        """
        values = params.as_dict()
        spec = self._apply_spec(spec=targets.spec, params=params)
        preset = targets.exit_preset
        sizing = targets.sizing

        for axis in self.axes:
            if axis.target is AxisTarget.EXIT_PRESET:
                preset_id = values[axis.name]
                if targets.exit_library is None:
                    raise ValueError(
                        f"axis {axis.name!r} chooses an exit preset by id, but no exit library "
                        "was supplied to resolve it against"
                    )
                chosen = targets.exit_library.get(str(preset_id))
                if chosen is None:
                    raise ValueError(
                        f"axis {axis.name!r} names exit preset {preset_id!r}, which the library "
                        f"does not hold; it has {sorted(targets.exit_library)}"
                    )
                preset = chosen
                # The spec's own exit_ref follows the binding, always. A run
                # whose recorded strategy names one exit while the engine
                # executed another is a run nobody can reproduce from what was
                # written down.
                spec = StrategySpec.model_validate(
                    {**spec.model_dump(mode="json"), "exit_ref": chosen.id}
                )

        param_axes = [axis for axis in self.axes if axis.target is AxisTarget.EXIT_PARAM]
        if param_axes:
            document = preset.model_dump(mode="json")
            for axis in param_axes:
                for pointer in axis.paths:
                    _write_pointer(document, pointer, values[axis.name])
            preset = ExitPresetSpec.model_validate(document)
            build_plan(preset)

        for axis in self.axes:
            if axis.target is AxisTarget.SIZING_METHOD:
                name = str(values[axis.name])
                if targets.sizing_library is None:
                    raise ValueError(
                        f"axis {axis.name!r} chooses a sizing method by name, but no sizing "
                        "library was supplied to resolve it against"
                    )
                picked = targets.sizing_library.get(name)
                if picked is None:
                    raise ValueError(
                        f"axis {axis.name!r} names sizing method {name!r}, which the library does "
                        f"not hold; it has {sorted(targets.sizing_library)}"
                    )
                sizing = picked

        run_axes = [axis for axis in self.axes if axis.target is AxisTarget.RUN]
        if run_axes:
            if sizing is None:
                raise ValueError(
                    f"axes {sorted(a.name for a in run_axes)} target the run's own knobs, but no "
                    "sizing config was supplied. RunInputs carries the built sizing method, not "
                    "the description it came from, so the description has to be passed in."
                )
            document = {"sizing": _SIZING_ADAPTER.dump_python(sizing, mode="json")}
            for axis in run_axes:
                for pointer in axis.paths:
                    _write_pointer(document, pointer, values[axis.name])
            sizing = _SIZING_ADAPTER.validate_python(document["sizing"])

        if sizing is not None and sizing is not targets.sizing:
            # The semantic half, same split as build_plan above: field bounds
            # cannot see that a maximum risk landed below its minimum.
            build_sizing_method(sizing)

        return replace(targets, spec=spec, exit_preset=preset, sizing=sizing)


# ---------------------------------------------------------------------------
# Trials
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TrialOutcome:
    """What evaluating one parameter set produced.

    Attributes:
        score: The aggregate objective value, or ``None`` if no sub-window
            could be scored at all.
        piece_scores: The objective on each evaluated sub-window — one entry
            with no cross-validation, ``k`` with it. ``None`` where that piece
            was unscoreable.
        dispersion: Standard deviation across ``piece_scores``, or ``None``
            with fewer than two scored pieces. Reported beside the aggregate
            rather than folded into it: disagreement between cross-validation
            pieces is itself a stability signal, and a single number hiding it
            would be strictly less informative than two.
        n_trades: Trades closed across every sub-window.
        returns: Daily returns concatenated across sub-windows — what a
            correlation-aware trial count would be computed from later.
        unscoreable: Why ``score`` is ``None``, verbatim from the objective.
    """

    score: float | None
    piece_scores: tuple[float | None, ...]
    dispersion: float | None
    n_trades: int
    returns: tuple[float, ...]
    unscoreable: str | None


@dataclass(frozen=True)
class TrialRecord:
    """One evaluated point: what was tried and what it produced."""

    params: ParamSet
    outcome: TrialOutcome

    def to_row(self) -> dict[str, Any]:
        """Flatten to a parquet row, one column per axis plus the outcome."""
        row: dict[str, Any] = dict(self.params.values)
        row["coords"] = list(self.params.coords)
        row["score"] = self.outcome.score
        row["dispersion"] = self.outcome.dispersion
        row["n_trades"] = self.outcome.n_trades
        row["unscoreable"] = self.outcome.unscoreable
        row["piece_scores"] = list(self.outcome.piece_scores)
        return row


@dataclass
class TrialLedger:
    """How many trials and runs a search has spent, per fold and in total.

    Mutable on purpose — it is a counter — but written to disk after every fold
    so that a crashed walk-forward resumes with its history intact rather than
    restarting a count that a later Deflated Sharpe would read.

    Attributes:
        per_fold: Trials spent on each fold index.
        runs_per_fold: Backtests spent on each fold index. Larger than
            ``per_fold`` by the cross-validation factor: with ``k`` pieces one
            trial costs ``k`` runs, and reporting only trials would understate
            the compute by that factor while overstating nothing.
    """

    per_fold: dict[int, int] = field(default_factory=dict)
    runs_per_fold: dict[int, int] = field(default_factory=dict)

    @property
    def total_trials(self) -> int:
        """Trials across every fold."""
        return sum(self.per_fold.values())

    @property
    def total_runs(self) -> int:
        """Backtests across every fold."""
        return sum(self.runs_per_fold.values())

    def record(self, fold_index: int, *, trials: int, runs: int) -> None:
        """Add one fold's spend.

        Args:
            fold_index: Which fold.
            trials: Parameter sets evaluated.
            runs: Backtests walked.

        Raises:
            ValueError: If the fold has already been recorded — a budget spent
                twice on one fold is the exact accounting error the
                "budget does not carry between folds" invariant exists to
                prevent, so it is refused rather than summed.
        """
        if fold_index in self.per_fold:
            raise ValueError(f"fold {fold_index} already recorded in the trial ledger")
        self.per_fold[fold_index] = trials
        self.runs_per_fold[fold_index] = runs

    def to_dict(self) -> dict[str, Any]:
        """Serialise to JSON-able data."""
        return {
            "per_fold": {str(k): v for k, v in sorted(self.per_fold.items())},
            "runs_per_fold": {str(k): v for k, v in sorted(self.runs_per_fold.items())},
            "total_trials": self.total_trials,
            "total_runs": self.total_runs,
        }


# ---------------------------------------------------------------------------
# The in-sample view: the only bars a search can reach
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ISWindowView:
    """Streams already cut to one in-sample window, refusing to hold anything later.

    The structural half of "the selector cannot read out-of-sample data". The
    constructor rejects any stream carrying a bar at or after
    :attr:`trade_end`, so an instance of this type is itself the proof — there
    is no later bar in it to read, whatever anyone does with it afterwards.

    Attributes:
        streams: Bars per stream, all ending before :attr:`trade_end`.
        data_start: Where publication begins; warmup, not evaluated on.
        trade_start: Earliest instant a signal may be recognised.
        trade_end: The in-sample boundary. No bar at or after this exists here.
    """

    streams: Mapping[StreamKey, OHLCVFrame]
    data_start: datetime
    trade_start: datetime
    trade_end: datetime

    def __post_init__(self) -> None:
        """Validate that no held bar reaches the in-sample boundary.

        Raises:
            ValueError: If any stream is empty or carries a bar at or after
                ``trade_end``.
        """
        if not self.streams:
            raise ValueError("ISWindowView needs at least one stream")
        for key, frame in self.streams.items():
            if frame.end is None:
                raise ValueError(f"stream {key} is empty; an in-sample view cannot be built on it")
            if frame.end >= self.trade_end:
                raise ValueError(
                    f"stream {key} carries a bar at {frame.end!r}, at or after the in-sample "
                    f"boundary {self.trade_end!r} — an ISWindowView may not hold one"
                )

    @classmethod
    def build(cls, streams: Mapping[StreamKey, OHLCVFrame], window: FoldWindow) -> "ISWindowView":
        """Cut full-coverage streams down to one in-sample window.

        Args:
            streams: The whole run's bars.
            window: The in-sample window to cut to.

        Returns:
            The view.
        """
        return cls(
            streams={
                key: frame.slice(window.data_start, window.trade_end)
                for key, frame in streams.items()
            },
            data_start=window.data_start,
            trade_start=window.trade_start,
            trade_end=window.trade_end,
        )

    def sub_windows(
        self, pieces: Sequence[tuple[datetime, datetime]] | None
    ) -> tuple[tuple[datetime, datetime], ...]:
        """The evaluation windows a trial is scored over.

        Args:
            pieces: Cross-validation test pieces, or ``None`` for the whole
                in-sample window as a single piece.

        Returns:
            ``(evaluation_start, evaluation_end)`` per piece.

        Raises:
            ValueError: If any piece falls outside this view's own window.
        """
        if pieces is None:
            return ((self.trade_start, self.trade_end),)
        for start, end in pieces:
            if start < self.trade_start or end > self.trade_end:
                raise ValueError(
                    f"cross-validation piece {start!r}..{end!r} falls outside the in-sample "
                    f"window {self.trade_start!r}..{self.trade_end!r}"
                )
        return tuple(pieces)


@dataclass(frozen=True)
class TrialRunner:
    """Evaluates one parameter set by walking it over in-sample bars only.

    The second structural half. Its template must carry exactly the view's
    streams: handing it a full-coverage
    :class:`~trading_system.backtest.spec.RunInputs` — the obvious mistake,
    since the selector holds one in order to return it — raises here instead of
    quietly optimising against the future.

    Attributes:
        view: The in-sample bars, and the only ones reachable.
        template: The run to vary. Its ``streams`` must be the view's.
        space: What may be varied.
        objective: How a walked run is scored.
        pieces: Cross-validation test pieces, or ``None`` to score over the
            whole in-sample window at once.
        libraries: Where an :attr:`AxisTarget.EXIT_PRESET` or
            :attr:`AxisTarget.SIZING_METHOD` axis resolves its names, and the
            sizing description a run-target axis edits. ``None`` when the space
            varies only the strategy spec, which is every space written before
            categorical axes existed.
    """

    view: ISWindowView
    template: RunInputs
    space: SearchSpace
    objective: Objective
    pieces: tuple[tuple[datetime, datetime], ...] | None = None
    libraries: VariationTargets | None = None

    def __post_init__(self) -> None:
        """Validate the template against the view and against what the space needs.

        Raises:
            ValueError: If the template's streams are not exactly the view's,
                if it binds other than exactly one strategy, or if the space
                varies a document this runner was given no way to resolve.
                The last is checked here rather than at the first trial so
                that a misconfigured search fails before it spends a fold's
                budget producing scores nobody can interpret.
        """
        if dict(self.template.streams) != dict(self.view.streams):
            raise ValueError(
                "TrialRunner's template must carry exactly the ISWindowView's streams; "
                "it was given a different set, which is how out-of-sample bars would enter"
            )
        if len(self.template.bindings) != 1:
            raise ValueError(
                f"TrialRunner optimises exactly one strategy binding, got "
                f"{len(self.template.bindings)}. Which strategy a SearchSpace addresses in a "
                "multi-strategy portfolio has no answer yet, and inventing one here would be "
                "a rule for a case that does not exist."
            )
        needs = {axis.target for axis in self.space.axes} - {AxisTarget.SPEC}
        if needs and self.libraries is None:
            raise ValueError(
                f"space carries axes targeting {sorted(t.value for t in needs)}, but the runner "
                "was given no libraries to resolve them against"
            )

    def _targets(self) -> VariationTargets:
        """The template's own documents, carrying whatever libraries were supplied."""
        binding = self.template.bindings[0]
        if self.libraries is None:
            return VariationTargets(spec=binding.spec, exit_preset=binding.exit_preset)
        return replace(self.libraries, spec=binding.spec, exit_preset=binding.exit_preset)

    @property
    def runs_per_trial(self) -> int:
        """How many backtests one trial costs."""
        return len(self.view.sub_windows(self.pieces))

    def evaluate(self, params: ParamSet) -> TrialOutcome:
        """Walk one parameter set over every sub-window and aggregate.

        Args:
            params: The point to evaluate.

        Returns:
            Its outcome. Aggregated by **median** across pieces, not mean: one
            cross-validation piece holding two trades can throw a wild Sortino,
            and a mean would let that single piece decide the trial.
        """
        binding = self.template.bindings[0]
        applied = self.space.apply_to(self._targets(), params)
        varied = applied.binding(binding.keys)
        sizing = (
            self.template.sizing if applied.sizing is None else build_sizing_method(applied.sizing)
        )
        scores: list[float | None] = []
        returns: list[float] = []
        trades = 0
        reasons: list[str] = []
        for start, end in self.view.sub_windows(self.pieces):
            config = self.template.config.model_copy(
                update={"evaluation_start": start, "evaluation_end": end}
            )
            inputs = replace(self.template, config=config, bindings=(varied,), sizing=sizing)
            result = inputs.run()
            trades += len(result.trades)
            try:
                scores.append(self.objective.score(result))
            except ValueError as error:
                scores.append(None)
                reasons.append(str(error))
            if result.curve:
                daily = daily_curve(result.curve)
                if len(daily.days) > 1:
                    returns.extend(simple_returns(daily))
        scored = [value for value in scores if value is not None]
        return TrialOutcome(
            score=statistics.median(scored) if scored else None,
            piece_scores=tuple(scores),
            dispersion=statistics.stdev(scored) if len(scored) > 1 else None,
            n_trades=trades,
            returns=tuple(returns),
            unscoreable=None if scored else "; ".join(dict.fromkeys(reasons)) or "no pieces scored",
        )


#: What a search calls to score a point. A plain callable rather than the
#: :class:`TrialRunner` itself, so a test can drive a search over an analytic
#: surface without constructing bars at all.
Evaluate = Callable[[ParamSet], TrialOutcome]


# ---------------------------------------------------------------------------
# The three searches
# ---------------------------------------------------------------------------


class ParameterSearch(Protocol):
    """Visits points of a space, within a budget, reporting what each scored."""

    @property
    def name(self) -> str:
        """Identifies the method in reports and stored selections."""
        ...

    def run(self, space: SearchSpace, evaluate: Evaluate, budget: int) -> list[TrialRecord]:
        """Spend up to ``budget`` trials on ``space``.

        Args:
            space: What may be varied.
            evaluate: How to score a point.
            budget: Maximum number of points to evaluate. Per fold, and per
                *parameter set* rather than per backtest — cross-validation
                multiplies the runs a trial costs, not the trials themselves,
                so all three methods spend a comparable budget on a fold
                whether or not it is cross-validated.

        Returns:
            One record per evaluated point, in evaluation order.
        """
        ...


@dataclass(frozen=True)
class GridSearch:
    """Every feasible point of the space, in the space's own fixed order.

    Has no state between trials, which is what makes it trivially
    deterministic: the order comes from :meth:`SearchSpace.enumerate`, a
    property of the space rather than of this object.
    """

    @property
    def name(self) -> str:
        """``"grid"``."""
        return "grid"

    def suggest(self, space: SearchSpace, budget: int) -> Iterator[ParamSet]:
        """Every feasible point, in order, without evaluating anything.

        Exposed separately from :meth:`run` because grid proposals genuinely do
        not depend on any outcome — which makes the determinism invariant
        testable without walking a single backtest.
        :class:`OptunaSearch` deliberately has no such method, and that absence
        is the honest statement that its proposals *are* feedback-dependent.

        Args:
            space: What to enumerate.
            budget: Checked against the feasible size; see :meth:`run`.

        Yields:
            Points.

        Raises:
            ValueError: If ``budget`` cannot cover the whole grid.
        """
        feasible = space.feasible_size()
        if budget < feasible:
            raise ValueError(
                f"GridSearch needs a budget of at least {feasible} to cover this space, got "
                f"{budget}. Truncating a grid is refused rather than silently applied: taking "
                f"the first {budget} points in enumeration order would bias the search toward "
                "the leading axis's low values, and any other truncation rule would change "
                "which space was searched without changing what the report says was searched."
            )
        yield from space.enumerate()

    def run(self, space: SearchSpace, evaluate: Evaluate, budget: int) -> list[TrialRecord]:
        """Evaluate every feasible point. See :meth:`ParameterSearch.run`."""
        return [
            TrialRecord(params=params, outcome=evaluate(params))
            for params in self.suggest(space, budget)
        ]


@dataclass(frozen=True)
class RandomSearch:
    """A seeded sample of feasible points, without replacement.

    Attributes:
        seed: Fixes the sample and its order.
    """

    seed: int = 0

    @property
    def name(self) -> str:
        """``"random"``."""
        return "random"

    def suggest(self, space: SearchSpace, budget: int) -> Iterator[ParamSet]:
        """A seeded sample of feasible points, without evaluating anything.

        Args:
            space: What to sample from.
            budget: How many points. A budget at or above the feasible size
                yields the whole space — shuffled, since a caller who wanted
                enumeration order would have asked for a grid.

        Yields:
            Points.

        Raises:
            ValueError: If ``budget`` is not positive.
        """
        if budget < 1:
            raise ValueError(f"budget must be at least 1, got {budget}")
        import random

        population = list(space.enumerate())
        rng = random.Random(self.seed)
        rng.shuffle(population)
        yield from population[:budget]

    def run(self, space: SearchSpace, evaluate: Evaluate, budget: int) -> list[TrialRecord]:
        """Evaluate a seeded sample. See :meth:`ParameterSearch.run`."""
        return [
            TrialRecord(params=params, outcome=evaluate(params))
            for params in self.suggest(space, budget)
        ]


@dataclass(frozen=True)
class OptunaSearch:
    """Tree-structured Parzen estimation over the same discrete space.

    **Trials are evaluated strictly in sequence, and this is not a limitation
    to be lifted later.** A TPE sampler's next proposal is a function of every
    completed trial before it, so running trials concurrently would make the
    proposal sequence depend on which worker finished first — the same class of
    defect that made P13 choose ``pool.map`` over ``as_completed``.
    Reproducibility here would then hold only for a fixed worker count, which
    is not reproducibility.

    Attributes:
        seed: Fixes the sampler's own random stream.
        n_startup_trials: How many points are drawn at random before TPE begins
            modelling. Named explicitly rather than left to optuna's default
            because it is part of the sampler's state: the same seed with a
            different startup count is a different sequence, and a default that
            moved between library versions would silently move the result.
    """

    seed: int = 0
    n_startup_trials: int = 10

    @property
    def name(self) -> str:
        """``"optuna"``."""
        return "optuna"

    def run(self, space: SearchSpace, evaluate: Evaluate, budget: int) -> list[TrialRecord]:
        """Spend the budget under TPE. See :meth:`ParameterSearch.run`.

        Raises:
            ValueError: If ``budget`` is not positive.
            ImportError: If optuna is not installed — it lives in the
                ``optimization`` extra, so that a plain backtest does not drag
                in a dependency only a search uses.
        """
        if budget < 1:
            raise ValueError(f"budget must be at least 1, got {budget}")
        try:
            import optuna
        except ImportError as error:  # pragma: no cover - exercised by environment, not tests
            raise ImportError(
                "OptunaSearch needs the 'optimization' extra: uv sync --extra optimization"
            ) from error

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        records: list[TrialRecord] = []

        def objective(trial: "optuna.Trial") -> float:
            coords = [
                trial.suggest_categorical(axis.name, list(range(len(axis.values))))
                for axis in space.axes
            ]
            params = space.point(coords)
            if not space.satisfies(params):
                # Pruned, not scored as a failure: an infeasible point is not
                # part of the space, and reporting it as a bad result would
                # teach the sampler that a legal region is unpromising.
                raise optuna.TrialPruned
            outcome = evaluate(params)
            records.append(TrialRecord(params=params, outcome=outcome))
            if outcome.score is None:
                raise optuna.TrialPruned
            return outcome.score

        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(
                seed=self.seed, n_startup_trials=self.n_startup_trials
            ),
        )
        study.optimize(objective, n_trials=budget, n_jobs=1, catch=())
        return records


# ---------------------------------------------------------------------------
# What one fold's optimisation produced
# ---------------------------------------------------------------------------


def _mean_abs_correlation(records: Sequence[TrialRecord]) -> tuple[float | None, int]:
    """Mean absolute pairwise correlation between trials' daily-return vectors.

    A scalar summary, computed here where the vectors already live, because the
    full matrix must not travel: 240 trials make a 240x240 matrix that costs
    1.15 MB per fold written as JSON — 9.2 MB across a walk-forward's manifest,
    the same "the whole object goes through the pipe" defect P13 documented,
    relocated into a file. The vectors themselves are written to
    ``returns.parquet`` so a correlation-aware trial count can still be derived
    later by whoever renders a verdict.

    Args:
        records: The evaluated trials.

    Returns:
        ``(mean absolute off-diagonal correlation, how many trials it covered)``.
        ``None`` when fewer than two trials share a return-vector length.
    """
    lengths = [len(record.outcome.returns) for record in records if record.outcome.returns]
    if not lengths:
        return None, 0
    modal = statistics.mode(lengths)
    usable = [record.outcome.returns for record in records if len(record.outcome.returns) == modal]
    if len(usable) < 2 or modal < 2:
        return None, len(usable)
    # A constant return vector has no correlation with anything — its standard
    # deviation is zero and every coefficient involving it is 0/0. Dropped
    # before the matrix is formed rather than filtered out of the result, so
    # the arithmetic never produces the NaN in the first place.
    varying = [vector for vector in usable if len(set(vector)) > 1]
    if len(varying) < 2:
        return None, len(usable)
    frame = pl.DataFrame({f"t{index}": list(vector) for index, vector in enumerate(varying)})
    matrix = frame.corr().to_numpy()
    total = 0.0
    count = 0
    for i in range(len(varying)):
        for j in range(i + 1, len(varying)):
            value = float(matrix[i][j])
            if value == value:  # a NaN would mean a column slipped the filter above
                total += abs(value)
                count += 1
    return (total / count if count else None), len(varying)


@dataclass(frozen=True)
class FoldOptimization:
    """What the search did on one fold, and which point came out of it.

    Attributes:
        fold_index: Which fold.
        method: :attr:`ParameterSearch.name`.
        budget: Trials the fold was allowed.
        n_trials: Trials actually evaluated.
        n_runs: Backtests those trials cost.
        n_scored: Trials that produced a score.
        cv_applied: Whether the in-sample window was cross-validated.
        cv_skip_reason: Why not, when it was not — never silent, so a report
            can never leave a reader guessing which folds were cross-validated.
        cv_k: Pieces used, when it was.
        selected: The point handed to the out-of-sample run.
        selected_score: Its in-sample score.
        best: The plain ``argmax`` point, for comparison.
        best_score: Its in-sample score.
        plateau: The full plateau analysis.
        mean_abs_correlation: How alike the trials' daily returns were — the
            scalar a correlation-aware Deflated Sharpe would start from.
        correlation_trials: How many trials that mean covered.
        selected_ranking_score: What selection actually ranked the chosen point
            by. Equal to :attr:`selected_score` when nothing adjusted it, and
            the excess over the point's own categorical null when something
            did. Reported separately rather than in place of the raw score:
            the raw score is what the run produced, the ranking score is what
            it was judged on, and collapsing them would leave a reader unable
            to tell a corrected comparison from an uncorrected one.
        best_ranking_score: The same for the plain ``argmax`` point.
        null_baselines: The fold's baseline calibration as plain data, or
            ``None`` when nothing was calibrated. Carried as a mapping rather
            than as the calibration object so that this module keeps no import
            of the one that produces it.
        trials_path: Where the per-trial table was written.
        returns_path: Where the per-trial daily returns were written.
    """

    fold_index: int
    method: str
    budget: int
    n_trials: int
    n_runs: int
    n_scored: int
    cv_applied: bool
    cv_skip_reason: str | None
    cv_k: int | None
    selected: ParamSet
    selected_score: float | None
    best: ParamSet
    best_score: float | None
    plateau: PlateauAnalysis | None
    mean_abs_correlation: float | None
    correlation_trials: int
    trials_path: Path
    returns_path: Path
    selected_ranking_score: float | None = None
    best_ranking_score: float | None = None
    null_baselines: Mapping[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialise to JSON-able data — scalars only, never the trial table."""
        plateau = None
        if self.plateau is not None:
            plateau = {
                "penalty_weight": self.plateau.penalty_weight,
                "tolerance": self.plateau.tolerance,
                "tolerance_sigmas": self.plateau.tolerance_sigmas,
                "plateau_size": self.plateau.plateau_size,
                "plateau_fraction": self.plateau.plateau_fraction,
                "axis_extent": list(self.plateau.axis_extent),
                "selection_shift": self.plateau.selection_shift,
                "score_gap": self.plateau.score_gap,
                "roughness_at_best": self.plateau.roughness_at_best,
                "roughness_at_selected": self.plateau.roughness_at_selected,
                "n_scored": self.plateau.n_scored,
                "n_without_neighbours": self.plateau.n_without_neighbours,
            }
        return {
            "fold_index": self.fold_index,
            "method": self.method,
            "budget": self.budget,
            "n_trials": self.n_trials,
            "n_runs": self.n_runs,
            "n_scored": self.n_scored,
            "cv_applied": self.cv_applied,
            "cv_skip_reason": self.cv_skip_reason,
            "cv_k": self.cv_k,
            "selected": self.selected.as_dict(),
            "selected_coords": list(self.selected.coords),
            "selected_score": self.selected_score,
            "best": self.best.as_dict(),
            "best_coords": list(self.best.coords),
            "best_score": self.best_score,
            "plateau": plateau,
            "mean_abs_correlation": self.mean_abs_correlation,
            "correlation_trials": self.correlation_trials,
            "selected_ranking_score": self.selected_ranking_score,
            "best_ranking_score": self.best_ranking_score,
            "null_baselines": None if self.null_baselines is None else dict(self.null_baselines),
            "trials_path": str(self.trials_path),
            "returns_path": str(self.returns_path),
        }


def write_fold_optimization(
    directory: Path, outcome: FoldOptimization, records: Sequence[TrialRecord]
) -> None:
    """Persist one fold's trial table, return vectors and selection.

    Args:
        directory: Where to write. Created if absent.
        outcome: The selection and its diagnostics.
        records: Every evaluated trial.
    """
    directory.mkdir(parents=True, exist_ok=True)
    pl.DataFrame([record.to_row() for record in records]).write_parquet(directory / TRIALS_FILE)
    pl.DataFrame(
        {
            "trial": list(range(len(records))),
            "returns": [list(record.outcome.returns) for record in records],
        }
    ).write_parquet(directory / RETURNS_FILE)
    (directory / SELECTION_FILE).write_text(json.dumps(outcome.to_dict(), indent=2))


def read_fold_selection(directory: Path) -> dict[str, Any] | None:
    """Read back one fold's selection, or ``None`` if it was never written.

    Args:
        directory: Where the fold's search wrote.

    Returns:
        The selection as stored, or ``None``.
    """
    path = directory / SELECTION_FILE
    if not path.exists():
        return None
    parsed: dict[str, Any] = json.loads(path.read_text())
    return parsed


def summarise(
    fold_index: int,
    search: ParameterSearch,
    records: Sequence[TrialRecord],
    *,
    budget: int,
    runs_per_trial: int,
    cv_applied: bool,
    cv_skip_reason: str | None,
    cv_k: int | None,
    directory: Path,
    tolerance_sigmas: float,
    penalty_weight: float,
    categorical: Sequence[bool] = (),
    ranking_scores: Sequence[float | None] | None = None,
    null_baselines: Mapping[str, Any] | None = None,
) -> FoldOptimization:
    """Turn a finished search into a selection plus its diagnostics.

    Args:
        fold_index: Which fold.
        search: The method used.
        records: Every evaluated trial, in evaluation order.
        budget: Trials allowed.
        runs_per_trial: Backtests each trial cost.
        cv_applied: Whether cross-validation was used.
        cv_skip_reason: Why not, when it was not.
        cv_k: Pieces used, when it was.
        directory: Where the tables will be written.
        tolerance_sigmas: Plateau margin, in standard deviations.
        penalty_weight: Instability penalty weight.
        categorical: Per axis, whether it is categorical —
            :attr:`SearchSpace.categorical_mask`. Passed through to the plateau
            analysis, which must not treat "the next value in the list" as a
            neighbour.
        ranking_scores: What to rank by, aligned with ``records``, or ``None``
            to rank by each trial's own score. Supplied when a null-baseline
            correction has been measured: the raw score stays on the record and
            in the stored table, and only the ranking changes. A ``None`` entry
            drops that trial from selection.
        null_baselines: The calibration behind ``ranking_scores``, as plain
            data, recorded on the outcome.

    Returns:
        The fold's outcome.

    Raises:
        ValueError: If no trial produced a rankable score — a fold where every
            parameter set was unscoreable has no basis for a selection, and
            returning the unchanged base parameters would report an
            optimisation that did not happen — or if ``ranking_scores`` does
            not align with ``records``.
    """
    if ranking_scores is None:
        ranks: Sequence[float | None] = [record.outcome.score for record in records]
    else:
        if len(ranking_scores) != len(records):
            raise ValueError(
                f"ranking_scores covers {len(ranking_scores)} trials but there are {len(records)}"
            )
        ranks = ranking_scores
    scored = [
        (record, value) for record, value in zip(records, ranks, strict=True) if value is not None
    ]
    if not scored:
        raise ValueError(
            f"fold {fold_index}: none of {len(records)} trials could be scored; "
            "there is nothing to select from"
        )
    points = [
        ScoredPoint(coords=record.params.coords, score=float(score)) for record, score in scored
    ]
    plateau = analyse_plateau(
        points,
        tolerance_sigmas=tolerance_sigmas,
        penalty_weight=penalty_weight,
        categorical=categorical,
    )
    best_record = scored[plateau.best_index][0]
    selected_record = scored[plateau.selected_index][0]
    correlation, covered = _mean_abs_correlation(records)
    return FoldOptimization(
        fold_index=fold_index,
        method=search.name,
        budget=budget,
        n_trials=len(records),
        n_runs=len(records) * runs_per_trial,
        n_scored=len(scored),
        cv_applied=cv_applied,
        cv_skip_reason=cv_skip_reason,
        cv_k=cv_k,
        selected=selected_record.params,
        selected_score=selected_record.outcome.score,
        best=best_record.params,
        best_score=best_record.outcome.score,
        plateau=plateau,
        mean_abs_correlation=correlation,
        correlation_trials=covered,
        trials_path=directory / TRIALS_FILE,
        returns_path=directory / RETURNS_FILE,
        selected_ranking_score=scored[plateau.selected_index][1],
        best_ranking_score=scored[plateau.best_index][1],
        null_baselines=null_baselines,
    )


__all__ = [
    "AxisOrdering",
    "AxisTarget",
    "AxisValue",
    "Evaluate",
    "FoldOptimization",
    "GridSearch",
    "ISWindowView",
    "OptunaSearch",
    "ParamSet",
    "ParameterAxis",
    "ParameterSearch",
    "RandomSearch",
    "SearchSpace",
    "TrialLedger",
    "TrialOutcome",
    "TrialRecord",
    "TrialRunner",
    "VariationTargets",
    "read_fold_selection",
    "summarise",
    "write_fold_optimization",
]
