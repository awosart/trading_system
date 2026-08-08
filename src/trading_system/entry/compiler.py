"""Turning a declarative :class:`EntrySpec` into something that runs per bar.

A spec is compiled once, into a tree of closures over already-resolved operands,
and then called once per bar. Nothing is parsed, looked up by string, or
type-dispatched at evaluation time, and nothing goes through ``eval()``: a
strategy file is configuration, and configuration that can execute arbitrary
Python is a different and much worse thing. Every error a spec can contain —
an unknown indicator, a channel that does not exist, a range with its bounds
swapped, a comparison against a range — surfaces at compile time, before the
first bar. A one-sided direction and a mandatory invalidation price are now
guaranteed by P04 itself rather than re-checked here.

One construct the schema accepts is still rejected here: **``regime_is``**. The
schema admits it so specs can be written against the eventual contract, but no
Regime module exists to classify a bar with, and a condition that silently never
fires is worse than one that refuses to compile. ``pattern_is`` and
``session_is`` do compile, reading the label columns of
:mod:`trading_system.entry.labels`.

Evaluation is stateful, because confirmation is. A trigger opens a *pending
setup*; each confirmation latches on the bar it first becomes true; the signal
fires on the bar the last one latches. State moves strictly forward — the only
thing carried between bars is which confirmations have latched and when — so an
evaluator cannot revisit a bar, and a run over ``[0..t]`` produces the same
signals as the prefix of a run over ``[0..t+n]``.
"""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import datetime
from enum import StrEnum
from types import MappingProxyType
from typing import Any

from trading_system.core.exceptions import ValidationError
from trading_system.core.logging import get_logger
from trading_system.core.types import Price, Side
from trading_system.entry.context import BarContext, BarSeries
from trading_system.entry.features import LABEL_OPERATORS, FeatureRegistry
from trading_system.entry.labels import LabelCategory
from trading_system.entry.operators import (
    Truth,
    and_all,
    between,
    cross_above,
    cross_below,
    falling,
    gt,
    gte,
    inside_range,
    label_is,
    lt,
    lte,
    negate,
    or_any,
    rising,
)
from trading_system.entry.signal import EntrySignal
from trading_system.strategies.schema import (
    AllOf,
    Condition,
    ConditionOp,
    Direction,
    EntryOrder,
    EntrySpec,
    FeatureRef,
    LabelSet,
    LeafCondition,
    LimitOrder,
    MarketOrder,
    Not,
    Operand,
    StrategySpec,
    operand_labels,
    operand_price_field,
    operand_shift,
)

logger = get_logger(__name__)

#: A compiled condition: everything it needs is bound, so it takes only the bar.
type ConditionFn = Callable[[BarContext], Truth]

#: A compiled operand, read at a lookback in bars from the current bar.
type OperandFn = Callable[[BarContext, int], float | None]

_COMPARISONS: dict[ConditionOp, Callable[[float | None, float | None], Truth]] = {
    ConditionOp.GT: gt,
    ConditionOp.GTE: gte,
    ConditionOp.LT: lt,
    ConditionOp.LTE: lte,
}

_CROSSINGS: dict[
    ConditionOp,
    Callable[[float | None, float | None, float | None, float | None], Truth],
] = {
    ConditionOp.CROSS_ABOVE: cross_above,
    ConditionOp.CROSS_BELOW: cross_below,
}

_RANGES: dict[ConditionOp, Callable[[float | None, float, float], Truth]] = {
    ConditionOp.BETWEEN: between,
    ConditionOp.INSIDE_RANGE: inside_range,
}

_SLOPES: dict[ConditionOp, Callable[[float | None, float | None], Truth]] = {
    ConditionOp.RISING: rising,
    ConditionOp.FALLING: falling,
}

_SIDES: dict[Direction, Side] = {Direction.LONG: Side.BUY, Direction.SHORT: Side.SELL}


def compile_operand(operand: Operand, registry: FeatureRegistry) -> OperandFn:
    """Resolve one leaf operand into a reader over a bar context.

    The operand's own ``shift`` is *added* to the lookback the operator asks
    for, rather than replacing it. That is what makes a shifted operand compose
    with a two-bar operator: ``close cross_above upper@1`` compares
    ``close(t)`` against ``upper(t-1)`` and ``close(t-1)`` against
    ``upper(t-2)`` — the crossing of a channel that never contained either bar
    being compared. Replacing the lookback instead would pin both sides of the
    comparison to one bar and turn every crossing into a level test.

    Args:
        operand: A feature reference, a ``price:<field>`` reference, or a
            constant.
        registry: Features available to the compiled entry.

    Returns:
        A function taking a context and a lookback in bars.

    Raises:
        ValidationError: If the operand is a feature this registry does not
            provide, or is not a recognisable operand at all.
    """
    shift = operand_shift(operand)

    if isinstance(operand, FeatureRef):
        key = registry.resolve(operand)
        return lambda ctx, lookback: ctx.feature(key, lookback + shift)

    price_field = operand_price_field(operand)
    if price_field is not None:
        return lambda ctx, lookback: ctx.price(price_field, lookback + shift)

    if isinstance(operand, float | int) and not isinstance(operand, bool):
        constant = float(operand)
        return lambda _ctx, _lookback: constant

    if isinstance(operand, LabelSet):
        raise ValidationError(
            f"a label set is not a price or a series and cannot be read as one: {operand!r}"
        )
    raise ValidationError(f"cannot compile operand {operand!r}")


def compile_condition(condition: Condition, registry: FeatureRegistry) -> ConditionFn:
    """Compile a condition tree into a single callable.

    Nesting is handled by recursion, so ``AllOf`` of ``Not`` of ``AnyOf`` costs
    exactly the closures it names and nothing else. Unknown propagates through
    every combinator per :mod:`trading_system.entry.operators`.

    Args:
        condition: Root of the tree.
        registry: Features available to the compiled entry.

    Returns:
        A function from a bar context to a three-valued outcome.

    Raises:
        ValidationError: If any leaf is malformed or uses an operator this stage
            cannot compile.
    """
    if isinstance(condition, LeafCondition):
        return _compile_leaf(condition, registry)
    if isinstance(condition, Not):
        inner = compile_condition(condition.condition, registry)
        return lambda ctx: negate(inner(ctx))
    children = tuple(compile_condition(child, registry) for child in condition.conditions)
    if isinstance(condition, AllOf):
        return lambda ctx: and_all(child(ctx) for child in children)
    return lambda ctx: or_any(child(ctx) for child in children)


def _compile_leaf(leaf: LeafCondition, registry: FeatureRegistry) -> ConditionFn:
    """Compile one comparison.

    Raises:
        ValidationError: If the operands do not fit the operator.
    """
    category = LABEL_OPERATORS.get(leaf.op)
    if category is not None:
        return _compile_label_test(leaf, category)

    if isinstance(leaf.left, LabelSet) or isinstance(leaf.right, LabelSet):
        raise ValidationError(
            f"operator {leaf.op.value!r} compares numbers; a label set is only meaningful to "
            f"{sorted(op.value for op in LABEL_OPERATORS)}"
        )

    left = _require_operand(leaf, "left")

    if leaf.op in _COMPARISONS:
        compare = _COMPARISONS[leaf.op]
        right = compile_operand(_require_operand(leaf, "right"), registry)
        left_fn = compile_operand(left, registry)
        return lambda ctx: compare(left_fn(ctx, 0), right(ctx, 0))

    if leaf.op in _CROSSINGS:
        cross = _CROSSINGS[leaf.op]
        right = compile_operand(_require_operand(leaf, "right"), registry)
        left_fn = compile_operand(left, registry)
        return lambda ctx: cross(left_fn(ctx, 0), left_fn(ctx, 1), right(ctx, 0), right(ctx, 1))

    if leaf.op in _RANGES:
        in_range = _RANGES[leaf.op]
        low, high = _require_bounds(leaf)
        left_fn = compile_operand(left, registry)
        return lambda ctx: in_range(left_fn(ctx, 0), low, high)

    if leaf.op in _SLOPES:
        slope = _SLOPES[leaf.op]
        lookback = _require_lookback(leaf)
        left_fn = compile_operand(left, registry)
        return lambda ctx: slope(left_fn(ctx, 0), left_fn(ctx, lookback))

    raise ValidationError(f"unsupported operator {leaf.op.value!r}")


def _compile_label_test(leaf: LeafCondition, category: LabelCategory) -> ConditionFn:
    """Compile one ``pattern_is`` / ``session_is`` / ``regime_is`` test.

    Raises:
        ValidationError: If the category has no source of labels yet, if the
            right operand is not a label set, or if a left operand was given —
            these operators test the bar itself, not a series.
    """
    if category is LabelCategory.REGIME:
        raise ValidationError(
            f"operator {leaf.op.value!r} cannot be compiled: there is no Regime module to "
            "classify bars with yet. The schema accepts it so specs can be written against the "
            "eventual contract; the compiler will honour it once regimes are computed."
        )

    labels = operand_labels(leaf.right)
    if labels is None:
        raise ValidationError(
            f"operator {leaf.op.value!r} requires a label set as its right operand, "
            f"got {leaf.right!r}"
        )
    if leaf.left is not None:
        raise ValidationError(
            f"operator {leaf.op.value!r} tests the bar itself, not a series; left must be omitted"
        )

    wanted = frozenset(labels.labels)
    column = category.value
    return lambda ctx: label_is(ctx.labels(column), wanted)


def _require_operand(leaf: LeafCondition, side: str) -> Operand:
    """Return the named operand, rejecting a missing or range-shaped one.

    Raises:
        ValidationError: If the operand is absent or is a pair of bounds.
    """
    operand = leaf.left if side == "left" else leaf.right
    if operand is None:
        raise ValidationError(f"operator {leaf.op.value!r} requires a {side} operand")
    if isinstance(operand, tuple):
        raise ValidationError(
            f"operator {leaf.op.value!r} takes a single {side} operand, got the range {operand!r}"
        )
    return operand


def _require_bounds(leaf: LeafCondition) -> tuple[float, float]:
    """Return the ``[low, high]`` pair a range operator needs.

    Raises:
        ValidationError: If ``right`` is not a pair, or the bounds are inverted.
            Inverted bounds are always empty, so the condition could never fire
            and the spec is a typo rather than a deliberate never-true rule.
    """
    bounds = leaf.right
    if not isinstance(bounds, tuple):
        raise ValidationError(
            f"operator {leaf.op.value!r} requires a [low, high] pair as its right operand, "
            f"got {bounds!r}"
        )
    low, high = float(bounds[0]), float(bounds[1])
    if low > high:
        raise ValidationError(
            f"operator {leaf.op.value!r} has inverted bounds [{low}, {high}]; no value can "
            "satisfy it"
        )
    return low, high


def _require_lookback(leaf: LeafCondition) -> int:
    """Return the bar distance a slope operator compares across.

    Raises:
        ValidationError: If ``right`` is present but is not a positive whole
            number of bars.
    """
    right = leaf.right
    if right is None:
        return 1
    if not isinstance(right, float | int) or isinstance(right, bool):
        raise ValidationError(
            f"operator {leaf.op.value!r} takes a lookback in bars as its right operand, "
            f"got {right!r}"
        )
    lookback = float(right)
    if lookback != int(lookback) or lookback < 1:
        raise ValidationError(
            f"operator {leaf.op.value!r} needs a whole lookback of at least 1 bar, got {lookback}"
        )
    return int(lookback)


@dataclass(frozen=True)
class _CompiledModifier:
    """One quality modifier, compiled.

    Attributes:
        condition: When true on the signal bar, ``delta`` applies.
        delta: Adjustment to the base quality.
        reason: Justification, carried into the signal's context.
    """

    condition: ConditionFn
    delta: float
    reason: str


class DropReason(StrEnum):
    """Why a confirmed setup failed to become a signal.

    Every one of these is a defect in the spec or in the data, not a normal
    outcome — a setup that expires or is invalidated is not counted here, because
    that is the machinery working. Counting them matters because a ``WARNING`` on
    a 200 000-bar run is a line nobody reads: the defect surfaces downstream as
    "this strategy took suspiciously few trades", which is indistinguishable from
    a strategy that is merely selective. A count carried out with the results
    tells the two apart.
    """

    #: The invalidation level landed on the wrong side of the entry, so the
    #: setup's own definition was contradictory on that bar.
    WRONG_SIDED_INVALIDATION = "wrong_sided_invalidation"

    #: The invalidation level had no value on the signal bar — an indicator still
    #: in warmup, or a swing that has not formed.
    INVALIDATION_UNAVAILABLE = "invalidation_unavailable"


def empty_drop_counts() -> dict[DropReason, int]:
    """A zeroed count for every reason.

    Every reason is present rather than only those that occurred, so that "no
    wrong-sided invalidations" is a recorded fact rather than a missing key that
    a reader has to interpret.

    Returns:
        Each :class:`DropReason` mapped to zero.
    """
    return dict.fromkeys(DropReason, 0)


@dataclass(frozen=True)
class EntryRun:
    """The result of running an entry over a whole series.

    Signals and drop counts travel together deliberately. Hanging the counts off
    the evaluator as an attribute to be read afterwards would make forgetting
    them the default, and they are meant to reach the backtest report.

    Attributes:
        signals: Every signal produced, oldest first.
        drops: How many confirmed setups were discarded, by reason.
        bars: Bars evaluated.
    """

    signals: tuple[EntrySignal, ...]
    drops: Mapping[DropReason, int]
    bars: int

    @property
    def dropped(self) -> int:
        """Total confirmed setups discarded as defective."""
        return sum(self.drops.values())


@dataclass
class _PendingSetup:
    """A triggered setup waiting for its confirmations.

    Attributes:
        trigger_index: Bar the trigger fired on.
        trigger_close_ts: Close time of that bar.
        latched: Bar index each confirmation first became true on, ``None`` while
            still outstanding.
    """

    trigger_index: int
    trigger_close_ts: datetime
    latched: list[int | None]

    @property
    def confirmed(self) -> bool:
        """Whether every confirmation has latched."""
        return all(index is not None for index in self.latched)


class EntryEvaluator:
    """A compiled entry, advanced one closed bar at a time.

    One instance tracks one pending setup, so it is bound to a single symbol's
    bar stream. Running the same strategy on several symbols means one evaluator
    each; :meth:`reset` returns an instance to its initial state so one can be
    reused across walk-forward folds.

    Semantics fixed here, where P04 leaves room:

    * A confirmation window of ``n`` gives ``n`` bars of opportunity **including
      the trigger bar** — so ``1`` means "confirm on the trigger bar itself",
      which is otherwise inexpressible.
    * Confirmations latch independently. "All hold within ``n`` bars of the
      trigger" is read as each becoming true at some point inside the window, not
      as all being true on one bar simultaneously.
    * Invalidation applies to a *pending* setup, on bars after the trigger, which
      is what P04's own wording says ("voids a pending setup before entry"). It
      is not applied to the trigger bar: an invalidation level drawn from a
      feature the trigger itself references would otherwise kill nearly every
      setup on the bar it was born, since the bar's low routinely pierces a level
      its close sits above.
    * While a setup is pending, a fresh trigger is ignored: the older setup owns
      the window until it confirms, expires, or is invalidated. The bar on which
      it dies is free to open a new one, though — a later trigger is a *different*
      setup with its own window and its own confirmations, and killing the first
      says nothing about the second. Blocking that bar instead would silently
      lose an edge trigger that happened to land on it.
    * A trigger is a condition, not an edge. One that stays true for ten bars
      opens ten setups; a strategy that wants the edge writes ``cross_above``.
      Throttling repeated entries is the Risk Engine's ``max_concurrent_positions``
      and ``cooldown_bars_after_loss``, not the Entry Engine's business.
    """

    def __init__(
        self,
        *,
        strategy_id: str,
        side: Side,
        trigger: ConditionFn,
        confirmations: Sequence[ConditionFn],
        confirmation_window_bars: int,
        invalidation_level: OperandFn,
        invalidation_condition: ConditionFn | None,
        base_quality: float,
        quality_modifiers: Sequence[_CompiledModifier],
        order: EntryOrder,
    ) -> None:
        """Assemble a compiled entry. Built by :func:`compile_entry`, not directly.

        Args:
            strategy_id: Id carried into every signal.
            side: Direction every signal from this entry takes.
            trigger: Compiled trigger condition.
            confirmations: Compiled confirmation conditions.
            confirmation_window_bars: Bars of opportunity, trigger bar included.
            invalidation_level: Reads the price at which the setup is void.
            invalidation_condition: Compiled invalidation condition, if any.
            base_quality: Quality before modifiers.
            quality_modifiers: Compiled condition-gated adjustments.
            order: How the entry order is placed, which fixes the reference price.
        """
        self._strategy_id = strategy_id
        self._side = side
        self._trigger = trigger
        self._confirmations = tuple(confirmations)
        self._window = confirmation_window_bars
        self._invalidation_level = invalidation_level
        self._invalidation_condition = invalidation_condition
        self._base_quality = base_quality
        self._modifiers = tuple(quality_modifiers)
        self._order = order
        self._price_offset = _reference_offset(order, side)
        self._pending: _PendingSetup | None = None
        self._drops = empty_drop_counts()

    @property
    def strategy_id(self) -> str:
        """Id of the strategy this was compiled from."""
        return self._strategy_id

    @property
    def side(self) -> Side:
        """Direction every signal from this entry takes."""
        return self._side

    @property
    def has_pending_setup(self) -> bool:
        """Whether a triggered setup is currently awaiting confirmation."""
        return self._pending is not None

    @property
    def drops(self) -> Mapping[DropReason, int]:
        """Confirmed setups discarded as defective since the last :meth:`reset`."""
        return MappingProxyType(self._drops)

    def reset(self) -> None:
        """Discard any pending setup and zero the drop counts.

        Both, because the two describe one run: keeping counts across a reset
        would tally several walk-forward folds together, and keeping the pending
        setup would carry one fold's state into the next.
        """
        self._pending = None
        self._drops = empty_drop_counts()

    def evaluate(self, ctx: BarContext) -> EntrySignal | None:
        """Advance by one closed bar.

        Args:
            ctx: View of the bar that has just closed and the bars before it.

        Returns:
            A signal if the entry completed on this bar, otherwise ``None``.
        """
        setup = self._pending
        if setup is not None and self._setup_is_dead(setup, ctx):
            self._pending = setup = None

        if setup is None:
            if self._trigger(ctx) is not True:
                return None
            setup = _PendingSetup(
                trigger_index=ctx.index,
                trigger_close_ts=ctx.bar_close_ts,
                latched=[None] * len(self._confirmations),
            )
            self._pending = setup

        for position, condition in enumerate(self._confirmations):
            if setup.latched[position] is None and condition(ctx) is True:
                setup.latched[position] = ctx.index

        if not setup.confirmed:
            return None

        self._pending = None
        return self._build_signal(ctx, setup)

    def _setup_is_dead(self, setup: _PendingSetup, ctx: BarContext) -> bool:
        """Whether a pending setup expired or was invalidated on this bar."""
        if ctx.index - setup.trigger_index >= self._window:
            return True
        if self._invalidation_condition is not None and self._invalidation_condition(ctx) is True:
            return True
        level = self._invalidation_level(ctx, 0)
        if level is None:
            return False
        if self._side is Side.BUY:
            low = ctx.price("low")
            return low is not None and low <= level
        high = ctx.price("high")
        return high is not None and high >= level

    def _build_signal(self, ctx: BarContext, setup: _PendingSetup) -> EntrySignal | None:
        """Assemble the signal for a confirmed setup, or drop a defective one."""
        close = ctx.price("close")
        level = self._invalidation_level(ctx, 0)
        if close is None or level is None:
            self._drops[DropReason.INVALIDATION_UNAVAILABLE] += 1
            logger.warning(
                "entry.signal_dropped",
                reason=DropReason.INVALIDATION_UNAVAILABLE.value,
                strategy_id=self._strategy_id,
                symbol=ctx.symbol,
                bar_index=ctx.index,
            )
            return None

        reference = close + self._price_offset
        wrong_side = level >= reference if self._side is Side.BUY else level <= reference
        if wrong_side:
            # A defective setup definition, not a risk decision: the Risk Engine
            # could only "fix" this with abs(), turning an inverted stop into a
            # plausible size computed from a meaningless distance.
            self._drops[DropReason.WRONG_SIDED_INVALIDATION] += 1
            logger.warning(
                "entry.signal_dropped",
                reason=DropReason.WRONG_SIDED_INVALIDATION.value,
                strategy_id=self._strategy_id,
                symbol=ctx.symbol,
                bar_index=ctx.index,
                side=self._side.value,
                reference_price=reference,
                invalidation_price=level,
            )
            return None

        applied = [
            (modifier.reason, modifier.delta)
            for modifier in self._modifiers
            if modifier.condition(ctx) is True
        ]
        raw_quality = self._base_quality + sum(delta for _, delta in applied)
        quality = min(1.0, max(0.0, raw_quality))

        return EntrySignal(
            strategy_id=self._strategy_id,
            symbol=ctx.symbol,
            bar_close_ts=ctx.bar_close_ts,
            side=self._side,
            reference_price=Price(reference),
            invalidation_price=Price(level),
            quality=quality,
            context=self._signal_context(ctx, setup, applied, raw_quality),
        )

    def _signal_context(
        self,
        ctx: BarContext,
        setup: _PendingSetup,
        applied: Sequence[tuple[str, float]],
        raw_quality: float,
    ) -> dict[str, Any]:
        """Assemble the audit trail: what fired, when, and what it was worth."""
        return {
            "trigger_bar_index": setup.trigger_index,
            "trigger_bar_close_ts": setup.trigger_close_ts,
            "signal_bar_index": ctx.index,
            "bars_to_confirm": ctx.index - setup.trigger_index,
            "confirmation_window_bars": self._window,
            "confirmation_bars": tuple(setup.latched),
            "base_quality": self._base_quality,
            "quality_before_clamp": raw_quality,
            "quality_modifiers": tuple(applied),
            "entry_order": self._order.type,
            "reference_offset": self._price_offset,
            "features": ctx.feature_snapshot(),
        }


class EntryEngine:
    """Every leg of one strategy, advanced together bar by bar.

    A strategy has one or two entries, at most one per direction. They are driven
    as a unit because a bar can complete both, and only something holding both
    results can say so.

    **Both signals are emitted when that happens.** Choosing between them would
    be a portfolio decision taken without any portfolio input: whether opposing
    exposure is a contradiction, a hedge, or simply two fills that net out
    depends on the book, the instrument and the prop rules, none of which the
    Entry Engine can see. Picking the higher quality would look like a decision
    while being a guess, and dropping both would destroy the more informative
    outcome of the two. What the Entry Engine *can* do is state the fact, so the
    Risk Engine is never in the position of inferring it: every signal carries
    ``context["concurrent_sides"]``, listing the sides that fired on its bar.
    Present always, not only on conflict — an absent key would mean "no conflict"
    and "an older signal" at the same time.
    """

    def __init__(self, strategy_id: str, evaluators: Sequence[EntryEvaluator]) -> None:
        """Bundle a strategy's compiled legs.

        Args:
            strategy_id: Id every signal is attributed to.
            evaluators: One evaluator per direction the strategy trades.
        """
        self._strategy_id = strategy_id
        self._evaluators = tuple(evaluators)

    @property
    def strategy_id(self) -> str:
        """Id of the strategy this was compiled from."""
        return self._strategy_id

    @property
    def evaluators(self) -> tuple[EntryEvaluator, ...]:
        """The compiled legs, in spec order."""
        return self._evaluators

    @property
    def sides(self) -> tuple[Side, ...]:
        """Directions this strategy trades."""
        return tuple(evaluator.side for evaluator in self._evaluators)

    @property
    def drops(self) -> Mapping[DropReason, int]:
        """Defective setups discarded across every leg since the last reset."""
        totals = empty_drop_counts()
        for evaluator in self._evaluators:
            for reason, count in evaluator.drops.items():
                totals[reason] += count
        return MappingProxyType(totals)

    def reset(self) -> None:
        """Return every leg to its pre-run state."""
        for evaluator in self._evaluators:
            evaluator.reset()

    def evaluate(self, ctx: BarContext) -> tuple[EntrySignal, ...]:
        """Advance every leg by one closed bar.

        Args:
            ctx: View of the bar that has just closed and the bars before it.

        Returns:
            Every signal completed on this bar, in leg order. Usually empty,
            occasionally one, and — for a symmetric strategy at a level both legs
            watch — occasionally two opposing ones.
        """
        fired = [
            signal
            for evaluator in self._evaluators
            if (signal := evaluator.evaluate(ctx)) is not None
        ]
        if not fired:
            return ()
        concurrent = tuple(signal.side.value for signal in fired)
        return tuple(
            replace(signal, context={**signal.context, "concurrent_sides": concurrent})
            for signal in fired
        )

    def run(self, series: BarSeries) -> EntryRun:
        """Evaluate every bar of a series in order, from a clean state.

        Args:
            series: Bars, features and labels to run over.

        Returns:
            The signals produced and the defects discarded, together.
        """
        self.reset()
        signals = tuple(signal for ctx in series.contexts() for signal in self.evaluate(ctx))
        return EntryRun(signals=signals, drops=self.drops, bars=len(series))


def compile_entry(strategy: StrategySpec, registry: FeatureRegistry) -> EntryEngine:
    """Compile every entry of a strategy into a runnable engine.

    Takes the whole :class:`StrategySpec` rather than an ``EntrySpec`` because a
    signal needs two things an entry does not carry: the strategy id it is
    attributed to, and the ``base_quality``/``quality_modifiers`` that score it.
    Those modifiers are gated on bar conditions, so only something holding a
    :class:`~trading_system.entry.context.BarContext` can evaluate them, and this
    is that thing. The risk profile is shared across the legs, which is the point
    of them living under one spec.

    Args:
        strategy: Strategy whose entries to compile.
        registry: Features that will be available at evaluation time, normally
            :meth:`FeatureRegistry.from_strategy` of the same spec.

    Returns:
        A fresh engine, bound to one symbol's bar stream.

    Raises:
        ValidationError: If any entry cannot be compiled — ``regime_is`` before
            the Regime module exists, a malformed leaf, or a feature the registry
            does not provide. Every one is a defect in the spec, and surfacing it
            here means it cannot appear mid-backtest.
    """
    return EntryEngine(
        strategy.id,
        [_compile_leg(strategy, entry, registry) for entry in strategy.entries],
    )


def _compile_leg(
    strategy: StrategySpec, entry: EntrySpec, registry: FeatureRegistry
) -> EntryEvaluator:
    """Compile one direction of a strategy.

    Raises:
        ValidationError: If the entry contains something uncompilable.
    """
    invalidation = entry.invalidation
    return EntryEvaluator(
        strategy_id=strategy.id,
        side=_SIDES[entry.direction],
        trigger=compile_condition(entry.trigger, registry),
        confirmations=[compile_condition(condition, registry) for condition in entry.confirmation],
        confirmation_window_bars=max(entry.confirmation_window_bars, 1),
        invalidation_level=compile_operand(invalidation.price_level, registry),
        invalidation_condition=(
            compile_condition(invalidation.condition, registry)
            if invalidation.condition is not None
            else None
        ),
        base_quality=strategy.risk_profile.base_quality,
        quality_modifiers=[
            _CompiledModifier(
                condition=compile_condition(modifier.condition, registry),
                delta=modifier.delta,
                reason=modifier.reason or "",
            )
            for modifier in strategy.risk_profile.quality_modifiers
        ],
        order=entry.entry_order.order,
    )


def _reference_offset(order: EntryOrder, side: Side) -> float:
    """Signed displacement from the signal bar's close to the intended entry price.

    A market order is anchored to the close itself — the last price actually
    observed, and the best causal estimate of where ``open[t+1]`` will be. A limit
    order sits at the favourable side of it, a stop order beyond it in the
    direction of the move being confirmed.

    Args:
        order: Order specification from the entry.
        side: Direction the entry takes.

    Returns:
        The offset to add to ``close[t]``.
    """
    if isinstance(order, MarketOrder):
        return 0.0
    long_side = side is Side.BUY
    if isinstance(order, LimitOrder):
        return -order.offset if long_side else order.offset
    return order.offset if long_side else -order.offset
