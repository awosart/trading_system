"""Turning a declarative :class:`EntrySpec` into something that runs per bar.

A spec is compiled once, into a tree of closures over already-resolved operands,
and then called once per bar. Nothing is parsed, looked up by string, or
type-dispatched at evaluation time, and nothing goes through ``eval()``: a
strategy file is configuration, and configuration that can execute arbitrary
Python is a different and much worse thing. Every error a spec can contain —
an unknown indicator, a channel that does not exist, a range with its bounds
swapped, a comparison against a range — surfaces at compile time, before the
first bar.

Three constructs P04 permits are rejected here rather than guessed at:

* **``pattern_is`` / ``regime_is`` / ``session_is``.** P04's ``Operand`` is
  ``FeatureRef | PriceRef | float``, and ``PriceRef`` only matches
  ``price:<field>``. There is no way to write *which* pattern, regime or session
  is meant, so these operators cannot be given an argument at all. Compiling them
  would mean inventing an encoding that the schema, the validator and the JSON
  Schema know nothing about. The Regime module does not exist yet either, and
  sessions are already covered by ``SessionFilter``.
* **``direction: BOTH``.** When a trigger fires there is nothing in the tree
  saying which branch was the long one, so the signal's side is not derivable.
  Mirroring the tree automatically works on ``cross_above``/``cross_below`` and
  quietly produces nonsense on thresholds — the mirror of ``rsi < 30`` is
  ``rsi > 70``, not ``rsi > 30``.
* **An entry with no ``invalidation.price_level``.** A signal without a price at
  which it is disproven cannot be sized, and the Entry Engine has no business
  deriving one from ``risk_profile.stop_reference`` — that is the Risk Engine's
  input, and ``FIXED_PIPS`` needs instrument metadata the Entry Engine does not
  have.

Evaluation is stateful, because confirmation is. A trigger opens a *pending
setup*; each confirmation latches on the bar it first becomes true; the signal
fires on the bar the last one latches. State moves strictly forward — the only
thing carried between bars is which confirmations have latched and when — so an
evaluator cannot revisit a bar, and a run over ``[0..t]`` produces the same
signals as the prefix of a run over ``[0..t+n]``.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from trading_system.core.exceptions import ValidationError
from trading_system.core.logging import get_logger
from trading_system.core.types import Price, Side
from trading_system.entry.context import BarContext, BarSeries
from trading_system.entry.features import FeatureRegistry
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
    FeatureRef,
    LeafCondition,
    LimitOrder,
    MarketOrder,
    Not,
    Operand,
    StrategySpec,
    operand_price_field,
)

logger = get_logger(__name__)

#: A compiled condition: everything it needs is bound, so it takes only the bar.
type ConditionFn = Callable[[BarContext], Truth]

#: A compiled operand, read at a lookback in bars from the current bar.
type OperandFn = Callable[[BarContext, int], float | None]

#: Operators whose argument P04 cannot express. See the module docstring.
_CATEGORICAL_OPS = frozenset(
    {ConditionOp.PATTERN_IS, ConditionOp.REGIME_IS, ConditionOp.SESSION_IS}
)

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
    if isinstance(operand, FeatureRef):
        key = registry.resolve(operand)
        return lambda ctx, lookback: ctx.feature(key, lookback)

    price_field = operand_price_field(operand)
    if price_field is not None:
        return lambda ctx, lookback: ctx.price(price_field, lookback)

    if isinstance(operand, float | int) and not isinstance(operand, bool):
        constant = float(operand)
        return lambda _ctx, _lookback: constant

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
    if leaf.op in _CATEGORICAL_OPS:
        raise ValidationError(
            f"operator {leaf.op.value!r} needs a categorical operand (a pattern, regime or "
            "session name), which P04's Operand union cannot express: it admits only a "
            "FeatureRef, a 'price:<field>' reference, or a number. Extend the strategy schema "
            "before using it."
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

    def reset(self) -> None:
        """Discard any pending setup, returning to the pre-run state."""
        self._pending = None

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

    def run(self, series: BarSeries) -> list[EntrySignal]:
        """Evaluate every bar of a series in order, from a clean state.

        Args:
            series: Bars and features to run over.

        Returns:
            Every signal produced, oldest first.
        """
        self.reset()
        return [signal for ctx in series.contexts() if (signal := self.evaluate(ctx)) is not None]

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
            logger.warning(
                "entry.signal_dropped",
                reason="invalidation price unavailable on the signal bar",
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
            logger.warning(
                "entry.signal_dropped",
                reason="invalidation price on the wrong side of the reference price",
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


def compile_entry(strategy: StrategySpec, registry: FeatureRegistry) -> EntryEvaluator:
    """Compile a strategy's entry into an evaluator.

    Takes the whole :class:`StrategySpec` rather than its ``entry`` alone because
    a signal needs two things the ``EntrySpec`` does not carry: the strategy id it
    is attributed to, and the ``base_quality``/``quality_modifiers`` that score
    it. Those modifiers are gated on bar conditions, so only something holding a
    :class:`~trading_system.entry.context.BarContext` can evaluate them, and this
    is that thing.

    Args:
        strategy: Strategy whose entry to compile.
        registry: Features that will be available at evaluation time, normally
            :meth:`FeatureRegistry.from_strategy` of the same spec.

    Returns:
        A fresh evaluator, bound to one symbol's bar stream.

    Raises:
        ValidationError: If the entry cannot be compiled — a ``BOTH`` direction,
            a missing ``invalidation.price_level``, a categorical operator, or a
            malformed leaf. Every one of these is a defect in the spec, and
            surfacing it here means it cannot appear mid-backtest.
    """
    entry = strategy.entry
    side = _SIDES.get(entry.direction)
    if side is None:
        raise ValidationError(
            f"{strategy.id}: direction {entry.direction.value} cannot be compiled — when the "
            "trigger fires there is nothing in the condition tree saying which branch was the "
            "long one, so the signal's side is not derivable. Split the spec into a LONG and a "
            "SHORT variant."
        )

    invalidation = entry.invalidation
    if invalidation is None or invalidation.price_level is None:
        raise ValidationError(
            f"{strategy.id}: entry defines no invalidation.price_level, so no signal it emits "
            "could say where its thesis is disproven and the Risk Engine would have nothing to "
            "size from. Deriving one from risk_profile.stop_reference is the Risk Engine's job, "
            "not the Entry Engine's."
        )

    return EntryEvaluator(
        strategy_id=strategy.id,
        side=side,
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
