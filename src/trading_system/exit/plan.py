"""Composing exit rules: what happens when several of them fire on one bar.

An exit is never one rule. A plan runs its whole set on every bar and then has
to answer a question OHLC data cannot: in what order did those things happen
inside the bar? :class:`ExitPlan` answers it in two passes with two different
kinds of certainty.

**Phase 1 — exit rules**, checked against the stop as it stood at the *start*
of this bar — i.e. as :attr:`ManagedPosition.stop` was left by the previous
bar's modifiers, not this one's. Decisions split by
:class:`~trading_system.exit.base.ExitTrigger`:

* ``LEVEL_TOUCH`` decisions were resting orders and filled *inside* bar ``t``.
  Their relative order is unknowable, so it is decided by
  :class:`~trading_system.exit.base.IntrabarPolicy`, applied worst-outcome-first
  by default. Nothing here is a configurable "stop beats take" priority: a stop
  is worse than a take by construction, so under ``PESSIMISTIC`` it wins as a
  consequence rather than as a setting somebody could edit.
* ``BAR_CLOSE`` decisions were taken on ``close(t)`` and cannot fill before
  ``open(t+1)``. They are handed back as :attr:`BarOutcome.deferred` and applied
  by :meth:`ExitPlan.run` on the next bar, at that bar's open — before that
  bar's levels are examined, since the open precedes every intrabar price.

At most one ``BAR_CLOSE`` decision is deferred per bar, chosen by priority.
Nothing is lost by dropping the others: a close-decided rule is level-free and
will simply say the same thing on the next bar if it still holds.

**Phase 2 — stop modifiers**, run last rather than first. A trailing or ATR
modifier typically reads *this* bar's own high or low to propose its level; if
phase 1 then tested that level against this same bar's opposite extreme, a rule
would be testing a level against the very data used to derive it moments
earlier — indistinguishable from testing an entry's invalidation against its
own trigger bar, which P06 already ruled out for the same reason. A modifier's
proposal was only ever knowable at ``close(t)``, exactly like a ``BAR_CLOSE``
decision, so ``open(t+1)`` is the earliest its effect can honestly reach a
rule — which phase 2 running last, and the next bar's phase 1 seeing the
result, is what provides.

A plan cannot be built without a protective stop. ``protective_stop`` is a
required argument with no default and no ``None``, so an unprotected position is
not expressible — the continuation of P06's decision to make ``invalidation``
mandatory. Exactly one rule ever turns a stop level into an exit; every trailing
and breakeven rule is a modifier that feeds it.
"""

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from types import MappingProxyType

from trading_system.core.exceptions import ValidationError
from trading_system.core.logging import get_logger
from trading_system.core.types import Price
from trading_system.exit.base import (
    ExitDecision,
    ExitDropReason,
    ExitRule,
    ExitTrigger,
    IntrabarPolicy,
    StopModifier,
    empty_drop_counts,
)
from trading_system.exit.context import ExitContext
from trading_system.exit.fills import approaches_from_below, resting_fill_price
from trading_system.exit.position import ClosedLeg, ManagedPosition

logger = get_logger(__name__)


@dataclass(frozen=True)
class ExitFill:
    """One decision, carried out.

    Attributes:
        rule: Name of the rule that produced the decision.
        decision: What was instructed.
        decided_bar_index: Bar the decision was taken on.
        bar_index: Bar it was filled on. Equal to ``decided_bar_index`` for a
            ``LEVEL_TOUCH`` fill and one greater for a ``BAR_CLOSE`` one, so the
            two-bar delay is visible rather than inferred.
        leg: The position leg that was booked.
    """

    rule: str
    decision: ExitDecision
    decided_bar_index: int
    bar_index: int
    leg: ClosedLeg


@dataclass(frozen=True)
class DeferredExit:
    """A decision taken on ``close(t)`` and awaiting the open of ``t+1``.

    Attributes:
        rule: Name of the rule that produced it.
        decision: What was instructed. Always ``BAR_CLOSE``, so it names no
            price — the open it fills at is not knowable when it is made.
        decided_bar_index: Bar it was decided on.
    """

    rule: str
    decision: ExitDecision
    decided_bar_index: int


@dataclass(frozen=True)
class BarOutcome:
    """What a plan did, and undertook to do, on one bar.

    Attributes:
        applied: Fills realised inside this bar, in the order they were applied.
        deferred: A decision taken on this bar's close, to be filled at the next
            bar's open. ``None`` when there is none.
    """

    applied: tuple[ExitFill, ...]
    deferred: DeferredExit | None


@dataclass(frozen=True)
class ExitRun:
    """The result of running a plan over a position's whole life.

    Fills and drop counts travel together, for P06's reason: hanging the counts
    off the plan as an attribute to be read afterwards would make forgetting
    them the default, and they belong in the backtest report.

    Attributes:
        fills: Every fill, oldest first.
        drops: How many instructions could not be carried out as stated, by
            reason. Every reason is present, including those that never fired.
        bars: Bars consumed.
        closed: Whether the position ended flat. ``False`` means the data ran
            out with the position still open, which is not an exit.
    """

    fills: tuple[ExitFill, ...]
    drops: Mapping[ExitDropReason, int]
    bars: int
    closed: bool

    @property
    def dropped(self) -> int:
        """Total instructions that could not be carried out as stated."""
        return sum(self.drops.values())


class ExitPlan:
    """A composed exit: modifiers, rules, and the policy resolving their races."""

    def __init__(
        self,
        *,
        exit_id: str,
        protective_stop: ExitRule,
        rules: Sequence[ExitRule] = (),
        stop_modifiers: Sequence[StopModifier] = (),
        intrabar_policy: IntrabarPolicy = IntrabarPolicy.PESSIMISTIC,
    ) -> None:
        """Compose one exit.

        Args:
            exit_id: Stable id this plan is registered under, the thing a
                strategy's ``exit_ref`` resolves to.
            protective_stop: The rule that turns the position's stop level into
                an exit. Mandatory and positional-by-name: a plan that could be
                built without one would make an unprotected position expressible.
            rules: Everything else that may close the position.
            stop_modifiers: Rules that move the stop without closing anything.
                Applied before the exit rules on every bar.
            intrabar_policy: How to resolve several levels touched in one bar.
                Pessimistic unless deliberately changed.

        Raises:
            ValidationError: If ``exit_id`` is empty.
        """
        if not exit_id:
            raise ValidationError("exit_id must be a non-empty identifier")
        self._exit_id = exit_id
        self._protective_stop = protective_stop
        self._rules: tuple[ExitRule, ...] = (protective_stop, *rules)
        self._stop_modifiers = tuple(stop_modifiers)
        self._intrabar_policy = intrabar_policy
        self._drops = empty_drop_counts()

    def __repr__(self) -> str:
        """Compact description naming the id and the composition."""
        return (
            f"ExitPlan({self._exit_id}, rules={[rule.name for rule in self._rules]}, "
            f"modifiers={[modifier.name for modifier in self._stop_modifiers]}, "
            f"policy={self._intrabar_policy.value})"
        )

    @property
    def exit_id(self) -> str:
        """Id this plan is registered under."""
        return self._exit_id

    @property
    def rules(self) -> tuple[ExitRule, ...]:
        """Every exit rule, protective stop first."""
        return self._rules

    @property
    def stop_modifiers(self) -> tuple[StopModifier, ...]:
        """Every stop modifier, in application order."""
        return self._stop_modifiers

    @property
    def intrabar_policy(self) -> IntrabarPolicy:
        """How simultaneous level touches are resolved."""
        return self._intrabar_policy

    @property
    def drops(self) -> Mapping[ExitDropReason, int]:
        """Instructions that could not be carried out as stated, since reset."""
        return MappingProxyType(dict(self._drops))

    def smallest_fraction(self) -> Decimal | None:
        """The smallest partial any rule in this plan can ever request.

        Static, derived from the rules' configuration rather than from a run.
        This is the hook the minimum-lot problem is solved through, and it is
        deliberately the only part of that problem this layer touches: the Exit
        Engine speaks in fractions of a position and has no idea what a lot is,
        so it cannot tell whether "close 50%" is executable. The Risk Engine
        can — it holds both the size in lots and the instrument's ``min_lot`` —
        and with this it can refuse an unexecutable pairing *at position open*
        instead of discovering mid-trade that a rung of the ladder rounds to
        nothing.

        Returns:
            The smallest fraction, or ``None`` if this plan only closes in full.
        """
        fractions = [fraction for rule in self._rules for fraction in rule.partial_fractions]
        return min(fractions) if fractions else None

    def reset(self) -> None:
        """Return every rule, modifier and counter to its pre-run state."""
        for rule in self._rules:
            rule.reset()
        for modifier in self._stop_modifiers:
            modifier.reset()
        self._drops = empty_drop_counts()

    def on_bar(self, position: ManagedPosition, ctx: ExitContext) -> BarOutcome:
        """Advance the plan by one closed bar.

        Args:
            position: The position being managed. Mutated in place by any fill
                and by any accepted stop move.
            ctx: View of the bar that has just closed and the bars before it.

        Returns:
            What was filled inside this bar, and what was deferred to the next
            bar's open.
        """
        decisions = [
            (rule.name, decision)
            for rule in self._rules
            if (decision := rule.on_bar(position, ctx)) is not None
        ]
        touched = [
            (name, decision)
            for name, decision in decisions
            if decision.trigger is ExitTrigger.LEVEL_TOUCH
        ]
        at_close = [
            (name, decision)
            for name, decision in decisions
            if decision.trigger is ExitTrigger.BAR_CLOSE
        ]

        applied: list[ExitFill] = []
        for name, decision, fill_price in self._intrabar_order(position, ctx, touched):
            if not position.is_open:
                break
            # Timestamped at the bar's close: the instant by which the fill had
            # certainly happened. Where inside the bar it happened is exactly
            # what OHLC does not record.
            leg = self._book(position, decision, fill_price, ctx.bar_close_ts)
            applied.append(
                ExitFill(
                    rule=name,
                    decision=decision,
                    decided_bar_index=ctx.index,
                    bar_index=ctx.index,
                    leg=leg,
                )
            )

        deferred: DeferredExit | None = None
        if position.is_open and at_close:
            name, decision = min(at_close, key=lambda item: item[1].priority)
            deferred = DeferredExit(rule=name, decision=decision, decided_bar_index=ctx.index)

        # Modifiers run last, not first, and their effect is on the NEXT bar's
        # rule-check, not this one's. A modifier that reads this bar's own high
        # to propose a level and a rule that reads this bar's own low to test
        # it would otherwise be testing a level against the very data used to
        # derive it — the same hazard P06 ruled out by not applying an
        # invalidation to its own trigger bar. The level a modifier proposes on
        # bar t was only ever knowable at close(t), so bar t's own intrabar
        # prices cannot have reacted to it; open(t+1) is the earliest that is
        # honest, which is exactly the rule everything else decided on
        # close(t) already follows.
        for modifier in self._stop_modifiers:
            if not position.is_open:
                break
            level = modifier.on_bar(position, ctx)
            if level is not None:
                position.tighten_stop(level)

        return BarOutcome(applied=tuple(applied), deferred=deferred)

    def run(self, position: ManagedPosition, contexts: Iterable[ExitContext]) -> ExitRun:
        """Manage a position bar by bar until it closes or the bars run out.

        A deferred decision from bar ``t`` is filled at the open of bar ``t+1``,
        before that bar's levels are examined — the open precedes every other
        price in the bar, so nothing resting inside it can have come first.

        Args:
            position: The position to manage, opened and stopped already.
            contexts: One context per bar, oldest first, normally from
                :func:`~trading_system.exit.context.exit_contexts`.

        Returns:
            Every fill, the drop counts, and whether the position ended flat.
        """
        self.reset()
        fills: list[ExitFill] = []
        deferred: DeferredExit | None = None
        bars = 0

        for ctx in contexts:
            bars += 1
            if deferred is not None:
                leg = self._book(
                    position, deferred.decision, Price(_bar_open(ctx)), ctx.bar_open_ts
                )
                fills.append(
                    ExitFill(
                        rule=deferred.rule,
                        decision=deferred.decision,
                        decided_bar_index=deferred.decided_bar_index,
                        bar_index=ctx.index,
                        leg=leg,
                    )
                )
                deferred = None
                if not position.is_open:
                    break

            outcome = self.on_bar(position, ctx)
            fills.extend(outcome.applied)
            if not position.is_open:
                break
            deferred = outcome.deferred

        return ExitRun(
            fills=tuple(fills),
            drops=self.drops,
            bars=bars,
            closed=not position.is_open,
        )

    def _intrabar_order(
        self,
        position: ManagedPosition,
        ctx: ExitContext,
        touched: Sequence[tuple[str, ExitDecision]],
    ) -> list[tuple[str, ExitDecision, Price]]:
        """Order the levels touched in one bar, and price each of them.

        The sort key is the decision's outcome in R at its own fill price, which
        makes "worst first" a definition rather than a table of rule names, and
        keeps the ordering meaningful for rules that do not exist yet. Ties fall
        back to priority and then to declaration order, so a run is reproducible.
        """
        priced = [
            (
                declared,
                name,
                decision,
                resting_fill_price(
                    _level(decision),
                    bar_open=_bar_open(ctx),
                    exit_side=position.exit_side,
                    order_type=decision.order_type,
                ),
            )
            for declared, (name, decision) in enumerate(touched)
        ]
        if len(priced) > 1:
            policy = self._intrabar_policy
            if policy is IntrabarPolicy.TICK_BASED and not ctx.has_ticks:
                self._count(
                    ExitDropReason.TICKS_UNAVAILABLE,
                    symbol=position.symbol,
                    bar_index=ctx.index,
                )
                policy = IntrabarPolicy.PESSIMISTIC

            if policy is IntrabarPolicy.TICK_BASED:
                priced.sort(
                    key=lambda item: (
                        _first_touching_tick(ctx, position, item[2]),
                        item[2].priority,
                        item[0],
                    )
                )
            else:
                sign = 1.0 if policy is IntrabarPolicy.PESSIMISTIC else -1.0
                priced.sort(
                    key=lambda item: (
                        sign * position.r_multiple(item[3]),
                        item[2].priority,
                        item[0],
                    )
                )
        return [(name, decision, fill) for _, name, decision, fill in priced]

    def _book(
        self,
        position: ManagedPosition,
        decision: ExitDecision,
        fill_price: Price,
        ts: datetime,
    ) -> ClosedLeg:
        """Apply one decision to the position, promoting an oversized partial."""
        fraction = decision.fraction
        if fraction is None or fraction >= position.remaining_fraction:
            if fraction is not None and fraction > position.remaining_fraction:
                self._count(
                    ExitDropReason.PARTIAL_EXCEEDS_REMAINDER,
                    symbol=position.symbol,
                    exit_reason=decision.reason.value,
                    requested=str(fraction),
                    remaining=str(position.remaining_fraction),
                )
            return position.close_all(price=fill_price, ts=ts, reason=decision.reason)
        return position.close(fraction, price=fill_price, ts=ts, reason=decision.reason)

    def _count(self, drop: ExitDropReason, **details: object) -> None:
        """Record a drop and log it once, with enough context to find the bar."""
        self._drops[drop] += 1
        logger.warning("exit.drop", reason=drop.value, exit_id=self._exit_id, **details)


def _level(decision: ExitDecision) -> Price:
    """The level a ``LEVEL_TOUCH`` decision rests at.

    Raises:
        ValidationError: If the decision carries no level, which
            :class:`~trading_system.exit.base.ExitDecision` should have made
            impossible.
    """
    if decision.price is None:
        raise ValidationError(f"{decision.reason.value}: a LEVEL_TOUCH decision has no level")
    return decision.price


def _bar_open(ctx: ExitContext) -> float:
    """The open of the bar a context sits at.

    Raises:
        ValidationError: If the bar has no open price.
    """
    bar_open = ctx.price("open")
    if bar_open is None:
        raise ValidationError(f"{ctx.symbol} bar {ctx.index} has no open price")
    return bar_open


def _first_touching_tick(
    ctx: ExitContext, position: ManagedPosition, decision: ExitDecision
) -> int:
    """Index of the first tick that reached a decision's level.

    A level the ticks never reach sorts last: the bar's OHLC said it was touched
    and the tick stream disagrees, which is a data defect, and putting it first
    would let a bad tick file reorder good fills.
    """
    level = _level(decision)
    rising = approaches_from_below(position.exit_side, decision.order_type)
    for index, tick in enumerate(ctx.ticks):
        if (tick.price >= level) if rising else (tick.price <= level):
            return index
    return len(ctx.ticks)
