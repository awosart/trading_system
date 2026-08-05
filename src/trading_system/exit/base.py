"""The Exit Engine's vocabulary: what a rule may say, and how it will be read.

Every exit in this library is one of two things, and the difference is not a
nuance — it decides where the trade fills:

* A **resting order** that was already in the market during bar ``t`` and filled
  the moment price reached its level (:attr:`ExitTrigger.LEVEL_TOUCH`). The rule
  names the level, because the level is what it decided.
* A **decision taken on ``close(t)``** from information available at ``close(t)``,
  which cannot fill before ``open(t+1)`` (:attr:`ExitTrigger.BAR_CLOSE`) — the
  same no-lookahead rule entries obey. The rule names no price at all, because
  it does not know ``open(t+1)`` and must not appear to.

The distinction lives in :class:`ExitDecision` rather than in the execution
layer's guesswork, and it is not merely documented there: ``LEVEL_TOUCH``
requires a price, ``BAR_CLOSE`` forbids one. A downstream consumer handed a
``BAR_CLOSE`` decision has no price to misuse, so "which of these two is it" is
never a question it has to answer by inference.

Three things fall out of that field:

* :class:`IntrabarPolicy` acquires a well-defined domain. The order of events
  inside a bar is unknowable without ticks, but the ambiguity only ever exists
  *between ``LEVEL_TOUCH`` decisions*; a ``BAR_CLOSE`` decision fills on the next
  bar's open and races nothing.
* "The stop outranks the take" stops being a configurable priority and becomes a
  consequence. Under :attr:`IntrabarPolicy.PESSIMISTIC` the touched levels are
  applied worst-outcome-first, and a stop is worse than a take by construction.
  A priority table that could be edited to say otherwise does not exist.
* The default cannot be talked into optimism. ``PESSIMISTIC`` is the default,
  and :attr:`IntrabarPolicy.TICK_BASED` without ticks falls back to it and
  *counts* the fallback rather than quietly resolving in the trade's favour.
  An optimistic default is the single most effective way to draw a return that
  never existed.

The rule protocols take an :class:`~trading_system.exit.context.ExitContext`
rather than a bare ``BarContext``. A bare context could not carry the ticks
``TICK_BASED`` already needs, let alone the news and volume feeds P16/P17 will
bring, and every one of those would have forced a change to the protocol
signature — the rework this interface exists to avoid. The wrapper delegates the
whole ``BarContext`` surface and inherits its no-lookahead guarantee, which the
same equivalence test re-proves.
"""

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from decimal import Decimal
from enum import IntEnum, StrEnum
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from trading_system.core.types import OrderType, Price

if TYPE_CHECKING:
    from trading_system.exit.context import ExitContext
    from trading_system.exit.position import ManagedPosition


class ExitKind(StrEnum):
    """How much of the position a decision closes."""

    #: Close the remainder, whatever it is. Carries no fraction: "all of it" has
    #: exactly one spelling, so ``FULL`` with a fraction of 0.5 is unsayable.
    FULL = "FULL"

    #: Close a fraction of the ORIGINAL size, strictly between nothing and all.
    PARTIAL = "PARTIAL"


class ExitTrigger(StrEnum):
    """When the decision was made, and therefore where it fills."""

    #: A resting order was in the market through bar ``t`` and price reached it.
    #: Fills intrabar at the level in :attr:`ExitDecision.price`, or at the bar's
    #: open if the bar gapped through it.
    LEVEL_TOUCH = "LEVEL_TOUCH"

    #: Decided on ``close(t)`` from bars ``<= t``. Fills no earlier than
    #: ``open(t+1)``. :attr:`ExitDecision.price` is ``None``: the Exit Engine
    #: does not know the next open and does not pretend to.
    BAR_CLOSE = "BAR_CLOSE"


class ExitReason(StrEnum):
    """Which class of rule produced a decision.

    Kept to what the implemented rules actually emit. A reason with no rule
    behind it would be a vocabulary entry that nothing can produce, and the
    priority table below has to stay total over this enum.
    """

    #: The position's protective stop was reached. Also what fires when a
    #: StopModifier's proposed level — ATR-based, structure-based, trailing or
    #: breakeven — was the one most recently accepted: there is exactly one
    #: level in the market, so there is exactly one reason for it firing.
    PROTECTIVE_STOP = "PROTECTIVE_STOP"

    #: A profit target measured in multiples of the initial risk was reached.
    TAKE_PROFIT = "TAKE_PROFIT"

    #: One rung of a partial-close ladder was reached.
    PARTIAL_TAKE_PROFIT = "PARTIAL_TAKE_PROFIT"

    #: Closed by elapsed time, session boundary or the approach of a market
    #: closure rather than by price reaching any level.
    TIME_EXIT = "TIME_EXIT"

    #: Closed because an opposing entry signal was recognised on this bar.
    SIGNAL_REVERSAL = "SIGNAL_REVERSAL"


class ExitPriority(IntEnum):
    """Tiebreak among decisions that are equally good or bad. Lower wins.

    Only a tiebreak. Which decision fills first is decided by outcome under the
    intrabar policy, not here; this settles the rare exact tie so that a run is
    reproducible rather than dependent on rule declaration order alone.
    """

    #: Capital preservation.
    STOP = 0

    #: Forced flat — time, session, weekend.
    FLATTEN = 1

    #: Profit taking.
    TARGET = 2


#: Priority of each reason. Total over :class:`ExitReason` by test, so a reason
#: added without a priority fails the suite rather than sorting arbitrarily.
REASON_PRIORITY: dict[ExitReason, ExitPriority] = {
    ExitReason.PROTECTIVE_STOP: ExitPriority.STOP,
    ExitReason.TAKE_PROFIT: ExitPriority.TARGET,
    ExitReason.PARTIAL_TAKE_PROFIT: ExitPriority.TARGET,
    ExitReason.TIME_EXIT: ExitPriority.FLATTEN,
    ExitReason.SIGNAL_REVERSAL: ExitPriority.FLATTEN,
}


class IntrabarPolicy(StrEnum):
    """How to resolve several resting orders touched inside the same bar.

    Without ticks the order of events inside a bar is genuinely unknown, so this
    is a modelling choice that has to be stated rather than defaulted into.
    """

    #: The worst reachable outcome happened first. The default, and a *bound*
    #: rather than a reconstruction: where several adverse levels were touched it
    #: assumes the most adverse of them. Erring in this direction understates a
    #: strategy; erring in the other invents money.
    PESSIMISTIC = "PESSIMISTIC"

    #: The best outcome happened first. For measuring how much of a result is an
    #: artefact of the intrabar assumption — never as a reporting default.
    OPTIMISTIC = "OPTIMISTIC"

    #: Order the touched levels by the tick that reached each one first. Falls
    #: back to ``PESSIMISTIC`` when the bar carries no ticks, and records the
    #: fallback in :class:`ExitDropReason` — a silent fallback would let a run
    #: configured for tick precision report optimistic-looking fills it never
    #: earned.
    TICK_BASED = "TICK_BASED"


class ExitDropReason(StrEnum):
    """Where an exit instruction could not be carried out as stated.

    Counted rather than logged, for P06's reason: a ``WARNING`` on a 200 000-bar
    run is a line nobody reads, and the defect surfaces downstream as "this exit
    behaved oddly", which is indistinguishable from an exit that is merely
    conservative. Every reason is present with a count of zero, so "no ticks
    were missing" is a recorded fact rather than an absent key.
    """

    #: A partial asked for more than the position had left. It was promoted to a
    #: full close — the only executable reading of the instruction.
    PARTIAL_EXCEEDS_REMAINDER = "partial_exceeds_remainder"

    #: ``TICK_BASED`` was configured but the bar carried no ticks, so the bar was
    #: resolved pessimistically.
    TICKS_UNAVAILABLE = "ticks_unavailable"

    #: A partial rounded to nothing against the instrument's minimum lot, so no
    #: order could be emitted. Never raised by this layer, which knows fractions
    #: and not lots: it is incremented by the executor that owns the instrument
    #: metadata, and lives here because this is where exit counts are collected.
    #: The Risk Engine should have refused the pairing at position open — see
    #: :meth:`~trading_system.exit.plan.ExitPlan.smallest_closing_fraction`.
    PARTIAL_QUANTIZED_TO_ZERO = "partial_quantized_to_zero"

    #: A resting order whose level the bar reached did not execute. Only an
    #: injected :class:`~trading_system.exit.pricing.RestingExitPricer` can
    #: produce this — the reference model has no notion of it — and in practice
    #: only for a limit resting exactly at the bar's extreme, where the order sat
    #: behind a queue at that price which the bar did not clear. A stop that is
    #: reached always fills, so a protective stop is never turned away here.
    RESTING_ORDER_NOT_FILLED = "resting_order_not_filled"


def empty_drop_counts() -> dict[ExitDropReason, int]:
    """A zeroed count for every reason.

    Returns:
        Every :class:`ExitDropReason` mapped to zero.
    """
    return dict.fromkeys(ExitDropReason, 0)


@dataclass(frozen=True)
class ExitDecision:
    """One rule's instruction for one bar.

    Attributes:
        reason: Which class of rule this came from.
        kind: Whether it closes the remainder or a fraction of the original size.
        trigger: Whether it was a resting order filled intrabar, or a decision
            taken on the close and filled at the next open.
        price: The level the resting order sat at, for ``LEVEL_TOUCH``. ``None``
            for ``BAR_CLOSE``, always.
        fraction: Share of the ORIGINAL position size, for ``PARTIAL``. ``None``
            for ``FULL``, always.
        order_type: How the resting order was placed. This is not decoration —
            it is what tells the fill model which side of the level price
            approached from, and therefore whether a gap through the level fills
            better or worse than the level. ``BAR_CLOSE`` decisions are
            ``MARKET`` by construction, there being no order resting anywhere.
        context: What the rule saw, for trade forensics. Never read by any
            control-flow decision; excluded from equality for the same reason.
    """

    reason: ExitReason
    kind: ExitKind
    trigger: ExitTrigger
    price: Price | None = None
    fraction: Decimal | None = None
    order_type: OrderType = OrderType.MARKET
    context: Mapping[str, Any] = field(default_factory=dict, compare=False)

    def __post_init__(self) -> None:
        """Enforce the field dependencies that keep a decision unambiguous.

        Raises:
            ValueError: If a ``FULL`` decision carries a fraction, a ``PARTIAL``
                one lacks a fraction strictly inside ``(0, 1)``, a
                ``LEVEL_TOUCH`` decision lacks a finite price, or a
                ``BAR_CLOSE`` decision carries a price or a non-market order.
        """
        object.__setattr__(self, "context", MappingProxyType(dict(self.context)))

        if self.kind is ExitKind.FULL:
            if self.fraction is not None:
                raise ValueError(
                    f"a FULL exit closes the remainder and carries no fraction, got "
                    f"{self.fraction}; use kind=PARTIAL to close a share of the position"
                )
        else:
            if not isinstance(self.fraction, Decimal):
                raise ValueError(
                    "a PARTIAL exit requires a Decimal fraction of the original position size, "
                    f"got {self.fraction!r}"
                )
            if not Decimal(0) < self.fraction < Decimal(1):
                raise ValueError(
                    f"a PARTIAL fraction must be strictly inside (0, 1), got {self.fraction}; "
                    "closing all of it is kind=FULL"
                )

        if self.trigger is ExitTrigger.LEVEL_TOUCH:
            if self.price is None or not math.isfinite(self.price):
                raise ValueError(
                    f"a LEVEL_TOUCH exit is a resting order and must name its level, got "
                    f"{self.price!r}"
                )
        else:
            if self.price is not None:
                raise ValueError(
                    f"a BAR_CLOSE exit fills at open(t+1), which the Exit Engine cannot see; "
                    f"it must not name a price, got {self.price!r}"
                )
            if self.order_type is not OrderType.MARKET:
                raise ValueError(
                    f"a BAR_CLOSE exit has no order resting in the market, so it can only be "
                    f"MARKET, got {self.order_type.value}"
                )

    @property
    def is_full(self) -> bool:
        """Whether this closes the whole remainder."""
        return self.kind is ExitKind.FULL

    @property
    def priority(self) -> ExitPriority:
        """Tiebreak rank of this decision's reason."""
        return REASON_PRIORITY[self.reason]


@runtime_checkable
class ExitRule(Protocol):
    """Something that can decide to close all or part of a position.

    Stateless by preference: an implementation that carries state across bars
    must clear it in :meth:`reset` so one instance can be reused across
    walk-forward folds without leaking a previous fold's state into the next.
    """

    @property
    def name(self) -> str:
        """Stable identifier of this rule, used in forensics and library files."""
        ...

    @property
    def partial_fractions(self) -> tuple[Decimal, ...]:
        """Every partial fraction this rule can ever request, empty if none.

        Static — read off the rule's configuration, never from a simulation.
        This is what lets the Risk Engine check, at the moment it opens a
        position, whether the smallest partial in the plan will survive the
        instrument's minimum lot, instead of discovering mid-trade that a rung
        of the ladder rounds to nothing. The Exit Engine itself never applies
        this knowledge: it speaks in fractions and has no idea what a lot is.
        """
        ...

    def on_bar(self, position: "ManagedPosition", ctx: "ExitContext") -> ExitDecision | None:
        """Decide what to do about ``position`` on the bar ``ctx`` sits at.

        Args:
            position: The position being managed, with its live stop.
            ctx: View of the closed bar ``t`` and everything before it.

        Returns:
            The instruction, or ``None`` to do nothing on this bar.
        """
        ...

    def reset(self) -> None:
        """Return the rule to its pre-run state."""
        ...


@runtime_checkable
class StopModifier(Protocol):
    """Something that moves a position's stop without ever closing it.

    Every rule that only relocates the stop is a modifier, not an exit —
    trailing stops and breakeven moves, but equally an ATR- or structure-based
    stop that recomputes over the trade's life. Making any of them emit exits
    would put two competing stop levels in the market — the position's own and
    the rule's — racing each other every bar, and would spread the "a stop may
    only tighten" invariant across every rule that has one. As modifiers they
    all funnel through
    :meth:`~trading_system.exit.position.ManagedPosition.tighten_stop`, where the
    ratchet is enforced once, and exactly one rule
    (:class:`~trading_system.exit.rules.protective_stop.ProtectiveStop`) ever
    converts a stop level into an exit — which is also why none of these
    modifiers carries its own :class:`ExitReason`: whichever one most recently
    tightened the level, the level firing is always attributed to
    ``PROTECTIVE_STOP``.
    """

    @property
    def name(self) -> str:
        """Stable identifier of this modifier."""
        ...

    def on_bar(self, position: "ManagedPosition", ctx: "ExitContext") -> Price | None:
        """Propose a stop level for ``position`` on the bar ``ctx`` sits at.

        A proposal that would loosen the stop is declined by the position and is
        not an error: recomputing a trailing level on a pullback produces one as
        a matter of course.

        Args:
            position: The position being managed.
            ctx: View of the closed bar ``t`` and everything before it.

        Returns:
            The proposed level, or ``None`` to leave the stop alone.
        """
        ...

    def reset(self) -> None:
        """Return the modifier to its pre-run state."""
        ...
