"""Where a touched exit level actually fills, as an injected port.

:mod:`trading_system.exit.fills` says in its own docstring that it is a reference
model and that P13 replaces it. This module is the seam through which that
replacement happens without the Exit Engine learning what a lot, a spread or a
commission is.

**Why a port rather than repricing the legs afterwards.** The obvious
alternative is to let :class:`~trading_system.exit.plan.ExitPlan` book its legs
at the reference price and have the backtest adjust them once it has the real
execution stack. That fails twice:

* :meth:`~trading_system.exit.plan.ExitPlan._intrabar_order` sorts touched levels
  by ``position.r_multiple(fill_price)`` — worst outcome first *is* the
  definition of ``PESSIMISTIC``. Sorting on the reference price and booking the
  executable one orders the bar's events by a number nobody trades at.
* The journal would carry two prices for one fill, and every downstream figure
  (``realized_r``, the equity curve, the trade list) would have to say which one
  it meant.

**Why the port may answer ``None``.** P12's
:class:`~trading_system.execution.fill_model.LimitTouch` can decide that a limit
resting exactly at the bar's extreme did not fill — at the level you are behind
everything already queued at that price. :func:`resting_fill_price` cannot
express that outcome at all; it always returns a price. So the port returns
``Price | None``, and a ``None`` drops the decision for that bar and is
**counted** as :attr:`~trading_system.exit.base.ExitDropReason.RESTING_ORDER_NOT_FILLED`
rather than vanishing. The blast radius is small and known: a protective stop is
a ``STOP`` order and P12's stop model never declines a level it reached, so only
targets and partial rungs can be turned away.
"""

from typing import Protocol, runtime_checkable

from trading_system.core.exceptions import ValidationError
from trading_system.core.types import Price
from trading_system.exit.base import ExitDecision
from trading_system.exit.context import ExitContext
from trading_system.exit.fills import resting_fill_price
from trading_system.exit.position import ManagedPosition


@runtime_checkable
class RestingExitPricer(Protocol):
    """Prices one ``LEVEL_TOUCH`` decision against the bar it was touched on."""

    def price(
        self, decision: ExitDecision, *, position: ManagedPosition, ctx: ExitContext
    ) -> Price | None:
        """The price this resting order executed at, or ``None`` if it did not.

        Args:
            decision: The instruction, always ``LEVEL_TOUCH`` and so always
                carrying a level and an order type.
            position: The position being closed, for the side the order takes.
            ctx: The bar the level was touched on.

        Returns:
            The executable price, or ``None`` when the order did not fill after
            all — which the caller counts rather than treats as an error.
        """
        ...


class ReferencePricer:
    """P07's model: a stop fills at the worse of level and open, a limit the better.

    The default, so that a plan built without a pricer behaves exactly as it did
    before this seam existed. Optimistic about a bad fill by its own admission —
    a run that wants the P12 execution stack injects a pricer built over it.
    """

    __slots__ = ()

    def price(
        self, decision: ExitDecision, *, position: ManagedPosition, ctx: ExitContext
    ) -> Price | None:
        """Price the level off the bar's open.

        Args:
            decision: The touched decision.
            position: The position being closed.
            ctx: The bar it was touched on.

        Returns:
            The reference fill price. Never ``None``: this model has no notion of
            an order that reached its level and failed to execute.
        """
        return resting_fill_price(
            decision_level(decision),
            bar_open=bar_open(ctx),
            exit_side=position.exit_side,
            order_type=decision.order_type,
        )


def decision_level(decision: ExitDecision) -> Price:
    """The level a ``LEVEL_TOUCH`` decision rests at.

    Args:
        decision: The decision to read.

    Returns:
        Its level.

    Raises:
        ValidationError: If the decision carries no level, which
            :class:`~trading_system.exit.base.ExitDecision` should have made
            impossible.
    """
    if decision.price is None:
        raise ValidationError(f"{decision.reason.value}: a LEVEL_TOUCH decision has no level")
    return decision.price


def bar_open(ctx: ExitContext) -> float:
    """The open of the bar a context sits at.

    Args:
        ctx: The context to read.

    Returns:
        Its bar's open price.

    Raises:
        ValidationError: If the bar has no open price.
    """
    opening = ctx.price("open")
    if opening is None:
        raise ValidationError(f"{ctx.symbol} bar {ctx.index} has no open price")
    return opening
