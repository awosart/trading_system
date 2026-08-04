"""Close on an opposing entry signal.

Reads :attr:`~trading_system.exit.context.ExitContext.reverse_signal_side`
rather than re-evaluating any entry condition itself — the compiled trigger
that decides "is there an opposing setup here" lives in the Entry Engine, and
re-running it from inside Exit would mean importing
:mod:`trading_system.entry.compiler` and, with it, an ``EntrySpec`` for Exit to
hold onto. That is precisely the coupling ``exit_ref``/N×M composition exists to
avoid: an :class:`~trading_system.exit.plan.ExitPlan` would stop being
combinable with an arbitrary entry and would instead be built *for* one.
Whether an opposing signal fired on this bar is decided once, by the Entry
Engine, and handed in as a plain fact.
"""

from decimal import Decimal

from trading_system.exit.base import ExitDecision, ExitKind, ExitReason, ExitTrigger
from trading_system.exit.context import ExitContext
from trading_system.exit.position import ManagedPosition


class SignalReverseExit:
    """Close in full when an opposing entry signal is recognised on this bar.

    An :class:`EntrySignal` is itself only knowable once ``close(t)`` — the
    same instant this decision is made — so ``BAR_CLOSE`` is not a choice here,
    it is the only trigger a signal-driven exit could honestly use: there is no
    order that could have been resting in the market ahead of a fact not yet
    known.
    """

    @property
    def name(self) -> str:
        """Stable identifier."""
        return "signal_reverse_exit"

    @property
    def partial_fractions(self) -> tuple[Decimal, ...]:
        """None: a signal reversal closes in full."""
        return ()

    def on_bar(self, position: ManagedPosition, ctx: ExitContext) -> ExitDecision | None:
        """Close in full if this bar's opposing signal side is not our own.

        Args:
            position: Position being managed.
            ctx: View of the closed bar ``t``, carrying
                :attr:`~trading_system.exit.context.ExitContext.reverse_signal_side`.

        Returns:
            A ``BAR_CLOSE`` full exit, or ``None`` if no signal fired or it
            agreed with the position's own side.
        """
        side = ctx.reverse_signal_side
        if side is None or side is position.side:
            return None
        return ExitDecision(
            reason=ExitReason.SIGNAL_REVERSAL,
            kind=ExitKind.FULL,
            trigger=ExitTrigger.BAR_CLOSE,
            context={"reverse_side": side.value, "bar_index": ctx.index},
        )

    def reset(self) -> None:
        """No state to clear; every decision is read fresh from the context."""
