"""The one rule that turns a position's stop level into an exit.

Every other stop in this library — ATR, structure, trailing, breakeven — is a
:class:`~trading_system.exit.base.StopModifier` that moves
:attr:`~trading_system.exit.position.ManagedPosition.stop`. This rule is what
reads that level and closes the position when price reaches it. Keeping it
singular is the point: with one level in the market there is nothing for two
stops to race, and the "a stop may only tighten" invariant has exactly one
enforcement site.
"""

from decimal import Decimal

from trading_system.core.types import OrderType
from trading_system.exit.base import ExitDecision, ExitKind, ExitReason, ExitTrigger
from trading_system.exit.context import ExitContext
from trading_system.exit.fills import resting_order_filled
from trading_system.exit.position import ManagedPosition


class ProtectiveStop:
    """Close the whole position when its current stop level is reached."""

    @property
    def name(self) -> str:
        """Stable identifier."""
        return "protective_stop"

    @property
    def partial_fractions(self) -> tuple[Decimal, ...]:
        """None: a protective stop closes everything or nothing."""
        return ()

    def on_bar(self, position: ManagedPosition, ctx: ExitContext) -> ExitDecision | None:
        """Emit a full exit if the bar reached the position's stop.

        Args:
            position: Position being managed, carrying the live stop.
            ctx: View of the closed bar ``t``.

        Returns:
            A ``LEVEL_TOUCH`` full exit at the stop level, or ``None``.
        """
        high = ctx.price("high")
        low = ctx.price("low")
        if high is None or low is None:
            return None
        level = position.stop
        if not resting_order_filled(
            level,
            high=high,
            low=low,
            exit_side=position.exit_side,
            order_type=OrderType.STOP,
        ):
            return None
        return ExitDecision(
            reason=ExitReason.PROTECTIVE_STOP,
            kind=ExitKind.FULL,
            trigger=ExitTrigger.LEVEL_TOUCH,
            price=level,
            order_type=OrderType.STOP,
            context={
                "stop": level,
                "initial_stop": position.initial_stop,
                "stop_source": position.stop_source,
                "bar_index": ctx.index,
            },
        )

    def reset(self) -> None:
        """No state to clear; the level lives on the position."""
