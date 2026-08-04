"""Take profit at a fixed multiple of the position's initial risk."""

from decimal import Decimal

from trading_system.core.types import OrderType, Price, Side
from trading_system.exit.base import ExitDecision, ExitKind, ExitReason, ExitTrigger
from trading_system.exit.context import ExitContext
from trading_system.exit.fills import resting_order_filled
from trading_system.exit.position import ManagedPosition


class FixedRR:
    """Close the whole position at ``r_multiple`` times its initial risk.

    The target is measured against
    :attr:`~trading_system.exit.position.ManagedPosition.initial_risk_distance`,
    which is frozen at entry. So a breakeven move or a trailing stop tightening
    underneath does not drag the target around: the trade was sized against the
    original risk and the reward is quoted in those same units for its whole
    life.
    """

    def __init__(self, r_multiple: float) -> None:
        """Configure the target.

        Args:
            r_multiple: Multiple of the initial risk to take profit at.

        Raises:
            ValueError: If ``r_multiple`` is not positive — a target at or below
                the entry is a stop wearing the wrong name.
        """
        if r_multiple <= 0:
            raise ValueError(f"r_multiple must be positive, got {r_multiple}")
        self._r_multiple = r_multiple

    @property
    def name(self) -> str:
        """Stable identifier, e.g. ``fixed_rr_2``."""
        return f"fixed_rr_{self._r_multiple:g}"

    @property
    def r_multiple(self) -> float:
        """Multiple of initial risk this rule targets."""
        return self._r_multiple

    @property
    def partial_fractions(self) -> tuple[Decimal, ...]:
        """None: this rule takes the whole position off at its target."""
        return ()

    def target_price(self, position: ManagedPosition) -> Price:
        """The level this rule rests its order at.

        Args:
            position: Position being managed.

        Returns:
            The target price, above the entry for a long and below for a short.
        """
        direction = 1.0 if position.side is Side.BUY else -1.0
        return Price(
            position.entry_price + direction * self._r_multiple * position.initial_risk_distance
        )

    def on_bar(self, position: ManagedPosition, ctx: ExitContext) -> ExitDecision | None:
        """Emit a full exit if the bar reached the target.

        Args:
            position: Position being managed.
            ctx: View of the closed bar ``t``.

        Returns:
            A ``LEVEL_TOUCH`` full exit at the target, or ``None``.
        """
        high = ctx.price("high")
        low = ctx.price("low")
        if high is None or low is None:
            return None
        level = self.target_price(position)
        if not resting_order_filled(
            level,
            high=high,
            low=low,
            exit_side=position.exit_side,
            order_type=OrderType.LIMIT,
        ):
            return None
        return ExitDecision(
            reason=ExitReason.TAKE_PROFIT,
            kind=ExitKind.FULL,
            trigger=ExitTrigger.LEVEL_TOUCH,
            price=level,
            order_type=OrderType.LIMIT,
            context={
                "r_multiple": self._r_multiple,
                "target": level,
                "bar_index": ctx.index,
            },
        )

    def reset(self) -> None:
        """No state to clear; the target is derived from the position."""
