"""A ladder of partial closes at successive multiples of the initial risk."""

from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal

from trading_system.core.exceptions import ValidationError
from trading_system.core.types import OrderType, Price, Side
from trading_system.exit.base import ExitDecision, ExitKind, ExitReason, ExitTrigger
from trading_system.exit.context import ExitContext
from trading_system.exit.fills import resting_order_filled
from trading_system.exit.position import ManagedPosition


@dataclass(frozen=True)
class PartialRung:
    """One rung of a partial-close ladder.

    Attributes:
        r_multiple: Multiple of the initial risk this rung's target sits at.
        fraction: Share of the ORIGINAL position size this rung closes, in
            ``(0, 1)`` — the same open interval
            :class:`~trading_system.exit.base.ExitDecision` enforces for a
            ``PARTIAL`` exit, checked here too so a malformed rung fails at
            load time rather than the first time it fires.
    """

    r_multiple: float
    fraction: Decimal

    def __post_init__(self) -> None:
        """Reject a non-positive multiple or a fraction outside ``(0, 1)``.

        Raises:
            ValueError: If either bound is violated.
        """
        if self.r_multiple <= 0:
            raise ValueError(f"r_multiple must be positive, got {self.r_multiple}")
        if not Decimal(0) < self.fraction < Decimal(1):
            raise ValueError(f"fraction must be strictly inside (0, 1), got {self.fraction}")


class PartialClose:
    """Close successive slices of the position as it reaches each rung's target.

    At most one rung fires per bar, the nearest untouched one first. A single
    bar large enough to cross two rungs at once still only closes the nearer:
    the untouched rung beyond it waits for the next bar to be checked again.
    This is a stated simplification, the same kind
    :class:`~trading_system.exit.base.IntrabarPolicy` already makes for the
    stop-versus-target race — without ticks, which of two events inside one
    bar happened "first" is not knowable, and closing the nearer rung is the
    conservative reading.
    """

    def __init__(self, rungs: Sequence[PartialRung]) -> None:
        """Configure the ladder.

        Args:
            rungs: One or more rungs. Sorted by ``r_multiple`` internally, so
                any input order is accepted.

        Raises:
            ValidationError: If ``rungs`` is empty or the fractions sum to more
                than the whole position.
        """
        if not rungs:
            raise ValidationError("a partial-close ladder needs at least one rung")
        total = sum((rung.fraction for rung in rungs), Decimal(0))
        if total > Decimal(1):
            raise ValidationError(
                f"partial fractions sum to {total}, exceeding the position (100%); "
                "a ladder that promises more than the position holds cannot be executed"
            )
        self._rungs = tuple(sorted(rungs, key=lambda rung: rung.r_multiple))
        self._fired = [False] * len(self._rungs)

    @property
    def name(self) -> str:
        """Stable identifier, e.g. ``partial_close_3``."""
        return f"partial_close_{len(self._rungs)}"

    @property
    def partial_fractions(self) -> tuple[Decimal, ...]:
        """Every rung's fraction, what ``ExitPlan.smallest_closing_fraction`` reads."""
        return tuple(rung.fraction for rung in self._rungs)

    def on_bar(self, position: ManagedPosition, ctx: ExitContext) -> ExitDecision | None:
        """Close the nearest untouched rung reached on this bar, if any.

        Args:
            position: Position being managed.
            ctx: View of the closed bar ``t``.

        Returns:
            A ``LEVEL_TOUCH`` partial exit at the rung's target, or ``None`` if
            no untouched rung was reached.
        """
        high = ctx.price("high")
        low = ctx.price("low")
        if high is None or low is None:
            return None

        direction = 1.0 if position.side is Side.BUY else -1.0
        for index, rung in enumerate(self._rungs):
            if self._fired[index]:
                continue
            level = Price(
                position.entry_price + direction * rung.r_multiple * position.initial_risk_distance
            )
            if not resting_order_filled(
                level,
                high=high,
                low=low,
                exit_side=position.exit_side,
                order_type=OrderType.LIMIT,
            ):
                continue
            self._fired[index] = True
            return ExitDecision(
                reason=ExitReason.PARTIAL_TAKE_PROFIT,
                kind=ExitKind.PARTIAL,
                trigger=ExitTrigger.LEVEL_TOUCH,
                price=level,
                fraction=rung.fraction,
                order_type=OrderType.LIMIT,
                context={
                    "r_multiple": rung.r_multiple,
                    "rung_index": index,
                    "bar_index": ctx.index,
                },
            )
        return None

    def reset(self) -> None:
        """Clear every rung's fired state for a fresh run."""
        self._fired = [False] * len(self._rungs)
