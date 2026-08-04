"""What a sizing method is, and why all four return money rather than lots.

Every method answers one question — **how much money should be at stake on this
trade** — and the engine turns that into lots with one shared division:

    ``size = risk_amount / (stop_distance_points * point_value)``

Keeping the division in one place is what makes the cap in the engine total. If a
method returned lots directly it would be computing risk implicitly, and the
invariant "risk never exceeds ``max_risk_pct`` of equity" would have to be
re-derived, and re-tested, once per method. As written, the cap is applied to one
number on one line, and a fifth method cannot escape it.

That framing costs :class:`~trading_system.risk.sizing.methods.VolatilityTargeting`
one step of algebra, and it is worth being explicit about. Its intent is that an
ATR-sized move be worth a target fraction of equity, i.e.
``size = equity * target / atr_value_per_lot``. Multiplying both sides by the
stop value per lot gives ``risk_amount = equity * target * stop_distance / atr``,
which is what the method returns. The behaviour is identical and the cap still
sees it.

A method may also refuse outright — that is what ``QualityScaled``'s quality
floor is. Refusal travels as :attr:`SizingOutcome.risk_amount` being ``None``
with a naming :class:`~trading_system.risk.models.RiskReason`, not as a risk of
zero: a zero would reach the engine as a size below the minimum lot and be
reported as "too small to trade", which is a different fact from "this setup was
not good enough to trade".
"""

from dataclasses import dataclass
from decimal import Decimal
from typing import Protocol, runtime_checkable

from trading_system.risk.models import RiskReason


@dataclass(frozen=True)
class SizingRequest:
    """Everything a sizing method is allowed to see.

    Deliberately narrow. A method gets the account's size, the signal's quality
    and the geometry of the trade — not the instrument, not the portfolio, not
    the strategy spec. Anything a method cannot see, it cannot accidentally start
    depending on.

    Attributes:
        equity: Sizing base, in account currency. See
            :class:`~trading_system.risk.models.AccountState` for why equity.
        quality: The signal's confidence in ``[0, 1]``.
        stop_distance_price: Entry-to-stop distance in price units, as placed by
            :mod:`trading_system.risk.stop_calculator`. Strictly positive.
        atr_price: ATR in price units on the signal bar, or ``None`` if it was
            not computed for this run.
    """

    equity: Decimal
    quality: float
    stop_distance_price: float
    atr_price: float | None


@dataclass(frozen=True)
class SizingOutcome:
    """A method's answer: an amount to risk, or a refusal to trade at all.

    Attributes:
        risk_amount: Money to put at stake, in account currency, or ``None`` to
            refuse the trade entirely.
        reason: Why — an explanation when sized, a refusal reason when not.
        detail: The specific numbers, for the decision's reason lines.
    """

    risk_amount: Decimal | None
    reason: RiskReason
    detail: str


@runtime_checkable
class SizingMethod(Protocol):
    """Turns a request into an amount of money to risk."""

    @property
    def name(self) -> str:
        """Stable identifier, reported in decisions and configuration."""
        ...

    def size(self, request: SizingRequest) -> SizingOutcome:
        """Decide how much to risk on this trade.

        Args:
            request: What the method is allowed to see.

        Returns:
            The amount to risk, or a refusal.
        """
        ...
