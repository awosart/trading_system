"""What the Risk Engine is told, and what it answers.

The engine's whole output is a :class:`RiskDecision`: how much to trade, how much
that puts at stake, where the stop went, and — in every case, approval or
refusal — why. The "why" is not decoration. A run that produces few trades is
indistinguishable from a selective strategy unless the refusals are recorded, so
:attr:`RiskDecision.rejection` carries a machine-readable reason the engine
counts, alongside the human-readable :attr:`RiskDecision.reasons` lines that name
the actual numbers. This is the same discipline P06 applied to dropped entry
signals and P07 to undeliverable exit instructions.

**Sizing measures against equity, not balance.** Three reasons, in order of
weight:

1. Prop firms compute daily and maximum drawdown against equity, floating losses
   included. Sizing from balance while the limit that can end the account watches
   equity puts the two figures out of step exactly during the drawdown the limit
   exists for.
2. Equity is anti-martingale for free. Three open losers shrink it before any of
   them is closed, so the next position is smaller; balance would keep full size
   while the account is already down.
3. The counter-argument — that floating *profit* raises size, compounding open
   risk — is real, and is answered by a cap on total open risk in stage 2, not by
   changing the base. Switching to balance would trade a bounded problem with a
   known fix for both of the properties above.

There is deliberately no ``sizing_basis`` switch. One basis, documented; changing
it is an edit with a test, not a config flag nobody remembers setting.
"""

import math
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from enum import StrEnum

from trading_system.core.types import Price, ensure_utc

#: Stop level on a refusal taken before any stop was computed.
NO_STOP = Price(0.0)


class RiskReason(StrEnum):
    """Why a decision came out the way it did, machine-readably.

    Every line in :attr:`RiskDecision.reasons` is prefixed with one of these, so
    the same text is both readable in a report and countable in aggregate.
    Members divide into refusals — at most one per decision, in
    :attr:`RiskDecision.rejection` — and explanations, which accumulate.
    """

    # --- refusals -------------------------------------------------------
    UNKNOWN_INSTRUMENT = "unknown_instrument"
    NON_POSITIVE_EQUITY = "non_positive_equity"
    ACCOUNT_CURRENCY_MISMATCH = "account_currency_mismatch"
    QUALITY_BELOW_FLOOR = "quality_below_floor"
    FX_RATE_UNAVAILABLE = "fx_rate_unavailable"
    ATR_UNAVAILABLE = "atr_unavailable"
    BELOW_MIN_LOT = "below_min_lot"
    ABOVE_MAX_LOT = "above_max_lot"
    EXIT_LADDER_UNEXECUTABLE = "exit_ladder_unexecutable"

    # --- explanations ---------------------------------------------------
    SIZED = "sized"
    STOP_FROM_INVALIDATION = "stop_from_invalidation"
    STOP_WIDENED_TO_BROKER_MINIMUM = "stop_widened_to_broker_minimum"
    RISK_CAPPED = "risk_capped"
    SIZE_ROUNDED_DOWN = "size_rounded_down"


#: Every refusal reason. A decision carrying one of these is never approved.
REJECTION_REASONS: frozenset[RiskReason] = frozenset(
    {
        RiskReason.UNKNOWN_INSTRUMENT,
        RiskReason.NON_POSITIVE_EQUITY,
        RiskReason.ACCOUNT_CURRENCY_MISMATCH,
        RiskReason.QUALITY_BELOW_FLOOR,
        RiskReason.FX_RATE_UNAVAILABLE,
        RiskReason.ATR_UNAVAILABLE,
        RiskReason.BELOW_MIN_LOT,
        RiskReason.ABOVE_MAX_LOT,
        RiskReason.EXIT_LADDER_UNEXECUTABLE,
    }
)


def reason_line(reason: RiskReason, detail: str) -> str:
    """Format one line of :attr:`RiskDecision.reasons`.

    Args:
        reason: The tag the line is about.
        detail: The specific numbers, in plain language.

    Returns:
        ``"<reason>: <detail>"``.
    """
    return f"{reason.value}: {detail}"


def empty_rejection_counts() -> dict[RiskReason, int]:
    """Build a zeroed counter over every refusal reason.

    Every reason is present from the start, including those that never fire.
    "No trade was ever refused for an unexecutable ladder" is then a recorded
    fact rather than a missing key indistinguishable from a forgotten counter.

    Returns:
        A fresh counter, all zero.
    """
    return dict.fromkeys(sorted(REJECTION_REASONS), 0)


@dataclass(frozen=True)
class AccountState:
    """The account as it stood when a signal arrived.

    Passed in explicitly rather than queried. There is no portfolio module yet,
    and when there is one it will hand this over rather than be reached into.

    Attributes:
        currency: Account denomination. Every money figure the engine returns is
            in this currency.
        balance: Closed-trade equity. Recorded for reporting and for stage 2;
            **not** what size is computed from.
        equity: Balance plus the floating result of open positions. This is the
            sizing base — see the module docstring.
        as_of: When this snapshot was taken, tz-aware. Also the instant FX rates
            are priced at.
        open_risk_amount: Money at risk across positions already open, in account
            currency. Stage 1 carries it without acting on it; the portfolio heat
            cap that consumes it is stage 2.
    """

    currency: str
    balance: Decimal
    equity: Decimal
    as_of: datetime
    open_risk_amount: Decimal = Decimal(0)

    def __post_init__(self) -> None:
        """Normalise the timestamp and reject impossible figures.

        Raises:
            ValueError: If the currency is empty, ``as_of`` is naive, or
                ``open_risk_amount`` is negative. A non-positive ``equity`` is
                *not* rejected here — a blown account is a real state, and the
                engine refuses to size against it rather than being unable to
                describe it.
        """
        object.__setattr__(self, "as_of", ensure_utc(self.as_of))
        if not self.currency:
            raise ValueError("account currency must be a non-empty code")
        if self.open_risk_amount < 0:
            raise ValueError(f"open_risk_amount must not be negative, got {self.open_risk_amount}")


@dataclass(frozen=True)
class RiskDecision:
    """The engine's verdict on one signal.

    Attributes:
        approved: Whether a position may be opened at all.
        size: Size in lots, already a whole multiple of the instrument's
            ``lot_step``. Zero when refused.
        risk_amount: Money at stake if the stop is hit, in account currency.
            Computed from the *quantised* ``size``, not from the size that was
            asked for, so it is what the account actually stands to lose rather
            than what the sizing method requested. Since quantisation only ever
            rounds down, this is at or below the cap, never above it.
        risk_pct: ``risk_amount`` as a **fraction** of equity — ``0.005`` is half
            a percent, not five thousandths of one. The same convention holds for
            every ``*_pct`` field in :mod:`trading_system.risk`.
        stop_price: Absolute price the protective stop was placed at. This is the
            level the Exit Engine will protect and the level ``risk_amount`` was
            measured against; the two cannot disagree because the size was
            derived from this number.
        reasons: Human-readable lines, each prefixed by a :class:`RiskReason`.
            Populated on approval as well as refusal — "why this size" is as much
            a question as "why no trade".
        rejection: The single reason a refusal happened, ``None`` on approval.
            Separate from ``reasons`` because counting refusals must not mean
            parsing prose.
        fx_rate: Rate applied to convert the instrument's quote currency into the
            account's; ``Decimal(1)`` when they match. Recorded so the decision
            can be reproduced without going back to the market data.
        point_value: What one point of this instrument was worth, per lot, in
            account currency, at the moment of the decision.
    """

    approved: bool
    size: Decimal
    risk_amount: Decimal
    risk_pct: float
    stop_price: Price
    reasons: tuple[str, ...]
    rejection: RiskReason | None
    fx_rate: Decimal
    point_value: Decimal

    def __post_init__(self) -> None:
        """Enforce that approval and refusal are each internally consistent.

        Raises:
            ValueError: If an approval carries a rejection reason or a
                non-positive size, if a refusal carries a size or a risk, if a
                refusal names no reason, or if ``rejection`` is not a refusal
                reason. These are construction errors in the engine, not user
                input, so they raise rather than producing a decision that says
                two things at once.
        """
        if self.approved:
            if self.rejection is not None:
                raise ValueError(f"an approved decision cannot be rejected for {self.rejection}")
            if self.size <= 0:
                raise ValueError(f"an approved decision must have a positive size, got {self.size}")
            if self.risk_amount <= 0:
                raise ValueError(
                    f"an approved decision must risk something, got {self.risk_amount}"
                )
        else:
            if self.rejection is None:
                raise ValueError("a refused decision must name the reason it was refused")
            if self.rejection not in REJECTION_REASONS:
                raise ValueError(f"{self.rejection} is an explanation, not a refusal reason")
            if self.size != 0 or self.risk_amount != 0:
                raise ValueError(
                    f"a refused decision must risk nothing, got size {self.size} "
                    f"risking {self.risk_amount}"
                )
        if not math.isfinite(self.stop_price):
            raise ValueError(f"stop_price must be finite, got {self.stop_price!r}")
        if self.fx_rate <= 0:
            raise ValueError(f"fx_rate must be positive, got {self.fx_rate}")

    @classmethod
    def refused(
        cls,
        reason: RiskReason,
        detail: str,
        *,
        stop_price: Price = NO_STOP,
        reasons: tuple[str, ...] = (),
    ) -> "RiskDecision":
        """Build a refusal, with everything monetary at zero.

        Args:
            reason: Why the trade was refused.
            detail: The specific numbers behind it.
            stop_price: The stop that had been computed, if the refusal happened
                after that point. Zero when it happened before.
            reasons: Lines accumulated before the refusal, kept so an approval
                and a refusal read the same way up to the point they diverge.

        Returns:
            The refusal.
        """
        return cls(
            approved=False,
            size=Decimal(0),
            risk_amount=Decimal(0),
            risk_pct=0.0,
            stop_price=stop_price,
            reasons=(*reasons, reason_line(reason, detail)),
            rejection=reason,
            fx_rate=Decimal(1),
            point_value=Decimal(0),
        )
