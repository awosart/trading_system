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
from dataclasses import dataclass, replace
from datetime import datetime
from decimal import Decimal
from enum import StrEnum

from trading_system.core.types import Price, Side, ensure_utc

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

    # --- refusals from portfolio limits (stage 2) -------------------------
    PORTFOLIO_HEAT_EXCEEDED = "portfolio_heat_exceeded"
    INSTRUMENT_LIMIT_EXCEEDED = "instrument_limit_exceeded"
    CLUSTER_LIMIT_EXCEEDED = "cluster_limit_exceeded"
    STRATEGY_LIMIT_EXCEEDED = "strategy_limit_exceeded"
    DIRECTION_LIMIT_EXCEEDED = "direction_limit_exceeded"

    # --- refusals from circuit breakers (stage 2) -------------------------
    DAILY_LOSS_LIMIT = "daily_loss_limit"
    WEEKLY_LOSS_LIMIT = "weekly_loss_limit"
    MONTHLY_LOSS_LIMIT = "monthly_loss_limit"
    CONSECUTIVE_LOSS_PAUSE = "consecutive_loss_pause"
    SLIPPAGE_ANOMALY_PAUSE = "slippage_anomaly_pause"

    # --- explanations ---------------------------------------------------
    SIZED = "sized"
    STOP_FROM_INVALIDATION = "stop_from_invalidation"
    STOP_WIDENED_TO_BROKER_MINIMUM = "stop_widened_to_broker_minimum"
    RISK_CAPPED = "risk_capped"
    SIZE_ROUNDED_DOWN = "size_rounded_down"

    #: Risk was cut to fit the headroom left under a portfolio limit. An
    #: explanation, not a refusal: the trade still happens, smaller.
    TRIMMED_TO_LIMIT = "trimmed_to_limit"

    #: Clustering fell back to the configured manual groups because the
    #: correlation window had too little overlapping history. Recorded rather
    #: than counted as a refusal — the signal was sized, on a prior instead of
    #: a measurement, and conflating that with "no trade" would hide both.
    CORRELATION_UNAVAILABLE = "correlation_unavailable"


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
        RiskReason.PORTFOLIO_HEAT_EXCEEDED,
        RiskReason.INSTRUMENT_LIMIT_EXCEEDED,
        RiskReason.CLUSTER_LIMIT_EXCEEDED,
        RiskReason.STRATEGY_LIMIT_EXCEEDED,
        RiskReason.DIRECTION_LIMIT_EXCEEDED,
        RiskReason.DAILY_LOSS_LIMIT,
        RiskReason.WEEKLY_LOSS_LIMIT,
        RiskReason.MONTHLY_LOSS_LIMIT,
        RiskReason.CONSECUTIVE_LOSS_PAUSE,
        RiskReason.SLIPPAGE_ANOMALY_PAUSE,
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
class OpenRisk:
    """What one already-open position has at stake.

    The portfolio limits need composition, not just a total: "0.5% of equity is
    already at risk" cannot answer whether another EURUSD position breaches the
    per-instrument limit, or whether three USD-shorts have become one oversized
    bet. Each open position contributes one of these.

    Attributes:
        symbol: Instrument held.
        strategy_id: Strategy that opened it, for the per-strategy limit.
        side: Direction of the exposure, for the per-direction limit.
        risk_amount: Money that would be lost if this position's stop is hit, in
            account currency. Positive. Recomputed by the caller when a stop is
            tightened or a partial is closed — a trailing stop that has moved to
            breakeven puts nothing at risk any more, and holding the value fixed
            at entry would keep charging the portfolio for a risk that no longer
            exists.
    """

    symbol: str
    strategy_id: str
    side: Side
    risk_amount: Decimal

    def __post_init__(self) -> None:
        """Reject a negative stake.

        Raises:
            ValueError: If ``risk_amount`` is negative. Zero is allowed: a
                position whose stop has moved past breakeven genuinely risks
                nothing.
        """
        if self.risk_amount < 0:
            raise ValueError(
                f"{self.symbol}: risk_amount must not be negative, got {self.risk_amount}"
            )


@dataclass(frozen=True)
class AccountState:
    """The account as it stood when a signal arrived.

    Passed in explicitly rather than queried. There is no portfolio module yet,
    and when there is one it will hand this over rather than be reached into.

    Attributes:
        currency: Account denomination. Every money figure the engine returns is
            in this currency.
        balance: Closed-trade equity. Recorded for reporting and for the weekly
            and monthly circuit breakers; **not** what size is computed from.
        equity: Balance plus the floating result of open positions. This is the
            sizing base — see the module docstring.
        as_of: When this snapshot was taken, tz-aware. Also the instant FX rates
            are priced at.
        open_risks: One entry per open position. The single source of truth for
            what the portfolio has at stake: :attr:`open_risk_amount` is derived
            from it rather than stored alongside it, because two fields holding
            the same quantity drift the moment one of them is updated and the
            other is not.
    """

    currency: str
    balance: Decimal
    equity: Decimal
    as_of: datetime
    open_risks: tuple[OpenRisk, ...] = ()

    def __post_init__(self) -> None:
        """Normalise the timestamp and reject impossible figures.

        Raises:
            ValueError: If the currency is empty or ``as_of`` is naive. A
                non-positive ``equity`` is *not* rejected here — a blown account
                is a real state, and the engine refuses to size against it rather
                than being unable to describe it.
        """
        object.__setattr__(self, "as_of", ensure_utc(self.as_of))
        object.__setattr__(self, "open_risks", tuple(self.open_risks))
        if not self.currency:
            raise ValueError("account currency must be a non-empty code")

    @property
    def open_risk_amount(self) -> Decimal:
        """Total money at risk across open positions, in account currency."""
        return sum((position.risk_amount for position in self.open_risks), Decimal(0))

    def with_opened(
        self, symbol: str, strategy_id: str, side: Side, risk: Decimal
    ) -> "AccountState":
        """This state, plus a position that has just been opened.

        The engine has no memory of what it approved, so two signals evaluated
        against the *same* snapshot can jointly breach a portfolio limit that
        neither breaches alone. The engine cannot close that on its own — it
        would need a reservation ledger with a confirm/discard protocol, and a
        forgotten discard is phantom heat that never expires. Instead the
        orchestrator feeds the approval back, and this makes that a one-liner it
        cannot compute incorrectly.

        Args:
            symbol: Instrument opened.
            strategy_id: Strategy that opened it.
            side: Direction of the exposure.
            risk: Money at stake, from :attr:`RiskDecision.risk_amount`.

        Returns:
            A new state including the position. Equity is unchanged: opening a
            position books no profit or loss.
        """
        return replace(
            self,
            open_risks=(
                *self.open_risks,
                OpenRisk(symbol=symbol, strategy_id=strategy_id, side=side, risk_amount=risk),
            ),
        )


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
