"""The last veto: does the firm's rulebook permit this order, at this size, now.

Sits after the Risk Engine and before execution, which is the position
CLAUDE.md's architecture line already draws — ``Risk Engine (sizing) → Prop
Guard (hard limits) → Execution``. After sizing rather than before, because a
:attr:`GuardDecision.REDUCE` verdict has to name a size, and no size exists
until the Risk Engine has computed one.

**Drawdown is measured on equity, floating included, and that costs nothing
extra.** P13's phase order is ``on_mark`` → ``on_snapshot`` → ``on_sizing``,
and :attr:`~trading_system.backtest.portfolio.Portfolio.equity` is balance plus
unrealised. So the account snapshot this guard receives already carries the
open positions' floating loss, along with every commission and swap booked so
far. A guard that recomputed any of it would be building a second answer to a
question the portfolio has already answered.

**The buffer is the point, not a safety margin bolted on.** Trading right up to
the limit means the last permitted trade is the one that breaches it: slippage,
a gap, or the spread widening on the fill are all one-sided against a position
opened with nothing to spare. So the usable allowance is
:data:`DEFAULT_BUFFER` of the distance to the floor — with 1.2% of equity left
before the daily limit, an order risking 1% is refused, because 0.96% is what
is actually available.

**No state, recomputed every call.** The same discipline
:mod:`trading_system.risk.circuit_breakers` states for its period breakers: a
"blocked" flag has to be cleared at the right instant, and a clear that is
forgotten silently blocks the rest of the run while looking exactly like a
strategy that stopped finding setups. Everything here is derived from the
account snapshot handed in, so the reset is a consequence of the day label
changing rather than an event anyone has to remember.
"""

from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal
from enum import StrEnum

from trading_system.core.instruments import InstrumentSpec
from trading_system.core.logging import get_logger
from trading_system.core.types import ensure_utc
from trading_system.data.resample import trading_day
from trading_system.prop.rules import DailyLossBasis, PropRules, TotalLossBasis

logger = get_logger(__name__)

#: Share of the remaining allowance an order may actually use.
#:
#: Eighty per cent, so that the last order the guard permits is not the one
#: that breaches the limit. Everything that moves a fill away from where it was
#: priced — slippage, a weekend gap, the spread widening at the moment of
#: execution — is one-sided against a position opened with the allowance
#: exactly consumed, and the daily limit is the one rule that ends an account
#: rather than costing it money.
DEFAULT_BUFFER = 0.80


class GuardDecision(StrEnum):
    """What the guard says about one proposed order."""

    #: The order may go through at the size it was sized at.
    ALLOW = "allow"

    #: The order may not go through at any size.
    REJECT = "reject"

    #: The order may go through, smaller. :attr:`GuardVerdict.allowed_size`
    #: names the size, already quantised to the instrument's lot step.
    REDUCE = "reduce"


class PropReason(StrEnum):
    """Which rule produced a verdict, machine-readably.

    Separate from :class:`~trading_system.risk.models.RiskReason` on purpose:
    those are the Risk Engine's own refusals, counted in its own tallies, and
    folding a firm rule into that enum would make "the sizing refused it" and
    "the firm's rulebook refused it" indistinguishable in aggregate.
    """

    DAILY_LOSS_LIMIT = "prop_daily_loss_limit"
    TOTAL_LOSS_LIMIT = "prop_total_loss_limit"
    RISK_EXCEEDS_REMAINING = "prop_risk_exceeds_remaining"
    REDUCED_TO_REMAINING = "prop_reduced_to_remaining"
    BELOW_MIN_LOT_AFTER_REDUCTION = "prop_below_min_lot_after_reduction"
    WITHIN_LIMITS = "prop_within_limits"


@dataclass(frozen=True)
class ProposedOrder:
    """One sized, Risk-Engine-approved order, as the guard sees it.

    Attributes:
        symbol: Instrument.
        size: Size in lots, already quantised.
        risk_amount: Money at stake if the stop is hit, account currency.
        instrument: Contract specification, for re-quantising a reduced size.
    """

    symbol: str
    size: Decimal
    risk_amount: Decimal
    instrument: InstrumentSpec


@dataclass(frozen=True)
class PropAccountState:
    """The account as the firm's rules measure it.

    Distinct from :class:`~trading_system.risk.models.AccountState`, which is
    what the Risk Engine sizes against: that one carries open risk composition
    and margin, this one carries the firm's own reference points. Both are
    built by the orchestrator from the same portfolio at the same instant.

    Attributes:
        at: The instant, tz-aware. What places the account in a firm-day.
        equity: Balance plus floating, account currency. What every drawdown
            limit here is measured against.
        balance: Closed-trade equity.
        day_start_balance: Balance as it stood at the firm-day's reset.
        day_start_equity: Equity as it stood at the firm-day's reset.
        high_water_mark: Highest equity the account has reached, for a trailing
            floor. Equal to the starting balance until the account first
            profits.
        day: The firm-day label this instant falls in, computed by the caller
            with :attr:`PropRules.day_origin` — passed in rather than derived
            here so that the guard and whatever produced ``day_start_*`` cannot
            disagree about which day they are describing.
    """

    at: datetime
    equity: Decimal
    balance: Decimal
    day_start_balance: Decimal
    day_start_equity: Decimal
    high_water_mark: Decimal
    day: date

    def __post_init__(self) -> None:
        """Normalise the instant.

        Raises:
            ValueError: If ``at`` is naive.
        """
        object.__setattr__(self, "at", ensure_utc(self.at))


@dataclass(frozen=True)
class GuardVerdict:
    """What the guard decided, and the numbers behind it.

    Attributes:
        decision: Allow, reject or reduce.
        reason: Which rule decided it.
        detail: The specific numbers, in plain language.
        allowed_size: The size to trade. Equal to the proposed size on
            :attr:`GuardDecision.ALLOW`, smaller on
            :attr:`GuardDecision.REDUCE`, zero on
            :attr:`GuardDecision.REJECT`.
        max_risk_now: What :meth:`PropGuard.max_allowed_risk_now` returned for
            this account. Carried on every verdict, approval included: "how
            much room was left when this trade was allowed" is as much a
            question as "why was it refused".
    """

    decision: GuardDecision
    reason: PropReason
    detail: str
    allowed_size: Decimal
    max_risk_now: Decimal

    @property
    def blocked(self) -> bool:
        """Whether no order goes through at all."""
        return self.decision is GuardDecision.REJECT


class PropGuard:
    """Applies one firm's rules to one order at one instant."""

    __slots__ = ("_buffer", "_counts", "_rules")

    def __init__(self, rules: PropRules, *, buffer: float = DEFAULT_BUFFER) -> None:
        """Configure the guard.

        Args:
            rules: The firm's plan.
            buffer: Share of the remaining allowance an order may use. See
                :data:`DEFAULT_BUFFER`.

        Raises:
            ValueError: If ``buffer`` is outside ``(0, 1]``. Zero would refuse
                every order and above one would spend allowance that does not
                exist.
        """
        if not 0 < buffer <= 1:
            raise ValueError(f"buffer must be in (0, 1], got {buffer}")
        self._rules = rules
        self._buffer = buffer
        self._counts: dict[PropReason, int] = dict.fromkeys(PropReason, 0)

    def __repr__(self) -> str:
        """Compact description naming the plan and the buffer."""
        return f"PropGuard({self._rules.name}, buffer={self._buffer:.0%})"

    @property
    def rules(self) -> PropRules:
        """The firm's plan."""
        return self._rules

    @property
    def counts(self) -> dict[PropReason, int]:
        """How often each reason fired. Every reason present, including zeros."""
        return dict(self._counts)

    def reset(self) -> None:
        """Zero the counters, for a new run or walk-forward fold."""
        self._counts = dict.fromkeys(PropReason, 0)

    def trading_day_of(self, at: datetime) -> date:
        """The firm-day an instant falls in.

        The one place this module turns an instant into a firm-day, and it
        delegates to :func:`~trading_system.data.resample.trading_day` — the
        same function a session VWAP resets on and the circuit breakers measure
        against, called with the firm's own origin rather than the run's. One
        definition of "what day is it", used with two different boundaries.

        Args:
            at: A tz-aware instant.

        Returns:
            The firm-day label.
        """
        return trading_day(ensure_utc(at), self._rules.day_origin)

    def daily_floor(self, account: PropAccountState) -> Decimal:
        """Equity at which the firm's daily limit is breached.

        Args:
            account: The account snapshot.

        Returns:
            The floor. Below it, trading stops until the next reset.
        """
        basis = (
            account.day_start_balance
            if self._rules.daily_loss_basis is DailyLossBasis.BALANCE_AT_DAY_START
            else account.day_start_equity
        )
        return basis * (1 - Decimal(str(self._rules.max_daily_loss_pct)))

    def total_floor(self, account: PropAccountState) -> Decimal:
        """Equity at which the account is gone.

        Args:
            account: The account snapshot.

        Returns:
            The floor — fixed below the starting balance, or trailing the
            high-water mark, per :attr:`PropRules.total_loss_basis`.
        """
        basis = (
            self._rules.account_size
            if self._rules.total_loss_basis is TotalLossBasis.STATIC
            else account.high_water_mark
        )
        return basis * (1 - Decimal(str(self._rules.max_total_loss_pct)))

    def max_allowed_risk_now(self, account: PropAccountState) -> Decimal:
        """The largest risk one new order may carry, right now.

        The distance from current equity to the nearer of the two floors,
        multiplied by the buffer. Never negative: an account already through a
        floor has no allowance rather than a negative one, and the caller
        distinguishes the two by asking :meth:`check`, which refuses outright.

        Args:
            account: The account snapshot.

        Returns:
            The ceiling, in account currency.
        """
        headroom = min(
            account.equity - self.daily_floor(account),
            account.equity - self.total_floor(account),
        )
        if headroom <= 0:
            return Decimal(0)
        return headroom * Decimal(str(self._buffer))

    def check(self, order: ProposedOrder, account: PropAccountState) -> GuardVerdict:
        """Whether this order may go through, and at what size.

        Args:
            order: The sized, Risk-Engine-approved order.
            account: The account as the firm's rules measure it.

        Returns:
            The verdict. Reduction is quantised down to the instrument's lot
            step, and falls back to a refusal when what is left is below
            ``min_lot`` — rounding up would spend allowance the rule says is
            not there, which is the one thing the buffer exists to prevent.
        """
        allowance = self.max_allowed_risk_now(account)

        daily_floor = self.daily_floor(account)
        if account.equity <= daily_floor:
            return self._verdict(
                GuardDecision.REJECT,
                PropReason.DAILY_LOSS_LIMIT,
                f"equity {account.equity} is at or below the firm's daily floor {daily_floor} "
                f"({self._rules.max_daily_loss_pct:.2%} of "
                f"{self._rules.daily_loss_basis.value}); no new position until the day resets "
                f"at {self._rules.daily_reset_time} {self._rules.daily_reset_tz}",
                Decimal(0),
                allowance,
            )

        total_floor = self.total_floor(account)
        if account.equity <= total_floor:
            return self._verdict(
                GuardDecision.REJECT,
                PropReason.TOTAL_LOSS_LIMIT,
                f"equity {account.equity} is at or below the account floor {total_floor} "
                f"({self._rules.max_total_loss_pct:.2%} below "
                f"{self._rules.total_loss_basis.value}); the account is gone",
                Decimal(0),
                allowance,
            )

        if order.risk_amount <= allowance:
            headroom = min(account.equity - daily_floor, account.equity - total_floor)
            return self._verdict(
                GuardDecision.ALLOW,
                PropReason.WITHIN_LIMITS,
                f"{order.symbol}: risking {order.risk_amount} against {allowance} available "
                f"({self._buffer:.0%} of the {headroom} left before the nearer floor)",
                order.size,
                allowance,
            )

        if allowance <= 0:
            return self._verdict(
                GuardDecision.REJECT,
                PropReason.RISK_EXCEEDS_REMAINING,
                f"{order.symbol}: risking {order.risk_amount} with no allowance left at all",
                Decimal(0),
                allowance,
            )

        # Risk scales linearly with size — the stop distance is fixed by the
        # time the guard sees the order — so the fraction of the allowance is
        # the fraction of the size.
        scaled = order.size * allowance / order.risk_amount
        reduced = order.instrument.round_volume(scaled)
        if reduced < order.instrument.min_lot:
            return self._verdict(
                GuardDecision.REJECT,
                PropReason.BELOW_MIN_LOT_AFTER_REDUCTION,
                f"{order.symbol}: {order.risk_amount} exceeds the {allowance} available, and "
                f"scaling {order.size} lots to fit gives {scaled} lots, which rounds to "
                f"{reduced}, below min_lot {order.instrument.min_lot}",
                Decimal(0),
                allowance,
            )
        return self._verdict(
            GuardDecision.REDUCE,
            PropReason.REDUCED_TO_REMAINING,
            f"{order.symbol}: {order.size} lots risking {order.risk_amount} cut to {reduced} "
            f"lots to fit the {allowance} still available before the firm's nearer floor",
            reduced,
            allowance,
        )

    def _verdict(
        self,
        decision: GuardDecision,
        reason: PropReason,
        detail: str,
        size: Decimal,
        allowance: Decimal,
    ) -> GuardVerdict:
        """Build a verdict and count it.

        Counting here rather than at each call site so that adding a path
        cannot silently add an uncounted one — the same reason
        :meth:`~trading_system.risk.engine.RiskEngine._refuse` exists.
        """
        self._counts[reason] += 1
        if decision is not GuardDecision.ALLOW:
            logger.debug("prop.guard", decision=decision.value, reason=reason.value, detail=detail)
        return GuardVerdict(
            decision=decision,
            reason=reason,
            detail=detail,
            allowed_size=size,
            max_risk_now=allowance,
        )
