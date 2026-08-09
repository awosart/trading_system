"""The Risk Engine: a signal in, a size or a refusal out.

The engine knows nothing about indicators. Its inputs are an
:class:`~trading_system.entry.signal.EntrySignal`, the account, and instrument
metadata; whichever strategy produced the signal and whichever pattern it fired
on are invisible here by design.

**Order of operations, and why it is this one.**

0. Refusals that need no arithmetic — unknown instrument, dead account — so
   nothing below is computed against nonsense.
1. **Circuit breakers.** Whether trading is permitted at all. First, because
   nothing else matters if it is not, and because a size computed past a
   tripped breaker is a number somebody will eventually read.
2. **Portfolio limits.** Which bucket — portfolio, instrument, cluster,
   strategy, direction — has the least headroom. No headroom anywhere means a
   refusal here, before a stop is even placed.
3. **The stop.** Size depends on the stop; the stop never depends on size.
   :mod:`trading_system.risk.stop_calculator` guarantees that direction.
4. **Sizing.** The method proposes an amount of money to risk, or refuses.
5. **The cap and the correlation adjustment.** The proposal is capped at
   ``max_risk_pct`` of equity, then trimmed to the tightest limit's headroom.
   Both act on one number on one line, so no sizing method can escape either.
6. **Lot rounding**, always down, then the instrument's bounds.
7. **Margin, then the firm's leverage cap**, then the exit plan's smallest
   close. Here rather than earlier because both are computed from the size that
   will actually be traded, and see :mod:`trading_system.risk.margin` for why
   they are two refusals rather than one.
8. ``risk_amount`` is recomputed from the quantised size. What the decision
   reports is what the account actually stands to lose, not what was asked for.

Step 8 is what makes both ceilings exact rather than approximate. Quantisation
only rounds down, so recomputing after it can only lower the figure — the
reported risk is at or below the cap *and* at or below the headroom for every
input, which is what the two property tests claim.

**The portfolio guarantee is per decision, not across concurrent ones.** The
engine has no memory of what it approved, so two signals evaluated against the
same snapshot can jointly breach a limit neither breaches alone. Closing that
inside the engine would take a reservation ledger with a confirm/discard
protocol, and a forgotten discard is phantom heat that never expires. Instead
the orchestrator feeds each approval back through
:meth:`~trading_system.risk.models.AccountState.with_opened`, which makes the
update a one-liner it cannot compute incorrectly.

**The exit ladder check lives here**, because this is the only place that holds
both a size in lots and the instrument's ``min_lot``. The Exit Engine deliberately
speaks in fractions of a position and cannot know whether "close 40%" is
executable; it exposes
:meth:`~trading_system.exit.plan.ExitPlan.smallest_closing_fraction` and stops
there. ``smallest_exit_fraction`` is a required keyword argument with no default:
a default would turn "forgot to pass it" into "check silently skipped", which is
the same class of defect that made ``source`` mandatory on
:meth:`~trading_system.exit.position.ManagedPosition.tighten_stop`. It is a plain
``Decimal`` rather than an ``ExitPlan`` so that this module does not import the
Exit Engine — the mirror of the Exit Engine not importing this one.
"""

from collections.abc import Mapping, Sequence
from decimal import ROUND_DOWN, Decimal, localcontext
from types import MappingProxyType

from pydantic import BaseModel, ConfigDict, Field

from trading_system.core.instruments import InstrumentRegistry
from trading_system.core.logging import get_logger
from trading_system.core.types import Price
from trading_system.entry.signal import EntrySignal
from trading_system.risk.circuit_breakers import CircuitBreakers, ClosedTrade
from trading_system.risk.conversion import FxConverter, FxRateUnavailableError
from trading_system.risk.correlation import CorrelationProvider
from trading_system.risk.margin import (
    PropProfile,
    margin_rate_for,
    notional_per_lot,
    quantise_up,
)
from trading_system.risk.models import (
    NO_STOP,
    AccountState,
    RiskDecision,
    RiskReason,
    empty_rejection_counts,
    reason_line,
)
from trading_system.risk.portfolio_risk import PortfolioRisk, binding_limit
from trading_system.risk.sizing.base import SizingMethod, SizingRequest
from trading_system.risk.stop_calculator import (
    StopBufferConfig,
    calculate_stop,
    requires_atr,
)
from trading_system.strategies.schema import StopReference

logger = get_logger(__name__)

#: Placeholder threshold used when no correlation provider is configured. With
#: no matrix, clustering is by manual groups alone and no comparison happens.
DEFAULT_CORRELATION_THRESHOLD = 1.0

#: Step the reported ``risk_amount`` is quantised to, always downwards.
#:
#: A hundredth of the account currency: the smallest amount the account can
#: actually move by, so the figure is money rather than an artefact of decimal
#: division. It is **not** an epsilon and not a tolerance — the comparisons
#: against the cap and the headroom stay exact ``<=`` — it is the observation
#: that a risk figure carrying twenty-five significant digits is claiming a
#: precision no account has. Currencies with no minor unit (JPY) are unharmed:
#: two extra decimal places on a figure that never uses them cost nothing, while
#: a per-currency table of minor units would be a second source of truth about
#: money for the sake of one rounding step.
#:
#: Downwards for the reason every quantisation in this layer is downwards: the
#: reported risk must be at or below what was asked for, never above, because a
#: figure above the cap is the one thing the cap exists to prevent. Rounding to
#: nearest here would let a decision report a risk a hundredth above its
#: ceiling, which is a smaller error than the one it replaced and the same kind.
RISK_AMOUNT_STEP = Decimal("0.01")


class RiskEngineConfig(BaseModel):
    """Settings that hold across every strategy the engine sizes for.

    Attributes:
        max_risk_pct: Hard ceiling on the fraction of equity one trade may risk,
            applied after the sizing method has had its say. Not the sizing
            method's own figure — that is per-method configuration — but the
            bound no method may exceed, whatever it was configured with.
        stop_buffer: How far past the invalidation level stops are pushed.
        prop_profile: The firm's account rules, or ``None`` for a run trading
            under the venue leverage already declared on each instrument. The
            margin check itself is **not** conditional on this — every
            instrument carries a mandatory ``margin_rate`` and every broker
            enforces it — so a profile only ever changes the rate used and adds
            the optional cap on total notional.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    max_risk_pct: float = Field(default=0.02, gt=0, le=1)
    stop_buffer: StopBufferConfig = Field(default_factory=StopBufferConfig)
    prop_profile: PropProfile | None = None


class RiskEngine:
    """Turns scored signals into sizes, and counts everything it refuses."""

    __slots__ = (
        "_breakers",
        "_config",
        "_converter",
        "_correlations",
        "_degradations",
        "_instruments",
        "_portfolio",
        "_rejections",
        "_sizing",
    )

    def __init__(
        self,
        *,
        instruments: InstrumentRegistry,
        sizing: SizingMethod,
        converter: FxConverter,
        config: RiskEngineConfig | None = None,
        portfolio: PortfolioRisk | None = None,
        breakers: CircuitBreakers | None = None,
        correlations: CorrelationProvider | None = None,
    ) -> None:
        """Wire the engine to its dependencies.

        Args:
            instruments: Contract specifications, keyed by symbol.
            sizing: How much money to put at stake per trade.
            converter: Where FX rates come from. Explicit and required: a default
                would have to be a converter that answers something, and every
                answer it could invent is a wrong position size.
            config: Engine-wide settings. Defaults are conservative.
            portfolio: Concurrent-risk limits. Defaults to
                :class:`~trading_system.risk.portfolio_risk.PortfolioRisk`'s own
                conservative ceilings with no manual groups.
            breakers: When to stop trading altogether. Defaults to the three
                period loss limits with no bar-counted pauses.
            correlations: Measured co-movement, or ``None`` to cluster by the
                configured manual groups alone. ``None`` is a *choice* — the run
                is stating it has no return history to measure — and is distinct
                from a provider that has too little history for a given pair,
                which is recorded as a degradation.
        """
        self._instruments = instruments
        self._sizing = sizing
        self._converter = converter
        self._config = config if config is not None else RiskEngineConfig()
        self._portfolio = portfolio if portfolio is not None else PortfolioRisk()
        self._breakers = breakers if breakers is not None else CircuitBreakers()
        self._correlations = correlations
        self._rejections = empty_rejection_counts()
        self._degradations: dict[RiskReason, int] = {RiskReason.CORRELATION_UNAVAILABLE: 0}

    @property
    def _correlation_threshold(self) -> float:
        """Absolute correlation at which two instruments become one cluster."""
        return (
            self._correlations.config.threshold
            if self._correlations is not None
            else DEFAULT_CORRELATION_THRESHOLD
        )

    @property
    def breakers(self) -> CircuitBreakers:
        """The circuit breakers, for the execution layer to report fills to."""
        return self._breakers

    @property
    def degradations(self) -> Mapping[RiskReason, int]:
        """How often a decision was made on a weaker basis than configured.

        Separate from :attr:`rejections` because these signals were *traded*.
        Folding "sized on the manual prior because the correlation window was
        short" into the refusal counts would hide both: the refusals would look
        inflated and the degradations would look like no-trades.
        """
        return MappingProxyType(dict(self._degradations))

    def __repr__(self) -> str:
        """Compact description naming the sizing method and the cap."""
        return (
            f"RiskEngine({self._sizing.name}, max_risk_pct={self._config.max_risk_pct}, "
            f"{len(self._instruments)} instruments)"
        )

    @property
    def rejections(self) -> Mapping[RiskReason, int]:
        """How many signals were refused, by reason, since the last reset.

        Every refusal reason is present, including those that never fired. A
        backtest that produced few trades is otherwise indistinguishable from a
        selective strategy, and "no signal was ever refused for an unexecutable
        ladder" is a result worth being able to state.
        """
        return MappingProxyType(dict(self._rejections))

    @property
    def refused(self) -> int:
        """Total signals refused since the last reset."""
        return sum(self._rejections.values())

    def reset(self) -> None:
        """Return every counter and pause to its pre-run state.

        Covers the breakers' bar-counted pauses too: a pause left armed from the
        previous walk-forward fold would silently suppress the opening bars of
        the next one.
        """
        self._rejections = empty_rejection_counts()
        self._degradations = {RiskReason.CORRELATION_UNAVAILABLE: 0}
        self._breakers.reset()

    def evaluate(
        self,
        signal: EntrySignal,
        *,
        account: AccountState,
        stop_reference: StopReference,
        smallest_exit_fraction: Decimal,
        bar_index: int,
        trades: Sequence[ClosedTrade],
        atr_price: float | None = None,
    ) -> RiskDecision:
        """Size one signal, or refuse it and say why.

        Args:
            signal: The scored, unsized entry opportunity.
            account: Account snapshot. Sizing measures against ``equity``.
            stop_reference: How the originating strategy derives its stop
                distance, from its ``risk_profile``.
            smallest_exit_fraction: The smallest fraction of the position its
                paired exit plan can ever close in one go, from
                :meth:`~trading_system.exit.plan.ExitPlan.smallest_closing_fraction`.
                Required, with no default — see the module docstring.
            bar_index: Index of the bar being evaluated, for the bar-counted
                pauses. Required rather than defaulted for the same reason
                ``smallest_exit_fraction`` is: a default would turn "forgot to
                pass it" into "the pause never expires" or "never starts".
            trades: Every trade realised so far, for the period loss limits and
                the losing-streak pause. Required and with no default: an empty
                tuple is a legitimate state at the start of a run, but omission
                would silently disable every breaker that was configured.
            atr_price: ATR in price units on the signal bar, when the run
                computes one. Required by some stop and sizing configurations,
                which refuse rather than substitute zero when it is absent.

        Returns:
            An approved size, or a refusal naming its reason.

        Raises:
            ValueError: If ``smallest_exit_fraction`` is outside ``(0, 1]``. That
                is a wiring error in the caller, not a market condition, so it
                raises instead of being counted as a refused signal.
        """
        if not 0 < smallest_exit_fraction <= 1:
            raise ValueError(
                f"smallest_exit_fraction must be a fraction in (0, 1], got {smallest_exit_fraction}"
            )

        instrument = self._instruments.get(signal.symbol)
        if instrument is None:
            return self._refuse(
                RiskReason.UNKNOWN_INSTRUMENT,
                f"{signal.symbol} is not in the instrument registry "
                f"({', '.join(self._instruments.symbols)})",
            )
        if account.equity <= 0:
            return self._refuse(
                RiskReason.NON_POSITIVE_EQUITY,
                f"equity is {account.equity}; there is nothing left to size against",
            )

        # --- 1. Circuit breakers, before anything is computed ----------------
        # Nothing below matters if trading is stopped, and a breaker that only
        # ran after sizing would burn the work and, worse, invite someone to
        # read the size it produced.
        tripped = self._breakers.check(
            at=signal.bar_close_ts, bar_index=bar_index, equity=account.equity, trades=trades
        )
        if tripped is not None:
            return self._refuse(tripped.reason, tripped.detail)

        # --- 2. Portfolio limits ---------------------------------------------
        matrix = (
            self._correlations.matrix(as_of=signal.bar_close_ts)
            if self._correlations is not None
            else None
        )
        reasons: list[str] = []
        if matrix is not None:
            unmeasured = [
                position.symbol
                for position in account.open_risks
                if position.symbol != signal.symbol
                and matrix.get(signal.symbol, position.symbol) is None
            ]
            if unmeasured:
                # Not a refusal: the signal is still sized, on the manual prior
                # instead of a measurement. Counted separately so that neither
                # fact hides the other.
                self._degradations[RiskReason.CORRELATION_UNAVAILABLE] += 1
                reasons.append(
                    reason_line(
                        RiskReason.CORRELATION_UNAVAILABLE,
                        f"{signal.symbol} has too little overlapping history against "
                        f"{sorted(set(unmeasured))}; clustered by the configured groups alone",
                    )
                )

        checks = self._portfolio.checks(
            account=account,
            symbol=signal.symbol,
            strategy_id=signal.strategy_id,
            side=signal.side,
            matrix=matrix,
            threshold=self._correlation_threshold,
        )
        binding = binding_limit(checks)
        if binding.headroom <= 0:
            return self._refuse(
                binding.reason,
                f"{binding.bucket} already carries {binding.used} against a ceiling of "
                f"{binding.ceiling}; no headroom for another position",
                reasons=tuple(reasons),
            )

        if requires_atr(stop_reference, self._config.stop_buffer) and atr_price is None:
            return self._refuse(
                RiskReason.ATR_UNAVAILABLE,
                f"{signal.symbol}: the stop configuration needs an ATR and none was computed "
                "for this bar",
                reasons=tuple(reasons),
            )

        # --- 3. The stop, which the size then adapts to ----------------------
        stop = calculate_stop(
            side=signal.side,
            reference_price=signal.reference_price,
            invalidation_price=signal.invalidation_price,
            instrument=instrument,
            stop_reference=stop_reference,
            buffer=self._config.stop_buffer,
            atr_price=atr_price,
        )
        reasons.extend(
            reason_line(RiskReason.STOP_FROM_INVALIDATION, line) for line in stop.reasons
        )

        # --- 4. Sizing -------------------------------------------------------
        outcome = self._sizing.size(
            SizingRequest(
                equity=account.equity,
                quality=signal.quality,
                stop_distance_price=stop.distance_price,
                atr_price=atr_price,
            )
        )
        if outcome.risk_amount is None:
            return self._refuse(
                outcome.reason,
                outcome.detail,
                stop_price=stop.stop_price,
                reasons=tuple(reasons),
            )
        reasons.append(reason_line(outcome.reason, outcome.detail))

        requested = outcome.risk_amount
        ceiling = account.equity * Decimal(str(self._config.max_risk_pct))
        if requested > ceiling:
            reasons.append(
                reason_line(
                    RiskReason.RISK_CAPPED,
                    f"{self._sizing.name} asked for {requested}, capped to {ceiling} "
                    f"({self._config.max_risk_pct:.4%} of equity {account.equity})",
                )
            )
            requested = ceiling

        # --- 5. Correlation adjustment: trim to the tightest limit's headroom -
        # Trimmed rather than refused. Cutting size to fit a risk budget is this
        # layer doing its job; contrast max_lot below, where hitting the venue's
        # ceiling means the sizing has diverged from what the instrument
        # supports and is worth surfacing instead of quietly shrinking.
        if requested > binding.headroom:
            reasons.append(
                reason_line(
                    RiskReason.TRIMMED_TO_LIMIT,
                    f"{requested} trimmed to {binding.headroom}, the headroom left under the "
                    f"{binding.reason.value} limit on {binding.bucket} "
                    f"({binding.used} of {binding.ceiling} used)",
                )
            )
            requested = binding.headroom

        try:
            fx_rate = self._converter.rate(
                base=instrument.quote_currency,
                quote=account.currency,
                at=signal.bar_close_ts,
            )
        except FxRateUnavailableError as error:
            return self._refuse(
                RiskReason.FX_RATE_UNAVAILABLE,
                f"{signal.symbol} is quoted in {instrument.quote_currency} and the account "
                f"is in {account.currency}: {error}",
                stop_price=stop.stop_price,
                reasons=tuple(reasons),
            )

        point_value = instrument.value_per_point_quote * fx_rate
        stop_value_per_lot = stop.distance_points_exact * point_value
        # Divided at the default half-even, deliberately. Rounding this division
        # downwards instead looks like the same discipline the lot quantisation
        # follows, and is not: ``stop_value_per_lot`` is itself an inexact
        # decimal whenever an FX rate is (1/150 is 0.006666…667, rounded *up*),
        # so the exact quotient sits a whisker below the answer the arithmetic
        # is meant to give. Rounding that whisker down costs a whole lot step —
        # GBPJPY at 150.00 sizes to 1.49 lots instead of 1.50 — which is a far
        # larger error than the 1e-24 residue it would be protecting against.
        # That residue is dealt with where it belongs, on the reported money
        # figure, at RISK_AMOUNT_STEP.
        unquantised = requested / stop_value_per_lot
        size = instrument.round_volume(unquantised)

        if size < instrument.min_lot:
            return self._refuse(
                RiskReason.BELOW_MIN_LOT,
                f"{signal.symbol}: risking {requested} against a stop {stop.distance_points:.6g} "
                f"points away sizes to {unquantised} lots, below min_lot {instrument.min_lot}. "
                "Refused rather than rounded up: rounding up would risk more than the cap "
                "allows, which is the one thing the cap exists to prevent",
                stop_price=stop.stop_price,
                reasons=tuple(reasons),
            )
        if size > instrument.max_lot:
            return self._refuse(
                RiskReason.ABOVE_MAX_LOT,
                f"{signal.symbol}: {size} lots exceeds max_lot {instrument.max_lot}; refused "
                "rather than trimmed, since a trimmed order is a different trade from the one "
                "that was sized",
                stop_price=stop.stop_price,
                reasons=tuple(reasons),
            )

        # --- 6.5. Margin, then the firm's ceiling on total exposure ----------
        # After quantisation, because the requirement is computed from the size
        # that will actually be traded: rounding is always downwards, so
        # checking the unquantised quotient would refuse trades that do fit, on
        # a number no run ever trades. Before the ladder check, because a
        # position the broker will not open makes "is the ladder executable
        # inside it" a question about nothing.
        rate = margin_rate_for(instrument, self._config.prop_profile)
        per_lot_notional = notional_per_lot(instrument, signal.reference_price, fx_rate)
        per_lot_margin = per_lot_notional * rate
        margin_amount = quantise_up(per_lot_margin * size)
        notional_amount = quantise_up(per_lot_notional * size)

        # Margin first, and the order decides only which reason a trade refused
        # by both reports. A requirement the account cannot meet is a physical
        # fact about the broker; the cap below is a rule the firm chose. When
        # both bind, the physical fact is the more informative of the two.
        free_margin = account.free_margin
        if margin_amount > free_margin:
            return self._refuse(
                RiskReason.INSUFFICIENT_MARGIN,
                f"{signal.symbol}: {size} lots at {signal.reference_price:.6g} is "
                f"{notional_amount} notional, needing {margin_amount} margin at a rate of "
                f"{rate:.6f}; equity {account.equity} less {account.used_margin} already "
                f"posted leaves {free_margin} free",
                stop_price=stop.stop_price,
                reasons=tuple(reasons),
            )

        cap = (
            self._config.prop_profile.leverage_cap
            if self._config.prop_profile is not None
            else None
        )
        if cap is not None:
            ceiling = account.equity * Decimal(str(cap))
            exposure = account.used_notional + notional_amount
            if exposure > ceiling:
                return self._refuse(
                    RiskReason.LEVERAGE_LIMIT_EXCEEDED,
                    f"{signal.symbol}: {notional_amount} notional on top of "
                    f"{account.used_notional} already open is {exposure}, over the "
                    f"{self._config.prop_profile.name if self._config.prop_profile else ''} "
                    f"ceiling of {cap:g}x equity {account.equity} = {ceiling}. The margin was "
                    f"there ({free_margin} free against {margin_amount} needed); the rule was "
                    "not met",
                    stop_price=stop.stop_price,
                    reasons=tuple(reasons),
                )

        smallest_close = instrument.round_volume(size * smallest_exit_fraction)
        if smallest_close < instrument.min_lot:
            return self._refuse(
                RiskReason.EXIT_LADDER_UNEXECUTABLE,
                f"{signal.symbol}: the exit plan's smallest close is {smallest_exit_fraction} of "
                f"{size} lots = {size * smallest_exit_fraction}, which rounds to "
                f"{smallest_close}, below min_lot {instrument.min_lot}. Refused at open rather "
                "than discovered mid-trade when the rung rounds to nothing",
                stop_price=stop.stop_price,
                reasons=tuple(reasons),
            )

        if size != unquantised:
            reasons.append(
                reason_line(
                    RiskReason.SIZE_ROUNDED_DOWN,
                    f"{unquantised} lots rounded down to {size} on a lot_step of "
                    f"{instrument.lot_step}",
                )
            )

        # Recomputed from the quantised size: what the account actually stands to
        # lose, which after rounding down is at or below what was asked for. Then
        # quantised to the account's minor unit, downwards — see RISK_AMOUNT_STEP.
        with localcontext() as context:
            context.rounding = ROUND_DOWN
            risk_amount = (size * stop_value_per_lot).quantize(RISK_AMOUNT_STEP)
        return RiskDecision(
            approved=True,
            size=size,
            risk_amount=risk_amount,
            risk_pct=float(risk_amount / account.equity),
            stop_price=stop.stop_price,
            reasons=tuple(reasons),
            rejection=None,
            fx_rate=fx_rate,
            point_value=point_value,
            margin_per_lot=per_lot_margin,
            notional_per_lot=per_lot_notional,
        )

    def _refuse(
        self,
        reason: RiskReason,
        detail: str,
        *,
        stop_price: Price = NO_STOP,
        reasons: tuple[str, ...] = (),
    ) -> RiskDecision:
        """Build a refusal and count it.

        Counting happens here rather than at each call site so that adding a
        refusal path cannot silently add an uncounted one.

        Args:
            reason: Why the trade was refused.
            detail: The specific numbers behind it.
            stop_price: The stop that had been computed, if any.
            reasons: Lines accumulated before the refusal.

        Returns:
            The refusal.
        """
        self._rejections[reason] += 1
        logger.debug("risk.refused", reason=reason.value, detail=detail)
        return RiskDecision.refused(reason, detail, stop_price=stop_price, reasons=reasons)
