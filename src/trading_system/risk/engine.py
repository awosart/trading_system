"""The Risk Engine: a signal in, a size or a refusal out.

The engine knows nothing about indicators. Its inputs are an
:class:`~trading_system.entry.signal.EntrySignal`, the account, and instrument
metadata; whichever strategy produced the signal and whichever pattern it fired
on are invisible here by design.

**Order of operations, and why it is this one.**

1. Refusals that need no arithmetic come first — unknown instrument, dead
   account — so nothing further is computed against nonsense.
2. The stop is placed. Size depends on the stop; the stop never depends on size.
   :mod:`trading_system.risk.stop_calculator` guarantees the one-way direction.
3. The sizing method proposes an amount of money to risk, or refuses.
4. The proposal is **capped** at ``max_risk_pct`` of equity. One cap, one line,
   applied to every method including a flat ``FixedAmount`` on a shrunken
   account.
5. Point value is converted into account currency. A missing rate refuses here,
   after the stop is known, so the refusal still reports where the stop would
   have gone.
6. The size is quantised **down** to the lot grid, then checked against the
   instrument's bounds and against the exit plan's smallest close.
7. ``risk_amount`` is recomputed from the quantised size. What the decision
   reports is what the account actually stands to lose, not what was asked for.

Step 7 is what makes the cap exact rather than approximate. Quantisation only
rounds down, so recomputing after it can only lower the figure — the reported
risk is at or below the cap for every input, which is the property test's claim.

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

from collections.abc import Mapping
from decimal import Decimal
from types import MappingProxyType

from pydantic import BaseModel, ConfigDict, Field

from trading_system.core.instruments import InstrumentRegistry
from trading_system.core.logging import get_logger
from trading_system.core.types import Price
from trading_system.entry.signal import EntrySignal
from trading_system.risk.conversion import FxConverter, FxRateUnavailableError
from trading_system.risk.models import (
    NO_STOP,
    AccountState,
    RiskDecision,
    RiskReason,
    empty_rejection_counts,
    reason_line,
)
from trading_system.risk.sizing.base import SizingMethod, SizingRequest
from trading_system.risk.stop_calculator import (
    StopBufferConfig,
    calculate_stop,
    requires_atr,
)
from trading_system.strategies.schema import StopReference

logger = get_logger(__name__)


class RiskEngineConfig(BaseModel):
    """Settings that hold across every strategy the engine sizes for.

    Attributes:
        max_risk_pct: Hard ceiling on the fraction of equity one trade may risk,
            applied after the sizing method has had its say. Not the sizing
            method's own figure — that is per-method configuration — but the
            bound no method may exceed, whatever it was configured with.
        stop_buffer: How far past the invalidation level stops are pushed.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    max_risk_pct: float = Field(default=0.02, gt=0, le=1)
    stop_buffer: StopBufferConfig = Field(default_factory=StopBufferConfig)


class RiskEngine:
    """Turns scored signals into sizes, and counts everything it refuses."""

    __slots__ = ("_config", "_converter", "_instruments", "_rejections", "_sizing")

    def __init__(
        self,
        *,
        instruments: InstrumentRegistry,
        sizing: SizingMethod,
        converter: FxConverter,
        config: RiskEngineConfig | None = None,
    ) -> None:
        """Wire the engine to its dependencies.

        Args:
            instruments: Contract specifications, keyed by symbol.
            sizing: How much money to put at stake per trade.
            converter: Where FX rates come from. Explicit and required: a default
                would have to be a converter that answers something, and every
                answer it could invent is a wrong position size.
            config: Engine-wide settings. Defaults are conservative.
        """
        self._instruments = instruments
        self._sizing = sizing
        self._converter = converter
        self._config = config if config is not None else RiskEngineConfig()
        self._rejections = empty_rejection_counts()

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
        """Zero the refusal counters, for the start of a new run or fold."""
        self._rejections = empty_rejection_counts()

    def evaluate(
        self,
        signal: EntrySignal,
        *,
        account: AccountState,
        stop_reference: StopReference,
        smallest_exit_fraction: Decimal,
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
        if requires_atr(stop_reference, self._config.stop_buffer) and atr_price is None:
            return self._refuse(
                RiskReason.ATR_UNAVAILABLE,
                f"{signal.symbol}: the stop configuration needs an ATR and none was computed "
                "for this bar",
            )

        stop = calculate_stop(
            side=signal.side,
            reference_price=signal.reference_price,
            invalidation_price=signal.invalidation_price,
            instrument=instrument,
            stop_reference=stop_reference,
            buffer=self._config.stop_buffer,
            atr_price=atr_price,
        )
        reasons = [reason_line(RiskReason.STOP_FROM_INVALIDATION, line) for line in stop.reasons]

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
        # lose, which after rounding down is at or below what was asked for.
        risk_amount = size * stop_value_per_lot
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
