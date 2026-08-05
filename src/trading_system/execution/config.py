"""Every number the cost model uses, as a validated config rather than a constant.

Nothing in :mod:`trading_system.execution.costs` contains a magic multiplier. The
session widening, the volatility coefficient, the slippage bases and the gap
penalty all arrive from here, because they are venue properties: a run against
a raw-spread ECN and a run against a market maker differ in these numbers and in
nothing else, and that comparison has to be a config edit rather than a patch.

**The check this module exists for, beyond validating fields.** P10's
``StopBufferConfig.spread_multiple`` already pushes the protective stop one
spread further from the invalidation level, for a real reason: a long is stopped
out on the *bid*, so a stop resting on a mid level is hit roughly half a spread
before the mid reaches it. That correction is right as long as the fill model
works in mid space, which it does today. The moment a run gets real bid/ask data
and the trigger test becomes quote-aware, the same half spread is being paid
twice — once by a stop placed further away than it needs to be, once by the
quote-aware trigger — and the symptom is a strategy that quietly stops out less
often than it should while sizing smaller than it should.

That is not left to whoever remembers reading a design document.
:class:`ExecutionRunConfig` holds both numbers and refuses to construct when they
contradict, so the run fails at config load with the reason written out. The
quote-aware source does not exist yet, so the check does not fire yet — it
exists *before* the thing it guards, which is the only ordering in which a guard
of this kind is ever actually present when it is needed.
"""

from collections.abc import Mapping
from datetime import time
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Any, Literal

import pydantic
import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from trading_system.core.exceptions import ValidationError
from trading_system.core.types import OrderType
from trading_system.data.resample import FX_DAY_ORIGIN, DayOrigin
from trading_system.data.sessions import Session
from trading_system.execution.market_state import NewsSeverity


class SpreadSource(StrEnum):
    """Where the modelled spread comes from.

    Ordered by how much the run knows. Each level subsumes the one above it: a
    run with quote data can always fall back to the typical spread, never the
    reverse.
    """

    #: ``InstrumentSpec.typical_spread_points``, widened by the configured
    #: session, volatility and news multipliers. The only source available to a
    #: run with nothing but OHLCV bars, which is every run today.
    TYPICAL = "TYPICAL"

    #: The observed spread carried on :class:`~trading_system.execution.market_state.MarketState`,
    #: when the run has one. Multipliers still apply on top, because an observed
    #: spread sampled at bar close is not the spread at the instant of a fill.
    #: A bar without one degrades to ``TYPICAL`` and is counted.
    QUOTED = "QUOTED"

    #: Real bid/ask series, with resting levels tested against the side that
    #: actually triggers them. **Not implemented.** Declaring it is what trips
    #: the double-count check below; constructing a cost model with it raises.
    QUOTE_AWARE = "QUOTE_AWARE"


class SpreadConfig(BaseModel):
    """How the base spread is widened by conditions.

    Attributes:
        source: Where the base figure comes from.
        session_multipliers: Multiplier per active session. Resolution takes the
            **minimum** over the active sessions, because liquidity adds: a bar
            during the London/New York overlap is quoted more tightly than one
            in London alone, so an additional open session can only narrow the
            spread. Configuring ``LONDON_NY_OVERLAP`` below both of its parents
            therefore just works, without a precedence rule to get wrong.
        off_session_multiplier: Applied when no session is active at all — the
            rollover hour and the Asian pre-open, where spreads are at their
            widest and where a backtest that assumes otherwise books trades
            nobody could have got.
        volatility_beta: Sensitivity of the spread to volatility. The multiplier
            is ``1 + beta * (atr_ratio - 1)``, so it is exactly ``1`` at typical
            volatility whatever ``beta`` is, and ``beta = 0`` disables the term
            without disabling the ATR input.
        volatility_multiplier_min: Floor on the volatility multiplier. Quiet
            markets narrow spreads, but not without limit and never to zero.
        volatility_multiplier_max: Ceiling on the volatility multiplier. A
            volatility spike is not unbounded permission to widen; past some
            point the honest model is that the venue stopped quoting, which is a
            P16 question rather than a bigger number here.
        news_multipliers: Multiplier per news grade. ``NONE`` must be present
            and is normally ``1``.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    source: SpreadSource = SpreadSource.TYPICAL
    session_multipliers: Mapping[Session, float] = Field(default_factory=dict)
    off_session_multiplier: float = Field(default=1.5, gt=0)
    volatility_beta: float = Field(default=0.5, ge=0)
    volatility_multiplier_min: float = Field(default=0.8, gt=0)
    volatility_multiplier_max: float = Field(default=4.0, gt=0)
    news_multipliers: Mapping[NewsSeverity, float] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check_multipliers(self) -> "SpreadConfig":
        """Reject multipliers that would produce a negative or inverted spread.

        Returns:
            The validated config.

        Raises:
            ValueError: If any multiplier is not positive or the volatility
                clamp is inverted.
        """
        for name, table in (
            ("session_multipliers", self.session_multipliers),
            ("news_multipliers", self.news_multipliers),
        ):
            for key, value in table.items():
                if value <= 0:
                    raise ValueError(f"{name}[{key}] must be positive, got {value}")
        if self.volatility_multiplier_min > self.volatility_multiplier_max:
            raise ValueError(
                f"volatility_multiplier_min {self.volatility_multiplier_min} is above "
                f"volatility_multiplier_max {self.volatility_multiplier_max}"
            )
        return self

    def session_multiplier(self, sessions: frozenset[Session]) -> float:
        """Resolve the session widening for a set of simultaneously open sessions.

        Args:
            sessions: Sessions active at the fill instant.

        Returns:
            The tightest configured multiplier among them, or
            ``off_session_multiplier`` when none is active or none is
            configured.
        """
        configured = [
            self.session_multipliers[s] for s in sessions if s in self.session_multipliers
        ]
        if not configured:
            return self.off_session_multiplier
        return min(configured)

    def news_multiplier(self, severity: NewsSeverity) -> float:
        """Resolve the news widening for a severity grade.

        Args:
            severity: Grade at the fill instant.

        Returns:
            The configured multiplier, or ``1.0`` when the grade is not
            configured — an unconfigured grade means "no view", not "free".
        """
        return self.news_multipliers.get(severity, 1.0)

    def volatility_multiplier(self, atr_ratio: float) -> float:
        """Resolve the volatility widening, clamped.

        Args:
            atr_ratio: Current ATR over its rolling mean. ``1.0`` is typical.

        Returns:
            The clamped multiplier, exactly ``1.0`` at ``atr_ratio == 1``.
        """
        raw = 1.0 + self.volatility_beta * (atr_ratio - 1.0)
        return min(max(raw, self.volatility_multiplier_min), self.volatility_multiplier_max)


class SlippageParams(BaseModel):
    """Slippage for one order type.

    The model is ``base + k * atr_ratio + news + U(0, jitter)``, in points, and
    it is always a cost: the result is clamped at zero so no configuration can
    turn slippage into a rebate.

    Attributes:
        base_points: Fixed component, in points.
        atr_coefficient: Multiplied by ``atr_ratio``. Note this is the ratio
            itself, not its excess over one, so the term contributes
            ``atr_coefficient`` at typical volatility — the coefficient and the
            base together set the normal-conditions figure.
        news_points: Additive penalty per news grade, in points.
        jitter_points: Width of a uniform draw added on top, in points. Zero
            makes the model fully deterministic, which is what a test asserting
            an exact figure wants.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    base_points: float = Field(default=0.0, ge=0)
    atr_coefficient: float = Field(default=0.0, ge=0)
    news_points: Mapping[NewsSeverity, float] = Field(default_factory=dict)
    jitter_points: float = Field(default=0.0, ge=0)

    @model_validator(mode="after")
    def _check_news(self) -> "SlippageParams":
        """Reject a negative news penalty.

        Returns:
            The validated params.

        Raises:
            ValueError: If any news penalty is negative.
        """
        for grade, value in self.news_points.items():
            if value < 0:
                raise ValueError(f"news_points[{grade}] must not be negative, got {value}")
        return self

    def news_penalty(self, severity: NewsSeverity) -> float:
        """Additive penalty for a news grade, in points.

        Args:
            severity: Grade at the fill instant.

        Returns:
            The configured penalty, or zero when the grade is not configured.
        """
        return self.news_points.get(severity, 0.0)


class SlippageConfig(BaseModel):
    """Slippage parameters per order type, with the asymmetry enforced.

    Three sets rather than one coefficient, because the three order types meet
    the book in genuinely different ways:

    * A **stop** that triggers becomes a market order into a book that is already
      moving away — that is the whole reason it triggered. It is the worst case
      and it is where the honest model lives.
    * A **market** order crosses the spread at a moment nobody chose for
      liquidity reasons.
    * A **limit** order never crosses the spread at all. It fills at its price or
      it does not fill, so its slippage is normally zero.

    The ordering ``stop >= market >= limit`` is validated rather than documented.
    A transposed configuration — the stop parameters pasted into the limit slot —
    produces a backtest in which breakout strategies look cheap to run, which is
    the single most flattering error available in this module.

    Attributes:
        market: Parameters for market orders.
        stop: Parameters for stop orders, including triggered protective stops.
        limit: Parameters for limit orders, including targets.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    market: SlippageParams = Field(default_factory=SlippageParams)
    stop: SlippageParams = Field(default_factory=SlippageParams)
    limit: SlippageParams = Field(default_factory=SlippageParams)

    @model_validator(mode="after")
    def _check_ordering(self) -> "SlippageConfig":
        """Reject a configuration in which a limit slips more than a stop.

        Returns:
            The validated config.

        Raises:
            ValueError: If any of the base, ATR or jitter components orders the
                three types the wrong way round.
        """
        for field_name in ("base_points", "atr_coefficient", "jitter_points"):
            stop = getattr(self.stop, field_name)
            market = getattr(self.market, field_name)
            limit = getattr(self.limit, field_name)
            if not stop >= market >= limit:
                raise ValueError(
                    f"slippage {field_name} must order stop >= market >= limit, got "
                    f"stop={stop}, market={market}, limit={limit}; a limit order does not "
                    "cross the spread and a triggered stop enters a book already moving "
                    "away, so the reverse ordering is a transposed config, not a venue"
                )
        return self

    def for_order_type(self, order_type: OrderType) -> SlippageParams:
        """Select the parameters governing one order type.

        Args:
            order_type: How the order reached the market.

        Returns:
            The matching parameters.
        """
        if order_type is OrderType.STOP:
            return self.stop
        if order_type is OrderType.LIMIT:
            return self.limit
        return self.market


class GapConfig(BaseModel):
    """What happens to a resting order the market gapped straight through.

    P07's :func:`~trading_system.exit.fills.resting_fill_price` fills such an
    order at the bar's open and says in its own docstring that this is optimistic:
    a stop swept through a hole in the book fills below the first printable
    price, not at it. This is the replacement.

    Attributes:
        penalty_base_points: Fixed penalty applied whenever a resting order
            filled through a gap, in points.
        penalty_fraction: Share of the gap itself added to the penalty. The
            reason the penalty is proportional rather than flat: a forty-point
            gap means the book was empty for forty points, and the chance of
            being filled at the very first print falls as the hole widens. A
            flat penalty asserts a five-point gap and a five-hundred-point gap
            cost the same, which is false in the direction that matters.
        grant_improvement_to_limits: Whether a limit order gapped through fills
            at the bar's open — better than its level — instead of at its level.
            **Default false**, which is a deliberate change to the P07 reference
            model. Free positive slippage on every gap through a target is
            exactly the kind of thing that makes a backtest curve prettier than
            the account. A venue that really does fill this way can turn it on.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    penalty_base_points: float = Field(default=0.0, ge=0)
    penalty_fraction: float = Field(default=0.0, ge=0)
    grant_improvement_to_limits: bool = False

    def penalty_points(self, gap_points: float) -> float:
        """Extra slippage for a fill that came through a gap.

        Args:
            gap_points: How far beyond the level the market had already moved.
                Zero for an ordinary fill.

        Returns:
            The penalty in points, zero when there was no gap.
        """
        if gap_points <= 0:
            return 0.0
        return self.penalty_base_points + self.penalty_fraction * gap_points


class LimitFillConfig(BaseModel):
    """When a limit order resting exactly at the touched price actually fills.

    Attributes:
        touch_fill_probability: Chance of filling when the bar's extreme reached
            the level exactly and went no further, in ``[0, 1]``. Zero by
            default: at the level you are behind every order already queued at
            that price, and a bar that turned around there is a bar where the
            queue was not cleared. Assuming otherwise credits mean-reversion
            strategies with their best fills, which are the ones that never
            happen.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    touch_fill_probability: float = Field(default=0.0, ge=0, le=1)


class PerLotRolloverSwap(BaseModel):
    """Overnight financing charged as a flat per-lot amount at each rollover.

    The FX and CFD convention. The instrument's ``swap_long``/``swap_short``
    supply the amount; this supplies when it is charged and when it is tripled.

    **The rollover instant is never constructed**, only the trading-day label is
    computed, through the same :func:`~trading_system.data.resample.trading_day`
    that resets a session VWAP and a daily loss limit. P10 stage 2 established
    why: building "17:00 local on date D" is undefined on the spring-forward
    Sunday and ambiguous on the autumn one. Counting distinct day labels between
    two instants needs no such instant to exist.

    The arithmetic falls out correctly and is worth checking by hand. Trading-day
    labels run Sunday through Saturday; the labels for Friday (Fri 17:00 to Sat
    17:00) and Saturday (Sat 17:00 to Sun 17:00) cover a closed market and carry
    no charge. That leaves Sunday, Monday, Tuesday, Wednesday and Thursday — five
    charges, with Wednesday tripled, for **seven per week**, which is what a week
    of financing should cost.

    Attributes:
        day_origin: Where the trading day starts. The FX default is 17:00 New
            York, tracking US daylight saving.
        triple_weekday: Weekday whose rollover carries three days of financing,
            as :meth:`datetime.date.weekday` numbers it — ``2`` for Wednesday,
            where value date T+2 lands on the following Monday.
        closed_weekdays: Trading-day labels that carry no rollover because the
            market is shut across them. Friday and Saturday by default.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["per_lot_rollover"] = "per_lot_rollover"
    day_origin: DayOrigin = FX_DAY_ORIGIN
    triple_weekday: int = Field(default=2, ge=0, le=6)
    closed_weekdays: tuple[int, ...] = (4, 5)

    @model_validator(mode="after")
    def _check_weekdays(self) -> "PerLotRolloverSwap":
        """Reject a tripled weekday that is also declared closed.

        Returns:
            The validated model.

        Raises:
            ValueError: If a closed weekday is out of range, or the tripled
                weekday is one of them — a rollover that never happens cannot
                also be the one that carries three days.
        """
        for weekday in self.closed_weekdays:
            if not 0 <= weekday <= 6:
                raise ValueError(f"closed_weekdays entries must be in 0..6, got {weekday}")
        if self.triple_weekday in self.closed_weekdays:
            raise ValueError(
                f"triple_weekday {self.triple_weekday} is also in closed_weekdays "
                f"{self.closed_weekdays}, so the tripled rollover would never be charged"
            )
        return self


class FundingRateSwap(BaseModel):
    """Perpetual-swap funding, charged as a rate on notional at fixed instants.

    Crypto perpetuals do not roll over; they exchange funding between longs and
    shorts several times a day, as a fraction of notional rather than a flat
    per-lot fee. There is no tripled day — the schedule is uniform and the
    weekend is not special, because the market never closes.

    The amount comes out in the instrument's **quote** currency, not the account's,
    since it is a fraction of a notional priced in the quote currency. Converting
    it is the caller's job, at the rate of the accrual instant — the same rule
    :mod:`trading_system.risk.pnl` applies to each closed leg.

    Attributes:
        times_utc: Instants of day at which funding is exchanged, UTC. The usual
            eight-hourly schedule by default.
        rate_per_interval: Fraction of notional exchanged at each instant.
            Signed by convention: **positive means longs pay shorts**, which is
            the normal state of a market in contango.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["funding_rate"] = "funding_rate"
    times_utc: tuple[time, ...] = (time(0, 0), time(8, 0), time(16, 0))
    rate_per_interval: float = 0.0001

    @model_validator(mode="after")
    def _check_schedule(self) -> "FundingRateSwap":
        """Reject an empty or duplicated funding schedule.

        Returns:
            The validated model.

        Raises:
            ValueError: If no instants are configured or one repeats.
        """
        if not self.times_utc:
            raise ValueError("times_utc must name at least one funding instant")
        if len(set(self.times_utc)) != len(self.times_utc):
            raise ValueError(f"times_utc contains a duplicate: {self.times_utc}")
        return self


SwapModel = Annotated[PerLotRolloverSwap | FundingRateSwap, Field(discriminator="kind")]


class CostConfig(BaseModel):
    """Everything the cost model needs that is not instrument metadata.

    Attributes:
        spread: How the base spread is widened.
        slippage: Slippage parameters per order type.
        gap: What a fill through a gap costs on top.
        limit_fill: When a limit resting exactly at the touch fills.
        swap: Financing model per asset class. An asset class absent from the
            mapping accrues no financing, which is a statement the config makes
            rather than a default it inherits.
        run_seed: Seed for the per-fill random streams. Two runs with the same
            seed produce identical fills regardless of what else they contain.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    spread: SpreadConfig = Field(default_factory=SpreadConfig)
    slippage: SlippageConfig = Field(default_factory=SlippageConfig)
    gap: GapConfig = Field(default_factory=GapConfig)
    limit_fill: LimitFillConfig = Field(default_factory=LimitFillConfig)
    swap: Mapping[str, SwapModel] = Field(default_factory=dict)
    run_seed: int = 0


class ExecutionRunConfig(BaseModel):
    """The costs of a run, together with the one Risk Engine number they interact with.

    The two fields are here together for a single reason: neither is wrong on its
    own, and the pair can be. See this module's docstring for the mechanism. A
    model that holds both makes the contradiction a load-time failure with the
    explanation attached, rather than a run that quietly pays for the same half
    spread twice.

    Attributes:
        costs: The cost configuration.
        stop_buffer_spread_multiple: A copy of P10's
            ``StopBufferConfig.spread_multiple``. A plain float rather than the
            model itself, so this package does not import
            :mod:`trading_system.risk` — the same import discipline that keeps
            Risk out of Exit and Exit out of Entry.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    costs: CostConfig = Field(default_factory=CostConfig)
    stop_buffer_spread_multiple: float = Field(default=1.0, ge=0)

    @model_validator(mode="after")
    def _check_spread_not_paid_twice(self) -> "ExecutionRunConfig":
        """Reject a quote-aware run that also buffers the stop by a spread.

        Returns:
            The validated config.

        Raises:
            ValueError: If the spread source is quote-aware while the stop
                buffer still widens by a spread.
        """
        if (
            self.costs.spread.source is SpreadSource.QUOTE_AWARE
            and self.stop_buffer_spread_multiple != 0
        ):
            raise ValueError(
                "spread source is QUOTE_AWARE while stop_buffer_spread_multiple is "
                f"{self.stop_buffer_spread_multiple}, so the same half spread is charged "
                "twice: once by a protective stop placed a spread further out than the "
                "invalidation level, and again by a trigger test that already knows which "
                "side of the quote fires it. Set risk stop_buffer.spread_multiple to 0 for "
                "quote-aware runs, or keep the mid-quoted spread source."
            )
        return self


def load_execution_config(path: Path) -> ExecutionRunConfig:
    """Load and validate the execution configuration from a YAML file.

    Args:
        path: Location of the YAML file.

    Returns:
        The validated configuration.

    Raises:
        ValidationError: If the file cannot be read, is not a mapping, or fails
            validation — including the spread double-count check.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as error:
        raise ValidationError(f"execution config {path}: cannot be read: {error}") from error

    loaded: Any = yaml.safe_load(text)
    if not isinstance(loaded, dict):
        raise ValidationError(
            f"execution config {path}: expected a mapping, got {type(loaded).__name__}"
        )
    try:
        return ExecutionRunConfig.model_validate(loaded)
    except pydantic.ValidationError as error:
        raise ValidationError(f"execution config {path}: {error}") from error
