"""What an order actually costs: spread, commission, slippage, financing.

This is the layer backtests lie in. Everything understated here comes back as
the gap between the curve and the account, and none of it is visible in a test
that does not compute the expected figure by hand.

**The spread is crossed once per fill, as half of it, and that is the only place
the model leaves mid space.**

Bars are mid, so every level derived from a bar is mid: an invalidation from P06,
a protective stop from P10, a target from P07. A fill therefore costs::

    fill = mid ± spread / 2        BUY adds, SELL subtracts

applied at exactly one point in the code, with the sign taken from the order's
side. A position is opened and closed, so the round turn crosses twice and pays
``spread/2 + spread/2`` — **one spread, not two and not none.** Worked through on
EURUSD with a one-point spread (``point_size`` 0.0001, ``contract_size`` 100 000,
so a point is 10 USD per lot), in a market that has not moved at all, mid at
1.10000 on both fills:

===========  ===========================  ===========================
             long                         short
===========  ===========================  ===========================
entry        BUY at 1.10000 + 0.00005     SELL at 1.10000 − 0.00005
             = 1.10005                    = 1.09995
exit         SELL at 1.10000 − 0.00005    BUY at 1.10000 + 0.00005
             = 1.09995                    = 1.10005
movement     −0.00010 = −1.0 point        −0.00010 = −1.0 point
per lot      −10.00 USD                   −10.00 USD
===========  ===========================  ===========================

The three ways this goes wrong all produce plausible numbers: charging the half
spread only at entry gives −0.5 points, charging the whole spread on both sides
gives −2.0, and charging the whole spread on whichever fill happens to be a
``BUY`` gives −1.0 on longs and 0.0 on shorts — which averages to the right
answer over a balanced sample and hides indefinitely. That last one is why the
round-turn test runs both directions and asserts they are equal.

Halving rather than charging the whole spread at entry also survives P07's
partial ladders: each leg pays half a spread on **its own fraction**, so the
fractions sum to exactly one spread over the original size whatever the ladder
looks like. Charging it all at entry would book the cost of three exits at the
moment of the first, and the first leg's realised R would be wrong by the cost of
two closes that had not happened.

**Rounding is against the trader.** Half a spread does not always land on the
tick grid — an instrument whose point is one tick, at an odd spread, puts the
fill half a tick off. The fill is snapped away from the trader (up for a ``BUY``,
down for a ``SELL``), because snapping to nearest hands back half a tick per fill
in a direction that looks random and is not. The exact invariant is therefore:
*a round turn costs at least the spread, exceeds it by at most one tick per
fill, and equals it exactly whenever the half spread lands on the grid* — which
it does on EURUSD, where a point is ten ticks.

**Interaction with P10, and why it is a load-time failure rather than a comment.**
``StopBufferConfig.spread_multiple`` already widens the protective stop by one
spread, because a long is stopped on the bid and a mid-space trigger test does
not know that. That is the right correction while fills are modelled in mid
space, and it becomes double-counting the moment a run gets real quotes.
:class:`~trading_system.execution.config.ExecutionRunConfig` holds both numbers
and refuses to construct when they contradict.

**Commission is normalised to a per-side figure exactly once**, in
:attr:`~trading_system.core.instruments.InstrumentSpec.commission_per_side`, and
charged on each fill. The round-turn halving is the whole factor-of-two risk in
this module and it lives on one line.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from decimal import Decimal
from enum import StrEnum
from types import MappingProxyType

from trading_system.core.instruments import InstrumentClass, InstrumentSpec
from trading_system.core.logging import get_logger
from trading_system.core.types import OrderType, Price, Side
from trading_system.data.resample import DayOrigin, trading_day
from trading_system.execution.config import (
    CostConfig,
    FundingRateSwap,
    PerLotRolloverSwap,
    SpreadSource,
    SwapModel,
)
from trading_system.execution.market_state import MarketState
from trading_system.execution.orders import ExecutionOrder, Fill
from trading_system.execution.rng import fill_rng

logger = get_logger(__name__)


class CostDegradation(StrEnum):
    """A cost that was computed on a fallback rather than on the input it wanted.

    Kept apart from anything resembling a refusal, exactly as P10 stage 2 keeps
    ``CORRELATION_UNAVAILABLE`` out of ``REJECTION_REASONS``: the order **was**
    filled and the trade **did** happen, just priced on a prior instead of a
    measurement. Folding the two together would hide both — refusals would look
    inflated and degradations would look like nothing at all.
    """

    #: No ATR ratio on this bar, so the volatility terms of the spread and the
    #: slippage fell back to typical conditions.
    ATR_UNAVAILABLE = "atr_unavailable"

    #: The run asked for observed spreads but this instant carried none, so the
    #: instrument's typical spread was used.
    QUOTED_SPREAD_UNAVAILABLE = "quoted_spread_unavailable"


@dataclass(frozen=True)
class CostStats:
    """How much of a run was priced on complete information.

    A count on its own does not answer the question anybody asks of it. "The ATR
    was missing on 40 fills" is a warm-up artefact in a run of ten thousand and a
    statement that half the trades were priced on a model known to be too cheap
    in a run of eighty. The denominator is what makes the count mean something,
    so it is carried alongside and the division is done here rather than left to
    whoever writes the report.

    The denominator is **fills**, not bars: the cost model is invoked per fill
    and has no view of bars it was not asked about. A run wanting the bar-level
    coverage of an indicator should ask the feature layer, which is where that
    fact lives.

    Attributes:
        fills: How many fills were priced.
        degradations: Count per degradation reason. Every reason is present,
            including at zero — "no fill was missing an ATR" is a recorded fact,
            not an absent key.
    """

    fills: int
    degradations: Mapping[CostDegradation, int]

    def fraction(self, reason: CostDegradation) -> float:
        """Share of priced fills that degraded for one reason.

        Args:
            reason: Which degradation.

        Returns:
            A fraction in ``[0, 1]``, and zero for a run with no fills — an
            empty run degraded on nothing rather than on everything.
        """
        if self.fills == 0:
            return 0.0
        return self.degradations[reason] / self.fills

    @property
    def fractions(self) -> Mapping[CostDegradation, float]:
        """Every degradation's share of priced fills.

        Exposed as well as :meth:`fraction` so a report cannot present the raw
        counts by simply forgetting to divide.
        """
        return MappingProxyType({reason: self.fraction(reason) for reason in CostDegradation})


def _empty_degradations() -> dict[CostDegradation, int]:
    """Start every degradation counter at zero.

    Returns:
        A counter per reason, all zero.
    """
    return dict.fromkeys(CostDegradation, 0)


class CostModel:
    """Turns an order resolved to a mid price into an executable fill.

    Holds the run's configuration and instrument metadata, plus the counters that
    say how much of the run was priced on fallbacks. Stateful only in those
    counters: the price of a fill depends on nothing but the order, the market
    state and the seed, which is what makes fills reproducible independently of
    each other.
    """

    __slots__ = ("_config", "_degradations", "_fills", "_instruments")

    def __init__(self, instruments: Mapping[str, InstrumentSpec], config: CostConfig) -> None:
        """Bind the model to a universe and a configuration.

        Args:
            instruments: Symbol to contract specification.
            config: Every number the model uses.

        Raises:
            NotImplementedError: If the configured spread source is
                ``QUOTE_AWARE``. The source is declarable so that the
                double-count check in
                :class:`~trading_system.execution.config.ExecutionRunConfig`
                exists before the feature does; silently pricing it as if it
                were ``QUOTED`` would make a run report a precision it does not
                have.
        """
        if config.spread.source is SpreadSource.QUOTE_AWARE:
            raise NotImplementedError(
                "spread source QUOTE_AWARE needs a real bid/ask series and a trigger test "
                "that knows which side of the quote fires a resting order; neither exists "
                "yet. Use TYPICAL or QUOTED."
            )
        self._instruments = dict(instruments)
        self._config = config
        self._fills = 0
        self._degradations = _empty_degradations()

    def __repr__(self) -> str:
        """Compact description naming the universe size and the fill count."""
        return f"CostModel({len(self._instruments)} instruments, {self._fills} fills)"

    @property
    def config(self) -> CostConfig:
        """The cost configuration this model was built with.

        Exposed because the pieces of it that are *not* about pricing a fill —
        the financing model per asset class, the gap and limit-fill parameters
        the resting-order models need, the run seed — belong to the same run and
        would otherwise have to be passed alongside this object and kept in step
        with it by hand.
        """
        return self._config

    @property
    def stats(self) -> CostStats:
        """How many fills were priced and how many degraded, with shares."""
        return CostStats(fills=self._fills, degradations=MappingProxyType(dict(self._degradations)))

    def reset(self) -> None:
        """Clear the counters for a fresh run.

        Both the fill count and the degradations, since together they describe
        one run — the same reasoning that makes
        :meth:`~trading_system.entry.compiler.EntryEvaluator.reset` clear its
        counters and its pending setups at once.
        """
        self._fills = 0
        self._degradations = _empty_degradations()

    def apply(self, order: ExecutionOrder, market_state: MarketState) -> Fill:
        """Price one order.

        Args:
            order: An order a fill model has already resolved to a mid price.
            market_state: Conditions at the fill instant.

        Returns:
            The fill, with the spread, slippage and commission decomposed.

        Raises:
            KeyError: If the symbol is not in the instrument universe. Never a
                stand-in specification: guessing a point size is guessing the
                cost of every trade in the instrument.
        """
        instrument = self._instrument(order.symbol)
        atr_ratio = self._atr_ratio(market_state)

        spread_points = self._spread_points(instrument, market_state, atr_ratio)
        slippage_points = self._slippage_points(order, market_state, atr_ratio)

        # The single point at which this system leaves mid space. A BUY lifts the
        # offer and a SELL hits the bid, so both pay; slippage is signed the same
        # way, since it is defined as "worse". `shift_price` snaps away from the
        # mid and does the arithmetic in decimal, so a half spread that lands on
        # the tick grid stays there.
        adverse_points = spread_points / 2 + slippage_points
        signed = adverse_points if order.side is Side.BUY else -adverse_points
        price = instrument.shift_price(order.mid_price, signed)

        self._fills += 1
        return Fill(
            order_id=order.order_id,
            symbol=order.symbol,
            side=order.side,
            size=order.size,
            ts=market_state.ts,
            mid_price=order.mid_price,
            price=price,
            spread_points=spread_points,
            slippage_points=slippage_points,
            commission=self.commission(order.symbol, order.size),
            gap_points=order.gap_points,
        )

    def commission(self, symbol: str, size: Decimal) -> Decimal:
        """Commission for one fill, in account currency.

        Args:
            symbol: Instrument traded.
            size: Size filled, in lots.

        Returns:
            ``commission_per_side * size``. Charged on each of the two fills, so
            a round turn costs the instrument's stated round-turn figure under
            ``ROUND_TURN`` and twice its stated figure under ``PER_SIDE`` —
            which is what each basis asserts.

        Raises:
            KeyError: If the symbol is unknown.
        """
        return self._instrument(symbol).commission_per_side * size

    def _instrument(self, symbol: str) -> InstrumentSpec:
        """Look up an instrument, refusing to invent one.

        Args:
            symbol: Symbol to look up.

        Returns:
            Its specification.

        Raises:
            KeyError: If it is not registered.
        """
        try:
            return self._instruments[symbol]
        except KeyError:
            raise KeyError(
                f"unknown instrument {symbol!r}; registered: {sorted(self._instruments)}"
            ) from None

    def _atr_ratio(self, market_state: MarketState) -> float:
        """Resolve the volatility ratio, counting the fallback when there is none.

        Args:
            market_state: Conditions at the fill instant.

        Returns:
            The ratio, or ``1.0`` when the indicator had not warmed up. The
            fallback is counted rather than silent: a run priced largely on
            typical volatility is a run whose costs are understated wherever it
            mattered most, and a count of zero is the only way to know it was
            not.
        """
        if market_state.atr_ratio is None:
            self._degradations[CostDegradation.ATR_UNAVAILABLE] += 1
            return 1.0
        return market_state.atr_ratio

    def _spread_points(
        self, instrument: InstrumentSpec, market_state: MarketState, atr_ratio: float
    ) -> float:
        """The full modelled spread at this instant, in points.

        Args:
            instrument: Contract specification.
            market_state: Conditions at the fill instant.
            atr_ratio: Already-resolved volatility ratio.

        Returns:
            The spread in points, never negative.
        """
        spread_config = self._config.spread
        base = instrument.typical_spread_points
        if spread_config.source is SpreadSource.QUOTED:
            if market_state.quoted_spread_points is None:
                self._degradations[CostDegradation.QUOTED_SPREAD_UNAVAILABLE] += 1
            else:
                base = market_state.quoted_spread_points

        return (
            base
            * spread_config.session_multiplier(market_state.sessions)
            * spread_config.volatility_multiplier(atr_ratio)
            * spread_config.news_multiplier(market_state.news_severity)
        )

    def _slippage_points(
        self, order: ExecutionOrder, market_state: MarketState, atr_ratio: float
    ) -> float:
        """Slippage beyond the half spread, in points, signed so positive is worse.

        Args:
            order: The order being priced.
            market_state: Conditions at the fill instant.
            atr_ratio: Already-resolved volatility ratio.

        Returns:
            The slippage. Non-negative except when a run has explicitly enabled
            gap improvement on limit orders, which is the only configuration
            under which a fill may come back better than its mid.
        """
        params = self._config.slippage.for_order_type(order.order_type)
        points = (
            params.base_points
            + params.atr_coefficient * atr_ratio
            + params.news_penalty(market_state.news_severity)
        )
        if params.jitter_points > 0:
            rng = fill_rng(
                self._config.run_seed,
                symbol=order.symbol,
                ts=market_state.ts,
                order_id=order.order_id,
            )
            points += rng.uniform(0.0, params.jitter_points)

        points += self._gap_points(order)
        return points

    def _gap_points(self, order: ExecutionOrder) -> float:
        """The gap component of slippage, in points.

        A limit order gapped through is the one case that can come back
        negative, and only when a run has asked for it. See
        :class:`~trading_system.execution.config.GapConfig`.

        Args:
            order: The order being priced.

        Returns:
            The signed gap contribution.
        """
        if order.gap_points <= 0:
            return 0.0
        if order.order_type is OrderType.LIMIT:
            # A limit is never made worse by a gap — it was already resting at a
            # price the market ran past. The only question is whether the run
            # credits it with the improvement, and by default it does not.
            return 0.0
        return self._config.gap.penalty_points(order.gap_points)


def realized_points(entry: Fill, close: Fill, instrument: InstrumentSpec) -> Decimal:
    """Price movement a round turn actually realised, in exact points.

    The figure the spread invariant is stated in. **Decimal, not float**, for the
    same reason ``StopCalculation.distance_points_exact`` is: subtracting two
    prices that both sit on the tick grid is exact in decimal and is not in
    binary — ``1.09995 - 1.10005`` comes out as ``-9.999999999998899e-05``, and a
    round turn that costs exactly one point would report ``-0.9999999999998899``.
    That is harmless for money and useless for an invariant, since an assertion
    against it has to carry a tolerance and a tolerance hides the very
    off-by-a-half-spread errors this is checking for.

    Signed by direction: positive is profit. In a market that has not moved, this
    returns exactly minus the spread — one spread for the round turn, not two and
    not half of one.

    Args:
        entry: The opening fill.
        close: The closing fill.
        instrument: Contract specification, for the point size.

    Returns:
        Movement in points, exact.

    Raises:
        ValueError: If the two fills are on the same side, which is not a round
            turn but two entries.
    """
    if entry.side is close.side:
        raise ValueError(f"a round turn closes on the opposite side; both fills are {entry.side}")
    entry_price = Decimal(str(entry.price))
    close_price = Decimal(str(close.price))
    move = close_price - entry_price if entry.side is Side.BUY else entry_price - close_price
    return move / Decimal(str(instrument.point_size))


@dataclass(frozen=True)
class SwapAccrual:
    """Financing charged over a holding period.

    Attributes:
        amount: The charge, signed so that negative is money leaving the
            account. Zero when no financing instant fell in the period.
        currency: What ``amount`` is denominated in. **Not always the account
            currency**: a per-lot rollover is quoted in account currency by the
            instrument specification, while perpetual funding is a fraction of a
            notional priced in the quote currency. Stating which is cheaper than
            a convention nobody can check at the call site, and the caller
            converts at the accrual instant the same way
            :mod:`trading_system.risk.pnl` converts each closed leg.
        charges: How many financing instants were crossed, counting a tripled
            Wednesday as three. Reported because "the swap looks too big" and
            "the swap was charged too often" are different bugs.
    """

    amount: Decimal
    currency: str
    charges: int


def accrue_swap(
    instrument: InstrumentSpec,
    *,
    side: Side,
    size: Decimal,
    held_from: datetime,
    held_to: datetime,
    model: SwapModel,
    account_currency: str,
    mark_price: Price | None = None,
) -> SwapAccrual:
    """Financing for holding a position across one or more rollovers.

    Args:
        instrument: Contract specification, supplying the per-lot swap figures.
        side: Direction held.
        size: Size held, in lots.
        held_from: When the position opened, tz-aware.
        held_to: When it closed, tz-aware.
        model: Which financing convention applies.
        account_currency: Currency the per-lot figures are quoted in.
        mark_price: Price to value the notional at, required by the funding-rate
            model and ignored by the per-lot one.

    Returns:
        The accrual, naming its own currency.

    Raises:
        ValueError: If the period runs backwards, or the funding model was given
            no mark price — a rate on notional with no notional is not a number.
    """
    if held_to < held_from:
        raise ValueError(f"holding period runs backwards: {held_from:%F %T%z} to {held_to:%F %T%z}")

    if isinstance(model, PerLotRolloverSwap):
        return _accrue_per_lot(
            instrument,
            side=side,
            size=size,
            held_from=held_from,
            held_to=held_to,
            model=model,
            account_currency=account_currency,
        )
    return _accrue_funding(
        instrument,
        side=side,
        size=size,
        held_from=held_from,
        held_to=held_to,
        model=model,
        mark_price=mark_price,
    )


def _accrue_per_lot(
    instrument: InstrumentSpec,
    *,
    side: Side,
    size: Decimal,
    held_from: datetime,
    held_to: datetime,
    model: PerLotRolloverSwap,
    account_currency: str,
) -> SwapAccrual:
    """Count rollovers by trading-day label and charge the per-lot figure at each.

    Args:
        instrument: Contract specification.
        side: Direction held.
        size: Size held, in lots.
        held_from: Open time.
        held_to: Close time.
        model: Rollover convention.
        account_currency: Currency the per-lot figures are quoted in.

    Returns:
        The accrual in account currency.
    """
    per_lot = instrument.swap_long if side is Side.BUY else instrument.swap_short
    charges = sum(
        _rollover_weight(label, model)
        for label in _labels_crossed(held_from, held_to, model.day_origin)
    )
    return SwapAccrual(
        amount=per_lot * size * charges,
        currency=account_currency,
        charges=charges,
    )


def _labels_crossed(held_from: datetime, held_to: datetime, day_origin: DayOrigin) -> list[date]:
    """Trading-day labels entered strictly after the opening one.

    Each such label is exactly one rollover: the day boundary that started it.
    The boundary **instant** is never constructed, only the labels either side
    of it are compared — the rule P10 stage 2 established, because "17:00 local
    on date D" does not exist on the spring-forward Sunday and happens twice on
    the autumn one.

    Args:
        held_from: Open time.
        held_to: Close time.
        day_origin: Where the trading day starts.

    Returns:
        The labels entered during the holding period, ascending.
    """
    first = trading_day(held_from, day_origin)
    last = trading_day(held_to, day_origin)
    crossed: list[date] = []
    label = first + timedelta(days=1)
    while label <= last:
        crossed.append(label)
        label += timedelta(days=1)
    return crossed


def _rollover_weight(label: date, model: PerLotRolloverSwap) -> int:
    """How many days of financing the rollover into ``label`` carries.

    Args:
        label: Trading-day label the rollover started.
        model: Rollover convention.

    Returns:
        Three on the tripled weekday, zero across a closed market, one
        otherwise. Over a full week this sums to seven, which is the arithmetic
        check that the convention was implemented rather than approximated:
        Sunday, Monday, Tuesday and Thursday contribute one each, Wednesday
        three, and the Friday and Saturday labels cover a shut market.
    """
    weekday = label.weekday()
    if weekday in model.closed_weekdays:
        return 0
    if weekday == model.triple_weekday:
        return 3
    return 1


def _accrue_funding(
    instrument: InstrumentSpec,
    *,
    side: Side,
    size: Decimal,
    held_from: datetime,
    held_to: datetime,
    model: FundingRateSwap,
    mark_price: Price | None,
) -> SwapAccrual:
    """Charge perpetual funding at each scheduled instant crossed.

    Args:
        instrument: Contract specification.
        side: Direction held.
        size: Size held, in lots.
        held_from: Open time.
        held_to: Close time.
        model: Funding schedule and rate.
        mark_price: Price the notional is valued at.

    Returns:
        The accrual in the instrument's quote currency.

    Raises:
        ValueError: If no mark price was supplied.
    """
    if mark_price is None:
        raise ValueError(
            f"{instrument.symbol}: the funding-rate model charges a fraction of notional and "
            "needs a mark_price to value it; there is no per-lot fallback"
        )
    charges = _funding_instants(held_from, held_to, model)
    notional = size * instrument.contract_size * Decimal(str(mark_price))
    rate = Decimal(str(model.rate_per_interval))
    # Positive rate means longs pay, so a long's accrual is negative money.
    direction = Decimal(-1) if side is Side.BUY else Decimal(1)
    return SwapAccrual(
        amount=direction * rate * notional * charges,
        currency=instrument.quote_currency,
        charges=charges,
    )


def _funding_instants(held_from: datetime, held_to: datetime, model: FundingRateSwap) -> int:
    """How many scheduled funding instants fall in ``(held_from, held_to]``.

    Args:
        held_from: Open time, exclusive.
        held_to: Close time, inclusive.
        model: Funding schedule.

    Returns:
        The count. Crypto never closes, so no day is skipped and no instant is
        tripled — the schedule is uniform and the weekend is not special.
    """
    count = 0
    day = held_from.date()
    while day <= held_to.date():
        for at in model.times_utc:
            instant = datetime.combine(day, at, tzinfo=held_from.tzinfo)
            if held_from < instant <= held_to:
                count += 1
        day += timedelta(days=1)
    return count


def warn_if_optimistic(instrument_class: InstrumentClass, model: SwapModel) -> None:
    """Log when a financing model does not match what the asset class does.

    Args:
        instrument_class: What is being traded.
        model: The configured financing convention.
    """
    crypto = instrument_class is InstrumentClass.CRYPTO
    funding = isinstance(model, FundingRateSwap)
    if crypto != funding:
        logger.warning(
            "execution.swap_model_mismatch",
            instrument_class=str(instrument_class),
            model=model.kind,
            detail=(
                "perpetuals exchange funding on notional and never roll over; "
                "everything else rolls over at a per-lot rate"
            ),
        )
