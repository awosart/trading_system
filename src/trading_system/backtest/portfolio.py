"""Positions, money, and the equity curve.

**The equity curve is sampled on the merged bar-close stream, one row per
instant.** Not per event — several rows at one instant would make max drawdown
depend on the order of things inside it and would make the row count a function
of the trade count, which breaks every annualisation downstream. And not on a
wall-clock grid — a regular grid either invents rows where no market printed,
where a weekend of flat equity deflates volatility and inflates Sharpe, or has to
be cut by a calendar that then becomes a second notion of "market open" able to
disagree with the bars.

**Marks are stale by construction, and honestly so.** At an M15 instant a EURUSD
H1 position is marked at its own last *closed* H1 bar. There is no forward fill
from a bar that has not closed, because :class:`~trading_system.backtest.engine.BarStore`
does not hold one.

**Every row carries its ``trading_day`` label**, from the one function the whole
system cuts days with — the same boundary a session VWAP resets on and a daily
loss limit measures against. P14 derives the daily curve as the last row per
label and never defines a day for itself.

.. warning::

   **The number of rows depends on the composition of the portfolio.** Adding a
   NAS100 M15 stream to a EURUSD H1 run adds rows, so drawdown duration measured
   in rows and the ``N`` in a Sharpe ratio both change without any strategy
   changing. That is honest — the portfolio really is different — and it makes
   one requirement on P14 non-negotiable: **annualise from elapsed time and the
   day labels, never from ``len(curve)``.** The label is in every row for
   exactly this reason.

**Money is decimal end to end.** ``balance`` moves only through booked leg
profit, commission and financing, each a :class:`~decimal.Decimal`, so
``balance == starting_balance + realized - commission + swap`` is an exact
identity rather than an equality to the cent. ``equity`` is ``balance`` plus the
floating result of what is still open.

**Multi-currency.** A leg's profit accrues in the instrument's quote currency and
is converted at *that leg's own fill time*, per P10 — a position opened in March
and closed in June lived through a currency move, and one averaged rate would
erase precisely that. A *mark* is different: it is an unrealised figure between
fills, so when the converter cannot price the instant it falls back to the rate
frozen at the position's entry and **counts** the fallback rather than failing
the run or silently substituting parity.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import Decimal

from trading_system.backtest.clock import StreamKey
from trading_system.core.instruments import InstrumentRegistry, InstrumentSpec
from trading_system.core.logging import get_logger
from trading_system.core.types import Price, Side, ensure_utc
from trading_system.data.resample import DayOrigin, trading_day, trading_day_of_close
from trading_system.data.sessions import TradingCalendar
from trading_system.execution.orders import Fill
from trading_system.exit.plan import DeferredExit, ExitPlan
from trading_system.exit.position import ClosedLeg, ManagedPosition
from trading_system.risk.circuit_breakers import ClosedTrade
from trading_system.risk.conversion import FxConverter, FxRateUnavailableError
from trading_system.risk.models import AccountState, OpenRisk

logger = get_logger(__name__)


@dataclass
class OpenPosition:
    """One live position and everything the run has spent on it.

    Attributes:
        position_id: Unique within the run.
        key: Stream the position is managed on — the symbol *and* the timeframe
            its exit plan steps at.
        position: The P07 position, owning size, stop, ratchet and closed legs.
        plan: This position's own exit plan instance. Not shared: rules carry
            state (a trailing activation, which rungs of a ladder have fired),
            so two concurrent positions sharing one plan would read each other's.
        strategy_id: Strategy that opened it.
        entry_fill: The fill that opened it, costs decomposed.
        entry_bar_index: Bar of ``key`` the entry filled on.
        entry_fx_rate: Quote-to-account rate frozen at sizing, per P10. Used for
            restating open risk and as the fallback when a mark cannot be priced.
        risk_amount: Money at stake if the stop is hit, in account currency.
            Restated whenever the stop moves or a partial closes.
        entry_quality: The opening signal's ``quality``. Carried so the closed
            trade can report it — nothing in this layer reads it to decide
            anything, which is the point: it is a record of what the Entry
            Engine scored the setup, not an input to sizing or exit.
        deferred_exit: A ``BAR_CLOSE`` decision awaiting this stream's next open.
            Held here, on the object that outlives the loop — not in a local
            variable, which is how it gets lost, and not on the plan, which is
            reused across positions.
        commission_paid: Commission booked against this position so far.
        swap_paid: Financing booked so far, signed; negative is money leaving.
        realized: Leg profit booked so far, account currency, gross of costs.
        last_financing_ts: Instant financing has been charged up to.
    """

    position_id: str
    key: StreamKey
    position: ManagedPosition
    plan: ExitPlan
    strategy_id: str
    entry_fill: Fill
    entry_bar_index: int
    entry_fx_rate: Decimal
    risk_amount: Decimal
    entry_quality: float
    deferred_exit: DeferredExit | None = None
    commission_paid: Decimal = Decimal(0)
    swap_paid: Decimal = Decimal(0)
    realized: Decimal = Decimal(0)
    last_financing_ts: datetime = field(default_factory=lambda: datetime.min)

    @property
    def symbol(self) -> str:
        """Instrument held."""
        return self.position.symbol

    @property
    def side(self) -> Side:
        """Direction held."""
        return self.position.side


@dataclass(frozen=True)
class EquityPoint:
    """The account at one instant of the merged stream.

    Attributes:
        ts: The instant, tz-aware UTC.
        day: Trading-day label, from the run's day origin. What P14 groups by.
        balance: Closed-trade equity: starting balance plus realised profit,
            less commission, plus financing.
        equity: ``balance`` plus the floating result of open positions.
        realized: Cumulative gross leg profit, account currency.
        unrealized: Floating result of what is still open, account currency.
        commission_paid: Cumulative commission, positive.
        swap_paid: Cumulative financing, signed; negative is money paid out.
        open_positions: How many positions were open when the row was written.
    """

    ts: datetime
    day: date
    balance: Decimal
    equity: Decimal
    realized: Decimal
    unrealized: Decimal
    commission_paid: Decimal
    swap_paid: Decimal
    open_positions: int


@dataclass(frozen=True)
class TradeRecord:
    """One position, from open to flat, decomposed.

    The decomposition is what makes the curve reconcilable: ``net`` is
    ``gross - commission + swap`` by construction, and the sum of ``net`` across
    every record plus the starting balance is the final balance, exactly.

    Attributes:
        position_id: Matches the :class:`OpenPosition` it came from.
        symbol: Instrument traded.
        strategy_id: Strategy that opened it.
        side: Direction held.
        size: Size opened, in lots.
        opened_at: Entry fill time.
        closed_at: Time of the last closing leg.
        entry_price: Executable entry price, after spread and slippage.
        quality: The opening signal's ``quality``, in ``[0, 1]``. A record, not
            a decision: by the time a trade is filed, quality has already done
            whatever sizing it was going to do. It is here so P14 can ask
            whether the score predicted anything — a question that needs the
            score and the realised R on the same row.
        initial_risk_distance: Entry price to initial stop, in price units,
            frozen at entry — the denominator ``realized_r`` was measured
            against. Carried for the same reason ``quality`` is: excursions
            (MAE/MFE) are only interpretable in R, and R needs a denominator
            that no consumer of a closed trade can otherwise reconstruct —
            ``realized_r`` is size-weighted across legs and does not invert.
        gross: Leg profit summed over the position's legs, account currency.
        commission: Commission across the entry fill and every closing fill.
        swap: Financing, signed.
        net: ``gross - commission + swap``.
        realized_r: Size-weighted result in multiples of the initial risk.
        legs: How many closes it took.
    """

    position_id: str
    symbol: str
    strategy_id: str
    side: Side
    size: Decimal
    opened_at: datetime
    closed_at: datetime
    entry_price: Price
    quality: float
    initial_risk_distance: float
    gross: Decimal
    commission: Decimal
    swap: Decimal
    net: Decimal
    realized_r: float
    legs: int

    def as_closed_trade(self) -> ClosedTrade:
        """Project onto what the circuit breakers read.

        Returns:
            The trade as :class:`~trading_system.risk.circuit_breakers.ClosedTrade`,
            carrying ``net`` — the breakers compare against a limit and want the
            figure the account actually moved by.
        """
        return ClosedTrade(
            closed_at=self.closed_at,
            pnl=self.net,
            symbol=self.symbol,
            strategy_id=self.strategy_id,
        )


class Portfolio:
    """Holdings, cash and the equity curve of one run."""

    def __init__(
        self,
        *,
        currency: str,
        starting_balance: Decimal,
        instruments: InstrumentRegistry,
        converter: FxConverter,
        day_origin: DayOrigin,
        calendar: TradingCalendar | None = None,
    ) -> None:
        """Open an account.

        Args:
            currency: Account denomination. Every figure here is in it.
            starting_balance: Opening balance.
            instruments: Contract specifications, for contract size and currency.
            converter: Where quote-to-account rates come from.
            day_origin: Where the trading day is cut, for the row labels.
            calendar: When the market is open, for folding a bar whose close
                lands in a shut window back onto the trading day that
                actually produced it (see ``backtest/clock.py``). ``None``
                keeps the plain ``trading_day(instant, day_origin)`` label
                unconditionally — the right default for a ``Portfolio`` built
                directly in a test, not passing through an
                :class:`~trading_system.backtest.orchestrator.Orchestrator`,
                which always derives and passes a real one when the run's
                instruments make one unambiguous.

        Raises:
            ValueError: If the starting balance is not positive.
        """
        if starting_balance <= 0:
            raise ValueError(f"starting balance must be positive, got {starting_balance}")
        self._currency = currency
        self._starting_balance = starting_balance
        self._instruments = instruments
        self._converter = converter
        self._day_origin = day_origin
        self._calendar = calendar

        self._realized = Decimal(0)
        self._commission = Decimal(0)
        self._swap = Decimal(0)
        self._open: dict[str, OpenPosition] = {}
        self._trades: list[TradeRecord] = []
        self._curve: list[EquityPoint] = []
        self._unrealized = Decimal(0)
        self._fx_fallback_marks = 0

    def __repr__(self) -> str:
        """Compact description naming the balance and what is open."""
        return (
            f"Portfolio({self._currency} balance={self.balance}, "
            f"open={len(self._open)}, trades={len(self._trades)})"
        )

    @property
    def currency(self) -> str:
        """Account denomination."""
        return self._currency

    @property
    def starting_balance(self) -> Decimal:
        """Opening balance."""
        return self._starting_balance

    @property
    def balance(self) -> Decimal:
        """Closed-trade equity: an exact function of what has been booked."""
        return self._starting_balance + self._realized - self._commission + self._swap

    @property
    def equity(self) -> Decimal:
        """Balance plus the floating result of open positions, as last marked."""
        return self.balance + self._unrealized

    @property
    def realized(self) -> Decimal:
        """Cumulative gross leg profit."""
        return self._realized

    @property
    def commission_paid(self) -> Decimal:
        """Cumulative commission, positive."""
        return self._commission

    @property
    def swap_paid(self) -> Decimal:
        """Cumulative financing, signed."""
        return self._swap

    @property
    def unrealized(self) -> Decimal:
        """Floating result of open positions, as of the last mark."""
        return self._unrealized

    @property
    def open_positions(self) -> Sequence[OpenPosition]:
        """Live positions, in the order they were opened."""
        return list(self._open.values())

    @property
    def trades(self) -> Sequence[TradeRecord]:
        """Completed trades, oldest first."""
        return list(self._trades)

    @property
    def closed_trades(self) -> Sequence[ClosedTrade]:
        """Completed trades as the circuit breakers read them."""
        return [trade.as_closed_trade() for trade in self._trades]

    @property
    def curve(self) -> Sequence[EquityPoint]:
        """The equity curve, one row per instant of the merged stream."""
        return list(self._curve)

    @property
    def fx_fallback_marks(self) -> int:
        """Marks priced at the entry rate because the instant could not be priced.

        Counted rather than logged for P06's reason: a warning in a run of two
        hundred thousand bars is a line nobody reads, and a stale mark shows up
        downstream as an equity curve that is subtly wrong in one instrument.
        """
        return self._fx_fallback_marks

    def get(self, position_id: str) -> OpenPosition | None:
        """One live position by id.

        Args:
            position_id: Id to look up.

        Returns:
            The position, or ``None`` if it is not open.
        """
        return self._open.get(position_id)

    def positions_on(self, key: StreamKey) -> list[OpenPosition]:
        """Live positions managed on one stream, oldest first.

        Args:
            key: Stream to filter by.

        Returns:
            The positions.
        """
        return [held for held in self._open.values() if held.key == key]

    def open(self, held: OpenPosition) -> None:
        """Record a position that has just been opened, and charge its entry fill.

        Args:
            held: The position, with its entry fill already priced.

        Raises:
            ValueError: If a position with the same id is already open.
        """
        if held.position_id in self._open:
            raise ValueError(f"position {held.position_id} is already open")
        held.last_financing_ts = held.entry_fill.ts
        held.commission_paid += held.entry_fill.commission
        self._commission += held.entry_fill.commission
        self._open[held.position_id] = held

    def book_leg(self, held: OpenPosition, leg: ClosedLeg, fill: Fill) -> Decimal:
        """Book one closing leg's profit and the commission on its fill.

        The leg's profit is converted at **the leg's own fill time**, never at
        one averaged rate for the position: a trade opened in March and closed
        in June lived through a currency move, and that is exactly what an
        average would erase.

        Args:
            held: The position being closed into.
            leg: The realised close, already priced by the execution stack.
            fill: The execution fill behind it, for its commission.

        Returns:
            The leg's gross profit in account currency, signed.
        """
        instrument = self._instrument(held.symbol)
        gross = self._to_account(
            leg.gross_quote_move * instrument.contract_size, instrument, at=leg.ts
        )
        self._realized += gross
        held.realized += gross
        self._commission += fill.commission
        held.commission_paid += fill.commission
        return gross

    def charge_financing(self, held: OpenPosition, amount: Decimal, upto: datetime) -> None:
        """Book financing for a position and advance its accrual mark.

        Args:
            held: The position charged.
            amount: The charge in account currency, signed; negative is paid out.
            upto: Instant the accrual now runs to.
        """
        self._swap += amount
        held.swap_paid += amount
        held.last_financing_ts = ensure_utc(upto)

    def close_out(self, held: OpenPosition, closed_at: datetime) -> TradeRecord:
        """Retire a position that has gone flat and file its trade record.

        Args:
            held: The position, whose P07 side reports no remaining size.
            closed_at: Time of its last closing leg.

        Returns:
            The filed record.

        Raises:
            ValueError: If the position is still open, or was never registered.
        """
        if held.position.is_open:
            raise ValueError(
                f"position {held.position_id} still holds "
                f"{held.position.remaining_size}; it has not gone flat"
            )
        if self._open.pop(held.position_id, None) is None:
            raise ValueError(f"position {held.position_id} is not open")
        record = TradeRecord(
            position_id=held.position_id,
            symbol=held.symbol,
            strategy_id=held.strategy_id,
            side=held.side,
            size=held.position.size,
            opened_at=held.position.opened_at,
            closed_at=ensure_utc(closed_at),
            entry_price=held.position.entry_price,
            quality=held.entry_quality,
            initial_risk_distance=held.position.initial_risk_distance,
            gross=held.realized,
            commission=held.commission_paid,
            swap=held.swap_paid,
            net=held.realized - held.commission_paid + held.swap_paid,
            realized_r=held.position.realized_r(),
            legs=len(held.position.legs),
        )
        self._trades.append(record)
        return record

    def restate_risk(self, held: OpenPosition) -> Decimal:
        """Recompute what a position has at stake, in account currency.

        Measured from the **entry price** to the current stop, not from the
        current mark: P10 states that a stop trailed to breakeven puts nothing at
        risk any more, and that reading is only true against the entry. Clamped
        at zero, because a stop past breakeven risks nothing rather than a
        negative amount. Uses the rate frozen at sizing, so restating never
        silently reprices a decision that was already taken.

        Args:
            held: The position to restate.

        Returns:
            The money at stake, non-negative.
        """
        instrument = self._instrument(held.symbol)
        position = held.position
        signed = (
            position.entry_price - position.stop
            if position.side is Side.BUY
            else position.stop - position.entry_price
        )
        distance = max(signed, 0.0)
        quote_risk = Decimal(str(distance)) * position.remaining_size * instrument.contract_size
        held.risk_amount = quote_risk * held.entry_fx_rate
        return held.risk_amount

    def mark(self, marks: Mapping[str, float], at: datetime) -> Decimal:
        """Mark every open position to the given prices.

        Args:
            marks: Latest closed price per symbol. A symbol absent from the
                mapping keeps its previous contribution — which is what a market
                that has not printed since looks like.
            at: The instant being marked, for the conversion rate.

        Returns:
            The total floating result in account currency.
        """
        total = Decimal(0)
        for held in self._open.values():
            price = marks.get(held.symbol)
            if price is None:
                continue
            total += self._floating(held, Price(price), at=at)
        self._unrealized = total
        return total

    def snapshot(self, at: datetime) -> EquityPoint:
        """Write one equity row for this instant and return it.

        Args:
            at: The instant, which must be the merged stream's.

        Returns:
            The row, which is also appended to :attr:`curve`.
        """
        instant = ensure_utc(at)
        day = (
            trading_day_of_close(instant, self._day_origin, self._calendar)
            if self._calendar is not None
            else trading_day(instant, self._day_origin)
        )
        point = EquityPoint(
            ts=instant,
            day=day,
            balance=self.balance,
            equity=self.equity,
            realized=self._realized,
            unrealized=self._unrealized,
            commission_paid=self._commission,
            swap_paid=self._swap,
            open_positions=len(self._open),
        )
        self._curve.append(point)
        return point

    def account_state(self, at: datetime) -> AccountState:
        """The account as the Risk Engine reads it.

        Args:
            at: The instant of the snapshot.

        Returns:
            Balance, equity and one :class:`~trading_system.risk.models.OpenRisk`
            per live position — the composition the portfolio limits need, not
            just a total.
        """
        return AccountState(
            currency=self._currency,
            balance=self.balance,
            equity=self.equity,
            as_of=ensure_utc(at),
            open_risks=tuple(
                OpenRisk(
                    symbol=held.symbol,
                    strategy_id=held.strategy_id,
                    side=held.side,
                    risk_amount=held.risk_amount,
                )
                for held in self._open.values()
            ),
        )

    def _floating(self, held: OpenPosition, price: Price, *, at: datetime) -> Decimal:
        """Unrealised result of one position at a mark price, account currency."""
        instrument = self._instrument(held.symbol)
        position = held.position
        move = (
            price - position.entry_price
            if position.side is Side.BUY
            else position.entry_price - price
        )
        quote = Decimal(str(move)) * position.remaining_size * instrument.contract_size
        return self._to_account(quote, instrument, at=at, fallback=held.entry_fx_rate)

    def _to_account(
        self,
        quote_amount: Decimal,
        instrument: InstrumentSpec,
        *,
        at: datetime,
        fallback: Decimal | None = None,
    ) -> Decimal:
        """Convert a quote-currency amount at ``at``, optionally with a fallback.

        A booked leg passes no fallback: an unconvertible realisation is a real
        failure and must not be silently valued. A mark passes the rate frozen at
        entry, because an unrealised figure between fills is a display value and
        stopping the run over it would be worse than carrying a stale rate — but
        the fallback is counted.
        """
        if instrument.quote_currency == self._currency:
            return quote_amount
        try:
            rate = self._converter.rate(base=instrument.quote_currency, quote=self._currency, at=at)
        except FxRateUnavailableError:
            if fallback is None:
                raise
            self._fx_fallback_marks += 1
            logger.warning(
                "portfolio.mark_fx_fallback",
                symbol=instrument.symbol,
                pair=f"{instrument.quote_currency}{self._currency}",
                at=at.isoformat(),
                rate=str(fallback),
            )
            rate = fallback
        return quote_amount * rate

    def _instrument(self, symbol: str) -> InstrumentSpec:
        """Look up an instrument, refusing to invent one."""
        instrument = self._instruments.get(symbol)
        if instrument is None:
            raise KeyError(f"{symbol} is not in the instrument registry")
        return instrument
