"""The order handed to the cost model, and the fill that comes back.

The split between this module and :mod:`trading_system.execution.fill_model` is
the split between two questions that get confused constantly:

* **Where would this have executed, in the price series the backtest reads?**
  That is the fill model's job, and its answer is a *mid* price — the open of
  the next bar, the level a resting order sat at, the price a gap printed at.
* **What does the account actually get, once the spread, the slippage and the
  commission are taken out?** That is the cost model's job, and the mid price is
  its input.

:class:`ExecutionOrder` is the handoff between them: an order that a fill model
has already resolved to a mid price, awaiting costing. It carries
``gap_points`` because the size of the hole a resting order was swept through is
an output of the first question and an input to the second, and there is nowhere
else for it to live.

**Every price in this system is a mid price except :attr:`Fill.price`.** Bars are
mid, so a level derived from a bar — an invalidation from P06, a stop from P10,
a target from P07 — is mid too. The single point at which the model leaves mid
space is the fill, and it leaves it by exactly half a spread. That is what makes
the round turn cost exactly one spread rather than two or none; see
:mod:`trading_system.execution.costs`.
"""

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal

from trading_system.core.types import OrderType, Price, Side, ensure_utc


@dataclass(frozen=True)
class ExecutionOrder:
    """An order resolved to a mid reference price, not yet costed.

    Attributes:
        order_id: Identifier unique within the run. Part of the material the
            per-fill random stream is seeded from, so two orders on the same bar
            of the same instrument draw different slippage — and so re-running
            draws the same values again.
        symbol: Instrument being traded.
        side: Direction of *this order*, not of the position. A ``SELL`` closing
            a long and a ``SELL`` opening a short both cross the spread the same
            way, which is why the cost model needs only this.
        order_type: How the order reached the market. Selects which slippage
            parameters apply — a stop that has just been triggered is a market
            order into a moving book, a limit never crosses the spread at all.
        size: Size in lots, already quantised by the Risk Engine. Positive.
        mid_price: The price a fill model resolved this order to, in mid space.
        gap_points: How far beyond its resting level the market had already
            moved when this order filled, in the instrument's points. Zero for
            market orders and for resting orders the market reached in the
            ordinary way; positive only when a bar opened through the level.
            Never negative — a level the market did not reach produces no order
            at all, not one with a negative gap.
    """

    order_id: str
    symbol: str
    side: Side
    order_type: OrderType
    size: Decimal
    mid_price: Price
    gap_points: float = 0.0

    def __post_init__(self) -> None:
        """Reject an order that cannot be costed meaningfully.

        Raises:
            ValueError: If the size is not positive, the mid price is not
                positive, or ``gap_points`` is negative.
        """
        if self.size <= 0:
            raise ValueError(f"{self.symbol}: order size must be positive, got {self.size}")
        if self.mid_price <= 0:
            raise ValueError(f"{self.symbol}: mid_price must be positive, got {self.mid_price}")
        if self.gap_points < 0:
            raise ValueError(
                f"{self.symbol}: gap_points must not be negative, got {self.gap_points}; "
                "a level the market never reached produces no order, not a negative gap"
            )


@dataclass(frozen=True)
class Fill:
    """What an order actually cost, decomposed.

    The decomposition is the point. A single "fill price" hides which of three
    unrelated mechanisms moved it, and the three behave differently: the spread
    is deterministic and symmetric, slippage is random and one-sided, commission
    is not in price space at all. A backtest that reports only the net price
    cannot answer why a strategy stopped working when the venue changed.

    Attributes:
        order_id: The order this filled.
        symbol: Instrument.
        side: Direction of the order.
        size: Size filled, in lots.
        ts: When it filled, tz-aware UTC.
        mid_price: The pre-cost price the fill model produced.
        price: The executable price, after the half spread and slippage. This
            is the only price in the system that is not a mid.
        spread_points: The **full** modelled spread at this instant. Half of it
            is what moved ``price``; the whole of it is what the round turn
            costs across two fills.
        slippage_points: Deviation beyond the half spread, in points. Signed and
            positive means worse, matching
            :class:`~trading_system.risk.circuit_breakers.SlippageReport`.
            Negative only when a run explicitly enables gap improvement on
            limits.
        commission: Commission for this fill alone, in account currency, already
            normalised to a per-side figure.
        gap_points: Copied from the order, so a report can separate a fill that
            slipped because the book was thin from one that slipped because a
            weekend gap jumped the level.
    """

    order_id: str
    symbol: str
    side: Side
    size: Decimal
    ts: datetime
    mid_price: Price
    price: Price
    spread_points: float
    slippage_points: float
    commission: Decimal
    gap_points: float

    def __post_init__(self) -> None:
        """Normalise the fill time.

        Raises:
            ValueError: If ``ts`` is naive.
        """
        object.__setattr__(self, "ts", ensure_utc(self.ts))

    @property
    def half_spread_points(self) -> float:
        """The share of the spread this one fill paid.

        Exactly half, always. Two fills make a round turn and pay one spread
        between them, which is the invariant the whole module is arranged
        around.
        """
        return self.spread_points / 2

    @property
    def cost_points(self) -> float:
        """Total price-space cost of this fill, in points.

        Commission is deliberately absent: it is in account currency, per lot,
        and adding it to a figure in points would produce a number that is
        neither.
        """
        return self.half_spread_points + self.slippage_points
