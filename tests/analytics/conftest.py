"""Shared builders for analytics tests.

Real :class:`~trading_system.backtest.portfolio.EquityPoint` and
:class:`~trading_system.backtest.portfolio.TradeRecord` objects, not
analytics-only stand-ins: ``metrics.py``'s contract is against the actual
backtest types, and a hand-rolled look-alike could drift from their real
shape without any test noticing.
"""

from collections.abc import Sequence
from datetime import UTC, date, datetime, time, timedelta
from decimal import Decimal

from trading_system.analytics.metrics import DailyCurve
from trading_system.backtest.portfolio import EquityPoint, TradeRecord
from trading_system.core.types import Price, Side


def point(
    day: date,
    equity: float,
    *,
    ts: datetime | None = None,
    balance: float | None = None,
    open_positions: int = 0,
) -> EquityPoint:
    """One equity-curve row. ``balance`` defaults to ``equity`` (nothing open)."""
    resolved_ts = ts if ts is not None else datetime.combine(day, time(0, 0), tzinfo=UTC)
    resolved_balance = balance if balance is not None else equity
    return EquityPoint(
        ts=resolved_ts,
        day=day,
        balance=Decimal(str(resolved_balance)),
        equity=Decimal(str(equity)),
        realized=Decimal(0),
        unrealized=Decimal(0),
        commission_paid=Decimal(0),
        swap_paid=Decimal(0),
        open_positions=open_positions,
    )


def trade(
    *,
    position_id: str = "t",
    symbol: str = "EURUSD",
    strategy_id: str = "s",
    side: Side = Side.BUY,
    size: Decimal = Decimal("1"),
    opened_at: datetime,
    closed_at: datetime,
    entry_price: float = 1.1,
    net: float,
    realized_r: float,
    legs: int = 1,
) -> TradeRecord:
    """One completed trade. ``gross`` mirrors ``net``; commission and swap are zero."""
    return TradeRecord(
        position_id=position_id,
        symbol=symbol,
        strategy_id=strategy_id,
        side=side,
        size=size,
        opened_at=opened_at,
        closed_at=closed_at,
        entry_price=Price(entry_price),
        gross=Decimal(str(net)),
        commission=Decimal(0),
        swap=Decimal(0),
        net=Decimal(str(net)),
        realized_r=realized_r,
        legs=legs,
    )


def curve_from_returns(
    returns: Sequence[Decimal],
    *,
    start: Decimal = Decimal(100),
    start_day: date = date(2024, 1, 1),
) -> DailyCurve:
    """A :class:`DailyCurve` whose ``simple_returns`` are exactly ``returns``.

    Built by exact :class:`~decimal.Decimal` compounding — ``100 * 1.05`` is
    exact in decimal arithmetic where it would carry float rounding noise, so
    the returns a test hand-picks come back out unchanged.

    Args:
        returns: Simple returns, oldest first.
        start: Equity of day zero.
        start_day: Label of day zero. Consecutive calendar days follow.

    Returns:
        The curve.
    """
    equity = [start]
    for r in returns:
        equity.append(equity[-1] * (1 + r))
    days = tuple(start_day + timedelta(days=i) for i in range(len(equity)))
    return DailyCurve(days=days, equity=tuple(equity), balance=tuple(equity))
