"""Fixtures for the execution layer.

The instruments come from the bundled ``configs/instruments.yaml`` wherever the
real numbers are what is being tested, for the reason the risk fixtures give:
a hand-built spec drifts from the shipped one, and then the tests pass against a
universe nobody trades.

The exception is the spread arithmetic. ``EURUSD`` ships with a typical spread of
0.8 points, and the round-turn invariant is far easier to read at exactly 1.0 —
so those tests use a spec derived from the shipped one with that single field
replaced, rather than a spec invented from nothing.
"""

from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

import pytest

from trading_system.core.instruments import InstrumentRegistry, InstrumentSpec, load_instruments
from trading_system.core.types import Bar, OrderType, Price, Side, Volume
from trading_system.execution.config import (
    CostConfig,
    GapConfig,
    LimitFillConfig,
    SlippageConfig,
    SpreadConfig,
)
from trading_system.execution.costs import CostModel
from trading_system.execution.market_state import MarketState
from trading_system.execution.orders import ExecutionOrder

REGISTRY_PATH = Path(__file__).resolve().parents[2] / "configs" / "instruments.yaml"

#: A weekday instant inside the London/New York overlap, so session lookups are
#: deterministic rather than depending on when the suite runs.
OVERLAP_TS = datetime(2024, 3, 6, 14, 0, tzinfo=UTC)


@pytest.fixture(scope="session")
def registry() -> InstrumentRegistry:
    """The bundled instrument registry."""
    return load_instruments(REGISTRY_PATH)


@pytest.fixture
def eurusd(registry: InstrumentRegistry) -> InstrumentSpec:
    """EURUSD with a spread of exactly one point.

    One point is 0.0001 in price and 10 USD per lot, which makes every figure in
    the round-turn tests readable without a calculator.
    """
    return registry["EURUSD"].model_copy(update={"typical_spread_points": 1.0})


@pytest.fixture
def flat_config() -> CostConfig:
    """Costs with the spread alone: no multipliers, no slippage, no jitter.

    Every multiplier is one and every slippage term is zero, so a fill's only
    cost is half a spread. That isolates the invariant under test — a round turn
    costs one spread — from every other mechanism that could coincidentally
    produce a similar number.
    """
    return CostConfig(
        spread=SpreadConfig(
            session_multipliers={},
            off_session_multiplier=1.0,
            volatility_beta=0.0,
        ),
        slippage=SlippageConfig(),
        gap=GapConfig(),
        limit_fill=LimitFillConfig(),
        run_seed=1234,
    )


@pytest.fixture
def flat_model(eurusd: InstrumentSpec, flat_config: CostConfig) -> CostModel:
    """A cost model over the one-point EURUSD, charging spread only."""
    return CostModel({eurusd.symbol: eurusd}, flat_config)


@pytest.fixture
def typical_state() -> MarketState:
    """Ordinary conditions: warmed-up ATR at its own mean, no news."""
    return MarketState(ts=OVERLAP_TS, atr_ratio=1.0)


def order(
    *,
    side: Side,
    mid: float,
    symbol: str = "EURUSD",
    order_type: OrderType = OrderType.MARKET,
    size: str = "1",
    gap_points: float = 0.0,
    order_id: str = "o-1",
) -> ExecutionOrder:
    """Build an execution order without repeating six keyword arguments.

    Args:
        side: Direction of the order.
        mid: Mid price the fill model resolved.
        symbol: Instrument.
        order_type: How the order reached the market.
        size: Size in lots, as a decimal string.
        gap_points: Distance the market had already moved past a resting level.
        order_id: Identifier, which seeds any random draw.

    Returns:
        The order.
    """
    return ExecutionOrder(
        order_id=order_id,
        symbol=symbol,
        side=side,
        order_type=order_type,
        size=Decimal(size),
        mid_price=Price(mid),
        gap_points=gap_points,
    )


def bar(
    *,
    open_: float,
    high: float,
    low: float,
    close: float,
    ts: datetime = OVERLAP_TS,
) -> Bar:
    """Build an OHLCV bar with a fixed timestamp and volume.

    Args:
        open_: Opening price.
        high: Bar high.
        low: Bar low.
        close: Closing price.
        ts: Bar OPEN time.

    Returns:
        The bar.
    """
    return Bar(
        timestamp=ts,
        open=Price(open_),
        high=Price(high),
        low=Price(low),
        close=Price(close),
        volume=Volume(1000.0),
    )
