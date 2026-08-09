"""Fixtures for the Risk Engine tests.

The bundled ``configs/instruments.yaml`` is used rather than a hand-built one:
the DoD numbers are stated against real contract specifications, and a test
registry would let the file and the tests drift apart silently.
"""

from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Protocol

import pytest

from trading_system.core.instruments import (
    InstrumentClass,
    InstrumentRegistry,
    load_instruments,
)
from trading_system.core.types import Price, Side
from trading_system.entry.signal import EntrySignal
from trading_system.risk.circuit_breakers import CircuitBreakerConfig, CircuitBreakers
from trading_system.risk.conversion import StaticFxConverter
from trading_system.risk.correlation import CorrelationProvider
from trading_system.risk.engine import RiskEngine, RiskEngineConfig
from trading_system.risk.margin import PropProfile
from trading_system.risk.models import AccountState
from trading_system.risk.portfolio_risk import PortfolioLimitsConfig, PortfolioRisk
from trading_system.risk.sizing.base import SizingMethod
from trading_system.risk.sizing.methods import FixedFractional
from trading_system.risk.stop_calculator import StopBufferConfig

REGISTRY_PATH = Path(__file__).resolve().parents[2] / "configs" / "instruments.yaml"

#: The instant every fixture prices things at.
NOW = datetime(2024, 3, 5, 12, 0, tzinfo=UTC)

#: USDJPY at the moment of the tests. The GBPJPY hand-computations depend on it,
#: so it is a named constant rather than a number repeated in three places.
USDJPY = Decimal("150")

#: Representative prices, only ever used as an anchor for a stop distance.
PRICES: dict[str, float] = {
    "EURUSD": 1.0850,
    "GBPJPY": 190.00,
    "XAUUSD": 2050.00,
    "NAS100": 18000.0,
    "US30": 39000.0,
    "BTCUSD": 65000.0,
}


@pytest.fixture(scope="session")
def registry() -> InstrumentRegistry:
    """The bundled instrument registry."""
    return load_instruments(REGISTRY_PATH)


@pytest.fixture
def converter() -> StaticFxConverter:
    """A converter that knows only USDJPY, which is all the universe needs.

    Every other instrument in the registry is quoted in USD, so a USD account
    needs no rate for them at all — that asymmetry is the point of question (a)
    and is exercised by leaving the table this bare.
    """
    return StaticFxConverter({("USD", "JPY"): USDJPY})


@pytest.fixture
def account() -> AccountState:
    """A 100k USD account, flat."""
    return AccountState(
        currency="USD",
        balance=Decimal("100000"),
        equity=Decimal("100000"),
        as_of=NOW,
    )


@pytest.fixture
def exact_buffer() -> StopBufferConfig:
    """A buffer of zero, so a stop distance in a test is the distance asserted.

    Not the production default — that pushes the stop past the invalidation
    level by a spread — but a hand-computed size needs the stop to be exactly
    where the test put it.
    """
    return StopBufferConfig(spread_multiple=0.0, fixed_points=0.0, atr_multiple=0.0)


#: Ceilings loose enough never to bind, for tests about something else.
PERMISSIVE_LIMITS = PortfolioLimitsConfig(
    max_portfolio_risk_pct=1.0,
    max_instrument_risk_pct=1.0,
    max_cluster_risk_pct=1.0,
    max_strategy_risk_pct=1.0,
    max_direction_risk_pct=1.0,
)

#: Breakers that never trip, for tests about something else.
NO_BREAKERS = CircuitBreakerConfig(
    max_daily_loss_pct=None, max_weekly_loss_pct=None, max_monthly_loss_pct=None
)

#: Leverage loose enough that margin never binds, for tests about something else.
#:
#: The default here rather than the real rates, and the reason is a finding
#: rather than convenience: several of the hand-computed sizing cases in this
#: directory are genuinely unaffordable at real leverage — 10 lots of gold on a
#: 100k account is 2.05M of notional and 102 500 of margin at 1:20 — and nothing
#: noticed until :mod:`trading_system.risk.margin` gave ``margin_rate`` a
#: reader. Those tests are about the sizing arithmetic, so they keep testing it;
#: that the same cases are refused under a real profile is asserted where it
#: belongs, in ``test_margin.py``.
NO_MARGIN_LIMIT = PropProfile(
    name="test-unconstrained",
    source="test fixture; not a firm",
    leverage=dict.fromkeys(InstrumentClass, 1e6),
    leverage_cap=None,
)


class EngineFactory(Protocol):
    """Builds a :class:`RiskEngine` with the pieces a test wants to vary."""

    def __call__(
        self,
        sizing: SizingMethod | None = None,
        *,
        max_risk_pct: float = 0.02,
        buffer: StopBufferConfig | None = None,
        portfolio: PortfolioRisk | None = None,
        breakers: CircuitBreakers | None = None,
        correlations: CorrelationProvider | None = None,
        prop_profile: PropProfile | None = NO_MARGIN_LIMIT,
    ) -> RiskEngine:
        """Build the engine."""
        ...


@pytest.fixture
def engine_factory(
    registry: InstrumentRegistry,
    converter: StaticFxConverter,
    exact_buffer: StopBufferConfig,
) -> EngineFactory:
    """Build an engine with a given sizing method and cap.

    Portfolio limits and breakers default to inert, so a test about sizing is
    not silently also a test of the portfolio ceilings. Tests that are about
    those pass their own.
    """

    def build(
        sizing: SizingMethod | None = None,
        *,
        max_risk_pct: float = 0.02,
        buffer: StopBufferConfig | None = None,
        portfolio: PortfolioRisk | None = None,
        breakers: CircuitBreakers | None = None,
        correlations: CorrelationProvider | None = None,
        prop_profile: PropProfile | None = NO_MARGIN_LIMIT,
    ) -> RiskEngine:
        return RiskEngine(
            instruments=registry,
            sizing=sizing if sizing is not None else FixedFractional(0.005),
            converter=converter,
            config=RiskEngineConfig(
                max_risk_pct=max_risk_pct,
                stop_buffer=buffer if buffer is not None else exact_buffer,
                prop_profile=prop_profile,
            ),
            portfolio=portfolio if portfolio is not None else PortfolioRisk(PERMISSIVE_LIMITS),
            breakers=breakers if breakers is not None else CircuitBreakers(NO_BREAKERS),
            correlations=correlations,
        )

    return build


def signal_with_stop_points(
    registry: InstrumentRegistry,
    symbol: str,
    *,
    points: float,
    side: Side = Side.BUY,
    quality: float = 0.8,
) -> EntrySignal:
    """Build a signal whose invalidation sits exactly ``points`` from the entry.

    Args:
        registry: Instrument registry, for the point size.
        symbol: Instrument to trade.
        points: Distance from entry to invalidation, in the instrument's points.
        side: Direction to trade.
        quality: Signal confidence.

    Returns:
        The signal.
    """
    instrument = registry[symbol]
    reference = PRICES[symbol]
    distance = instrument.points_to_price(points)
    invalidation = reference - distance if side is Side.BUY else reference + distance
    return EntrySignal(
        strategy_id="test-strategy",
        symbol=symbol,
        bar_close_ts=NOW,
        side=side,
        reference_price=Price(reference),
        invalidation_price=Price(invalidation),
        quality=quality,
    )
