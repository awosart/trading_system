"""Zero costs against realistic costs, on the same trades, in money.

The comparison this whole stage exists to make possible. A backtest with costs
switched off is not a slightly optimistic backtest — for a strategy whose edge is
measured in a few points it is a different sign. This file runs one fixed
sequence of trades through both models and states the difference as a number, so
that "costs matter" stops being an assertion and becomes an amount.

The realistic side deliberately uses the **shipped** ``configs/execution.yaml``
and ``configs/instruments.yaml`` rather than figures invented here. A test that
proves costs bite under numbers chosen to make them bite proves nothing about the
configuration anybody actually runs.
"""

from decimal import Decimal
from pathlib import Path

import pytest

from tests.execution.conftest import OVERLAP_TS, order
from trading_system.core.instruments import InstrumentRegistry, InstrumentSpec
from trading_system.core.types import Side
from trading_system.execution.config import (
    CostConfig,
    GapConfig,
    SlippageConfig,
    SpreadConfig,
    load_execution_config,
)
from trading_system.execution.costs import CostModel, Fill, realized_points
from trading_system.execution.market_state import MarketState

CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "execution.yaml"

#: Trades per simulated run. Large enough that the per-trade difference is not a
#: rounding artefact, small enough to read.
TRADES = 200

#: Gross edge a scalper is assumed to capture, in points, before any cost.
SCALP_EDGE_POINTS = 2.0

#: Gross edge a swing trade is assumed to capture. The same costs applied to it
#: are a rounding error, which is the contrast that makes the scalping result a
#: statement about scalping rather than about the cost model.
SWING_EDGE_POINTS = 60.0

SIZE = Decimal("1")


def _net_money(entry: Fill, close: Fill, instrument: InstrumentSpec) -> Decimal:
    """Account-currency result of one round turn, after commission.

    EURUSD is quoted in USD and the account is USD, so no conversion is needed
    and the arithmetic stays exact. Swap is absent because these trades open and
    close inside one trading day, which is what makes them scalps.

    Args:
        entry: Opening fill.
        close: Closing fill.
        instrument: Contract specification.

    Returns:
        Net profit, signed.
    """
    gross = realized_points(entry, close, instrument) * instrument.value_per_point_quote * SIZE
    return gross - entry.commission - close.commission


def _run(model: CostModel, instrument: InstrumentSpec, edge_points: float) -> Decimal:
    """Trade a fixed sequence of long round turns and total the result.

    Every trade captures exactly ``edge_points`` of gross movement, so the only
    thing that varies between two runs of this function is the cost model.

    Args:
        model: Cost model to price the fills with.
        instrument: What is being traded.
        edge_points: Gross movement captured per trade, in points.

    Returns:
        Total net profit over :data:`TRADES` trades.
    """
    state = MarketState(ts=OVERLAP_TS, atr_ratio=1.0)
    entry_mid = 1.10000
    exit_mid = entry_mid + instrument.points_to_price(edge_points)

    total = Decimal(0)
    for index in range(TRADES):
        entry = model.apply(order(side=Side.BUY, mid=entry_mid, order_id=f"in-{index}"), state)
        close = model.apply(order(side=Side.SELL, mid=exit_mid, order_id=f"out-{index}"), state)
        total += _net_money(entry, close, instrument)
    return total


@pytest.fixture
def costless(registry: InstrumentRegistry) -> tuple[CostModel, InstrumentSpec]:
    """A run in which trading is free: no spread, no commission, no slippage."""
    instrument = registry["EURUSD"].model_copy(
        update={"typical_spread_points": 0.0, "commission_per_lot": Decimal("0")}
    )
    config = CostConfig(
        spread=SpreadConfig(off_session_multiplier=1.0, volatility_beta=0.0),
        slippage=SlippageConfig(),
        gap=GapConfig(),
    )
    return CostModel({instrument.symbol: instrument}, config), instrument


@pytest.fixture
def realistic(registry: InstrumentRegistry) -> tuple[CostModel, InstrumentSpec]:
    """A run under the shipped instrument and the shipped venue configuration."""
    instrument = registry["EURUSD"]
    config = load_execution_config(CONFIG_PATH).costs
    return CostModel({instrument.symbol: instrument}, config), instrument


class TestScalpingTurnsNegative:
    """The headline: a two-point edge does not survive a real venue."""

    def test_the_costless_run_is_profitable(
        self, costless: tuple[CostModel, InstrumentSpec]
    ) -> None:
        """Two points on EURUSD is 20 USD a lot, two hundred times over."""
        model, instrument = costless
        assert _run(model, instrument, SCALP_EDGE_POINTS) == Decimal("4000.0")

    def test_the_realistic_run_is_a_loss(self, realistic: tuple[CostModel, InstrumentSpec]) -> None:
        """The same trades, the same edge, the opposite sign."""
        model, instrument = realistic
        assert _run(model, instrument, SCALP_EDGE_POINTS) < 0

    def test_the_difference_is_stated_as_an_amount_per_trade(
        self,
        costless: tuple[CostModel, InstrumentSpec],
        realistic: tuple[CostModel, InstrumentSpec],
    ) -> None:
        """24.03 USD a round turn, against a 20.00 USD gross edge.

        Decomposed against the shipped configuration, per lot, and measured
        rather than estimated:

        * **7.20** — spread. 0.8 points, tightened to 0.72 by the London/New York
          overlap multiplier of 0.9, paid once across the two fills.
        * **8.84** — market slippage. 0.2 + 0.15 x 1.0 a fill, plus a uniform
          draw of up to 0.2, so about 0.44 each and 0.88 for the pair.
        * **7.00** — commission. A 7.00 round-turn figure charged as 3.50 on each
          fill. Read as per-side instead it would be 14.00, and the run would
          still look entirely plausible — which is why the basis is a required
          field.
        * **0.99** — tick rounding. Each fill is snapped away from the mid, so
          the adverse move is never shortened; that costs up to a tick a fill and
          averages half of one. Small, systematic, and in one direction — exactly
          the shape of cost that a model rounding to nearest would hand back.

        A little over 24 USD to earn 20. No component is large enough to notice
        alone, which is the point: the failure this stage guards against is not
        one big omission but four small ones, and a backtest missing all four
        reports a 20% return on the same trades that lose 4%.
        """
        costless_model, costless_instrument = costless
        realistic_model, realistic_instrument = realistic

        free = _run(costless_model, costless_instrument, SCALP_EDGE_POINTS)
        real = _run(realistic_model, realistic_instrument, SCALP_EDGE_POINTS)
        per_trade = (free - real) / TRADES

        # Exact, not approximate: every draw is seeded from its own fill, so the
        # figure is reproducible to the cent. A change to the shipped config or
        # to the cost arithmetic should break this and be looked at.
        assert per_trade == Decimal("24.03")
        assert free == Decimal("4000.0")
        assert real == Decimal("-806.0")
        assert free > 0 > real

    def test_the_gross_edge_is_untouched_and_only_costs_differ(
        self,
        costless: tuple[CostModel, InstrumentSpec],
        realistic: tuple[CostModel, InstrumentSpec],
    ) -> None:
        """The two runs differ by costs alone, not by a different set of trades.

        Without this the previous test would also pass if the realistic run had
        quietly stopped taking trades.
        """
        costless_model, _ = costless
        realistic_model, _ = realistic
        _run(costless_model, costless[1], SCALP_EDGE_POINTS)
        _run(realistic_model, realistic[1], SCALP_EDGE_POINTS)
        assert costless_model.stats.fills == realistic_model.stats.fills == TRADES * 2


class TestSwingTradingSurvives:
    """The contrast that keeps the scalping result honest."""

    def test_a_sixty_point_edge_stays_profitable_under_the_same_costs(
        self, realistic: tuple[CostModel, InstrumentSpec]
    ) -> None:
        """The same venue, thirty times the edge, still clearly positive."""
        model, instrument = realistic
        assert _run(model, instrument, SWING_EDGE_POINTS) > 0

    def test_costs_take_the_same_absolute_amount_from_both(
        self,
        costless: tuple[CostModel, InstrumentSpec],
        realistic: tuple[CostModel, InstrumentSpec],
    ) -> None:
        """Costs are per round turn, so they scale with trade count, not edge.

        Stated because it is what makes trade frequency the dominant term: a
        strategy taking ten times as many trades pays ten times as much for the
        same gross movement.
        """
        costless_model, costless_instrument = costless
        realistic_model, realistic_instrument = realistic

        scalp_gap = _run(costless_model, costless_instrument, SCALP_EDGE_POINTS) - _run(
            realistic_model, realistic_instrument, SCALP_EDGE_POINTS
        )
        costless_model.reset()
        realistic_model.reset()
        swing_gap = _run(costless_model, costless_instrument, SWING_EDGE_POINTS) - _run(
            realistic_model, realistic_instrument, SWING_EDGE_POINTS
        )
        assert scalp_gap == pytest.approx(swing_gap, abs=Decimal("1"))
