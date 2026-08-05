"""Slippage: the asymmetry between order types, and reproducibility per fill."""

from datetime import timedelta

import pytest

from tests.execution.conftest import OVERLAP_TS, order
from trading_system.core.instruments import InstrumentSpec
from trading_system.core.types import OrderType, Side
from trading_system.execution.config import (
    CostConfig,
    GapConfig,
    SlippageConfig,
    SlippageParams,
    SpreadConfig,
)
from trading_system.execution.costs import CostModel
from trading_system.execution.market_state import MarketState, NewsSeverity
from trading_system.execution.rng import fill_seed

VENUE = SlippageConfig(
    stop=SlippageParams(
        base_points=0.6,
        atr_coefficient=0.4,
        jitter_points=0.5,
        news_points={NewsSeverity.HIGH: 4.0},
    ),
    market=SlippageParams(
        base_points=0.2,
        atr_coefficient=0.15,
        jitter_points=0.2,
        news_points={NewsSeverity.HIGH: 1.5},
    ),
    limit=SlippageParams(),
)


#: The same profile with the random term switched off. Used wherever a test
#: compares two fills against each other: two orders differ in ``order_id``, so
#: they seed different streams by design, and a jitter of up to half a point
#: would swamp the effect under test. Turning it off isolates the term being
#: measured instead of tolerating it with a wide approximation.
DETERMINISTIC_VENUE = SlippageConfig(
    stop=VENUE.stop.model_copy(update={"jitter_points": 0.0}),
    market=VENUE.market.model_copy(update={"jitter_points": 0.0}),
    limit=VENUE.limit,
)


def _model(
    instrument: InstrumentSpec,
    *,
    seed: int = 42,
    gap: GapConfig | None = None,
    slippage: SlippageConfig | None = None,
) -> CostModel:
    """A cost model with the venue slippage profile and no spread widening.

    Args:
        instrument: The instrument to trade.
        seed: Run seed.
        gap: Gap penalty configuration, defaulting to none.
        slippage: Slippage profile, defaulting to the jittered venue.

    Returns:
        The model.
    """
    return CostModel(
        {instrument.symbol: instrument},
        CostConfig(
            spread=SpreadConfig(off_session_multiplier=1.0, volatility_beta=0.0),
            slippage=slippage or VENUE,
            gap=gap or GapConfig(),
            run_seed=seed,
        ),
    )


class TestStopsSlipMoreThanLimits:
    """A triggered stop enters a book already moving away; a limit never crosses."""

    def test_a_stop_order_slips_more_than_a_limit_order(
        self, eurusd: InstrumentSpec, typical_state: MarketState
    ) -> None:
        """The headline asymmetry, on identical orders differing only in type."""
        model = _model(eurusd, slippage=DETERMINISTIC_VENUE)
        stop = model.apply(order(side=Side.SELL, mid=1.1, order_type=OrderType.STOP), typical_state)
        limit = model.apply(
            order(side=Side.SELL, mid=1.1, order_type=OrderType.LIMIT, order_id="o-2"),
            typical_state,
        )
        assert stop.slippage_points > limit.slippage_points
        assert limit.slippage_points == pytest.approx(0.0)

    def test_a_stop_order_slips_more_than_a_market_order(
        self, eurusd: InstrumentSpec, typical_state: MarketState
    ) -> None:
        """A stop is not just a market order with a trigger price."""
        model = _model(eurusd, slippage=DETERMINISTIC_VENUE)
        stop = model.apply(order(side=Side.SELL, mid=1.1, order_type=OrderType.STOP), typical_state)
        market = model.apply(
            order(side=Side.SELL, mid=1.1, order_type=OrderType.MARKET, order_id="o-2"),
            typical_state,
        )
        assert stop.slippage_points > market.slippage_points

    def test_a_worse_fill_price_follows_from_the_worse_slippage(
        self, eurusd: InstrumentSpec, typical_state: MarketState
    ) -> None:
        """The asymmetry reaches the price, not only the reported figure."""
        model = _model(eurusd, slippage=DETERMINISTIC_VENUE)
        stop = model.apply(order(side=Side.SELL, mid=1.1, order_type=OrderType.STOP), typical_state)
        limit = model.apply(
            order(side=Side.SELL, mid=1.1, order_type=OrderType.LIMIT, order_id="o-2"),
            typical_state,
        )
        # Both are sells: a lower price is the worse fill.
        assert stop.price < limit.price

    def test_a_transposed_config_does_not_load(self) -> None:
        """Pasting the stop parameters into the limit slot is a config error.

        The most flattering single mistake available in this module: it makes
        breakout strategies look cheap to run.
        """
        with pytest.raises(ValueError, match="stop >= market >= limit"):
            SlippageConfig(
                stop=SlippageParams(base_points=0.0),
                market=SlippageParams(base_points=0.2),
                limit=SlippageParams(base_points=0.6),
            )

    def test_volatility_and_news_both_increase_slippage(self, eurusd: InstrumentSpec) -> None:
        """Both terms of ``base + k * atr_ratio + news`` move the right way."""
        model = _model(eurusd, slippage=DETERMINISTIC_VENUE)
        calm = model.apply(
            order(side=Side.SELL, mid=1.1, order_type=OrderType.STOP),
            MarketState(ts=OVERLAP_TS, atr_ratio=1.0),
        )
        volatile = model.apply(
            order(side=Side.SELL, mid=1.1, order_type=OrderType.STOP, order_id="o-2"),
            MarketState(ts=OVERLAP_TS, atr_ratio=3.0),
        )
        newsy = model.apply(
            order(side=Side.SELL, mid=1.1, order_type=OrderType.STOP, order_id="o-3"),
            MarketState(ts=OVERLAP_TS, atr_ratio=1.0, news_severity=NewsSeverity.HIGH),
        )
        assert volatile.slippage_points > calm.slippage_points
        assert newsy.slippage_points > calm.slippage_points


class TestGapPenalty:
    """A stop swept through a hole fills worse than the first printable price."""

    def test_a_gap_adds_slippage_proportional_to_the_hole(
        self, eurusd: InstrumentSpec, typical_state: MarketState
    ) -> None:
        """Forty points of gap costs more than five, and not by a constant."""
        model = _model(
            eurusd,
            gap=GapConfig(penalty_base_points=0.5, penalty_fraction=0.1),
            slippage=DETERMINISTIC_VENUE,
        )
        small = model.apply(
            order(side=Side.SELL, mid=1.1, order_type=OrderType.STOP, gap_points=5.0),
            typical_state,
        )
        large = model.apply(
            order(
                side=Side.SELL, mid=1.1, order_type=OrderType.STOP, gap_points=40.0, order_id="o-2"
            ),
            typical_state,
        )
        extra_small = small.slippage_points
        extra_large = large.slippage_points
        # 0.5 + 0.1*5 = 1.0 against 0.5 + 0.1*40 = 4.5: a difference of 3.5,
        # which a flat penalty would have reported as zero.
        assert extra_large - extra_small == pytest.approx(3.5)

    def test_no_gap_means_no_penalty(
        self, eurusd: InstrumentSpec, typical_state: MarketState
    ) -> None:
        """An ordinary fill at its level pays nothing extra."""
        model = _model(
            eurusd,
            gap=GapConfig(penalty_base_points=0.5, penalty_fraction=0.1),
            slippage=DETERMINISTIC_VENUE,
        )
        ordinary = model.apply(
            order(side=Side.SELL, mid=1.1, order_type=OrderType.STOP), typical_state
        )
        gapped = model.apply(
            order(
                side=Side.SELL, mid=1.1, order_type=OrderType.STOP, gap_points=1.0, order_id="o-2"
            ),
            typical_state,
        )
        assert gapped.slippage_points - ordinary.slippage_points == pytest.approx(0.6)

    def test_a_gapped_limit_pays_no_penalty(
        self, eurusd: InstrumentSpec, typical_state: MarketState
    ) -> None:
        """A limit is not made worse by a gap it was already resting through."""
        model = _model(
            eurusd,
            gap=GapConfig(penalty_base_points=0.5, penalty_fraction=0.1),
            slippage=DETERMINISTIC_VENUE,
        )
        fill = model.apply(
            order(side=Side.SELL, mid=1.1, order_type=OrderType.LIMIT, gap_points=40.0),
            typical_state,
        )
        assert fill.slippage_points == pytest.approx(0.0)

    def test_the_gap_size_survives_into_the_fill(
        self, eurusd: InstrumentSpec, typical_state: MarketState
    ) -> None:
        """A report can separate a thin book from a weekend gap."""
        model = _model(eurusd, gap=GapConfig(penalty_base_points=0.5), slippage=DETERMINISTIC_VENUE)
        fill = model.apply(
            order(side=Side.SELL, mid=1.1, order_type=OrderType.STOP, gap_points=40.0),
            typical_state,
        )
        assert fill.gap_points == 40.0


class TestReproducibility:
    """The seed is per fill, so a run's composition cannot change its own fills."""

    def test_the_same_seed_gives_the_same_slippage(
        self, eurusd: InstrumentSpec, typical_state: MarketState
    ) -> None:
        """Two identically configured models agree fill for fill."""
        first = _model(eurusd, seed=7)
        second = _model(eurusd, seed=7)
        for index in range(20):
            spec = order(side=Side.SELL, mid=1.1, order_type=OrderType.STOP, order_id=f"o-{index}")
            assert first.apply(spec, typical_state).price == second.apply(spec, typical_state).price

    def test_a_different_seed_gives_different_slippage(
        self, eurusd: InstrumentSpec, typical_state: MarketState
    ) -> None:
        """Otherwise the seed is not doing anything."""
        spec = order(side=Side.SELL, mid=1.1, order_type=OrderType.STOP)
        drawn = {
            _model(eurusd, seed=seed).apply(spec, typical_state).slippage_points
            for seed in range(12)
        }
        assert len(drawn) > 1

    def test_adding_a_second_instrument_does_not_change_the_first_ones_fills(
        self, eurusd: InstrumentSpec, registry: InstrumentSpec, typical_state: MarketState
    ) -> None:
        """The reason the stream is keyed by fill identity rather than shared.

        A single generator advanced once per fill makes each draw depend on how
        many fills preceded it, so adding an instrument to a portfolio run
        silently rewrites the equity curve of every other instrument. Here the
        EURUSD fills are computed once alone and once interleaved with NAS100,
        and the two must be identical.
        """
        nas = registry["NAS100"]
        config = CostConfig(
            spread=SpreadConfig(off_session_multiplier=1.0, volatility_beta=0.0),
            slippage=VENUE,
            run_seed=99,
        )
        solo = CostModel({eurusd.symbol: eurusd}, config)
        portfolio = CostModel({eurusd.symbol: eurusd, nas.symbol: nas}, config)

        alone: list[float] = []
        interleaved: list[float] = []
        for index in range(10):
            eur = order(side=Side.SELL, mid=1.1, order_type=OrderType.STOP, order_id=f"eur-{index}")
            alone.append(solo.apply(eur, typical_state).price)
            # The portfolio run prices a NAS100 order between every EURUSD one,
            # which is exactly what a shared stream would be disturbed by.
            portfolio.apply(
                order(
                    side=Side.BUY,
                    mid=18000.0,
                    symbol="NAS100",
                    order_type=OrderType.STOP,
                    order_id=f"nas-{index}",
                ),
                typical_state,
            )
            interleaved.append(portfolio.apply(eur, typical_state).price)

        assert alone == interleaved

    def test_the_seed_depends_on_every_component_and_not_on_call_order(self) -> None:
        """Identity, not position in a stream."""
        base = {"symbol": "EURUSD", "ts": OVERLAP_TS, "order_id": "o-1"}
        seed = fill_seed(1, **base)
        assert fill_seed(1, **base) == seed
        assert fill_seed(2, **base) != seed
        assert fill_seed(1, **{**base, "symbol": "NAS100"}) != seed
        assert fill_seed(1, **{**base, "order_id": "o-2"}) != seed
        assert fill_seed(1, **{**base, "ts": OVERLAP_TS + timedelta(minutes=1)}) != seed

    def test_jitter_stays_inside_its_configured_width(
        self, eurusd: InstrumentSpec, typical_state: MarketState
    ) -> None:
        """The random term widens the cost and never narrows it below the base."""
        model = _model(eurusd)
        floor = VENUE.stop.base_points + VENUE.stop.atr_coefficient * 1.0
        for index in range(200):
            fill = model.apply(
                order(side=Side.SELL, mid=1.1, order_type=OrderType.STOP, order_id=f"o-{index}"),
                typical_state,
            )
            assert floor <= fill.slippage_points <= floor + VENUE.stop.jitter_points
