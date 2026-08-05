"""Fill models: when an order executes, and at what mid price."""

import pytest
from structlog.testing import capture_logs

from tests.execution.conftest import bar
from trading_system.core.instruments import InstrumentSpec
from trading_system.core.types import OrderType, Side
from trading_system.execution.config import GapConfig, LimitFillConfig
from trading_system.execution.fill_model import (
    GapFill,
    LimitTouch,
    NextBarOpen,
    RestingOrderModel,
    SameBarClose,
)


class TestMarketTiming:
    """A decision on ``close(t)`` cannot transact before ``open(t+1)``."""

    def test_next_bar_open_fills_at_the_following_open(self) -> None:
        """The default, and the only timing bar data actually supports."""
        decision = bar(open_=1.1000, high=1.1020, low=1.0990, close=1.1010)
        following = bar(open_=1.1015, high=1.1030, low=1.1005, close=1.1025)
        assert NextBarOpen().reference_price(decision, following) == pytest.approx(1.1015)

    def test_next_bar_open_refuses_the_last_bar_of_a_series(self) -> None:
        """The run ended before the order could execute.

        Falling back to the decision bar's close would put one free trade at the
        end of every backtest, always taken at a price the strategy already knew.
        """
        decision = bar(open_=1.1000, high=1.1020, low=1.0990, close=1.1010)
        assert NextBarOpen().reference_price(decision, None) is None

    def test_same_bar_close_warns_when_it_is_chosen(self) -> None:
        """The optimistic model is required to be noisy about itself.

        Once per instance rather than once per bar: a warning emitted two hundred
        thousand times is a line nobody reads, which is why P06 counts dropped
        signals instead of only logging them.
        """
        with capture_logs() as logs:
            SameBarClose()
        warnings = [entry for entry in logs if entry["log_level"] == "warning"]
        assert len(warnings) == 1
        assert warnings[0]["event"] == "execution.optimistic_fill_model"
        assert warnings[0]["model"] == "SameBarClose"

    def test_the_honest_model_says_nothing(self) -> None:
        """A warning that fires for both models tells a reader nothing."""
        with capture_logs() as logs:
            NextBarOpen()
        assert [entry for entry in logs if entry["log_level"] == "warning"] == []

    def test_same_bar_close_fills_at_the_decision_price(self) -> None:
        """It executes at the very price the decision was derived from."""
        decision = bar(open_=1.1000, high=1.1020, low=1.0990, close=1.1010)
        following = bar(open_=1.1015, high=1.1030, low=1.1005, close=1.1025)
        assert SameBarClose().reference_price(decision, following) == pytest.approx(1.1010)

    def test_the_optimistic_model_fills_better_than_the_honest_one(self) -> None:
        """On a bar that gapped in the trade's favour, and that is the point.

        The difference between the two models is the entire signal-to-fill gap,
        which is where a meaningful share of a short-timeframe edge lives.
        """
        decision = bar(open_=1.1000, high=1.1020, low=1.0990, close=1.1010)
        adverse = bar(open_=1.1018, high=1.1030, low=1.1010, close=1.1025)
        optimistic = SameBarClose().reference_price(decision, adverse)
        honest = NextBarOpen().reference_price(decision, adverse)
        assert optimistic is not None and honest is not None
        assert optimistic < honest  # a buyer pays less under the optimistic model


class TestLimitTouch:
    """A limit fills when price trades past it, not when it merely reaches it."""

    def _model(self, probability: float = 0.0) -> LimitTouch:
        """Build a limit model with a given touch-fill probability.

        Args:
            probability: Chance of filling on an exact touch.

        Returns:
            The model.
        """
        return LimitTouch(
            LimitFillConfig(touch_fill_probability=probability), GapConfig(), run_seed=5
        )

    def test_a_touch_exactly_at_the_level_does_not_fill(self, eurusd: InstrumentSpec) -> None:
        """The DoD boundary. At the level you are behind the whole queue.

        A bar whose extreme landed exactly on the level and turned around is a
        bar in which that queue was never cleared, so crediting the fill hands a
        mean-reversion strategy precisely the entries it would not have got.
        """
        # A sell limit at 1.1020 rests above; the bar's high reaches it exactly.
        touching = bar(open_=1.1000, high=1.1020, low=1.0990, close=1.1010)
        assert (
            self._model().fill(
                1.1020, bar=touching, side=Side.SELL, instrument=eurusd, order_id="o-1"
            )
            is None
        )

    def test_trading_one_tick_past_the_level_does_fill(self, eurusd: InstrumentSpec) -> None:
        """The comparison that makes the previous test a boundary and not a refusal."""
        through = bar(open_=1.1000, high=1.1021, low=1.0990, close=1.1010)
        fill = self._model().fill(
            1.1020, bar=through, side=Side.SELL, instrument=eurusd, order_id="o-1"
        )
        assert fill is not None
        assert fill.mid_price == pytest.approx(1.1020)

    def test_a_buy_limit_is_the_mirror_image(self, eurusd: InstrumentSpec) -> None:
        """A buy limit rests below and needs price to trade beneath it."""
        touching = bar(open_=1.1010, high=1.1020, low=1.0990, close=1.1000)
        through = bar(open_=1.1010, high=1.1020, low=1.0989, close=1.1000)
        model = self._model()
        assert (
            model.fill(1.0990, bar=touching, side=Side.BUY, instrument=eurusd, order_id="o-1")
            is None
        )
        assert (
            model.fill(1.0990, bar=through, side=Side.BUY, instrument=eurusd, order_id="o-1")
            is not None
        )

    def test_a_certain_touch_probability_does_fill(self, eurusd: InstrumentSpec) -> None:
        """The touch is improbable, not impossible, and the probability is configured."""
        touching = bar(open_=1.1000, high=1.1020, low=1.0990, close=1.1010)
        fill = self._model(probability=1.0).fill(
            1.1020, bar=touching, side=Side.SELL, instrument=eurusd, order_id="o-1"
        )
        assert fill is not None

    def test_the_touch_draw_is_reproducible(self, eurusd: InstrumentSpec) -> None:
        """Same order, same answer, run after run."""
        touching = bar(open_=1.1000, high=1.1020, low=1.0990, close=1.1010)
        outcomes = {
            self._model(probability=0.5).fill(
                1.1020, bar=touching, side=Side.SELL, instrument=eurusd, order_id="o-1"
            )
            is not None
            for _ in range(10)
        }
        assert len(outcomes) == 1

    def test_a_gapped_limit_fills_at_its_level_by_default(self, eurusd: InstrumentSpec) -> None:
        """The change to P07's reference model.

        P07 gave a gapped limit the better of level and open. Free positive
        slippage on every gap through a target is the most flattering assumption
        available in the module, so it is off unless a venue is known to work
        that way.
        """
        # A sell limit at 1.1020; the bar opens at 1.1050, well through it.
        gapped = bar(open_=1.1050, high=1.1060, low=1.1040, close=1.1055)
        fill = self._model().fill(
            1.1020, bar=gapped, side=Side.SELL, instrument=eurusd, order_id="o-1"
        )
        assert fill is not None
        assert fill.mid_price == pytest.approx(1.1020)
        assert fill.gap_points == pytest.approx(30.0)

    def test_the_improvement_can_be_switched_on(self, eurusd: InstrumentSpec) -> None:
        """A run that has measured its venue may ask for the old behaviour."""
        model = LimitTouch(
            LimitFillConfig(),
            GapConfig(grant_improvement_to_limits=True),
            run_seed=5,
        )
        gapped = bar(open_=1.1050, high=1.1060, low=1.1040, close=1.1055)
        fill = model.fill(1.1020, bar=gapped, side=Side.SELL, instrument=eurusd, order_id="o-1")
        assert fill is not None
        assert fill.mid_price == pytest.approx(1.1050)


class TestGapFill:
    """A stop that is reached always fills; a gap says where."""

    def test_a_stop_reached_in_the_ordinary_way_fills_at_its_level(
        self, eurusd: InstrumentSpec
    ) -> None:
        """No gap, no penalty, no ambiguity."""
        touching = bar(open_=1.1010, high=1.1020, low=1.0980, close=1.1000)
        fill = GapFill().fill(
            1.0990, bar=touching, side=Side.SELL, instrument=eurusd, order_id="o-1"
        )
        assert fill is not None
        assert fill.mid_price == pytest.approx(1.0990)
        assert fill.gap_points == 0.0

    def test_a_stop_touched_exactly_does_fill(self, eurusd: InstrumentSpec) -> None:
        """Unlike a limit: a stop becomes a market order, it does not queue.

        The asymmetry with :class:`TestLimitTouch` is deliberate and is the
        reason the two are separate models rather than one with a flag.
        """
        touching = bar(open_=1.1010, high=1.1020, low=1.0990, close=1.1000)
        fill = GapFill().fill(
            1.0990, bar=touching, side=Side.SELL, instrument=eurusd, order_id="o-1"
        )
        assert fill is not None

    def test_a_stop_gapped_through_reports_the_hole(self, eurusd: InstrumentSpec) -> None:
        """The long's stop at 1.0990, on a bar that opened forty points below it."""
        gapped = bar(open_=1.0950, high=1.0960, low=1.0940, close=1.0945)
        fill = GapFill().fill(1.0990, bar=gapped, side=Side.SELL, instrument=eurusd, order_id="o-1")
        assert fill is not None
        assert fill.mid_price == pytest.approx(1.0950)
        assert fill.gap_points == pytest.approx(40.0)

    def test_a_level_the_bar_never_reached_produces_no_fill(self, eurusd: InstrumentSpec) -> None:
        """And therefore never a negative gap."""
        away = bar(open_=1.1010, high=1.1020, low=1.1000, close=1.1005)
        assert (
            GapFill().fill(1.0990, bar=away, side=Side.SELL, instrument=eurusd, order_id="o-1")
            is None
        )

    def test_a_short_stop_gaps_upward(self, eurusd: InstrumentSpec) -> None:
        """Mirror image: a short's stop sits above and is gapped through by a rally."""
        gapped = bar(open_=1.1060, high=1.1070, low=1.1055, close=1.1065)
        fill = GapFill().fill(1.1020, bar=gapped, side=Side.BUY, instrument=eurusd, order_id="o-1")
        assert fill is not None
        assert fill.mid_price == pytest.approx(1.1060)
        assert fill.gap_points == pytest.approx(40.0)


class TestRestingOrderModel:
    """Dispatch by order type, and a refusal for the type that does not rest."""

    def test_it_routes_stops_and_limits_to_their_own_models(self, eurusd: InstrumentSpec) -> None:
        """The same touching bar fills a stop and does not fill a limit."""
        model = RestingOrderModel(
            limit=LimitTouch(LimitFillConfig(), GapConfig(), run_seed=1), stop=GapFill()
        )
        touching = bar(open_=1.1010, high=1.1020, low=1.0990, close=1.1000)
        as_stop = model.fill(
            1.0990,
            bar=touching,
            side=Side.SELL,
            order_type=OrderType.STOP,
            instrument=eurusd,
            order_id="o-1",
        )
        as_limit = model.fill(
            1.1020,
            bar=touching,
            side=Side.SELL,
            order_type=OrderType.LIMIT,
            instrument=eurusd,
            order_id="o-1",
        )
        assert as_stop is not None
        assert as_limit is None

    def test_a_market_order_does_not_rest(self, eurusd: InstrumentSpec) -> None:
        """It has no level to rest at, so asking is a caller bug."""
        model = RestingOrderModel(
            limit=LimitTouch(LimitFillConfig(), GapConfig(), run_seed=1), stop=GapFill()
        )
        with pytest.raises(ValueError, match="does not rest"):
            model.fill(
                1.0990,
                bar=bar(open_=1.1, high=1.1, low=1.09, close=1.095),
                side=Side.SELL,
                order_type=OrderType.MARKET,
                instrument=eurusd,
                order_id="o-1",
            )
