"""The spread and commission arithmetic, stated in numbers rather than prose."""

from decimal import Decimal

import pytest

from tests.execution.conftest import OVERLAP_TS, order
from trading_system.core.instruments import CommissionBasis, InstrumentSpec
from trading_system.core.types import OrderType, Side
from trading_system.data.sessions import Session
from trading_system.execution.config import CostConfig, SpreadConfig, SpreadSource
from trading_system.execution.costs import CostDegradation, CostModel, realized_points
from trading_system.execution.market_state import MarketState, NewsSeverity

#: A point of EURUSD, per lot, in quote currency: point_size * contract_size.
POINT_VALUE_EURUSD = Decimal("10")


class TestRoundTurnCostsExactlyOneSpread:
    """The invariant the whole cost module is arranged around.

    Three wrong implementations all produce plausible equity curves, so each one
    is named and excluded by a number rather than by an argument.
    """

    def test_a_long_round_turn_in_a_flat_market_costs_exactly_one_point(
        self, flat_model: CostModel, eurusd: InstrumentSpec, typical_state: MarketState
    ) -> None:
        """Buy at 1.10005, sell at 1.09995: one point, ten dollars, per lot."""
        entry = flat_model.apply(order(side=Side.BUY, mid=1.10000), typical_state)
        close = flat_model.apply(order(side=Side.SELL, mid=1.10000, order_id="o-2"), typical_state)

        assert entry.price == pytest.approx(1.10005)
        assert close.price == pytest.approx(1.09995)
        assert realized_points(entry, close, eurusd) == Decimal("-1")
        assert realized_points(entry, close, eurusd) * POINT_VALUE_EURUSD == Decimal("-10")

    def test_a_short_round_turn_costs_the_same_one_point(
        self, flat_model: CostModel, eurusd: InstrumentSpec, typical_state: MarketState
    ) -> None:
        """Mirror image, and the assertion that catches the asymmetric bug.

        Charging the whole spread on whichever fill happens to be a ``BUY`` gives
        one point on a long and nothing on a short. Averaged over a balanced
        sample of trades that is the right answer, so only comparing the two
        directions against each other finds it.
        """
        entry = flat_model.apply(order(side=Side.SELL, mid=1.10000), typical_state)
        close = flat_model.apply(order(side=Side.BUY, mid=1.10000, order_id="o-2"), typical_state)

        assert entry.price == pytest.approx(1.09995)
        assert close.price == pytest.approx(1.10005)
        assert realized_points(entry, close, eurusd) == Decimal("-1")

    def test_long_and_short_round_turns_are_equal(
        self, flat_model: CostModel, eurusd: InstrumentSpec, typical_state: MarketState
    ) -> None:
        """Neither direction is cheaper to trade in a market that has not moved."""
        long_cost = realized_points(
            flat_model.apply(order(side=Side.BUY, mid=1.10000, order_id="a"), typical_state),
            flat_model.apply(order(side=Side.SELL, mid=1.10000, order_id="b"), typical_state),
            eurusd,
        )
        short_cost = realized_points(
            flat_model.apply(order(side=Side.SELL, mid=1.10000, order_id="c"), typical_state),
            flat_model.apply(order(side=Side.BUY, mid=1.10000, order_id="d"), typical_state),
            eurusd,
        )
        assert long_cost == short_cost == Decimal("-1")

    def test_it_is_not_half_a_point(
        self, flat_model: CostModel, eurusd: InstrumentSpec, typical_state: MarketState
    ) -> None:
        """Excludes charging the half spread on entry only."""
        entry = flat_model.apply(order(side=Side.BUY, mid=1.10000), typical_state)
        close = flat_model.apply(order(side=Side.SELL, mid=1.10000, order_id="o-2"), typical_state)
        assert realized_points(entry, close, eurusd) != Decimal("-0.5")

    def test_it_is_not_two_points(
        self, flat_model: CostModel, eurusd: InstrumentSpec, typical_state: MarketState
    ) -> None:
        """Excludes charging the whole spread on both fills."""
        entry = flat_model.apply(order(side=Side.BUY, mid=1.10000), typical_state)
        close = flat_model.apply(order(side=Side.SELL, mid=1.10000, order_id="o-2"), typical_state)
        assert realized_points(entry, close, eurusd) != Decimal("-2")

    def test_each_fill_pays_exactly_half(
        self, flat_model: CostModel, typical_state: MarketState
    ) -> None:
        """The decomposition agrees with the round-turn figure it produces."""
        fill = flat_model.apply(order(side=Side.BUY, mid=1.10000), typical_state)
        assert fill.spread_points == pytest.approx(1.0)
        assert fill.half_spread_points == pytest.approx(0.5)
        assert fill.slippage_points == pytest.approx(0.0)

    def test_a_partial_ladder_still_pays_one_spread_in_total(
        self, flat_model: CostModel, eurusd: InstrumentSpec, typical_state: MarketState
    ) -> None:
        """Halving survives P07's ladders; charging it all at entry would not.

        Three closes of 50%, 25% and 25% each pay half a spread on their own
        fraction. Weighted by fraction the exits pay half a spread between them,
        matching the half the entry paid on the full size.
        """
        entry = flat_model.apply(order(side=Side.BUY, mid=1.10000, size="1"), typical_state)
        weighted = Decimal(0)
        for index, fraction in enumerate((Decimal("0.5"), Decimal("0.25"), Decimal("0.25"))):
            leg = flat_model.apply(
                order(side=Side.SELL, mid=1.10000, order_id=f"x{index}", size=str(fraction)),
                typical_state,
            )
            weighted += fraction * realized_points(entry, leg, eurusd)
        assert weighted == Decimal("-1")


class TestRoundingIsAgainstTheTrader:
    """A fill is snapped away from the mid, so the adverse move is never shortened."""

    def test_a_half_spread_landing_on_the_grid_costs_exactly_that(
        self, flat_model: CostModel, typical_state: MarketState
    ) -> None:
        """On EURUSD a point is ten ticks, so half a one-point spread is exact.

        The reason the arithmetic inside ``shift_price`` is decimal: as floats,
        ``1.1 + 0.00005`` is ``1.1000500000000001``, and snapping that away from
        the mid rounds up a whole tick. The round turn would then cost one spread
        plus two ticks, for no reason a reader could ever locate.
        """
        fill = flat_model.apply(order(side=Side.BUY, mid=1.10000), typical_state)
        assert fill.price == pytest.approx(1.10005)

    def test_an_off_grid_half_spread_rounds_away_from_the_mid(
        self, eurusd: InstrumentSpec, typical_state: MarketState
    ) -> None:
        """Never to nearest: rounding back hands the trader up to a tick a fill.

        A 0.3-point spread is 0.15 points a side, which is one and a half ticks —
        so the fill has to move two, not one.
        """
        narrow = eurusd.model_copy(update={"typical_spread_points": 0.3})
        model = CostModel(
            {narrow.symbol: narrow},
            CostConfig(spread=SpreadConfig(off_session_multiplier=1.0, volatility_beta=0.0)),
        )
        buy = model.apply(order(side=Side.BUY, mid=1.10000), typical_state)
        sell = model.apply(order(side=Side.SELL, mid=1.10000, order_id="o-2"), typical_state)
        assert buy.price == pytest.approx(1.10002)
        assert sell.price == pytest.approx(1.09998)

    def test_a_round_turn_never_costs_less_than_the_spread(
        self, eurusd: InstrumentSpec, typical_state: MarketState
    ) -> None:
        """The exact invariant, across spreads that do and do not fit the grid.

        At least the spread, over it by at most one tick a fill. Anything cheaper
        than the spread would mean rounding gave something back.
        """
        tick_points = Decimal(str(eurusd.tick_size)) / Decimal(str(eurusd.point_size))
        for spread in ("0.3", "0.8", "1.0", "1.7", "2.5"):
            instrument = eurusd.model_copy(update={"typical_spread_points": float(spread)})
            model = CostModel(
                {instrument.symbol: instrument},
                CostConfig(spread=SpreadConfig(off_session_multiplier=1.0, volatility_beta=0.0)),
            )
            entry = model.apply(order(side=Side.BUY, mid=1.10000), typical_state)
            close = model.apply(order(side=Side.SELL, mid=1.10000, order_id="o-2"), typical_state)
            cost = -realized_points(entry, close, instrument)
            assert Decimal(spread) <= cost <= Decimal(spread) + 2 * tick_points


class TestCommissionBasis:
    """The factor-of-two error, isolated to one property and one line."""

    def test_round_turn_basis_charges_half_on_each_fill(self, eurusd: InstrumentSpec) -> None:
        """A 7.00 round-turn figure is 3.50 per side, 7.00 for the pair."""
        spec = eurusd.model_copy(
            update={
                "commission_per_lot": Decimal("7.00"),
                "commission_basis": CommissionBasis.ROUND_TURN,
            }
        )
        assert spec.commission_per_side == Decimal("3.50")

    def test_per_side_basis_charges_the_whole_figure_on_each_fill(
        self, eurusd: InstrumentSpec
    ) -> None:
        """The same 7.00 read per side costs 14.00 for the round turn."""
        spec = eurusd.model_copy(
            update={
                "commission_per_lot": Decimal("7.00"),
                "commission_basis": CommissionBasis.PER_SIDE,
            }
        )
        assert spec.commission_per_side == Decimal("7.00")

    def test_the_two_bases_differ_by_exactly_two(self, eurusd: InstrumentSpec) -> None:
        """Stated as a ratio, because that is the size of the error being guarded."""
        per_side = eurusd.model_copy(
            update={"commission_basis": CommissionBasis.PER_SIDE}
        ).commission_per_side
        round_turn = eurusd.model_copy(
            update={"commission_basis": CommissionBasis.ROUND_TURN}
        ).commission_per_side
        assert per_side == round_turn * 2

    def test_the_field_has_no_default(self) -> None:
        """A registry entry that omits the basis does not load.

        The point of the whole change: an absent basis must be a load error, not
        an inherited guess, because both guesses produce believable backtests.
        """
        assert InstrumentSpec.model_fields["commission_basis"].is_required()

    def test_commission_scales_with_size(
        self, flat_model: CostModel, typical_state: MarketState
    ) -> None:
        """Two lots cost twice one lot, per side."""
        one = flat_model.apply(order(side=Side.BUY, mid=1.1, size="1"), typical_state)
        two = flat_model.apply(
            order(side=Side.BUY, mid=1.1, size="2", order_id="o-2"), typical_state
        )
        assert two.commission == one.commission * 2
        assert one.commission == Decimal("3.50")


class TestSpreadWidening:
    """Multipliers come from config and compose multiplicatively."""

    def test_an_off_session_fill_is_wider_than_an_overlap_fill(
        self, eurusd: InstrumentSpec
    ) -> None:
        """The rollover hour costs more than the London/New York overlap."""
        config = CostConfig(
            spread=SpreadConfig(
                session_multipliers={Session.LONDON_NY_OVERLAP: 0.9},
                off_session_multiplier=1.8,
                volatility_beta=0.0,
            )
        )
        model = CostModel({eurusd.symbol: eurusd}, config)
        overlap = model.apply(order(side=Side.BUY, mid=1.1), MarketState(ts=OVERLAP_TS))
        # 03:00 UTC on the same day: Tokyo and Sydney are open, neither configured.
        quiet = MarketState(ts=OVERLAP_TS.replace(hour=3))
        off = model.apply(order(side=Side.BUY, mid=1.1, order_id="o-2"), quiet)

        assert overlap.spread_points == pytest.approx(0.9)
        assert off.spread_points == pytest.approx(1.8)

    def test_liquidity_adds_so_the_tightest_open_session_wins(self) -> None:
        """Resolution is a minimum, so an extra open session can only narrow."""
        config = SpreadConfig(
            session_multipliers={
                Session.LONDON: 1.0,
                Session.NEWYORK: 1.05,
                Session.LONDON_NY_OVERLAP: 0.9,
            }
        )
        both = frozenset({Session.LONDON, Session.NEWYORK, Session.LONDON_NY_OVERLAP})
        assert config.session_multiplier(both) == 0.9
        assert config.session_multiplier(frozenset({Session.LONDON})) == 1.0

    def test_volatility_is_neutral_at_the_typical_ratio(self) -> None:
        """The multiplier is exactly one at ``atr_ratio == 1`` for any beta."""
        for beta in (0.0, 0.6, 3.0):
            config = SpreadConfig(volatility_beta=beta, volatility_multiplier_max=10.0)
            assert config.volatility_multiplier(1.0) == pytest.approx(1.0)

    def test_volatility_widening_is_clamped(self) -> None:
        """A spike is not unbounded permission to widen."""
        config = SpreadConfig(
            volatility_beta=1.0, volatility_multiplier_min=0.8, volatility_multiplier_max=4.0
        )
        assert config.volatility_multiplier(100.0) == 4.0
        assert config.volatility_multiplier(0.01) == 0.8

    def test_news_widens_the_spread(self, eurusd: InstrumentSpec) -> None:
        """A news grade multiplies on top of session and volatility."""
        config = CostConfig(
            spread=SpreadConfig(
                off_session_multiplier=1.0,
                volatility_beta=0.0,
                news_multipliers={NewsSeverity.NONE: 1.0, NewsSeverity.HIGH: 4.0},
            )
        )
        model = CostModel({eurusd.symbol: eurusd}, config)
        state = MarketState(ts=OVERLAP_TS, atr_ratio=1.0, news_severity=NewsSeverity.HIGH)
        assert model.apply(order(side=Side.BUY, mid=1.1), state).spread_points == pytest.approx(4.0)


class TestQuotedSpreadTakesPriority:
    """A real observed spread beats the instrument's representative figure."""

    def test_an_observed_spread_is_used_when_the_source_asks_for_it(
        self, eurusd: InstrumentSpec
    ) -> None:
        """The typical 1.0 gives way to the observed 2.5."""
        config = CostConfig(
            spread=SpreadConfig(
                source=SpreadSource.QUOTED, off_session_multiplier=1.0, volatility_beta=0.0
            )
        )
        model = CostModel({eurusd.symbol: eurusd}, config)
        state = MarketState(ts=OVERLAP_TS, atr_ratio=1.0, quoted_spread_points=2.5)
        assert model.apply(order(side=Side.BUY, mid=1.1), state).spread_points == pytest.approx(2.5)

    def test_a_missing_observation_degrades_to_typical_and_is_counted(
        self, eurusd: InstrumentSpec
    ) -> None:
        """Falling back is allowed; falling back silently is not."""
        config = CostConfig(
            spread=SpreadConfig(
                source=SpreadSource.QUOTED, off_session_multiplier=1.0, volatility_beta=0.0
            )
        )
        model = CostModel({eurusd.symbol: eurusd}, config)
        fill = model.apply(order(side=Side.BUY, mid=1.1), MarketState(ts=OVERLAP_TS, atr_ratio=1.0))
        assert fill.spread_points == pytest.approx(1.0)
        assert model.stats.degradations[CostDegradation.QUOTED_SPREAD_UNAVAILABLE] == 1

    def test_quote_aware_refuses_to_construct(self, eurusd: InstrumentSpec) -> None:
        """The source is declarable before it is implemented, and says so."""
        config = CostConfig(spread=SpreadConfig(source=SpreadSource.QUOTE_AWARE))
        with pytest.raises(NotImplementedError, match="QUOTE_AWARE"):
            CostModel({eurusd.symbol: eurusd}, config)


class TestDegradationIsReportedAsAShare:
    """A count without a denominator does not answer the question asked of it."""

    def test_a_missing_atr_falls_back_to_typical_volatility(self, flat_model: CostModel) -> None:
        """The fill happens; it is priced as if volatility were ordinary."""
        warmup = MarketState(ts=OVERLAP_TS, atr_ratio=None)
        fill = flat_model.apply(order(side=Side.BUY, mid=1.1), warmup)
        assert fill.spread_points == pytest.approx(1.0)
        assert flat_model.stats.degradations[CostDegradation.ATR_UNAVAILABLE] == 1

    def test_the_share_distinguishes_a_warmup_from_half_the_run(
        self, flat_model: CostModel, typical_state: MarketState
    ) -> None:
        """Two fills degraded out of ten is 0.2, and that is the number reported.

        The distinction the share exists for: the same count of two means a
        warm-up artefact in a long run and a badly mispriced run in a short one,
        and a bare counter cannot tell them apart.
        """
        warmup = MarketState(ts=OVERLAP_TS, atr_ratio=None)
        for index in range(8):
            flat_model.apply(order(side=Side.BUY, mid=1.1, order_id=f"ok{index}"), typical_state)
        for index in range(2):
            flat_model.apply(order(side=Side.BUY, mid=1.1, order_id=f"bad{index}"), warmup)

        stats = flat_model.stats
        assert stats.fills == 10
        assert stats.degradations[CostDegradation.ATR_UNAVAILABLE] == 2
        assert stats.fraction(CostDegradation.ATR_UNAVAILABLE) == pytest.approx(0.2)
        assert stats.fractions[CostDegradation.ATR_UNAVAILABLE] == pytest.approx(0.2)

    def test_every_reason_is_present_even_at_zero(self, flat_model: CostModel) -> None:
        """An absent key would read as 'not measured' rather than 'did not happen'."""
        stats = flat_model.stats
        for reason in CostDegradation:
            assert stats.degradations[reason] == 0
            assert stats.fractions[reason] == 0.0

    def test_an_empty_run_degraded_on_nothing(self, flat_model: CostModel) -> None:
        """Zero fills is a share of zero, not a division by zero."""
        assert flat_model.stats.fills == 0
        assert flat_model.stats.fraction(CostDegradation.ATR_UNAVAILABLE) == 0.0

    def test_reset_clears_counts_and_fills_together(self, flat_model: CostModel) -> None:
        """They describe one run, so they are cleared as one."""
        # No ATR on this state, so both counters are non-zero before the reset.
        flat_model.apply(order(side=Side.BUY, mid=1.1), MarketState(ts=OVERLAP_TS))
        assert flat_model.stats.degradations[CostDegradation.ATR_UNAVAILABLE] == 1
        flat_model.reset()
        assert flat_model.stats.fills == 0
        assert flat_model.stats.degradations[CostDegradation.ATR_UNAVAILABLE] == 0


class TestRefusals:
    """What the model will not guess at."""

    def test_an_unknown_instrument_raises_rather_than_assuming_one(
        self, flat_model: CostModel, typical_state: MarketState
    ) -> None:
        """Guessing a point size is guessing the cost of every trade in it."""
        with pytest.raises(KeyError, match="unknown instrument"):
            flat_model.apply(order(side=Side.BUY, mid=1.1, symbol="NOPE"), typical_state)

    def test_a_negative_gap_is_rejected(self) -> None:
        """A level the market never reached produces no order at all."""
        with pytest.raises(ValueError, match="gap_points"):
            order(side=Side.BUY, mid=1.1, order_type=OrderType.STOP, gap_points=-1.0)

    def test_a_zero_atr_ratio_is_rejected(self) -> None:
        """Use ``None`` for 'not warmed up'; zero reads as 'no volatility'."""
        with pytest.raises(ValueError, match="atr_ratio"):
            MarketState(ts=OVERLAP_TS, atr_ratio=0.0)

    def test_realized_points_rejects_two_fills_on_the_same_side(
        self, flat_model: CostModel, eurusd: InstrumentSpec, typical_state: MarketState
    ) -> None:
        """That is two entries, not a round turn."""
        first = flat_model.apply(order(side=Side.BUY, mid=1.1), typical_state)
        second = flat_model.apply(order(side=Side.BUY, mid=1.1, order_id="o-2"), typical_state)
        with pytest.raises(ValueError, match="opposite side"):
            realized_points(first, second, eurusd)
