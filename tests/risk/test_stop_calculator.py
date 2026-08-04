"""Stop placement: the invariant is that nothing here can ever tighten a stop."""

from decimal import Decimal

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from tests.risk.conftest import REGISTRY_PATH
from trading_system.core.instruments import InstrumentRegistry, load_instruments
from trading_system.core.types import Price, Side
from trading_system.risk.stop_calculator import (
    BROKER_MINIMUM,
    INVALIDATION_BUFFERED,
    STOP_REFERENCE,
    StopBufferConfig,
    calculate_stop,
    requires_atr,
)
from trading_system.strategies.schema import (
    AtrStop,
    FixedPipsStop,
    PercentStop,
    StopReference,
    StructureStop,
)

NO_BUFFER = StopBufferConfig(spread_multiple=0.0, fixed_points=0.0, atr_multiple=0.0)
#: Narrow enough never to be the binding candidate.
NEVER_BINDING: StopReference = FixedPipsStop(pips=0.001)


class TestTheStopOnlyEverWidens:
    """The rule the module exists to make structural."""

    def test_the_furthest_candidate_wins_not_the_nearest(
        self, registry: InstrumentRegistry
    ) -> None:
        # A 20-point structural stop against a 100-point stop_reference: the
        # wider one is used, and the size adapts down accordingly.
        instrument = registry["EURUSD"]
        result = calculate_stop(
            side=Side.BUY,
            reference_price=Price(1.0850),
            invalidation_price=Price(1.0830),
            instrument=instrument,
            stop_reference=FixedPipsStop(pips=100),
            buffer=NO_BUFFER,
            atr_price=None,
        )
        assert result.binding == STOP_REFERENCE
        assert result.distance_points == pytest.approx(100.0)

    def test_the_structural_level_wins_when_it_is_the_wider_one(
        self, registry: InstrumentRegistry
    ) -> None:
        result = calculate_stop(
            side=Side.BUY,
            reference_price=Price(1.0850),
            invalidation_price=Price(1.0750),
            instrument=registry["EURUSD"],
            stop_reference=FixedPipsStop(pips=20),
            buffer=NO_BUFFER,
            atr_price=None,
        )
        assert result.binding == INVALIDATION_BUFFERED
        assert result.distance_points == pytest.approx(100.0)

    def test_the_broker_minimum_widens_rather_than_the_trade_being_reshaped(
        self, registry: InstrumentRegistry
    ) -> None:
        # XAUUSD requires 30 points; a 5-point stop is pushed out to 30. The
        # size is then recomputed against the wider stop -- the stop is never
        # brought in to preserve the size.
        result = calculate_stop(
            side=Side.BUY,
            reference_price=Price(2050.00),
            invalidation_price=Price(2049.95),
            instrument=registry["XAUUSD"],
            stop_reference=NEVER_BINDING,
            buffer=NO_BUFFER,
            atr_price=None,
        )
        assert result.binding == BROKER_MINIMUM
        assert result.distance_points == pytest.approx(30.0)
        assert any("broker minimum" in line for line in result.reasons)

    @settings(max_examples=200, deadline=None)
    @given(
        symbol=st.sampled_from(["EURUSD", "GBPJPY", "XAUUSD", "NAS100", "BTCUSD"]),
        structural_points=st.floats(min_value=0.5, max_value=1000.0, allow_nan=False),
        reference_points=st.floats(min_value=0.5, max_value=1000.0, allow_nan=False),
        spread_multiple=st.floats(min_value=0.0, max_value=5.0, allow_nan=False),
        side=st.sampled_from([Side.BUY, Side.SELL]),
    )
    def test_the_stop_is_never_nearer_than_any_candidate_asked_for(
        self,
        symbol: str,
        structural_points: float,
        reference_points: float,
        spread_multiple: float,
        side: Side,
    ) -> None:
        # The property behind "never move the stop closer for a bigger size".
        # Whatever the inputs, the placed stop is at least as far away as each
        # candidate individually demanded.
        registry = load_instruments(REGISTRY_PATH)
        instrument = registry[symbol]
        reference = Price(100.0 * instrument.point_size * 1000)
        offset = instrument.points_to_price(structural_points)
        invalidation = Price(reference - offset if side is Side.BUY else reference + offset)

        result = calculate_stop(
            side=side,
            reference_price=reference,
            invalidation_price=invalidation,
            instrument=instrument,
            stop_reference=FixedPipsStop(pips=reference_points),
            buffer=StopBufferConfig(spread_multiple=spread_multiple),
            atr_price=None,
        )
        tolerance = instrument.price_to_points(instrument.tick_size)
        assert result.distance_points >= structural_points - tolerance
        assert result.distance_points >= reference_points - tolerance
        assert result.distance_points >= instrument.min_stop_distance_points - tolerance

    @settings(max_examples=100, deadline=None)
    @given(
        raw_points=st.floats(min_value=1.0, max_value=500.0, allow_nan=False),
        side=st.sampled_from([Side.BUY, Side.SELL]),
    )
    def test_tick_rounding_can_only_widen(self, raw_points: float, side: Side) -> None:
        # Snapping a stop to the nearest tick could shrink the distance the
        # size was computed from. Snapping away from entry cannot.
        registry = load_instruments(REGISTRY_PATH)
        instrument = registry["EURUSD"]
        reference = Price(1.08500)
        offset = instrument.points_to_price(raw_points)
        invalidation = Price(reference - offset if side is Side.BUY else reference + offset)
        result = calculate_stop(
            side=side,
            reference_price=reference,
            invalidation_price=invalidation,
            instrument=instrument,
            stop_reference=NEVER_BINDING,
            buffer=NO_BUFFER,
            atr_price=None,
        )
        assert result.distance_price >= offset - 1e-12


class TestBuffer:
    def test_the_spread_pushes_the_stop_past_the_visible_level(
        self, registry: InstrumentRegistry
    ) -> None:
        # EURUSD's typical spread is 0.8 points, so a 50-point structural stop
        # is placed at 50.8. A stop resting exactly on the level everyone can
        # see is taken out by the spread widening alone.
        result = calculate_stop(
            side=Side.BUY,
            reference_price=Price(1.0850),
            invalidation_price=Price(1.0800),
            instrument=registry["EURUSD"],
            stop_reference=NEVER_BINDING,
            buffer=StopBufferConfig(spread_multiple=1.0),
            atr_price=None,
        )
        assert result.distance_points == pytest.approx(50.8, abs=0.01)

    def test_a_fixed_noise_allowance_adds_on_top(self, registry: InstrumentRegistry) -> None:
        result = calculate_stop(
            side=Side.BUY,
            reference_price=Price(1.0850),
            invalidation_price=Price(1.0800),
            instrument=registry["EURUSD"],
            stop_reference=NEVER_BINDING,
            buffer=StopBufferConfig(spread_multiple=0.0, fixed_points=3.0),
            atr_price=None,
        )
        assert result.distance_points == pytest.approx(53.0, abs=0.01)


class TestAtrRequirement:
    def test_requires_atr_is_true_for_an_atr_referenced_stop(self) -> None:
        assert requires_atr(AtrStop(multiple=2.0), NO_BUFFER)

    def test_requires_atr_is_true_for_an_atr_buffer(self) -> None:
        assert requires_atr(NEVER_BINDING, StopBufferConfig(atr_multiple=0.5))

    def test_a_structure_stop_needs_atr_only_when_it_buffers_by_atr(self) -> None:
        assert not requires_atr(StructureStop(buffer_atr_multiple=0.0), NO_BUFFER)
        assert requires_atr(StructureStop(buffer_atr_multiple=0.5), NO_BUFFER)

    def test_a_missing_atr_raises_rather_than_defaulting_to_zero(
        self, registry: InstrumentRegistry
    ) -> None:
        # A zero buffer would silently tighten the stop and enlarge the
        # position -- the exact failure this module is built to prevent.
        with pytest.raises(ValueError, match="needs an ATR value"):
            calculate_stop(
                side=Side.BUY,
                reference_price=Price(1.0850),
                invalidation_price=Price(1.0800),
                instrument=registry["EURUSD"],
                stop_reference=AtrStop(multiple=2.0),
                buffer=NO_BUFFER,
                atr_price=None,
            )

    def test_an_atr_stop_measures_from_the_entry(self, registry: InstrumentRegistry) -> None:
        # 2 x ATR of 0.0030 = 0.0060 = 60 points, wider than the 50-point
        # structural level, so it binds.
        result = calculate_stop(
            side=Side.BUY,
            reference_price=Price(1.0850),
            invalidation_price=Price(1.0800),
            instrument=registry["EURUSD"],
            stop_reference=AtrStop(multiple=2.0),
            buffer=NO_BUFFER,
            atr_price=0.0030,
        )
        assert result.binding == STOP_REFERENCE
        assert result.distance_points == pytest.approx(60.0, abs=0.01)


class TestStopReferenceVariants:
    def test_percent_is_a_percentage_of_the_entry_price(self, registry: InstrumentRegistry) -> None:
        # 1% of 65 000 = 650 points on BTCUSD, whose point is 1.0.
        result = calculate_stop(
            side=Side.BUY,
            reference_price=Price(65000.0),
            invalidation_price=Price(64900.0),
            instrument=registry["BTCUSD"],
            stop_reference=PercentStop(percent=1.0),
            buffer=NO_BUFFER,
            atr_price=None,
        )
        assert result.binding == STOP_REFERENCE
        assert result.distance_points == pytest.approx(650.0)

    def test_structure_defers_to_the_entrys_own_invalidation(
        self, registry: InstrumentRegistry
    ) -> None:
        # The structural level IS the entry's invalidation. Re-deriving a swing
        # here would compute a second, competing level from data the Entry
        # Engine has already read.
        result = calculate_stop(
            side=Side.BUY,
            reference_price=Price(1.0850),
            invalidation_price=Price(1.0800),
            instrument=registry["EURUSD"],
            stop_reference=StructureStop(buffer_atr_multiple=0.0),
            buffer=NO_BUFFER,
            atr_price=None,
        )
        assert result.distance_points == pytest.approx(50.0)


class TestSides:
    def test_a_long_stop_sits_below_and_a_short_stop_above(
        self, registry: InstrumentRegistry
    ) -> None:
        for side, invalidation, comparison in (
            (Side.BUY, 1.0800, True),
            (Side.SELL, 1.0900, False),
        ):
            result = calculate_stop(
                side=side,
                reference_price=Price(1.0850),
                invalidation_price=Price(invalidation),
                instrument=registry["EURUSD"],
                stop_reference=NEVER_BINDING,
                buffer=NO_BUFFER,
                atr_price=None,
            )
            assert (result.stop_price < 1.0850) is comparison

    def test_a_wrong_sided_invalidation_is_rejected(self, registry: InstrumentRegistry) -> None:
        with pytest.raises(ValueError, match="must be strictly below"):
            calculate_stop(
                side=Side.BUY,
                reference_price=Price(1.0850),
                invalidation_price=Price(1.0900),
                instrument=registry["EURUSD"],
                stop_reference=NEVER_BINDING,
                buffer=NO_BUFFER,
                atr_price=None,
            )


class TestExactDistance:
    def test_the_money_distance_is_exact_where_the_float_one_is_not(
        self, registry: InstrumentRegistry
    ) -> None:
        # 1.0850 - 1.0800 in binary floating point is 0.004999999999999893,
        # which turns a risk of exactly 500 into 499.99999999998934. Both
        # endpoints are exact decimals, so subtracting them in decimal is not.
        result = calculate_stop(
            side=Side.BUY,
            reference_price=Price(1.0850),
            invalidation_price=Price(1.0800),
            instrument=registry["EURUSD"],
            stop_reference=NEVER_BINDING,
            buffer=NO_BUFFER,
            atr_price=None,
        )
        assert result.distance_points_exact == Decimal("50")
        assert result.distance_points != 50.0  # the float one carries the error
