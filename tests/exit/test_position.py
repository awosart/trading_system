"""The position ledger: the ratchet, the fraction invariant, and what R means."""

from datetime import UTC, datetime
from decimal import Decimal

import pytest

from trading_system.core.types import Price, Side
from trading_system.exit.base import ExitReason
from trading_system.exit.position import INITIAL_STOP_SOURCE, ManagedPosition

from .conftest import START, SYMBOL, bar_close_ts, long_position, short_position

STOP = ExitReason.PROTECTIVE_STOP
TAKE = ExitReason.TAKE_PROFIT


class TestConstruction:
    def test_a_position_without_a_stop_is_not_expressible(self) -> None:
        # There is no default and no None to pass: the only way to reach a
        # position object is through a stop level.
        with pytest.raises(TypeError):
            ManagedPosition(  # type: ignore[call-arg]
                symbol=SYMBOL,
                side=Side.BUY,
                entry_price=Price(1.10),
                size=Decimal("1"),
                opened_at=START,
            )

    @pytest.mark.parametrize(
        ("side", "entry", "stop"),
        [
            (Side.BUY, 1.1000, 1.1100),
            (Side.BUY, 1.1000, 1.1000),
            (Side.SELL, 1.1000, 1.0900),
            (Side.SELL, 1.1000, 1.1000),
        ],
    )
    def test_a_stop_on_the_wrong_side_of_the_entry_is_refused(
        self, side: Side, entry: float, stop: float
    ) -> None:
        with pytest.raises(ValueError, match="must be strictly"):
            ManagedPosition(
                symbol=SYMBOL,
                side=side,
                entry_price=Price(entry),
                size=Decimal("1"),
                initial_stop=Price(stop),
                opened_at=START,
            )

    def test_a_pip_distance_mistaken_for_a_level_lands_on_the_wrong_side(self) -> None:
        # 15 "pips" passed where a level belongs sits far above a long's entry.
        with pytest.raises(ValueError, match="not distances"):
            long_position(entry=1.1000, stop=15.0)

    def test_size_must_be_positive(self) -> None:
        with pytest.raises(ValueError, match="size must be positive"):
            long_position(size=Decimal("0"))

    def test_a_naive_open_time_is_refused(self) -> None:
        with pytest.raises(ValueError, match="tz-aware"):
            long_position(opened_at=datetime(2024, 1, 2, 9, 0))  # noqa: DTZ001


class TestRMultiple:
    def test_one_r_is_the_entry_to_initial_stop_distance(self) -> None:
        position = long_position(entry=1.1000, stop=1.0900)
        assert position.initial_risk_distance == pytest.approx(0.0100)
        assert position.r_multiple(Price(1.1100)) == pytest.approx(1.0)
        assert position.r_multiple(Price(1.0900)) == pytest.approx(-1.0)
        assert position.r_multiple(Price(1.1000)) == pytest.approx(0.0)

    def test_r_is_mirrored_for_a_short(self) -> None:
        position = short_position(entry=1.1000, stop=1.1100)
        assert position.r_multiple(Price(1.0900)) == pytest.approx(1.0)
        assert position.r_multiple(Price(1.1100)) == pytest.approx(-1.0)

    def test_tightening_the_stop_does_not_move_r(self) -> None:
        # The denominator is the risk the position was SIZED against. If it
        # tracked the live stop, a breakeven move would divide by zero and every
        # R threshold in the plan would become unreachable or infinite.
        position = long_position(entry=1.1000, stop=1.0900)
        before = position.r_multiple(Price(1.1200))
        position.tighten_stop(Price(1.1000), source="test")
        assert position.stop == pytest.approx(1.1000)
        assert position.initial_risk_distance == pytest.approx(0.0100)
        assert position.r_multiple(Price(1.1200)) == pytest.approx(before)

    def test_a_partial_close_does_not_move_r(self) -> None:
        # If R were measured against the remainder, closing half at 1R would
        # rescale the 2R and 3R rungs of the same ladder.
        position = long_position(entry=1.1000, stop=1.0900)
        position.close(Decimal("0.5"), price=Price(1.1100), ts=bar_close_ts(1), reason=TAKE)
        assert position.r_multiple(Price(1.1200)) == pytest.approx(2.0)


class TestStopRatchet:
    def test_a_long_stop_only_moves_up(self) -> None:
        position = long_position(entry=1.1000, stop=1.0900)
        assert position.tighten_stop(Price(1.0950), source="test") is True
        assert position.stop == pytest.approx(1.0950)
        # A trailing rule recomputing on a pullback proposes a wider stop as a
        # matter of course. That is declined, not raised on.
        assert position.tighten_stop(Price(1.0910), source="test") is False
        assert position.stop == pytest.approx(1.0950)

    def test_a_short_stop_only_moves_down(self) -> None:
        position = short_position(entry=1.1000, stop=1.1100)
        assert position.tighten_stop(Price(1.1050), source="test") is True
        assert position.tighten_stop(Price(1.1090), source="test") is False
        assert position.stop == pytest.approx(1.1050)

    def test_the_stop_is_monotonic_across_a_noisy_sequence(self) -> None:
        position = long_position(entry=1.1000, stop=1.0900)
        seen = [position.stop]
        for level in (1.0910, 1.0905, 1.0930, 1.0800, 1.0931, 1.0931):
            position.tighten_stop(Price(level), source="test")
            seen.append(position.stop)
        assert seen == sorted(seen)
        assert position.stop == pytest.approx(1.0931)

    def test_the_stop_of_a_flat_position_cannot_be_moved(self) -> None:
        position = long_position()
        position.close_all(price=Price(1.1100), ts=bar_close_ts(1), reason=TAKE)
        with pytest.raises(ValueError, match="closed position"):
            position.tighten_stop(Price(1.1000), source="test")


class TestStopSource:
    def test_starts_at_the_initial_stop_sentinel(self) -> None:
        position = long_position(entry=1.1000, stop=1.0900)
        assert position.stop_source == INITIAL_STOP_SOURCE

    def test_records_whoever_tightened_it_last(self) -> None:
        position = long_position(entry=1.1000, stop=1.0900)
        position.tighten_stop(Price(1.0950), source="atr_stop_14_1.5")
        assert position.stop_source == "atr_stop_14_1.5"
        position.tighten_stop(Price(1.0980), source="breakeven_1r")
        assert position.stop_source == "breakeven_1r"

    def test_a_declined_looser_proposal_does_not_overwrite_the_source(self) -> None:
        # The ratchet declines the move; the attribution must not update either
        # — otherwise a rejected proposal could still relabel who is "in charge"
        # of a level it never actually set.
        position = long_position(entry=1.1000, stop=1.0900)
        position.tighten_stop(Price(1.0950), source="atr_stop_14_1.5")
        position.tighten_stop(Price(1.0910), source="some_looser_rule")
        assert position.stop_source == "atr_stop_14_1.5"

    def test_the_closed_leg_carries_the_stop_source_at_close_time(self) -> None:
        position = long_position(entry=1.1000, stop=1.0900)
        position.tighten_stop(Price(1.0950), source="breakeven_1r")
        leg = position.close_all(price=Price(1.0950), ts=bar_close_ts(1), reason=STOP)
        assert leg.stop_source == "breakeven_1r"

    def test_a_leg_closed_before_any_tighten_carries_the_initial_sentinel(self) -> None:
        position = long_position(entry=1.1000, stop=1.0900)
        leg = position.close_all(price=Price(1.0900), ts=bar_close_ts(1), reason=STOP)
        assert leg.stop_source == INITIAL_STOP_SOURCE

    def test_a_take_profit_leg_still_carries_whatever_the_stop_source_was(self) -> None:
        # Not particularly meaningful for a non-stop reason, but present and
        # honest rather than silently omitted.
        position = long_position(entry=1.1000, stop=1.0900)
        position.tighten_stop(Price(1.0950), source="atr_stop_14_1.5")
        leg = position.close_all(price=Price(1.1200), ts=bar_close_ts(1), reason=TAKE)
        assert leg.stop_source == "atr_stop_14_1.5"


class TestFractionInvariant:
    def test_a_second_close_of_60_percent_is_refused_not_clamped(self) -> None:
        position = long_position(entry=1.1000, stop=1.0900, size=Decimal("0.10"))

        position.close(Decimal("0.6"), price=Price(1.1100), ts=bar_close_ts(1), reason=TAKE)
        assert position.remaining_fraction == Decimal("0.4")

        with pytest.raises(ValueError, match=r"0\.6 of the original position, 0\.4 remains"):
            position.close(Decimal("0.6"), price=Price(1.1200), ts=bar_close_ts(2), reason=TAKE)

        # Atomic: the rejected close booked no leg, no PnL and no reduction.
        assert position.remaining_fraction == Decimal("0.4")
        assert len(position.legs) == 1
        assert position.realized_quote_move == Decimal("0.10") * Decimal("0.6") * Decimal("0.01")

    def test_a_ladder_sums_to_exactly_one(self) -> None:
        # Exactly, not to within a tolerance — which is why fractions are Decimal.
        position = long_position(entry=1.1000, stop=1.0900)
        for index, (fraction, price) in enumerate(
            [("0.5", 1.1100), ("0.25", 1.1200), ("0.25", 1.1300)]
        ):
            position.close(
                Decimal(fraction), price=Price(price), ts=bar_close_ts(index + 1), reason=TAKE
            )
        assert sum(leg.fraction for leg in position.legs) == Decimal("1.00")
        assert position.remaining_fraction == Decimal("0.00")
        assert position.is_open is False

    def test_closing_a_flat_position_is_refused(self) -> None:
        position = long_position()
        position.close_all(price=Price(1.1100), ts=bar_close_ts(1), reason=TAKE)
        with pytest.raises(ValueError, match="already flat"):
            position.close(Decimal("0.1"), price=Price(1.12), ts=bar_close_ts(2), reason=TAKE)

    def test_a_float_fraction_is_refused(self) -> None:
        # 0.1 + 0.2 in binary floating point does not equal 0.3, and "the
        # position is flat" would become a comparison against a tolerance.
        position = long_position()
        with pytest.raises(ValueError, match="must be a Decimal"):
            position.close(0.5, price=Price(1.11), ts=bar_close_ts(1), reason=TAKE)  # type: ignore[arg-type]

    @pytest.mark.parametrize("fraction", ["0", "-0.5"])
    def test_a_non_positive_fraction_is_refused(self, fraction: str) -> None:
        position = long_position()
        with pytest.raises(ValueError, match="must be positive"):
            position.close(Decimal(fraction), price=Price(1.11), ts=bar_close_ts(1), reason=TAKE)


class TestRealisedResults:
    def test_realized_r_weights_each_leg_by_its_share(self) -> None:
        # 50% at 1R plus 50% at 3R is 2R, not 4R.
        position = long_position(entry=1.1000, stop=1.0900)
        position.close(Decimal("0.5"), price=Price(1.1100), ts=bar_close_ts(1), reason=TAKE)
        position.close(Decimal("0.5"), price=Price(1.1300), ts=bar_close_ts(2), reason=TAKE)
        assert [leg.r_multiple for leg in position.legs] == pytest.approx([1.0, 3.0])
        assert position.realized_r() == pytest.approx(2.0)

    def test_open_and_realized_r_add_up_to_the_result_if_closed_now(self) -> None:
        position = long_position(entry=1.1000, stop=1.0900)
        position.close(Decimal("0.5"), price=Price(1.1100), ts=bar_close_ts(1), reason=TAKE)
        assert position.realized_r() == pytest.approx(0.5)
        assert position.open_r(Price(1.1300)) == pytest.approx(1.5)
        assert position.total_r(Price(1.1300)) == pytest.approx(2.0)

    def test_pnl_sums_across_partials_and_is_decimal(self) -> None:
        position = long_position(entry=1.1000, stop=1.0900, size=Decimal("2"))
        position.close(Decimal("0.5"), price=Price(1.1100), ts=bar_close_ts(1), reason=TAKE)
        position.close_all(price=Price(1.0950), ts=bar_close_ts(2), reason=STOP)

        # 2 * 0.5 * (+0.01) + 2 * 0.5 * (-0.005)
        assert position.realized_quote_move == Decimal("0.005")
        assert isinstance(position.realized_quote_move, Decimal)
        assert position.is_open is False

    def test_a_short_books_profit_when_price_falls(self) -> None:
        position = short_position(entry=1.1000, stop=1.1100, size=Decimal("1"))
        position.close_all(price=Price(1.0900), ts=bar_close_ts(1), reason=TAKE)
        assert position.realized_quote_move == Decimal("0.0100")
        assert position.realized_r() == pytest.approx(1.0)

    def test_a_leg_records_where_and_why_it_closed(self) -> None:
        position = long_position(entry=1.1000, stop=1.0900)
        leg = position.close_all(price=Price(1.0900), ts=bar_close_ts(3), reason=STOP)
        assert leg.reason is STOP
        assert leg.price == pytest.approx(1.0900)
        assert leg.ts == bar_close_ts(3)
        assert leg.ts.tzinfo is UTC
        assert leg.fraction == Decimal("1")

    def test_remaining_size_tracks_the_remaining_fraction(self) -> None:
        position = long_position(size=Decimal("0.30"))
        position.close(Decimal("0.5"), price=Price(1.1100), ts=bar_close_ts(1), reason=TAKE)
        assert position.size == Decimal("0.30")
        assert position.remaining_size == Decimal("0.150")
