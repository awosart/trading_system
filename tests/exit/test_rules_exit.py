"""PartialClose, TimeExit, SignalReverseExit: the three new genuine ExitRules.

Unlike the stop modifiers, these emit :class:`~trading_system.exit.base.
ExitDecision` themselves and each carries its own
:class:`~trading_system.exit.base.ExitReason`.
"""

from datetime import UTC, datetime
from decimal import Decimal

import pytest

from trading_system.core.exceptions import ValidationError
from trading_system.core.types import Side
from trading_system.data.sessions import AssetClass, Session
from trading_system.entry.context import PRICE_FIELDS, BarSeries
from trading_system.exit.base import ExitDropReason, ExitKind, ExitReason, ExitRule, ExitTrigger
from trading_system.exit.context import exit_contexts
from trading_system.exit.plan import ExitPlan
from trading_system.exit.rules import PartialClose, PartialRung, ProtectiveStop, SignalReverseExit
from trading_system.exit.rules.time_exit import TimeExit, TimeExitMode

from .conftest import TIMEFRAME, Bar, bar_open_ts, long_position, series, short_position


def plan_with(rule: ExitRule) -> ExitPlan:
    """A plan pairing one exit rule with the protective stop."""
    return ExitPlan(exit_id="rule-scenario", protective_stop=ProtectiveStop(), rules=[rule])


class TestPartialRung:
    def test_construction_rejects_a_non_positive_r_multiple(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            PartialRung(r_multiple=0.0, fraction=Decimal("0.5"))

    @pytest.mark.parametrize("fraction", ["0", "1", "1.5", "-0.1"])
    def test_construction_rejects_a_fraction_outside_the_open_unit_interval(
        self, fraction: str
    ) -> None:
        with pytest.raises(ValueError, match="strictly inside"):
            PartialRung(r_multiple=1.0, fraction=Decimal(fraction))


class TestPartialCloseLoad:
    def test_a_ladder_summing_to_more_than_the_position_fails_at_load(self) -> None:
        with pytest.raises(ValidationError, match="exceeding the position"):
            PartialClose(
                [
                    PartialRung(r_multiple=1.0, fraction=Decimal("0.6")),
                    PartialRung(r_multiple=2.0, fraction=Decimal("0.6")),
                ]
            )

    def test_a_ladder_summing_to_exactly_the_position_loads_fine(self) -> None:
        PartialClose(
            [
                PartialRung(r_multiple=1.0, fraction=Decimal("0.5")),
                PartialRung(r_multiple=2.0, fraction=Decimal("0.25")),
                PartialRung(r_multiple=3.0, fraction=Decimal("0.25")),
            ]
        )

    def test_an_empty_ladder_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="at least one rung"):
            PartialClose([])

    def test_partial_fractions_reports_every_rung(self) -> None:
        rule = PartialClose(
            [
                PartialRung(r_multiple=2.0, fraction=Decimal("0.25")),
                PartialRung(r_multiple=1.0, fraction=Decimal("0.5")),
            ]
        )
        # Sorted by r_multiple internally, but the fraction set is what matters.
        assert set(rule.partial_fractions) == {Decimal("0.5"), Decimal("0.25")}


class TestPartialCloseScenario:
    def test_each_rung_fires_on_the_bar_its_target_is_reached(self) -> None:
        bars: list[Bar] = [
            (1.1000, 1.1110, 1.0990, 1.1100),  # 0: reaches 1R (1.1100)
            (1.1100, 1.1210, 1.1090, 1.1200),  # 1: reaches 2R (1.1200)
        ]
        rule = PartialClose(
            [
                PartialRung(r_multiple=1.0, fraction=Decimal("0.5")),
                PartialRung(r_multiple=2.0, fraction=Decimal("0.5")),
            ]
        )
        position = long_position(entry=1.1000, stop=1.0900)
        result = plan_with(rule).run(position, exit_contexts(series(bars)))

        assert [fill.bar_index for fill in result.fills] == [0, 1]
        assert [fill.leg.fraction for fill in result.fills] == [Decimal("0.5"), Decimal("0.5")]
        assert [fill.decision.reason for fill in result.fills] == [
            ExitReason.PARTIAL_TAKE_PROFIT,
            ExitReason.PARTIAL_TAKE_PROFIT,
        ]
        assert all(fill.decision.trigger is ExitTrigger.LEVEL_TOUCH for fill in result.fills)
        assert all(fill.decision.kind is ExitKind.PARTIAL for fill in result.fills)
        assert position.remaining_fraction == Decimal("0")

    def test_a_short_ladder_fires_as_price_falls(self) -> None:
        bars: list[Bar] = [(1.1000, 1.1020, 1.0800, 1.0850)]  # reaches 2R (1.0800)
        rule = PartialClose([PartialRung(r_multiple=2.0, fraction=Decimal("0.5"))])
        position = short_position(entry=1.1000, stop=1.1100)
        result = plan_with(rule).run(position, exit_contexts(series(bars)))

        assert result.fills[0].leg.price == pytest.approx(1.0800)
        assert result.fills[0].leg.fraction == Decimal("0.5")

    def test_a_bar_crossing_two_rungs_at_once_fires_only_the_nearer_one(self) -> None:
        # A single huge bar reaches both 1R (1.1100) and 2R (1.1200); only the
        # nearer, untouched rung fires this bar.
        bars: list[Bar] = [(1.1000, 1.1300, 1.0990, 1.1250)]
        rule = PartialClose(
            [
                PartialRung(r_multiple=1.0, fraction=Decimal("0.5")),
                PartialRung(r_multiple=2.0, fraction=Decimal("0.5")),
            ]
        )
        position = long_position(entry=1.1000, stop=1.0900)
        result = plan_with(rule).run(position, exit_contexts(series(bars)))

        assert len(result.fills) == 1
        assert result.fills[0].leg.price == pytest.approx(1.1100)
        assert position.remaining_fraction == Decimal("0.5")

    def test_a_rung_never_fires_twice(self) -> None:
        bars: list[Bar] = [
            (1.1000, 1.1110, 1.0990, 1.1100),
            (1.1100, 1.1150, 1.1090, 1.1120),  # still above 1R, rung already fired
        ]
        rule = PartialClose([PartialRung(r_multiple=1.0, fraction=Decimal("0.5"))])
        position = long_position(entry=1.1000, stop=1.0900)
        result = plan_with(rule).run(position, exit_contexts(series(bars)))
        assert len(result.fills) == 1

    def test_summing_to_less_than_the_position_leaves_a_runner_open(self) -> None:
        bars: list[Bar] = [(1.1000, 1.1110, 1.0990, 1.1100)]
        rule = PartialClose([PartialRung(r_multiple=1.0, fraction=Decimal("0.5"))])
        position = long_position(entry=1.1000, stop=1.0900)
        result = plan_with(rule).run(position, exit_contexts(series(bars)))
        assert result.closed is False
        assert position.remaining_fraction == Decimal("0.5")


class TestTimeExitConstruction:
    def test_max_bars_held_requires_a_positive_count(self) -> None:
        with pytest.raises(ValidationError, match="MAX_BARS_HELD"):
            TimeExit(TimeExitMode.MAX_BARS_HELD)
        with pytest.raises(ValidationError, match="MAX_BARS_HELD"):
            TimeExit(TimeExitMode.MAX_BARS_HELD, max_bars_held=0)

    def test_session_close_requires_a_session(self) -> None:
        with pytest.raises(ValidationError, match="SESSION_CLOSE"):
            TimeExit(TimeExitMode.SESSION_CLOSE)

    def test_before_weekend_needs_nothing_extra(self) -> None:
        TimeExit(TimeExitMode.BEFORE_WEEKEND)

    def test_partial_fractions_is_always_empty(self) -> None:
        assert TimeExit(TimeExitMode.MAX_BARS_HELD, max_bars_held=1).partial_fractions == ()


class TestTimeExitMaxBarsHeld:
    def test_closes_on_the_nth_bars_close_and_fills_on_the_n_plus_1th_open(self) -> None:
        # BAR_CLOSE, like every other close-decided exit: the condition is met
        # on bar 2's close (the third bar, held-count starting at one), and can
        # only fill at bar 3's open — a 4th bar is needed to see the fill land.
        bars: list[Bar] = [(1.1000, 1.1020, 1.0990, 1.1010)] * 3 + [
            (1.1050, 1.1060, 1.1040, 1.1055)
        ]
        rule = TimeExit(TimeExitMode.MAX_BARS_HELD, max_bars_held=3)
        position = long_position(entry=1.1000, stop=1.0900)
        result = plan_with(rule).run(position, exit_contexts(series(bars)))

        assert len(result.fills) == 1
        fill = result.fills[0]
        assert fill.decided_bar_index == 2
        assert fill.bar_index == 3
        assert fill.decision.trigger is ExitTrigger.BAR_CLOSE
        assert fill.decision.price is None
        assert fill.decision.reason is ExitReason.TIME_EXIT
        assert fill.decision.kind is ExitKind.FULL
        assert fill.leg.price == pytest.approx(1.1050)
        assert fill.leg.ts == bar_open_ts(3)
        assert result.closed is True

    def test_does_not_fire_before_the_bar_count_is_reached(self) -> None:
        bars: list[Bar] = [(1.1000, 1.1020, 1.0990, 1.1010)] * 2
        rule = TimeExit(TimeExitMode.MAX_BARS_HELD, max_bars_held=5)
        position = long_position(entry=1.1000, stop=1.0900)
        result = plan_with(rule).run(position, exit_contexts(series(bars)))
        assert result.fills == ()
        assert result.closed is False

    def test_reset_zeroes_the_bar_counter_between_runs(self) -> None:
        bars: list[Bar] = [(1.1000, 1.1020, 1.0990, 1.1010)] * 2
        rule = TimeExit(TimeExitMode.MAX_BARS_HELD, max_bars_held=5)
        plan = plan_with(rule)
        plan.run(long_position(), exit_contexts(series(bars)))
        # A second run over the same short series must not inherit bar 1's
        # count of 2 and somehow be one bar closer to firing.
        second = plan.run(long_position(), exit_contexts(series(bars)))
        assert second.fills == ()


class TestTimeExitSessionClose:
    def test_closes_on_the_bar_that_leaves_the_session(self) -> None:
        # LONDON is 08:00-17:00 Europe/London (UTC in winter). Build two bars
        # straddling the close: one whose open is inside London, whose close
        # (== the next bar's open) is not.
        opens = [
            datetime(2024, 1, 2, 16, 45, tzinfo=UTC),  # inside LONDON
            datetime(2024, 1, 2, 17, 0, tzinfo=UTC),  # LONDON has just closed
            datetime(2024, 1, 2, 17, 15, tzinfo=UTC),
        ]
        prices = {
            "open": [1.10, 1.11, 1.111],
            "high": [1.101, 1.112, 1.112],
            "low": [1.099, 1.109, 1.110],
            "close": [1.1005, 1.111, 1.1115],
            "volume": [1_000.0, 1_000.0, 1_000.0],
        }
        custom_series = BarSeries(
            symbol="TESTFX",
            timeframe=TIMEFRAME,
            timestamps=opens,
            prices={field: prices[field] for field in PRICE_FIELDS},
            features={},
        )
        rule = TimeExit(TimeExitMode.SESSION_CLOSE, session=Session.LONDON)
        position = long_position(entry=1.10, stop=1.09, opened_at=opens[0])
        result = plan_with(rule).run(position, exit_contexts(custom_series))

        assert len(result.fills) == 1
        assert result.fills[0].decided_bar_index == 0
        assert result.fills[0].bar_index == 1
        assert result.fills[0].leg.price == pytest.approx(1.11)


class TestTimeExitBeforeWeekend:
    def test_closes_on_the_last_bar_before_the_fx_week_closes(self) -> None:
        # FX closes Friday 17:00 New York (22:00 UTC in January). Bar 0's close
        # (21:45 UTC = 16:45 NY) is still inside the trading week, but the next
        # bar boundary (22:00 UTC = 17:00 NY exactly) is not — that is the
        # condition, decided on bar 0's close.
        opens = [
            datetime(2024, 1, 5, 21, 30, tzinfo=UTC),
            datetime(2024, 1, 5, 21, 45, tzinfo=UTC),
            datetime(2024, 1, 5, 22, 0, tzinfo=UTC),  # already past the close
        ]
        prices = {
            "open": [1.10, 1.11, 1.111],
            "high": [1.101, 1.112, 1.112],
            "low": [1.099, 1.109, 1.110],
            "close": [1.1005, 1.111, 1.1115],
            "volume": [1_000.0, 1_000.0, 1_000.0],
        }
        custom_series = BarSeries(
            symbol="TESTFX",
            timeframe=TIMEFRAME,
            timestamps=opens,
            prices={field: prices[field] for field in PRICE_FIELDS},
            features={},
        )
        rule = TimeExit(TimeExitMode.BEFORE_WEEKEND, asset_class=AssetClass.FX)
        position = long_position(entry=1.10, stop=1.09, opened_at=opens[0])
        result = plan_with(rule).run(position, exit_contexts(custom_series))

        assert len(result.fills) == 1
        assert result.fills[0].decided_bar_index == 0
        assert result.fills[0].bar_index == 1
        assert result.fills[0].leg.price == pytest.approx(1.11)


class TestSignalReverseExit:
    def test_no_signal_no_exit(self) -> None:
        bars: list[Bar] = [(1.1000, 1.1020, 1.0990, 1.1010)] * 2
        position = long_position(entry=1.1000, stop=1.0900)
        result = plan_with(SignalReverseExit()).run(position, exit_contexts(series(bars)))
        assert result.fills == ()

    def test_a_signal_agreeing_with_the_positions_side_does_not_exit(self) -> None:
        bars: list[Bar] = [(1.1000, 1.1020, 1.0990, 1.1010)] * 2
        position = long_position(entry=1.1000, stop=1.0900)
        result = plan_with(SignalReverseExit()).run(
            position, exit_contexts(series(bars), reverse_signals={0: Side.BUY})
        )
        assert result.fills == ()

    def test_an_opposing_signal_closes_in_full_at_the_next_bars_open(self) -> None:
        bars: list[Bar] = [
            (1.1000, 1.1020, 1.0990, 1.1010),  # reverse signal recognised here
            (1.1005, 1.1030, 1.0995, 1.1020),  # filled here, at the open
        ]
        position = long_position(entry=1.1000, stop=1.0900)
        result = plan_with(SignalReverseExit()).run(
            position, exit_contexts(series(bars), reverse_signals={0: Side.SELL})
        )

        assert len(result.fills) == 1
        fill = result.fills[0]
        assert fill.decided_bar_index == 0
        assert fill.bar_index == 1
        assert fill.decision.reason is ExitReason.SIGNAL_REVERSAL
        assert fill.decision.trigger is ExitTrigger.BAR_CLOSE
        assert fill.decision.kind is ExitKind.FULL
        assert fill.decision.price is None
        assert fill.leg.price == pytest.approx(1.1005)
        assert fill.leg.ts == bar_open_ts(1)

    def test_a_short_is_reversed_by_an_opposing_buy_signal(self) -> None:
        bars: list[Bar] = [
            (1.1000, 1.1010, 1.0990, 1.1000),
            (1.1000, 1.1010, 1.0990, 1.1000),
        ]
        position = short_position(entry=1.1000, stop=1.1100)
        result = plan_with(SignalReverseExit()).run(
            position, exit_contexts(series(bars), reverse_signals={0: Side.BUY})
        )
        assert len(result.fills) == 1
        assert result.fills[0].decision.reason is ExitReason.SIGNAL_REVERSAL

    def test_partial_fractions_is_always_empty(self) -> None:
        assert SignalReverseExit().partial_fractions == ()

    def test_the_context_records_which_side_reversed_it(self) -> None:
        bars: list[Bar] = [(1.1000, 1.1010, 1.0990, 1.1000)] * 2
        position = long_position(entry=1.1000, stop=1.0900)
        result = plan_with(SignalReverseExit()).run(
            position, exit_contexts(series(bars), reverse_signals={0: Side.SELL})
        )
        assert result.fills[0].decision.context["reverse_side"] == "SELL"


class TestReasonBookkeeping:
    def test_partial_and_time_exit_reasons_are_present_in_drop_counts_bookkeeping(self) -> None:
        # Not literally about ExitDropReason, but a smoke test that composing
        # both new ExitRule kinds together with the stop doesn't choke the
        # drop-counting machinery.
        bars: list[Bar] = [(1.1000, 1.1020, 1.0990, 1.1010)]
        plan = ExitPlan(
            exit_id="combo",
            protective_stop=ProtectiveStop(),
            rules=[
                PartialClose([PartialRung(r_multiple=1.0, fraction=Decimal("0.5"))]),
                TimeExit(TimeExitMode.MAX_BARS_HELD, max_bars_held=100),
            ],
        )
        result = plan.run(long_position(), exit_contexts(series(bars)))
        assert set(result.drops) == set(ExitDropReason)
