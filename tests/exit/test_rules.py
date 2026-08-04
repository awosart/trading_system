"""Per-rule scenarios: a sequence of bars, one exit, on a named bar at a named price."""

import pytest

from trading_system.core.types import OrderType, Price
from trading_system.exit.base import ExitKind, ExitReason, ExitTrigger
from trading_system.exit.context import exit_contexts
from trading_system.exit.plan import ExitPlan
from trading_system.exit.rules import FixedRR, ProtectiveStop

from .conftest import Bar, bar_close_ts, long_position, series, short_position


def plan(*, take: float | None = None) -> ExitPlan:
    """A plan with the protective stop and, optionally, a fixed-R target."""
    return ExitPlan(
        exit_id="scenario",
        protective_stop=ProtectiveStop(),
        rules=[FixedRR(take)] if take is not None else [],
    )


class TestProtectiveStop:
    def test_a_long_exits_on_the_bar_whose_low_reaches_the_stop(self) -> None:
        bars: list[Bar] = [
            (1.1000, 1.1030, 1.0980, 1.1020),  # 0: nowhere near
            (1.1020, 1.1040, 1.0950, 1.1000),  # 1: still above 1.0900
            (1.1000, 1.1010, 1.0900, 1.0910),  # 2: low touches the stop exactly
            (1.0910, 1.0990, 1.0900, 1.0980),  # 3: never reached
        ]
        position = long_position(entry=1.1000, stop=1.0900)
        result = plan().run(position, exit_contexts(series(bars)))

        assert len(result.fills) == 1
        fill = result.fills[0]
        assert fill.bar_index == 2
        assert fill.rule == "protective_stop"
        assert fill.decision.reason is ExitReason.PROTECTIVE_STOP
        assert fill.decision.trigger is ExitTrigger.LEVEL_TOUCH
        assert fill.decision.order_type is OrderType.STOP
        assert fill.leg.price == pytest.approx(1.0900)
        assert fill.leg.ts == bar_close_ts(2)
        assert position.realized_r() == pytest.approx(-1.0)
        assert result.closed is True

    def test_a_long_stop_gapped_through_fills_at_the_open_not_the_level(self) -> None:
        bars: list[Bar] = [
            (1.1000, 1.1030, 1.0980, 1.1020),
            (1.0850, 1.0870, 1.0800, 1.0820),  # 1: opened below the stop
        ]
        position = long_position(entry=1.1000, stop=1.0900)
        result = plan().run(position, exit_contexts(series(bars)))

        assert result.fills[0].bar_index == 1
        assert result.fills[0].leg.price == pytest.approx(1.0850)
        assert position.realized_r() == pytest.approx(-1.5)

    def test_a_short_exits_when_the_high_reaches_the_stop(self) -> None:
        bars: list[Bar] = [
            (1.1000, 1.1050, 1.0950, 1.0980),
            (1.0980, 1.1100, 1.0960, 1.1090),  # 1: high touches 1.1100
        ]
        position = short_position(entry=1.1000, stop=1.1100)
        result = plan().run(position, exit_contexts(series(bars)))

        assert result.fills[0].bar_index == 1
        assert result.fills[0].leg.price == pytest.approx(1.1100)
        assert position.realized_r() == pytest.approx(-1.0)

    def test_a_tightened_stop_is_the_one_that_fires(self) -> None:
        position = long_position(entry=1.1000, stop=1.0900)
        position.tighten_stop(Price(1.0990), source="test")
        bars: list[Bar] = [(1.1010, 1.1030, 1.0985, 1.1000)]  # low pierces 1.0990 only
        result = plan().run(position, exit_contexts(series(bars)))

        assert result.fills[0].leg.price == pytest.approx(1.0990)
        assert result.fills[0].decision.context["initial_stop"] == pytest.approx(1.0900)
        assert result.fills[0].decision.context["stop_source"] == "test"
        assert result.fills[0].leg.stop_source == "test"

    def test_a_position_the_bars_never_reach_stays_open(self) -> None:
        bars: list[Bar] = [(1.1000, 1.1030, 1.0980, 1.1020)] * 5
        position = long_position(entry=1.1000, stop=1.0900)
        result = plan().run(position, exit_contexts(series(bars)))

        assert result.fills == ()
        assert result.closed is False
        assert position.is_open is True
        assert result.bars == 5


class TestFixedRR:
    def test_a_long_takes_profit_on_the_bar_that_reaches_the_target(self) -> None:
        # One R is 0.0100, so a 2R target sits at 1.1200.
        bars: list[Bar] = [
            (1.1000, 1.1150, 1.0950, 1.1100),  # 0: 1.5R at best, not enough
            (1.1100, 1.1199, 1.1050, 1.1180),  # 1: one tick short
            (1.1180, 1.1220, 1.1150, 1.1210),  # 2: through the target
        ]
        position = long_position(entry=1.1000, stop=1.0900)
        result = plan(take=2.0).run(position, exit_contexts(series(bars)))

        assert len(result.fills) == 1
        fill = result.fills[0]
        assert fill.bar_index == 2
        assert fill.rule == "fixed_rr_2"
        assert fill.decision.reason is ExitReason.TAKE_PROFIT
        assert fill.decision.order_type is OrderType.LIMIT
        assert fill.decision.kind is ExitKind.FULL
        assert fill.leg.price == pytest.approx(1.1200)
        assert position.realized_r() == pytest.approx(2.0)

    def test_a_short_takes_profit_when_the_low_reaches_the_target(self) -> None:
        bars: list[Bar] = [
            (1.1000, 1.1020, 1.0950, 1.0960),
            (1.0960, 1.0980, 1.0800, 1.0820),  # target at 1.0800 for 2R
        ]
        position = short_position(entry=1.1000, stop=1.1100)
        result = plan(take=2.0).run(position, exit_contexts(series(bars)))

        assert result.fills[0].bar_index == 1
        assert result.fills[0].leg.price == pytest.approx(1.0800)
        assert position.realized_r() == pytest.approx(2.0)

    def test_a_target_gapped_through_fills_better_than_the_level(self) -> None:
        bars: list[Bar] = [
            (1.1000, 1.1100, 1.0950, 1.1090),
            (1.1260, 1.1300, 1.1250, 1.1280),  # opened above the 1.1200 target
        ]
        position = long_position(entry=1.1000, stop=1.0900)
        result = plan(take=2.0).run(position, exit_contexts(series(bars)))

        assert result.fills[0].leg.price == pytest.approx(1.1260)
        assert position.realized_r() == pytest.approx(2.6)

    def test_the_target_does_not_move_when_the_stop_is_tightened(self) -> None:
        rule = FixedRR(2.0)
        position = long_position(entry=1.1000, stop=1.0900)
        before = rule.target_price(position)
        position.tighten_stop(Price(1.1000), source="test")
        assert rule.target_price(position) == pytest.approx(before)
        assert rule.target_price(position) == pytest.approx(1.1200)

    def test_a_non_positive_target_is_refused(self) -> None:
        with pytest.raises(ValueError, match="must be positive"):
            FixedRR(0.0)

    def test_neither_rule_requests_a_partial(self) -> None:
        assert ProtectiveStop().partial_fractions == ()
        assert FixedRR(2.0).partial_fractions == ()
