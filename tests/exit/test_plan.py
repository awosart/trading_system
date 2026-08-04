"""Composition: phases, the intrabar race, deferral, and the fraction backstop.

The stub rules here exist because stage 1 ships two real ones, both of which
close in full on a touched level. Partial ladders and close-decided exits arrive
in stage 2, but the composition machinery that has to handle them is here now,
and it needs something to compose.
"""

from decimal import Decimal

import pytest

from trading_system.core.exceptions import ValidationError
from trading_system.core.types import OrderType, Price, Side
from trading_system.exit.base import (
    ExitDecision,
    ExitDropReason,
    ExitKind,
    ExitReason,
    ExitTrigger,
    IntrabarPolicy,
)
from trading_system.exit.context import ExitContext, exit_contexts
from trading_system.exit.fills import resting_order_filled
from trading_system.exit.plan import ExitPlan
from trading_system.exit.position import ManagedPosition
from trading_system.exit.rules import FixedRR, ProtectiveStop

from .conftest import Bar, bar_open_ts, long_position, series, ticks


class PartialAt:
    """Stub: close a fixed share of the original size at an R level, once."""

    def __init__(self, r_multiple: float, fraction: str) -> None:
        """Close ``fraction`` of the original size the first time ``r_multiple`` is reached."""
        self._r_multiple = r_multiple
        self._fraction = Decimal(fraction)
        self._fired = False

    @property
    def name(self) -> str:
        return f"partial_{self._r_multiple:g}r"

    @property
    def partial_fractions(self) -> tuple[Decimal, ...]:
        return (self._fraction,)

    def on_bar(self, position: ManagedPosition, ctx: ExitContext) -> ExitDecision | None:
        high, low = ctx.price("high"), ctx.price("low")
        if self._fired or high is None or low is None:
            return None
        direction = 1.0 if position.side is Side.BUY else -1.0
        level = Price(
            position.entry_price + direction * self._r_multiple * position.initial_risk_distance
        )
        if not resting_order_filled(
            level, high=high, low=low, exit_side=position.exit_side, order_type=OrderType.LIMIT
        ):
            return None
        self._fired = True
        return ExitDecision(
            reason=ExitReason.TAKE_PROFIT,
            kind=ExitKind.PARTIAL,
            trigger=ExitTrigger.LEVEL_TOUCH,
            price=level,
            fraction=self._fraction,
            order_type=OrderType.LIMIT,
        )

    def reset(self) -> None:
        self._fired = False


class CloseOnBar:
    """Stub: a close-decided full exit on one named bar."""

    def __init__(self, bar_index: int) -> None:
        """Decide a full exit on the close of bar ``bar_index``."""
        self._bar_index = bar_index

    @property
    def name(self) -> str:
        return "close_on_bar"

    @property
    def partial_fractions(self) -> tuple[Decimal, ...]:
        return ()

    def on_bar(
        self,
        position: ManagedPosition,  # noqa: ARG002
        ctx: ExitContext,
    ) -> ExitDecision | None:
        if ctx.index != self._bar_index:
            return None
        return ExitDecision(
            reason=ExitReason.TAKE_PROFIT,
            kind=ExitKind.FULL,
            trigger=ExitTrigger.BAR_CLOSE,
        )

    def reset(self) -> None:
        """No state."""


class MoveStopTo:
    """Stub stop modifier: propose one fixed level, every bar."""

    def __init__(self, level: float) -> None:
        """Propose ``level`` as the stop on every bar."""
        self._level = Price(level)
        self.proposals = 0

    @property
    def name(self) -> str:
        return "move_stop_to"

    def on_bar(
        self,
        position: ManagedPosition,  # noqa: ARG002
        ctx: ExitContext,  # noqa: ARG002
    ) -> Price | None:
        self.proposals += 1
        return self._level

    def reset(self) -> None:
        self.proposals = 0


#: A bar that touches both a long's 1.0900 stop and its 1.1200 target, opening
#: between the two so neither is reached by a gap.
CONFLICT_BAR: Bar = (1.1100, 1.1250, 1.0850, 1.1000)


def conflict_plan(policy: IntrabarPolicy) -> ExitPlan:
    """The stop-versus-target plan, under one intrabar policy."""
    return ExitPlan(
        exit_id="conflict",
        protective_stop=ProtectiveStop(),
        rules=[FixedRR(2.0)],
        intrabar_policy=policy,
    )


class TestConstruction:
    def test_a_plan_without_a_protective_stop_is_not_expressible(self) -> None:
        # No default and no None: an unprotected position cannot be reached
        # through this constructor at all.
        with pytest.raises(TypeError):
            ExitPlan(exit_id="unprotected")  # type: ignore[call-arg]

    def test_an_empty_exit_id_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="non-empty"):
            ExitPlan(exit_id="", protective_stop=ProtectiveStop())

    def test_the_protective_stop_leads_the_rule_list(self) -> None:
        plan = ExitPlan(exit_id="ordering", protective_stop=ProtectiveStop(), rules=[FixedRR(2.0)])
        assert [rule.name for rule in plan.rules] == ["protective_stop", "fixed_rr_2"]


class TestIntrabarConflict:
    def test_pessimistic_resolves_a_stop_and_target_bar_as_a_loss(self) -> None:
        # The DoD case. Both levels were touched; without ticks the order is
        # unknowable; the default assumes the loss came first.
        position = long_position(entry=1.1000, stop=1.0900)
        result = conflict_plan(IntrabarPolicy.PESSIMISTIC).run(
            position, exit_contexts(series([CONFLICT_BAR]))
        )

        assert len(result.fills) == 1
        assert result.fills[0].decision.reason is ExitReason.PROTECTIVE_STOP
        assert result.fills[0].leg.price == pytest.approx(1.0900)
        assert position.realized_r() == pytest.approx(-1.0)
        assert position.realized_quote_move < 0

    def test_optimistic_resolves_the_same_bar_as_a_win(self) -> None:
        position = long_position(entry=1.1000, stop=1.0900)
        result = conflict_plan(IntrabarPolicy.OPTIMISTIC).run(
            position, exit_contexts(series([CONFLICT_BAR]))
        )

        assert result.fills[0].decision.reason is ExitReason.TAKE_PROFIT
        assert position.realized_r() == pytest.approx(2.0)

    def test_the_stop_wins_under_pessimism_without_any_priority_table(self) -> None:
        # The ordering key is the outcome in R, so "the stop outranks the take"
        # is a consequence of a stop being worse, not a setting anyone can edit.
        position = long_position(entry=1.1000, stop=1.0900)
        plan = conflict_plan(IntrabarPolicy.PESSIMISTIC)
        ctx = ExitContext(series([CONFLICT_BAR]).context(0))
        outcome = plan.on_bar(position, ctx)
        assert [fill.decision.reason for fill in outcome.applied] == [ExitReason.PROTECTIVE_STOP]

    def test_tick_data_puts_the_bar_in_the_order_it_actually_happened(self) -> None:
        for prices, expected_reason, expected_r in (
            ([1.1210, 1.0880], ExitReason.TAKE_PROFIT, 2.0),
            ([1.0880, 1.1210], ExitReason.PROTECTIVE_STOP, -1.0),
        ):
            position = long_position(entry=1.1000, stop=1.0900)
            result = conflict_plan(IntrabarPolicy.TICK_BASED).run(
                position, exit_contexts(series([CONFLICT_BAR]), {0: ticks(0, prices)})
            )
            assert result.fills[0].decision.reason is expected_reason
            assert position.realized_r() == pytest.approx(expected_r)
            assert result.drops[ExitDropReason.TICKS_UNAVAILABLE] == 0

    def test_tick_based_without_ticks_falls_back_to_pessimistic_and_says_so(self) -> None:
        # A silent fallback would let a run configured for tick precision report
        # fills it never earned, or hide that the tick file was missing.
        position = long_position(entry=1.1000, stop=1.0900)
        result = conflict_plan(IntrabarPolicy.TICK_BASED).run(
            position, exit_contexts(series([CONFLICT_BAR]))
        )

        assert result.fills[0].decision.reason is ExitReason.PROTECTIVE_STOP
        assert result.drops[ExitDropReason.TICKS_UNAVAILABLE] == 1
        assert result.dropped == 1

    def test_a_bar_touching_only_one_level_needs_no_policy(self) -> None:
        only_target: Bar = (1.1100, 1.1250, 1.1050, 1.1200)
        for policy in IntrabarPolicy:
            position = long_position(entry=1.1000, stop=1.0900)
            result = conflict_plan(policy).run(position, exit_contexts(series([only_target])))
            assert result.fills[0].decision.reason is ExitReason.TAKE_PROFIT
            # Nothing raced, so no tick data was needed to resolve anything.
            assert result.drops[ExitDropReason.TICKS_UNAVAILABLE] == 0


class TestPhases:
    def test_a_modifiers_proposal_on_bar_t_is_not_checkable_until_bar_t_plus_1(self) -> None:
        # A rule seeing this bar's own tightened level tested against this same
        # bar's own low would be testing a level against the data used to
        # derive it — the same hazard P06 ruled out for an invalidation on its
        # own trigger bar. Bar 0's low (1.0985) is below the modifier's
        # proposal (1.0990), so a same-bar effect would wrongly stop it out
        # here; the position must instead survive bar 0 and only be caught once
        # the tightened level is live, on bar 1.
        bars: list[Bar] = [
            (1.1010, 1.1030, 1.0985, 1.1000),
            (1.1000, 1.1010, 1.0985, 1.0995),
        ]
        position = long_position(entry=1.1000, stop=1.0900)
        modifier = MoveStopTo(1.0990)
        plan = ExitPlan(
            exit_id="phases", protective_stop=ProtectiveStop(), stop_modifiers=[modifier]
        )
        result = plan.run(position, exit_contexts(series(bars)))

        # The modifier proposes once, on bar 0; bar 1 closes the position
        # during the rule phase, before the modifier phase ever runs again.
        assert modifier.proposals == 1
        assert len(result.fills) == 1
        assert result.fills[0].bar_index == 1
        assert result.fills[0].leg.price == pytest.approx(1.0990)

    def test_a_modifier_never_gets_to_react_to_its_own_bars_data(self) -> None:
        # The general form of the above: a modifier that reads this bar's own
        # high to compute a level must never have that level tested against
        # this same bar's own low, however tight the two are.
        bars: list[Bar] = [(1.1150, 1.1200, 1.1140, 1.1180)]  # range spans the proposal
        position = long_position(entry=1.1000, stop=1.0900)
        plan = ExitPlan(
            exit_id="same-bar-immunity",
            protective_stop=ProtectiveStop(),
            stop_modifiers=[MoveStopTo(1.1150)],  # inside this very bar's [1.1140, 1.1200]
        )
        result = plan.run(position, exit_contexts(series(bars)))

        assert result.fills == ()
        assert position.stop == pytest.approx(1.1150)
        assert position.is_open is True

    def test_a_modifier_cannot_widen_the_stop_through_the_plan_either(self) -> None:
        bars: list[Bar] = [(1.1010, 1.1030, 1.0995, 1.1000)] * 3
        position = long_position(entry=1.1000, stop=1.0950)
        plan = ExitPlan(
            exit_id="ratchet",
            protective_stop=ProtectiveStop(),
            stop_modifiers=[MoveStopTo(1.0800)],
        )
        result = plan.run(position, exit_contexts(series(bars)))

        assert position.stop == pytest.approx(1.0950)
        assert result.fills == ()


class TestDeferredDecisions:
    def test_a_close_decided_exit_fills_at_the_next_bars_open(self) -> None:
        bars: list[Bar] = [
            (1.1000, 1.1030, 1.0980, 1.1020),
            (1.1020, 1.1040, 1.0990, 1.1010),  # decided here, on the close
            (1.1050, 1.1070, 1.1040, 1.1060),  # filled here, at the open
        ]
        position = long_position(entry=1.1000, stop=1.0900)
        plan = ExitPlan(exit_id="deferred", protective_stop=ProtectiveStop(), rules=[CloseOnBar(1)])
        result = plan.run(position, exit_contexts(series(bars)))

        assert len(result.fills) == 1
        fill = result.fills[0]
        assert fill.decided_bar_index == 1
        assert fill.bar_index == 2
        assert fill.decision.trigger is ExitTrigger.BAR_CLOSE
        assert fill.decision.price is None
        assert fill.leg.price == pytest.approx(1.1050)
        assert fill.leg.ts == bar_open_ts(2)

    def test_a_deferred_exit_fills_before_the_next_bars_levels_are_examined(self) -> None:
        # The open precedes every other price in the bar, so nothing resting
        # inside it can have come first — not even the stop.
        bars: list[Bar] = [
            (1.1000, 1.1030, 1.0980, 1.1020),
            (1.1020, 1.1040, 1.0990, 1.1010),  # decided here
            (1.1000, 1.1010, 1.0800, 1.0850),  # opens at 1.1000, then hits the stop
        ]
        position = long_position(entry=1.1000, stop=1.0900)
        plan = ExitPlan(
            exit_id="deferred-race", protective_stop=ProtectiveStop(), rules=[CloseOnBar(1)]
        )
        result = plan.run(position, exit_contexts(series(bars)))

        assert len(result.fills) == 1
        assert result.fills[0].leg.price == pytest.approx(1.1000)
        assert position.realized_r() == pytest.approx(0.0)

    def test_nothing_is_deferred_from_a_bar_that_already_closed_the_position(self) -> None:
        bars: list[Bar] = [
            (1.1000, 1.1030, 1.0980, 1.1020),
            (1.1020, 1.1040, 1.0850, 1.0860),  # stop hit on the deciding bar
            (1.0860, 1.0900, 1.0800, 1.0850),
        ]
        position = long_position(entry=1.1000, stop=1.0900)
        plan = ExitPlan(
            exit_id="deferred-moot", protective_stop=ProtectiveStop(), rules=[CloseOnBar(1)]
        )
        result = plan.run(position, exit_contexts(series(bars)))

        assert len(result.fills) == 1
        assert result.fills[0].decision.reason is ExitReason.PROTECTIVE_STOP
        assert result.bars == 2


class TestPartialComposition:
    def test_partials_sum_to_the_whole_position_and_pnl_adds_up(self) -> None:
        bars: list[Bar] = [
            (1.1000, 1.1110, 1.0990, 1.1100),  # 0: reaches 1R only
            (1.1100, 1.1210, 1.1090, 1.1200),  # 1: reaches 2R
        ]
        position = long_position(entry=1.1000, stop=1.0900, size=Decimal("1"))
        plan = ExitPlan(
            exit_id="ladder",
            protective_stop=ProtectiveStop(),
            rules=[PartialAt(1.0, "0.5"), PartialAt(2.0, "0.5")],
        )
        result = plan.run(position, exit_contexts(series(bars)))

        assert [fill.bar_index for fill in result.fills] == [0, 1]
        assert [fill.leg.price for fill in result.fills] == pytest.approx([1.1100, 1.1200])
        assert sum(leg.fraction for leg in position.legs) == Decimal("1.0")
        assert position.remaining_fraction == Decimal("0.0")
        # 0.5 * 1R + 0.5 * 2R, and the money agrees: 0.5*0.01 + 0.5*0.02.
        assert position.realized_r() == pytest.approx(1.5)
        assert position.realized_quote_move == Decimal("0.015")
        assert result.closed is True
        assert result.dropped == 0

    def test_a_partial_larger_than_the_remainder_is_promoted_and_counted(self) -> None:
        # 60% then 60% of the same position: the second is unexecutable as
        # stated, and closing what is left is its only executable reading.
        bars: list[Bar] = [(1.1000, 1.1160, 1.0990, 1.1150)]
        position = long_position(entry=1.1000, stop=1.0900)
        plan = ExitPlan(
            exit_id="oversized",
            protective_stop=ProtectiveStop(),
            rules=[PartialAt(1.0, "0.6"), PartialAt(1.5, "0.6")],
        )
        result = plan.run(position, exit_contexts(series(bars)))

        assert [leg.fraction for leg in position.legs] == [Decimal("0.6"), Decimal("0.4")]
        assert position.remaining_fraction == Decimal("0.0")
        assert result.drops[ExitDropReason.PARTIAL_EXCEEDS_REMAINDER] == 1
        assert result.fills[1].decision.kind is ExitKind.PARTIAL
        assert result.fills[1].leg.fraction == Decimal("0.4")
        assert position.realized_r() == pytest.approx(0.6 * 1.0 + 0.4 * 1.5)

    def test_smallest_partial_fraction_reports_the_tightest_rung_statically(self) -> None:
        # Must answer without a run: there is no position yet when it is asked.
        plan = ExitPlan(
            exit_id="ladder",
            protective_stop=ProtectiveStop(),
            rules=[PartialAt(1.0, "0.5"), PartialAt(2.0, "0.25")],
        )
        assert plan.smallest_partial_fraction() == Decimal("0.25")

    def test_a_full_close_only_plan_requests_no_partials(self) -> None:
        plan = ExitPlan(
            exit_id="all-or-nothing", protective_stop=ProtectiveStop(), rules=[FixedRR(2.0)]
        )
        assert plan.smallest_partial_fraction() is None

    def test_a_full_close_only_plans_smallest_close_is_the_whole_position(self) -> None:
        # Not None: the Risk Engine's min_lot check then needs no special case,
        # it degenerates into the ordinary "is this size tradable" question.
        plan = ExitPlan(
            exit_id="all-or-nothing", protective_stop=ProtectiveStop(), rules=[FixedRR(2.0)]
        )
        assert plan.smallest_closing_fraction() == Decimal("1")

    def test_the_residual_counts_as_a_close_even_though_no_rule_requests_it(self) -> None:
        # The hole this method exists to close. A 0.5 + 0.4 ladder requests
        # nothing below 0.4, but after both rungs fire, 0.1 of the position is
        # still open and gets closed in one go by whatever fires next. It is
        # that 0.1 which decides whether the pairing survives min_lot, and
        # reading the rungs alone would clear a plan whose last close is four
        # times too small to execute.
        plan = ExitPlan(
            exit_id="leaves-a-tail",
            protective_stop=ProtectiveStop(),
            rules=[PartialAt(1.0, "0.5"), PartialAt(2.0, "0.4")],
        )
        assert plan.smallest_partial_fraction() == Decimal("0.4")
        assert plan.smallest_closing_fraction() == Decimal("0.1")

    def test_a_ladder_that_closes_the_position_exactly_leaves_no_residual(self) -> None:
        plan = ExitPlan(
            exit_id="exact",
            protective_stop=ProtectiveStop(),
            rules=[PartialAt(1.0, "0.5"), PartialAt(2.0, "0.25"), PartialAt(3.0, "0.25")],
        )
        assert plan.smallest_closing_fraction() == Decimal("0.25")


class TestRunAccounting:
    def test_every_drop_reason_is_present_even_at_zero(self) -> None:
        # "No ticks were missing" is a recorded fact, not an absent key.
        position = long_position(entry=1.1000, stop=1.0900)
        result = conflict_plan(IntrabarPolicy.PESSIMISTIC).run(
            position, exit_contexts(series([CONFLICT_BAR]))
        )
        assert set(result.drops) == set(ExitDropReason)
        assert result.drops[ExitDropReason.PARTIAL_QUANTIZED_TO_ZERO] == 0

    def test_reset_clears_the_counters_between_runs(self) -> None:
        plan = conflict_plan(IntrabarPolicy.TICK_BASED)
        first = plan.run(long_position(), exit_contexts(series([CONFLICT_BAR])))
        second = plan.run(long_position(), exit_contexts(series([CONFLICT_BAR])))
        assert first.drops[ExitDropReason.TICKS_UNAVAILABLE] == 1
        assert second.drops[ExitDropReason.TICKS_UNAVAILABLE] == 1

    def test_an_unfinished_position_is_reported_as_not_closed(self) -> None:
        bars: list[Bar] = [(1.1000, 1.1030, 1.0980, 1.1020)] * 4
        position = long_position(entry=1.1000, stop=1.0900)
        result = conflict_plan(IntrabarPolicy.PESSIMISTIC).run(
            position, exit_contexts(series(bars))
        )
        assert result.closed is False
        assert result.bars == 4
        assert position.remaining_fraction == Decimal("1")
