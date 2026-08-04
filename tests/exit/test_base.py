"""ExitDecision's field dependencies — the part that stops P13 from guessing."""

from decimal import Decimal

import pytest

from trading_system.core.types import OrderType, Price
from trading_system.exit.base import (
    REASON_PRIORITY,
    ExitDecision,
    ExitDropReason,
    ExitKind,
    ExitReason,
    ExitTrigger,
    IntrabarPolicy,
    empty_drop_counts,
)


class TestKindAndFraction:
    def test_a_full_exit_carries_no_fraction(self) -> None:
        # "All of it" has exactly one spelling. FULL with fraction=0.5 would be
        # two contradictory instructions in one object.
        with pytest.raises(ValueError, match="closes the remainder and carries no fraction"):
            ExitDecision(
                reason=ExitReason.TAKE_PROFIT,
                kind=ExitKind.FULL,
                trigger=ExitTrigger.BAR_CLOSE,
                fraction=Decimal("0.5"),
            )

    def test_a_partial_exit_requires_a_fraction(self) -> None:
        with pytest.raises(ValueError, match="requires a Decimal fraction"):
            ExitDecision(
                reason=ExitReason.TAKE_PROFIT,
                kind=ExitKind.PARTIAL,
                trigger=ExitTrigger.BAR_CLOSE,
            )

    def test_a_partial_fraction_must_be_a_decimal(self) -> None:
        with pytest.raises(ValueError, match="requires a Decimal fraction"):
            ExitDecision(
                reason=ExitReason.TAKE_PROFIT,
                kind=ExitKind.PARTIAL,
                trigger=ExitTrigger.BAR_CLOSE,
                fraction=0.5,  # type: ignore[arg-type]
            )

    @pytest.mark.parametrize("fraction", ["0", "1", "1.5", "-0.5"])
    def test_a_partial_fraction_outside_the_open_unit_interval_is_refused(
        self, fraction: str
    ) -> None:
        with pytest.raises(ValueError, match="strictly inside"):
            ExitDecision(
                reason=ExitReason.TAKE_PROFIT,
                kind=ExitKind.PARTIAL,
                trigger=ExitTrigger.BAR_CLOSE,
                fraction=Decimal(fraction),
            )


class TestTriggerAndPrice:
    def test_a_level_touch_exit_must_name_its_level(self) -> None:
        # It was a resting order. An order without a price is not an order.
        with pytest.raises(ValueError, match="must name its level"):
            ExitDecision(
                reason=ExitReason.PROTECTIVE_STOP,
                kind=ExitKind.FULL,
                trigger=ExitTrigger.LEVEL_TOUCH,
            )

    def test_a_bar_close_exit_must_not_name_a_price(self) -> None:
        # It fills at open(t+1), which the Exit Engine cannot see. Leaving room
        # for a price here is exactly how a consumer ends up inventing one.
        with pytest.raises(ValueError, match="must not name a price"):
            ExitDecision(
                reason=ExitReason.TAKE_PROFIT,
                kind=ExitKind.FULL,
                trigger=ExitTrigger.BAR_CLOSE,
                price=Price(1.1000),
            )

    def test_a_bar_close_exit_can_only_be_a_market_order(self) -> None:
        with pytest.raises(ValueError, match="can only be\nMARKET|can only be MARKET"):
            ExitDecision(
                reason=ExitReason.TAKE_PROFIT,
                kind=ExitKind.FULL,
                trigger=ExitTrigger.BAR_CLOSE,
                order_type=OrderType.LIMIT,
            )

    def test_a_non_finite_level_is_refused(self) -> None:
        with pytest.raises(ValueError, match="must name its level"):
            ExitDecision(
                reason=ExitReason.PROTECTIVE_STOP,
                kind=ExitKind.FULL,
                trigger=ExitTrigger.LEVEL_TOUCH,
                price=Price(float("inf")),
            )

    def test_the_two_classes_are_distinguishable_without_inference(self) -> None:
        resting = ExitDecision(
            reason=ExitReason.PROTECTIVE_STOP,
            kind=ExitKind.FULL,
            trigger=ExitTrigger.LEVEL_TOUCH,
            price=Price(1.0900),
            order_type=OrderType.STOP,
        )
        decided = ExitDecision(
            reason=ExitReason.TAKE_PROFIT,
            kind=ExitKind.FULL,
            trigger=ExitTrigger.BAR_CLOSE,
        )
        assert resting.price is not None and resting.trigger is ExitTrigger.LEVEL_TOUCH
        assert decided.price is None and decided.trigger is ExitTrigger.BAR_CLOSE


class TestAuditTrail:
    def test_context_is_read_only_and_out_of_equality(self) -> None:
        decision = ExitDecision(
            reason=ExitReason.PROTECTIVE_STOP,
            kind=ExitKind.FULL,
            trigger=ExitTrigger.LEVEL_TOUCH,
            price=Price(1.0900),
            order_type=OrderType.STOP,
            context={"stop": 1.09},
        )
        with pytest.raises(TypeError):
            decision.context["stop"] = 2.0  # type: ignore[index]
        same = ExitDecision(
            reason=ExitReason.PROTECTIVE_STOP,
            kind=ExitKind.FULL,
            trigger=ExitTrigger.LEVEL_TOUCH,
            price=Price(1.0900),
            order_type=OrderType.STOP,
            context={"stop": 999.0},
        )
        assert decision == same


class TestVocabulary:
    def test_every_reason_has_a_priority(self) -> None:
        # The priority table has to stay total: a reason added in a later stage
        # without one would otherwise sort by whatever the dict lookup raised.
        assert set(REASON_PRIORITY) == set(ExitReason)

    def test_the_default_intrabar_policy_is_pessimistic(self) -> None:
        # Guarded by a test rather than by a comment: an optimistic default is
        # the single most effective way to draw a return that never existed.
        from trading_system.exit.plan import ExitPlan
        from trading_system.exit.rules import ProtectiveStop

        plan = ExitPlan(exit_id="default-check", protective_stop=ProtectiveStop())
        assert plan.intrabar_policy is IntrabarPolicy.PESSIMISTIC

    def test_drop_counts_start_at_zero_for_every_reason(self) -> None:
        counts = empty_drop_counts()
        assert set(counts) == set(ExitDropReason)
        assert set(counts.values()) == {0}
