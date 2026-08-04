"""EntrySignal invariants — chiefly the unit and side guards on the two prices."""

from collections.abc import Mapping
from datetime import UTC, datetime

import pytest

from trading_system.core.types import Price, Side
from trading_system.entry.signal import EntrySignal

CLOSE_TS = datetime(2024, 1, 2, 9, 15, tzinfo=UTC)


def build(
    *,
    side: Side = Side.BUY,
    reference: float = 1.0847,
    invalidation: float = 1.0820,
    quality: float = 0.6,
    context: Mapping[str, object] | None = None,
) -> EntrySignal:
    """Build a signal, defaulting to a well-formed long."""
    return EntrySignal(
        strategy_id="test-entry",
        symbol="EURUSD",
        bar_close_ts=CLOSE_TS,
        side=side,
        reference_price=Price(reference),
        invalidation_price=Price(invalidation),
        quality=quality,
        context=context or {},
    )


class TestSideGuard:
    """A stop on the wrong side of the entry is a defect, not a risk decision."""

    def test_a_long_with_its_stop_below_is_accepted(self) -> None:
        assert build().invalidation_price == 1.0820

    def test_a_short_with_its_stop_above_is_accepted(self) -> None:
        assert build(side=Side.SELL, reference=1.0820, invalidation=1.0847).side is Side.SELL

    def test_a_long_with_its_stop_above_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="strictly below"):
            build(invalidation=1.0900)

    def test_a_short_with_its_stop_below_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="strictly above"):
            build(side=Side.SELL, reference=1.0847, invalidation=1.0820)

    def test_a_stop_equal_to_the_entry_is_rejected(self) -> None:
        # Zero risk distance divides by zero in every sizing formula there is.
        with pytest.raises(ValueError, match="strictly below"):
            build(invalidation=1.0847)

    def test_a_pip_distance_passed_as_a_level_fails_the_side_check(self) -> None:
        # The mistake question (a) is about: 15 pips written where 1.0820 was
        # meant. It lands above the entry of a long and is rejected here rather
        # than becoming a position size two orders of magnitude wrong.
        with pytest.raises(ValueError, match="not distances or pips"):
            build(invalidation=15.0)


class TestInvariants:
    """Everything else the dataclass refuses to construct."""

    @pytest.mark.parametrize("quality", [-0.01, 1.01])
    def test_quality_outside_the_unit_interval_is_rejected(self, quality: float) -> None:
        with pytest.raises(ValueError, match="quality"):
            build(quality=quality)

    @pytest.mark.parametrize("quality", [0.0, 1.0])
    def test_the_bounds_themselves_are_allowed(self, quality: float) -> None:
        assert build(quality=quality).quality == quality

    def test_a_non_finite_price_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="finite"):
            build(reference=float("inf"))

    def test_a_naive_timestamp_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="tz-aware"):
            EntrySignal(
                strategy_id="test-entry",
                symbol="EURUSD",
                bar_close_ts=datetime(2024, 1, 2, 9, 15),  # noqa: DTZ001
                side=Side.BUY,
                reference_price=Price(1.0847),
                invalidation_price=Price(1.0820),
                quality=0.5,
            )

    def test_a_non_utc_timestamp_is_normalised(self) -> None:
        from zoneinfo import ZoneInfo

        signal = EntrySignal(
            strategy_id="test-entry",
            symbol="EURUSD",
            bar_close_ts=datetime(2024, 1, 2, 11, 15, tzinfo=ZoneInfo("Europe/Riga")),
            side=Side.BUY,
            reference_price=Price(1.0847),
            invalidation_price=Price(1.0820),
            quality=0.5,
        )
        assert signal.bar_close_ts == CLOSE_TS


class TestDerivedValues:
    """The distance exists once, derived, so it cannot disagree with the levels."""

    def test_risk_distance_is_the_gap_between_the_two_levels(self) -> None:
        assert build(reference=1.0847, invalidation=1.0820).risk_distance == pytest.approx(0.0027)

    def test_risk_distance_is_positive_for_a_short_too(self) -> None:
        signal = build(side=Side.SELL, reference=1.0820, invalidation=1.0847)
        assert signal.risk_distance == pytest.approx(0.0027)

    def test_risk_distance_is_never_zero_because_the_side_guard_forbids_it(self) -> None:
        assert build().risk_distance > 0.0

    def test_the_signal_carries_no_size_or_money(self) -> None:
        # The architectural claim, asserted rather than left to the docstring.
        fields = set(EntrySignal.__dataclass_fields__)
        assert not fields & {"size", "risk_amount", "notional", "account_balance", "stop_loss"}

    def test_to_signal_projects_onto_the_core_contract(self) -> None:
        core = build().to_signal()
        assert core.strategy_id == "test-entry"
        assert core.symbol == "EURUSD"
        assert core.bar_close_ts == CLOSE_TS
        assert core.direction is Side.BUY
        assert core.quality == 0.6
        assert core.invalidation_price == 1.0820


class TestContext:
    """The audit trail is read-only and is not part of identity."""

    def test_context_cannot_be_mutated_by_a_consumer(self) -> None:
        signal = build(context={"trigger_bar_index": 7})
        with pytest.raises(TypeError):
            signal.context["trigger_bar_index"] = 8  # type: ignore[index]

    def test_context_is_detached_from_the_dict_passed_in(self) -> None:
        source = {"trigger_bar_index": 7}
        signal = build(context=source)
        source["trigger_bar_index"] = 8
        assert signal.context["trigger_bar_index"] == 7

    def test_two_signals_differing_only_in_context_are_equal(self) -> None:
        assert build(context={"a": 1}) == build(context={"b": 2})

    def test_signals_remain_hashable(self) -> None:
        assert len({build(context={"a": 1}), build(context={"b": 2})}) == 1
