"""Each firm rule on its own: a scenario of trades in, one verdict out.

Every account figure here is written by hand rather than produced by a
backtest. The guard's whole contract is arithmetic over a snapshot, and a test
that built the snapshot by running the engine would be checking the engine.
"""

from datetime import UTC, datetime, time
from decimal import Decimal
from pathlib import Path

import pytest

from trading_system.core.instruments import InstrumentRegistry, InstrumentSpec, load_instruments
from trading_system.data.resample import FX_DAY_ORIGIN, DayOrigin
from trading_system.prop.guard import (
    DEFAULT_BUFFER,
    GuardDecision,
    PropAccountState,
    PropGuard,
    ProposedOrder,
    PropReason,
)
from trading_system.prop.rules import (
    DailyLossBasis,
    PropRules,
    TotalLossBasis,
    day_origin_divergence,
    load_prop_rules,
)

REGISTRY_PATH = Path(__file__).resolve().parents[2] / "configs" / "instruments.yaml"
RULES_PATH = Path(__file__).resolve().parents[2] / "configs" / "prop_rules.yaml"
NOW = datetime(2024, 3, 5, 12, 0, tzinfo=UTC)


@pytest.fixture(scope="module")
def registry() -> InstrumentRegistry:
    """The bundled instrument registry."""
    return load_instruments(REGISTRY_PATH)


@pytest.fixture(scope="module")
def eurusd(registry: InstrumentRegistry) -> InstrumentSpec:
    """One instrument, for reduction arithmetic."""
    return registry["EURUSD"]


def rules(**overrides: object) -> PropRules:
    """A rule set with FTMO-like defaults, overridable field by field."""
    base: dict[str, object] = {
        "name": "test-plan",
        "prop_profile": "ftmo_swing",
        "source": "test",
        "account_size": Decimal("100000"),
        "profit_target_pct": 0.10,
        "max_daily_loss_pct": 0.05,
        "daily_loss_basis": DailyLossBasis.BALANCE_AT_DAY_START,
        "daily_reset_time": time(0, 0),
        "daily_reset_tz": "Europe/Prague",
        "max_total_loss_pct": 0.10,
        "total_loss_basis": TotalLossBasis.STATIC,
        "max_single_day_profit_share": 0.5,
        "min_trading_days": 4,
    }
    base.update(overrides)
    return PropRules.model_validate(base)


def account(
    equity: str,
    *,
    balance: str | None = None,
    day_start_balance: str = "100000",
    day_start_equity: str = "100000",
    high_water: str = "100000",
    at: datetime = NOW,
) -> PropAccountState:
    """One account snapshot, every figure named."""
    return PropAccountState(
        at=at,
        equity=Decimal(equity),
        balance=Decimal(balance if balance is not None else equity),
        day_start_balance=Decimal(day_start_balance),
        day_start_equity=Decimal(day_start_equity),
        high_water_mark=Decimal(high_water),
        day=at.date(),
    )


def order(eurusd: InstrumentSpec, *, risk: str, size: str = "1.00") -> ProposedOrder:
    """One sized order."""
    return ProposedOrder(
        symbol="EURUSD", size=Decimal(size), risk_amount=Decimal(risk), instrument=eurusd
    )


class TestDailyLossLimit:
    def test_at_the_daily_floor_the_guard_refuses(self, eurusd: InstrumentSpec) -> None:
        # 5% of a 100 000 day start puts the floor at 95 000.
        guard = PropGuard(rules())
        verdict = guard.check(order(eurusd, risk="100"), account("95000"))
        assert verdict.decision is GuardDecision.REJECT
        assert verdict.reason is PropReason.DAILY_LOSS_LIMIT
        assert verdict.allowed_size == 0

    def test_above_the_floor_with_room_to_spare_it_trades(self, eurusd: InstrumentSpec) -> None:
        guard = PropGuard(rules())
        # 99 000 leaves 4 000 above the daily floor and 9 000 above the total
        # one; the nearer is 4 000, and 80% of that is 3 200 against a 100 risk.
        verdict = guard.check(order(eurusd, risk="100"), account("99000"))
        assert verdict.decision is GuardDecision.ALLOW

    def test_a_hair_above_the_floor_leaves_too_little_to_trade(
        self, eurusd: InstrumentSpec
    ) -> None:
        """Above the floor is not the same as allowed to trade — that is the buffer.

        A cent of headroom buys 0.8 of a cent of allowance, so a 100 risk does
        not fit and cannot be scaled down to anything the instrument can trade.
        """
        guard = PropGuard(rules())
        verdict = guard.check(order(eurusd, risk="100"), account("95000.01"))
        assert verdict.decision is GuardDecision.REJECT
        assert verdict.reason is PropReason.BELOW_MIN_LOT_AFTER_REDUCTION

    def test_the_basis_decides_whether_floating_carried_over_counts(
        self, eurusd: InstrumentSpec
    ) -> None:
        """The one field that separates the two readings, on one snapshot.

        The day opened with 100 000 closed but 98 000 of equity: 2 000 was
        floating red on a position carried through the reset. Under a
        balance basis the floor is 95 000; under an equity basis it is 93 100.
        Equity is 94 000 — below one and above the other.
        """
        snapshot = account("94000", day_start_balance="100000", day_start_equity="98000")

        by_balance = PropGuard(rules(daily_loss_basis=DailyLossBasis.BALANCE_AT_DAY_START))
        assert by_balance.daily_floor(snapshot) == Decimal("95000.00")
        assert by_balance.check(order(eurusd, risk="100"), snapshot).blocked

        by_equity = PropGuard(rules(daily_loss_basis=DailyLossBasis.EQUITY))
        assert by_equity.daily_floor(snapshot) == Decimal("93100.00")
        assert not by_equity.check(order(eurusd, risk="100"), snapshot).blocked


class TestTotalLossLimit:
    def test_static_floor_does_not_move_with_profit(self) -> None:
        guard = PropGuard(rules(total_loss_basis=TotalLossBasis.STATIC))
        # Peaked at 120 000, now back at 91 000. The static floor is 90 000,
        # so the account survives.
        snapshot = account(
            "91000", day_start_balance="120000", day_start_equity="120000", high_water="120000"
        )
        assert guard.total_floor(snapshot) == Decimal("90000.00")

    def test_trailing_floor_ratchets_up_behind_the_peak(self, eurusd: InstrumentSpec) -> None:
        guard = PropGuard(rules(total_loss_basis=TotalLossBasis.TRAILING_HIGH_WATER))
        # The day opened at 109 000, so its own floor is 103 550 — below the
        # trailing floor, which is what leaves the total limit as the binding
        # one rather than the daily one the guard checks first.
        snapshot = account(
            "107000", day_start_balance="109000", day_start_equity="109000", high_water="120000"
        )
        # 10% below the 120 000 peak is 108 000; 107 000 is through it, even
        # though the account is 7 000 above where it started.
        assert guard.total_floor(snapshot) == Decimal("108000.00")
        assert guard.daily_floor(snapshot) == Decimal("103550.00")
        verdict = guard.check(order(eurusd, risk="100"), snapshot)
        assert verdict.decision is GuardDecision.REJECT
        assert verdict.reason is PropReason.TOTAL_LOSS_LIMIT

    def test_profit_buys_a_buffer_under_static_and_none_under_trailing(self) -> None:
        """The whole practical difference between the two bases, on one number."""
        snapshot = account(
            "110000", day_start_balance="110000", day_start_equity="110000", high_water="110000"
        )
        static = PropGuard(rules(total_loss_basis=TotalLossBasis.STATIC))
        trailing = PropGuard(rules(total_loss_basis=TotalLossBasis.TRAILING_HIGH_WATER))
        # Static: 20 000 of room below. Trailing: 11 000, and it rises again
        # with the next new peak.
        assert snapshot.equity - static.total_floor(snapshot) == Decimal("20000.00")
        assert snapshot.equity - trailing.total_floor(snapshot) == Decimal("11000.00")


class TestFloatingDrawdownBlocks:
    """DoD: an open floating loss stops new trades, with nothing realised."""

    def test_unrealised_loss_alone_reaches_the_daily_floor(self, eurusd: InstrumentSpec) -> None:
        guard = PropGuard(rules(daily_loss_basis=DailyLossBasis.BALANCE_AT_DAY_START))
        # Nothing closed all day — balance is untouched at 100 000 — but open
        # positions are 5 200 in the red, so equity is 94 800, through the
        # 95 000 floor. A guard reading realised P&L would allow this.
        snapshot = account("94800", balance="100000")
        verdict = guard.check(order(eurusd, risk="100"), snapshot)
        assert verdict.decision is GuardDecision.REJECT
        assert verdict.reason is PropReason.DAILY_LOSS_LIMIT
        assert snapshot.balance == Decimal("100000")


class TestMaxAllowedRiskNowDecays:
    """DoD: the ceiling falls as the day's loss grows."""

    def test_the_ceiling_shrinks_monotonically_towards_the_floor(self) -> None:
        guard = PropGuard(rules())
        equities = ["100000", "98000", "96000", "95500", "95100"]
        allowances = [guard.max_allowed_risk_now(account(equity)) for equity in equities]
        assert allowances == sorted(allowances, reverse=True)
        assert all(
            later < earlier for earlier, later in zip(allowances, allowances[1:], strict=False)
        )

    def test_the_ceiling_is_the_buffered_distance_to_the_nearer_floor(self) -> None:
        guard = PropGuard(rules())
        # Daily floor 95 000, total floor 90 000: the daily one is nearer.
        # 96 200 - 95 000 = 1 200, and 80% of that is 960.
        assert guard.max_allowed_risk_now(account("96200")) == Decimal("960.000")

    def test_the_brief_example_holds_exactly(self, eurusd: InstrumentSpec) -> None:
        """1.2% left, a 1% risk, and the answer is no.

        The stated reason the buffer exists: without it the last permitted
        trade is the one that breaches the limit.
        """
        guard = PropGuard(rules())
        snapshot = account("96200")  # 1.2% of 100 000 above the daily floor
        assert guard.max_allowed_risk_now(snapshot) == Decimal("960.000")
        verdict = guard.check(order(eurusd, risk="1000"), snapshot)
        assert verdict.decision is not GuardDecision.ALLOW

    def test_below_the_floor_the_ceiling_is_zero_not_negative(self) -> None:
        guard = PropGuard(rules())
        assert guard.max_allowed_risk_now(account("94000")) == Decimal(0)

    def test_the_total_floor_takes_over_when_it_is_the_nearer_one(self) -> None:
        guard = PropGuard(rules())
        # A day that opened at 92 000: its daily floor is 87 400, below the
        # static total floor of 90 000, so the total floor binds.
        snapshot = account("91000", day_start_balance="92000", day_start_equity="92000")
        assert guard.daily_floor(snapshot) == Decimal("87400.00")
        assert guard.total_floor(snapshot) == Decimal("90000.00")
        assert guard.max_allowed_risk_now(snapshot) == Decimal("800.000")


class TestReduction:
    def test_an_oversized_order_is_cut_to_fit_rather_than_refused(
        self, eurusd: InstrumentSpec
    ) -> None:
        guard = PropGuard(rules())
        # 960 available against a 1 000 risk on 1.00 lot: 0.96 lots, which
        # quantises cleanly on a 0.01 step.
        verdict = guard.check(order(eurusd, risk="1000", size="1.00"), account("96200"))
        assert verdict.decision is GuardDecision.REDUCE
        assert verdict.reason is PropReason.REDUCED_TO_REMAINING
        assert verdict.allowed_size == Decimal("0.96")

    def test_a_reduction_below_min_lot_is_a_refusal_not_a_rounding_up(
        self, eurusd: InstrumentSpec
    ) -> None:
        guard = PropGuard(rules())
        # Barely any headroom: scaling 1.00 lot to fit gives less than the
        # 0.01 minimum, and rounding up would spend allowance the rule says is
        # not there.
        snapshot = account("95000.50")
        verdict = guard.check(order(eurusd, risk="1000", size="1.00"), snapshot)
        assert verdict.decision is GuardDecision.REJECT
        assert verdict.reason is PropReason.BELOW_MIN_LOT_AFTER_REDUCTION
        assert verdict.allowed_size == 0

    def test_reduction_never_exceeds_the_allowance(self, eurusd: InstrumentSpec) -> None:
        guard = PropGuard(rules())
        snapshot = account("96200")
        verdict = guard.check(order(eurusd, risk="1000", size="1.00"), snapshot)
        implied_risk = Decimal("1000") * verdict.allowed_size / Decimal("1.00")
        assert implied_risk <= verdict.max_risk_now


class TestBufferIsConfigurableAndBounded:
    def test_a_full_buffer_permits_exactly_the_headroom(self) -> None:
        guard = PropGuard(rules(), buffer=1.0)
        assert guard.max_allowed_risk_now(account("96200")) == Decimal("1200.0")

    def test_the_default_is_eighty_percent(self) -> None:
        assert DEFAULT_BUFFER == 0.80

    @pytest.mark.parametrize("bad", [0.0, -0.1, 1.5])
    def test_a_buffer_outside_the_unit_interval_is_refused(self, bad: float) -> None:
        with pytest.raises(ValueError, match="buffer must be in"):
            PropGuard(rules(), buffer=bad)


class TestCounters:
    def test_every_reason_is_present_from_the_start(self) -> None:
        guard = PropGuard(rules())
        assert set(guard.counts) == set(PropReason)
        assert all(count == 0 for count in guard.counts.values())

    def test_a_refusal_is_counted_and_a_reset_clears_it(self, eurusd: InstrumentSpec) -> None:
        guard = PropGuard(rules())
        guard.check(order(eurusd, risk="100"), account("95000"))
        assert guard.counts[PropReason.DAILY_LOSS_LIMIT] == 1
        guard.reset()
        assert guard.counts[PropReason.DAILY_LOSS_LIMIT] == 0


class TestRulesValidation:
    def test_a_daily_limit_at_or_above_the_total_one_is_refused(self) -> None:
        with pytest.raises(Exception, match="is not below"):
            rules(max_daily_loss_pct=0.10, max_total_loss_pct=0.10)

    def test_the_shipped_rule_sets_all_load_and_name_a_real_profile(self) -> None:
        from trading_system.risk.margin import load_prop_profiles

        library = load_prop_rules(RULES_PATH)
        profiles = load_prop_profiles(
            Path(__file__).resolve().parents[2] / "configs" / "prop_profiles.yaml"
        )
        assert set(library.names) == {
            "ftmo_normal",
            "ftmo_swing",
            "the5ers_bootcamp",
            "the5ers_high_stakes",
        }
        for name in library.names:
            library.get(name).resolve_profile(profiles)

    def test_no_rule_set_restates_leverage(self) -> None:
        """Leverage lives in prop_profiles.yaml; a second copy is a second number."""
        text = RULES_PATH.read_text()
        assert "max_leverage" not in text
        assert "leverage_cap" not in text
        assert set(PropRules.model_fields) & {"max_leverage", "leverage_cap"} == set()

    def test_an_unknown_rule_set_is_refused_rather_than_ignored(self) -> None:
        library = load_prop_rules(RULES_PATH)
        with pytest.raises(Exception, match="no prop rules named"):
            library.get("ftmo_swng")


class TestDayOriginDivergence:
    def test_the_firm_day_and_the_run_day_disagreeing_produces_a_sentence(self) -> None:
        message = day_origin_divergence(rules(), FX_DAY_ORIGIN)
        assert message is not None
        assert "Europe/Prague" in message
        assert "America/New_York" in message

    def test_agreement_produces_nothing(self) -> None:
        prague = DayOrigin(tz="Europe/Prague", at=time(0, 0))
        assert day_origin_divergence(rules(), prague) is None
