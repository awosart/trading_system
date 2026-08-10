"""Margin, the firm's leverage ceiling, and the difference between them.

Every margin figure here is computed by hand in the comment above its assertion,
from the published contract specification and the leverage the shipped profile
declares — a test that recomputed it the way
:mod:`trading_system.risk.margin` does would agree with any bug that module has.
"""

from datetime import UTC, datetime, time
from decimal import Decimal
from pathlib import Path

import pytest

from tests.risk.conftest import (
    NOW,
    PRICES,
    EngineFactory,
    signal_with_stop_points,
)
from trading_system.backtest.clock import StreamKey
from trading_system.backtest.portfolio import OpenPosition, Portfolio
from trading_system.core.exceptions import ValidationError
from trading_system.core.instruments import (
    InstrumentClass,
    InstrumentRegistry,
    load_instruments,
)
from trading_system.core.types import Price, Side, Timeframe
from trading_system.data.resample import DayOrigin
from trading_system.execution.orders import Fill
from trading_system.exit.base import ExitReason
from trading_system.exit.library import ExitLibrarySpec, build_plan
from trading_system.exit.position import ManagedPosition
from trading_system.risk.conversion import SameCurrencyConverter
from trading_system.risk.margin import (
    MARGIN_STEP,
    PropProfile,
    load_prop_profiles,
    margin_rate_for,
    notional_per_lot,
    quantise_up,
)
from trading_system.risk.models import REJECTION_REASONS, AccountState, OpenRisk, RiskReason
from trading_system.risk.sizing.methods import FixedFractional
from trading_system.strategies.schema import FixedPipsStop, StopReference

PROFILES_PATH = Path(__file__).resolve().parents[2] / "configs" / "prop_profiles.yaml"
EXIT_LIBRARY_PATH = (
    Path(__file__).resolve().parents[2] / "src" / "trading_system" / "exit" / "library.json"
)

#: Same convention as ``test_engine``: the strategy's own stop view is kept out
#: of the arithmetic so the signal's invalidation drives the size alone.
NEVER_BINDING: StopReference = FixedPipsStop(pips=0.001)

WHOLE = Decimal("1")

#: A day origin for the portfolio fixture; nothing here depends on which.
FX_DAY_ORIGIN = DayOrigin(tz="America/New_York", at=time(17, 0))


@pytest.fixture(scope="module")
def profiles() -> object:
    """The shipped prop-firm profiles."""
    return load_prop_profiles(PROFILES_PATH)


def _profile(name: str) -> PropProfile:
    """One shipped profile by name."""
    return load_prop_profiles(PROFILES_PATH).get(name)


def _evaluate(
    engine_factory: EngineFactory,
    registry: InstrumentRegistry,
    account: AccountState,
    *,
    symbol: str,
    points: float,
    profile: PropProfile | None,
    risk_pct: float = 0.005,
) -> object:
    """Size one signal under a given profile."""
    return engine_factory(FixedFractional(risk_pct), prop_profile=profile).evaluate(
        signal_with_stop_points(registry, symbol, points=points),
        account=account,
        stop_reference=NEVER_BINDING,
        smallest_exit_fraction=WHOLE,
        bar_index=0,
        trades=(),
    )


class TestMarginRefusesWhatTheAccountCannotPost:
    """DoD: a position needing more free margin than there is, is refused."""

    def test_a_position_beyond_free_margin_is_refused(
        self,
        registry: InstrumentRegistry,
        account: AccountState,
        engine_factory: EngineFactory,
    ) -> None:
        # XAUUSD, 0.5% of 100 000 = 500 USD risked, 1 USD per point per lot.
        # 500 / (40 * 1) = 12.50 lots at 2050 = 100 * 2050 * 12.5 = 2 562 500
        # notional. FundedNext metals 1:25 -> 102 500 of margin, against 100 000
        # of equity and nothing already posted. Short by 2 500.
        decision = _evaluate(
            engine_factory,
            registry,
            account,
            symbol="XAUUSD",
            points=40,
            profile=_profile("fundednext"),
        )
        assert not decision.approved
        assert decision.rejection is RiskReason.INSUFFICIENT_MARGIN
        assert "102500.00" in decision.reasons[-1]
        assert "100000" in decision.reasons[-1]

    def test_the_same_position_is_approved_at_looser_leverage(
        self,
        registry: InstrumentRegistry,
        account: AccountState,
        engine_factory: EngineFactory,
    ) -> None:
        """The control: the rate was the obstacle, not the sizing arithmetic.

        Same instrument, same stop, same account, same 12.50 lots. E8 publishes
        metals at 1:50 rather than 1:25, halving the requirement to 51 250,
        which fits. Nothing about the trade changed.
        """
        approved = _evaluate(
            engine_factory, registry, account, symbol="XAUUSD", points=40, profile=_profile("e8")
        )
        assert approved.approved
        assert approved.size == Decimal("12.50")
        assert approved.notional_amount == Decimal("2562500.00")
        assert approved.margin_amount == Decimal("51250.00")

    def test_margin_already_posted_is_what_makes_the_difference(
        self,
        registry: InstrumentRegistry,
        engine_factory: EngineFactory,
    ) -> None:
        """``free_margin``, not ``equity``, is the figure the check is against."""
        # EURUSD, 500 risked, 10 USD per point per lot, 50-point stop -> 1.00 lot
        # at 1.0850 = 108 500 notional. FTMO swing FX 1:30 -> 3616.67 of margin.
        profile = _profile("ftmo_swing")
        flat = AccountState(
            currency="USD", balance=Decimal("100000"), equity=Decimal("100000"), as_of=NOW
        )
        assert _evaluate(
            engine_factory, registry, flat, symbol="EURUSD", points=50, profile=profile
        ).approved

        encumbered = AccountState(
            currency="USD",
            balance=Decimal("100000"),
            equity=Decimal("100000"),
            as_of=NOW,
            open_risks=(
                OpenRisk(
                    symbol="XAUUSD",
                    strategy_id="other",
                    side=Side.BUY,
                    risk_amount=Decimal("0"),
                    margin=Decimal("99000"),
                    notional=Decimal("2475000"),
                ),
            ),
        )
        assert encumbered.free_margin == Decimal("1000")
        refused = _evaluate(
            engine_factory, registry, encumbered, symbol="EURUSD", points=50, profile=profile
        )
        assert not refused.approved
        assert refused.rejection is RiskReason.INSUFFICIENT_MARGIN


class TestTheTwoRulesGiveDifferentReasons:
    """DoD: leverage and margin are distinguishable refusals, not one."""

    def test_the_cap_refuses_while_the_margin_was_there(
        self,
        registry: InstrumentRegistry,
        account: AccountState,
        engine_factory: EngineFactory,
    ) -> None:
        # EURUSD, 500 risked over a 5-point stop -> 500 / (5 * 10) = 10.00 lots
        # at 1.0850 = 1 085 000 notional. Finotive FX is 1:100, so the margin
        # requirement is only 10 850 against 100 000 of equity — affordable.
        # Its published exposure ceiling is 12.5x equity = 1 250 000... which
        # this clears. At 4 points the size is 12.50 lots = 1 356 250 notional,
        # margin 13 562.50, and it is the ceiling alone that binds.
        decision = _evaluate(
            engine_factory,
            registry,
            account,
            symbol="EURUSD",
            points=4,
            profile=_profile("finotive"),
        )
        assert not decision.approved
        assert decision.rejection is RiskReason.LEVERAGE_LIMIT_EXCEEDED
        assert "1250000" in decision.reasons[-1]
        # The refusal states that the money was there, which is the whole
        # reason the two reasons are two.
        assert "the rule was" in decision.reasons[-1]

    def test_the_same_trade_passes_once_the_cap_is_lifted(
        self,
        registry: InstrumentRegistry,
        account: AccountState,
        engine_factory: EngineFactory,
    ) -> None:
        """Isolates the cap: only ``leverage_cap`` differs between the two runs."""
        finotive = _profile("finotive")
        uncapped = finotive.model_copy(update={"leverage_cap": None})
        assert _evaluate(
            engine_factory, registry, account, symbol="EURUSD", points=4, profile=uncapped
        ).approved

    def test_margin_and_leverage_are_distinct_refusal_reasons(self) -> None:
        assert RiskReason.INSUFFICIENT_MARGIN is not RiskReason.LEVERAGE_LIMIT_EXCEEDED
        assert RiskReason.INSUFFICIENT_MARGIN in REJECTION_REASONS
        assert RiskReason.LEVERAGE_LIMIT_EXCEEDED in REJECTION_REASONS

    def test_a_profile_without_a_cap_never_produces_the_leverage_refusal(
        self,
        registry: InstrumentRegistry,
        account: AccountState,
        engine_factory: EngineFactory,
    ) -> None:
        """Nine of the ten shipped profiles publish no ceiling; they must not invent one."""
        # A trade far past any plausible ceiling, under FTMO swing (no cap).
        # It is refused, but on margin: 1:30 on 1 356 250 needs 45 208.34.
        big = _evaluate(
            engine_factory,
            registry,
            account,
            symbol="EURUSD",
            points=4,
            profile=_profile("ftmo_swing"),
            risk_pct=0.05,
        )
        assert big.rejection is not RiskReason.LEVERAGE_LIMIT_EXCEEDED


class TestPartialClosesReleaseMarginInProportion:
    """DoD: the release rule decided in (b) — proportional, at the leg's fill."""

    @staticmethod
    def _portfolio_with_a_position(
        registry: InstrumentRegistry,
    ) -> tuple[Portfolio, OpenPosition]:
        """A 4-lot EURUSD long whose per-lot margin is a round 1 000."""
        library = ExitLibrarySpec.model_validate_json(EXIT_LIBRARY_PATH.read_text())
        portfolio = Portfolio(
            currency="USD",
            starting_balance=Decimal("100000"),
            instruments=registry,
            converter=SameCurrencyConverter(),
            day_origin=FX_DAY_ORIGIN,
        )
        opened_at = datetime(2024, 3, 5, 12, 0, tzinfo=UTC)
        position = ManagedPosition(
            symbol="EURUSD",
            side=Side.BUY,
            entry_price=Price(1.0850),
            size=Decimal("4"),
            initial_stop=Price(1.0800),
            opened_at=opened_at,
            strategy_id="test-strategy",
        )
        held = OpenPosition(
            position_id="p1",
            key=StreamKey("EURUSD", Timeframe.H1),
            position=position,
            plan=build_plan(library.presets[0]),
            strategy_id="test-strategy",
            entry_fill=Fill(
                order_id="p1",
                symbol="EURUSD",
                side=Side.BUY,
                size=Decimal("4"),
                ts=opened_at,
                mid_price=Price(1.0850),
                price=Price(1.0850),
                spread_points=0.0,
                slippage_points=0.0,
                commission=Decimal(0),
                gap_points=0.0,
            ),
            entry_bar_index=0,
            entry_fx_rate=Decimal(1),
            risk_amount=Decimal("2000"),
            entry_quality=0.6,
            margin_per_lot=Decimal("1000"),
            notional_per_lot=Decimal("30000"),
        )
        portfolio.open(held)
        return portfolio, held

    def test_a_fifty_percent_close_releases_half_the_margin(
        self, registry: InstrumentRegistry
    ) -> None:
        portfolio, held = self._portfolio_with_a_position(registry)
        assert portfolio.used_margin == Decimal("4000")
        assert portfolio.used_notional == Decimal("120000")

        held.position.close(
            Decimal("0.5"),
            price=Price(1.0900),
            ts=datetime(2024, 3, 5, 13, 0, tzinfo=UTC),
            reason=ExitReason.TAKE_PROFIT,
        )
        assert portfolio.used_margin == Decimal("2000")
        assert portfolio.used_notional == Decimal("60000")

    def test_a_ladder_releases_it_rung_by_rung_and_the_last_rung_frees_the_rest(
        self, registry: InstrumentRegistry
    ) -> None:
        """50/25/25 against the *original* size, as P07 books partials."""
        portfolio, held = self._portfolio_with_a_position(registry)
        at = datetime(2024, 3, 5, 13, 0, tzinfo=UTC)
        for fraction, expected in (
            (Decimal("0.5"), Decimal("2000")),
            (Decimal("0.25"), Decimal("1000")),
        ):
            held.position.close(fraction, price=Price(1.0900), ts=at, reason=ExitReason.TAKE_PROFIT)
            assert portfolio.used_margin == expected
        held.position.close_all(price=Price(1.0900), ts=at, reason=ExitReason.TAKE_PROFIT)
        assert portfolio.used_margin == Decimal("0")
        # And once the position is retired it is gone from the sum entirely,
        # rather than lingering at zero.
        portfolio.close_out(held, at)
        assert portfolio.used_margin == Decimal("0")
        assert portfolio.account_state(at).open_risks == ()

    def test_the_account_state_the_engine_reads_carries_the_released_figure(
        self, registry: InstrumentRegistry
    ) -> None:
        """The release has to reach the Risk Engine, not just the portfolio."""
        portfolio, held = self._portfolio_with_a_position(registry)
        at = datetime(2024, 3, 5, 13, 0, tzinfo=UTC)
        held.position.close(
            Decimal("0.5"), price=Price(1.0900), ts=at, reason=ExitReason.TAKE_PROFIT
        )
        state = portfolio.account_state(at)
        assert state.used_margin == Decimal("2000")
        assert state.used_notional == Decimal("60000")
        assert state.free_margin == state.equity - Decimal("2000")


class TestTheReservationHoldsAcrossOneInstant:
    """Two signals at one instant must not both spend the same free margin."""

    def test_the_second_signal_sees_what_the_first_reserved(
        self,
        registry: InstrumentRegistry,
        account: AccountState,
        engine_factory: EngineFactory,
    ) -> None:
        profile = _profile("ftmo_swing")
        engine = engine_factory(FixedFractional(0.005), prop_profile=profile)
        signal = signal_with_stop_points(registry, "XAUUSD", points=100)
        # 500 / (100 * 1) = 5.00 lots at 2050 = 1 025 000 notional; 1:15 -> 68 333.34.
        # The stop was widened from 50 points when ftmo_swing's metals leverage
        # was corrected from 1:30 to 1:15: halving the size against a doubled
        # rate leaves the reserved figure identical, so this test still asserts
        # the reservation and not an arithmetic coincidence of the old profile.
        # At 50 points the first signal alone would now need 136 666.67 on a
        # 100k account and be refused, leaving nothing to reserve.
        first = engine.evaluate(
            signal,
            account=account,
            stop_reference=NEVER_BINDING,
            smallest_exit_fraction=WHOLE,
            bar_index=0,
            trades=(),
        )
        assert first.approved
        assert first.margin_amount == Decimal("68333.34")

        after = account.with_opened(
            "XAUUSD",
            "test-strategy",
            Side.BUY,
            first.risk_amount,
            margin=first.margin_amount,
            notional=first.notional_amount,
        )
        second = engine.evaluate(
            signal,
            account=after,
            stop_reference=NEVER_BINDING,
            smallest_exit_fraction=WHOLE,
            bar_index=0,
            trades=(),
        )
        assert not second.approved
        assert second.rejection is RiskReason.INSUFFICIENT_MARGIN

    def test_margin_and_notional_are_mandatory_on_with_opened(self) -> None:
        """No default, for the reason a default would be wrong: silent over-commitment."""
        state = AccountState(currency="USD", balance=Decimal("1"), equity=Decimal("1"), as_of=NOW)
        with pytest.raises(TypeError):
            state.with_opened("EURUSD", "s", Side.BUY, Decimal("1"))  # type: ignore[call-arg]


class TestWhereTheConstraintStartsToBind:
    """DoD: the numbers from question (в), on the shipped specifications.

    The identity behind every row: ``contract_size`` cancels out of
    ``notional = risk_money / relative_stop``, so effective leverage is
    ``risk_pct / relative_stop`` and margin utilisation is
    ``margin_rate * risk_pct / relative_stop`` — instrument-independent. Margin
    therefore exhausts a flat account exactly when the stop, as a fraction of
    price, drops to ``margin_rate * risk_pct``.
    """

    @pytest.mark.parametrize(
        ("points", "lots", "notional", "margin"),
        [
            # XAUUSD at 2050: 1 USD per point per lot, 0.5% of 100k = 500 risked.
            # 500 / (500 * 1) = 1.00 lot; 100 * 2050 * 1 = 205 000; /25 = 8 200.
            (500.0, "1.00", "205000.00", "8200.00"),
            # 500 / (100 * 1) = 5.00 lots; 1 025 000 notional; /25 = 41 000.
            (100.0, "5.00", "1025000.00", "41000.00"),
            # 500 / (50 * 1) = 10.00 lots; 2 050 000; /25 = 82 000 — 82% of equity.
            (50.0, "10.00", "2050000.00", "82000.00"),
        ],
    )
    def test_xauusd_margin_at_one_to_twenty_five(
        self,
        registry: InstrumentRegistry,
        account: AccountState,
        engine_factory: EngineFactory,
        points: float,
        lots: str,
        notional: str,
        margin: str,
    ) -> None:
        decision = _evaluate(
            engine_factory,
            registry,
            account,
            symbol="XAUUSD",
            points=points,
            profile=_profile("fundednext"),  # metals 1:25
        )
        assert decision.approved
        assert decision.size == Decimal(lots)
        assert decision.notional_amount == Decimal(notional)
        assert decision.margin_amount == Decimal(margin)

    def test_xauusd_binds_only_below_forty_one_points(
        self,
        registry: InstrumentRegistry,
        account: AccountState,
        engine_factory: EngineFactory,
    ) -> None:
        """``relative_stop <= margin_rate * risk_pct`` = 0.04 * 0.005 = 0.0002.

        At 2050 that is 0.41 of price, which is 41 points. A stop wider than
        that is affordable at 0.5% risk; a stop narrower is not — and 41 points
        on gold is 0.41 USD, well inside the 30-point broker minimum but far
        below anything a strategy in this repository trades.
        """
        profile = _profile("fundednext")
        assert _evaluate(
            engine_factory, registry, account, symbol="XAUUSD", points=42, profile=profile
        ).approved
        refused = _evaluate(
            engine_factory, registry, account, symbol="XAUUSD", points=40, profile=profile
        )
        assert not refused.approved
        assert refused.rejection is RiskReason.INSUFFICIENT_MARGIN

    @pytest.mark.parametrize(
        ("points", "lots", "notional", "margin"),
        [
            # NAS100 at 18000: point_size 1.0 * contract_size 20 = 20 USD per
            # point per lot. 500 / (100 * 20) = 0.25 lots; 20 * 18000 * 0.25 =
            # 90 000 notional; The5ers indices 1:25 -> 3 600.
            (100.0, "0.2", "72000.00", "2880.00"),
            # 500 / (50 * 20) = 0.50 lots; 180 000 notional; 7 200 of margin.
            (50.0, "0.5", "180000.00", "7200.00"),
            # 500 / (10 * 20) = 2.50 lots; 900 000 notional; 36 000 of margin.
            (10.0, "2.5", "900000.00", "36000.00"),
        ],
    )
    def test_nas100_margin_at_one_to_twenty_five(
        self,
        registry: InstrumentRegistry,
        account: AccountState,
        engine_factory: EngineFactory,
        points: float,
        lots: str,
        notional: str,
        margin: str,
    ) -> None:
        decision = _evaluate(
            engine_factory,
            registry,
            account,
            symbol="NAS100",
            points=points,
            profile=_profile("the5ers"),  # indices 1:25
        )
        assert decision.approved
        assert decision.size == Decimal(lots)
        assert decision.notional_amount == Decimal(notional)
        assert decision.margin_amount == Decimal(margin)

    def test_nas100_never_binds_on_margin_at_half_a_percent(
        self,
        registry: InstrumentRegistry,
        account: AccountState,
        engine_factory: EngineFactory,
    ) -> None:
        """The threshold sits below the broker's own minimum stop distance.

        ``0.04 * 0.005 * 18000 = 3.6`` points, and
        ``min_stop_distance_points`` is 5. The instrument refuses the stop
        before the account runs out of collateral, so on this instrument at this
        risk the margin check can never be the binding constraint — which is the
        answer to (в) for an index, and the reason the firm's own notional cap
        is the rule that actually bites.
        """
        assert registry["NAS100"].min_stop_distance_points == 5.0
        assert _evaluate(
            engine_factory,
            registry,
            account,
            symbol="NAS100",
            points=5,
            profile=_profile("the5ers"),
        ).approved

    def test_effective_leverage_does_not_depend_on_the_instrument(
        self,
        registry: InstrumentRegistry,
        account: AccountState,
        engine_factory: EngineFactory,
    ) -> None:
        """``contract_size`` cancels: equal relative stops give equal notional.

        A 0.5%-of-price stop is 10.25 on gold at 2050 (1 025 points) and 90.0 on
        NAS100 at 18000 (90 points). Both must produce the same notional at the
        same risk budget, whatever the contract sizes are: ``500 / 0.005 =
        100 000``.

        The tolerance is one lot step's worth of notional, not an epsilon. That
        is the whole of the discrepancy and it is not noise: gold quantises to
        0.01 lots (2 050 of notional) and NAS100 to 0.1 lots (36 000), so the
        identity holds exactly on the unquantised size and only to within a lot
        step on the traded one.
        """
        loose = PropProfile(
            name="loose",
            source="test",
            leverage=dict.fromkeys(InstrumentClass, 1000.0),
            leverage_cap=None,
        )
        for symbol, points in (("XAUUSD", 1025.0), ("NAS100", 90.0)):
            decision = _evaluate(
                engine_factory, registry, account, symbol=symbol, points=points, profile=loose
            )
            assert decision.approved
            step_notional = registry[symbol].lot_step * decision.notional_per_lot
            assert abs(decision.notional_amount - Decimal("100000")) < step_notional


class TestTheShippedProfiles:
    """The file is data, and the things that must be true of it are asserted."""

    def test_every_profile_states_its_cap_explicitly(self) -> None:
        """``leverage_cap`` has no default, so "none published" is a decision."""
        library = load_prop_profiles(PROFILES_PATH)
        raw = PROFILES_PATH.read_text()
        assert raw.count("leverage_cap:") == len(library.profiles)

    def test_finotive_is_the_only_profile_carrying_a_cap(self) -> None:
        library = load_prop_profiles(PROFILES_PATH)
        capped = {p.name for p in library.profiles if p.leverage_cap is not None}
        assert capped == {"finotive"}
        assert library.get("finotive").leverage_cap == 12.5

    def test_the_default_profile_exists_and_is_the_strictest_on_fx(self) -> None:
        library = load_prop_profiles(PROFILES_PATH)
        swing = library.get("ftmo_swing")
        lowest = min(p.leverage[InstrumentClass.FX] for p in library.profiles)
        assert swing.leverage[InstrumentClass.FX] == lowest

    def test_a_class_a_profile_omits_falls_back_to_the_instrument(self) -> None:
        """The registry stays the authority on anything the firm does not amend."""
        registry = load_instruments(
            Path(__file__).resolve().parents[2] / "configs" / "instruments.yaml"
        )
        partial = PropProfile(
            name="fx-only", source="test", leverage={InstrumentClass.FX: 50.0}, leverage_cap=None
        )
        assert margin_rate_for(registry["EURUSD"], partial) == Decimal(1) / Decimal("50.0")
        assert margin_rate_for(registry["XAUUSD"], partial) == Decimal(str(0.05))
        assert margin_rate_for(registry["XAUUSD"], None) == Decimal(str(0.05))

    def test_an_unknown_profile_name_is_refused_rather_than_ignored(self) -> None:
        library = load_prop_profiles(PROFILES_PATH)
        with pytest.raises(ValidationError, match="no prop profile named"):
            library.get("ftmo_swng")

    def test_leverage_must_be_positive(self) -> None:
        with pytest.raises(Exception, match="leverage must be positive"):
            PropProfile(
                name="broken",
                source="test",
                leverage={InstrumentClass.FX: 0.0},
                leverage_cap=None,
            )


class TestArithmetic:
    """The two places the sign of a rounding error matters."""

    def test_a_margin_requirement_rounds_up_not_down(self) -> None:
        """Against the trader, unlike every other quantisation in this layer."""
        assert quantise_up(Decimal("1.001")) == Decimal("1.01")
        assert quantise_up(Decimal("1.000")) == Decimal("1.00")
        assert Decimal("0.01") == MARGIN_STEP

    def test_notional_is_contract_size_times_price_times_the_rate(self) -> None:
        registry = load_instruments(
            Path(__file__).resolve().parents[2] / "configs" / "instruments.yaml"
        )
        # One EURUSD lot at 1.0850 is 100 000 * 1.0850 = 108 500 USD.
        assert notional_per_lot(registry["EURUSD"], PRICES["EURUSD"], Decimal(1)) == Decimal(
            "108500.000000"
        )
        # One GBPJPY lot at 190.00 is 19 000 000 JPY, which at 150 JPY per USD
        # is 126 666.66… USD — the conversion is the caller's rate, applied here.
        jpy = notional_per_lot(registry["GBPJPY"], PRICES["GBPJPY"], Decimal(1) / Decimal(150))
        assert abs(jpy - Decimal("126666.67")) < Decimal("0.01")
