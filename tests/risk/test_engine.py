"""The Risk Engine end to end: sizes, refusals, and the cap that binds them.

Every expected size in this module is computed by hand in the comment above the
assertion, from the instrument's published contract specification. A test that
recomputed the size the way the engine does would agree with any bug the engine
has.
"""

from decimal import Decimal

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from tests.risk.conftest import (
    NOW,
    PRICES,
    REGISTRY_PATH,
    USDJPY,
    EngineFactory,
    signal_with_stop_points,
)
from trading_system.core.instruments import InstrumentRegistry, load_instruments
from trading_system.core.types import Price, Side
from trading_system.entry.signal import EntrySignal
from trading_system.risk.conversion import StaticFxConverter
from trading_system.risk.engine import RISK_AMOUNT_STEP, RiskEngine, RiskEngineConfig
from trading_system.risk.models import AccountState, RiskReason
from trading_system.risk.sizing.base import SizingMethod
from trading_system.risk.sizing.methods import (
    FixedAmount,
    FixedFractional,
    QualityScaled,
    VolatilityTargeting,
)
from trading_system.risk.stop_calculator import StopBufferConfig
from trading_system.strategies.schema import AtrStop, FixedPipsStop, StopReference

#: Sizing is driven by the signal's invalidation level throughout this module;
#: a fixed-pips stop_reference wide enough to never be the binding candidate
#: keeps the strategy's own stop view out of the arithmetic.
NEVER_BINDING: StopReference = FixedPipsStop(pips=0.001)

WHOLE = Decimal("1")


class TestHandComputedSizes:
    """DoD: the size for a 0.5% risk on a 50-point stop, per instrument.

    Risk budget is 0.5% of 100 000 USD = 500 USD in every case. What differs is
    the money one point is worth, which is ``point_size * contract_size``
    converted into USD.
    """

    def test_eurusd(
        self, registry: InstrumentRegistry, account: AccountState, engine_factory: EngineFactory
    ) -> None:
        # 0.0001 * 100 000 = 10 USD per point per lot, already in USD.
        # 500 / (50 * 10) = 1.00 lot.
        decision = engine_factory().evaluate(
            signal_with_stop_points(registry, "EURUSD", points=50),
            account=account,
            stop_reference=NEVER_BINDING,
            smallest_exit_fraction=WHOLE,
            bar_index=0,
            trades=(),
        )
        assert decision.approved
        assert decision.size == Decimal("1.00")
        assert decision.risk_amount == Decimal("500")
        assert decision.risk_pct == pytest.approx(0.005)
        assert decision.point_value == Decimal("10")
        assert decision.fx_rate == Decimal(1)

    def test_gbpjpy_needs_the_usdjpy_rate(
        self, registry: InstrumentRegistry, account: AccountState, engine_factory: EngineFactory
    ) -> None:
        # 0.01 * 100 000 = 1 000 JPY per point per lot. At USDJPY 150 that is
        # 1000 / 150 = 6.666... USD. 500 / (50 * 6.666...) = 1.50 lots.
        # This is the one instrument in the registry whose size cannot be
        # computed without an exchange rate.
        decision = engine_factory().evaluate(
            signal_with_stop_points(registry, "GBPJPY", points=50),
            account=account,
            stop_reference=NEVER_BINDING,
            smallest_exit_fraction=WHOLE,
            bar_index=0,
            trades=(),
        )
        assert decision.approved
        assert decision.size == Decimal("1.50")
        assert decision.risk_amount == pytest.approx(Decimal("500"))
        assert decision.fx_rate == Decimal(1) / Decimal("150")

    def test_xauusd(
        self, registry: InstrumentRegistry, account: AccountState, engine_factory: EngineFactory
    ) -> None:
        # 0.01 * 100 oz = 1 USD per point per lot.
        # 500 / (50 * 1) = 10 lots.
        decision = engine_factory().evaluate(
            signal_with_stop_points(registry, "XAUUSD", points=50),
            account=account,
            stop_reference=NEVER_BINDING,
            smallest_exit_fraction=WHOLE,
            bar_index=0,
            trades=(),
        )
        assert decision.approved
        assert decision.size == Decimal("10")
        assert decision.point_value == Decimal("1")

    def test_nas100_carries_the_exchange_multiplier(
        self, registry: InstrumentRegistry, account: AccountState, engine_factory: EngineFactory
    ) -> None:
        # 1.0 * 20 = 20 USD per index point per lot.
        # 500 / (50 * 20) = 0.5 lots. If contract_size were ignored this would
        # come out at 10 lots -- twenty times the intended exposure.
        decision = engine_factory().evaluate(
            signal_with_stop_points(registry, "NAS100", points=50),
            account=account,
            stop_reference=NEVER_BINDING,
            smallest_exit_fraction=WHOLE,
            bar_index=0,
            trades=(),
        )
        assert decision.approved
        assert decision.size == Decimal("0.5")
        assert decision.point_value == Decimal("20")

    def test_btcusd(
        self, registry: InstrumentRegistry, account: AccountState, engine_factory: EngineFactory
    ) -> None:
        # 1.0 * 1 coin = 1 USD per point per lot.
        # 500 / (50 * 1) = 10 lots, which is exactly max_lot.
        decision = engine_factory().evaluate(
            signal_with_stop_points(registry, "BTCUSD", points=50),
            account=account,
            stop_reference=NEVER_BINDING,
            smallest_exit_fraction=WHOLE,
            bar_index=0,
            trades=(),
        )
        assert decision.approved
        assert decision.size == Decimal("10")

    def test_us30(
        self, registry: InstrumentRegistry, account: AccountState, engine_factory: EngineFactory
    ) -> None:
        # 1.0 * 5 = 5 USD per index point per lot.
        # 500 / (50 * 5) = 2.0 lots.
        decision = engine_factory().evaluate(
            signal_with_stop_points(registry, "US30", points=50),
            account=account,
            stop_reference=NEVER_BINDING,
            smallest_exit_fraction=WHOLE,
            bar_index=0,
            trades=(),
        )
        assert decision.approved
        assert decision.size == Decimal("2")

    def test_a_short_sizes_identically_to_a_long(
        self, registry: InstrumentRegistry, account: AccountState, engine_factory: EngineFactory
    ) -> None:
        # Risk is a distance; direction changes which side the stop sits, not
        # how much money is at stake.
        engine = engine_factory()
        long = engine.evaluate(
            signal_with_stop_points(registry, "EURUSD", points=50, side=Side.BUY),
            account=account,
            stop_reference=NEVER_BINDING,
            smallest_exit_fraction=WHOLE,
            bar_index=0,
            trades=(),
        )
        short = engine.evaluate(
            signal_with_stop_points(registry, "EURUSD", points=50, side=Side.SELL),
            account=account,
            stop_reference=NEVER_BINDING,
            smallest_exit_fraction=WHOLE,
            bar_index=0,
            trades=(),
        )
        assert long.size == short.size
        assert long.stop_price < PRICES["EURUSD"] < short.stop_price


class TestQualityFloor:
    """DoD: below the floor the trade is refused, not shrunk."""

    @pytest.fixture
    def scaled(self) -> QualityScaled:
        return QualityScaled(min_risk_pct=0.002, max_risk_pct=0.01, quality_floor=0.6)

    def test_below_the_floor_is_a_refusal_not_a_micro_position(
        self,
        registry: InstrumentRegistry,
        account: AccountState,
        engine_factory: EngineFactory,
        scaled: QualityScaled,
    ) -> None:
        decision = engine_factory(scaled).evaluate(
            signal_with_stop_points(registry, "EURUSD", points=50, quality=0.59),
            account=account,
            stop_reference=NEVER_BINDING,
            smallest_exit_fraction=WHOLE,
            bar_index=0,
            trades=(),
        )
        assert not decision.approved
        assert decision.rejection is RiskReason.QUALITY_BELOW_FLOOR
        assert decision.size == 0
        assert decision.risk_amount == 0

    def test_exactly_at_the_floor_trades_at_the_minimum_risk(
        self,
        registry: InstrumentRegistry,
        account: AccountState,
        engine_factory: EngineFactory,
        scaled: QualityScaled,
    ) -> None:
        # The floor is the smallest tradable signal, so it gets min_risk_pct --
        # 0.2% of 100k = 200 USD, and 200 / (50 * 10) = 0.4 lots. Scaling from
        # zero instead would jump straight to an interior size here.
        decision = engine_factory(scaled).evaluate(
            signal_with_stop_points(registry, "EURUSD", points=50, quality=0.6),
            account=account,
            stop_reference=NEVER_BINDING,
            smallest_exit_fraction=WHOLE,
            bar_index=0,
            trades=(),
        )
        assert decision.approved
        assert decision.risk_amount == Decimal("200")
        assert decision.size == Decimal("0.40")

    def test_perfect_quality_trades_at_the_maximum_risk(
        self,
        registry: InstrumentRegistry,
        account: AccountState,
        engine_factory: EngineFactory,
        scaled: QualityScaled,
    ) -> None:
        # 1% of 100k = 1 000 USD, and 1000 / (50 * 10) = 2 lots.
        decision = engine_factory(scaled).evaluate(
            signal_with_stop_points(registry, "EURUSD", points=50, quality=1.0),
            account=account,
            stop_reference=NEVER_BINDING,
            smallest_exit_fraction=WHOLE,
            bar_index=0,
            trades=(),
        )
        assert decision.approved
        assert decision.risk_amount == Decimal("1000")
        assert decision.size == Decimal("2.00")

    def test_the_refusal_is_counted(
        self,
        registry: InstrumentRegistry,
        account: AccountState,
        engine_factory: EngineFactory,
        scaled: QualityScaled,
    ) -> None:
        engine = engine_factory(scaled)
        for _ in range(3):
            engine.evaluate(
                signal_with_stop_points(registry, "EURUSD", points=50, quality=0.1),
                account=account,
                stop_reference=NEVER_BINDING,
                smallest_exit_fraction=WHOLE,
                bar_index=0,
                trades=(),
            )
        assert engine.rejections[RiskReason.QUALITY_BELOW_FLOOR] == 3
        assert engine.refused == 3

    def test_every_reason_is_present_even_at_zero(self, engine_factory: EngineFactory) -> None:
        # "No trade was ever refused for an unexecutable ladder" is a recorded
        # fact, not a missing key indistinguishable from a forgotten counter.
        rejections = engine_factory().rejections
        assert rejections[RiskReason.EXIT_LADDER_UNEXECUTABLE] == 0
        assert set(rejections) >= {
            RiskReason.BELOW_MIN_LOT,
            RiskReason.FX_RATE_UNAVAILABLE,
            RiskReason.QUALITY_BELOW_FLOOR,
        }

    def test_reset_clears_the_counters(
        self,
        registry: InstrumentRegistry,
        account: AccountState,
        engine_factory: EngineFactory,
        scaled: QualityScaled,
    ) -> None:
        engine = engine_factory(scaled)
        engine.evaluate(
            signal_with_stop_points(registry, "EURUSD", points=50, quality=0.1),
            account=account,
            stop_reference=NEVER_BINDING,
            smallest_exit_fraction=WHOLE,
            bar_index=0,
            trades=(),
        )
        engine.reset()
        assert engine.refused == 0


class TestMinimumLot:
    """DoD: a size below the minimum lot is refused, never rounded up."""

    def test_too_small_to_trade_is_refused(
        self, registry: InstrumentRegistry, engine_factory: EngineFactory
    ) -> None:
        # 500 USD of equity at 0.5% = 2.50 USD of risk. Against a 50-point
        # EURUSD stop worth 500 USD per lot that is 0.005 lots, half the 0.01
        # minimum. Rounding up to 0.01 would risk 5 USD -- twice the budget.
        tiny = AccountState(
            currency="USD", balance=Decimal("500"), equity=Decimal("500"), as_of=NOW
        )
        decision = engine_factory().evaluate(
            signal_with_stop_points(registry, "EURUSD", points=50),
            account=tiny,
            stop_reference=NEVER_BINDING,
            smallest_exit_fraction=WHOLE,
            bar_index=0,
            trades=(),
        )
        assert not decision.approved
        assert decision.rejection is RiskReason.BELOW_MIN_LOT
        assert decision.size == 0

    def test_the_refusal_still_reports_where_the_stop_would_have_gone(
        self, registry: InstrumentRegistry, engine_factory: EngineFactory
    ) -> None:
        # The stop is computed before sizing, so a refusal can say what the
        # trade would have looked like -- which is what makes it diagnosable.
        tiny = AccountState(
            currency="USD", balance=Decimal("500"), equity=Decimal("500"), as_of=NOW
        )
        decision = engine_factory().evaluate(
            signal_with_stop_points(registry, "EURUSD", points=50),
            account=tiny,
            stop_reference=NEVER_BINDING,
            smallest_exit_fraction=WHOLE,
            bar_index=0,
            trades=(),
        )
        assert decision.stop_price == pytest.approx(1.0800)

    def test_a_size_between_two_lot_steps_rounds_down(
        self, registry: InstrumentRegistry, engine_factory: EngineFactory
    ) -> None:
        # 0.5% of 27 000 = 135 USD; 135 / 500 = 0.27 lots exactly. Nudge the
        # equity so the raw size lands at 0.2718 and check it becomes 0.27,
        # not 0.28.
        account = AccountState(
            currency="USD", balance=Decimal("27180"), equity=Decimal("27180"), as_of=NOW
        )
        decision = engine_factory().evaluate(
            signal_with_stop_points(registry, "EURUSD", points=50),
            account=account,
            stop_reference=NEVER_BINDING,
            smallest_exit_fraction=WHOLE,
            bar_index=0,
            trades=(),
        )
        assert decision.approved
        assert decision.size == Decimal("0.27")
        # And the reported risk is what 0.27 lots actually risks, not the 135.9
        # that was asked for.
        assert decision.risk_amount == Decimal("135")
        assert any("rounded down" in line for line in decision.reasons)

    def test_above_the_maximum_lot_is_refused_not_trimmed(
        self, registry: InstrumentRegistry, engine_factory: EngineFactory
    ) -> None:
        # A trimmed order is a different trade from the one that was sized.
        big = AccountState(
            currency="USD",
            balance=Decimal("100000000"),
            equity=Decimal("100000000"),
            as_of=NOW,
        )
        decision = engine_factory().evaluate(
            signal_with_stop_points(registry, "BTCUSD", points=50),
            account=big,
            stop_reference=NEVER_BINDING,
            smallest_exit_fraction=WHOLE,
            bar_index=0,
            trades=(),
        )
        assert not decision.approved
        assert decision.rejection is RiskReason.ABOVE_MAX_LOT


class TestExitLadderExecutability:
    """The check ``smallest_closing_fraction`` was introduced on P07 to enable."""

    def test_a_rung_that_rounds_to_nothing_refuses_the_trade_at_open(
        self, registry: InstrumentRegistry, engine_factory: EngineFactory
    ) -> None:
        # Equity sized so the position is exactly 0.10 lots on EURUSD:
        # 0.10 * 500 USD per lot = 50 USD of risk = 0.5% of 10 000.
        # A ladder whose smallest close is 5% wants 0.005 lots of that, which
        # rounds to 0.00 -- below the 0.01 minimum. Refused now rather than
        # discovered when the rung fires.
        account = AccountState(
            currency="USD", balance=Decimal("10000"), equity=Decimal("10000"), as_of=NOW
        )
        engine = engine_factory()
        signal = signal_with_stop_points(registry, "EURUSD", points=50)

        whole = engine.evaluate(
            signal,
            account=account,
            stop_reference=NEVER_BINDING,
            smallest_exit_fraction=WHOLE,
            bar_index=0,
            trades=(),
        )
        assert whole.approved
        assert whole.size == Decimal("0.10")

        laddered = engine.evaluate(
            signal,
            account=account,
            stop_reference=NEVER_BINDING,
            smallest_exit_fraction=Decimal("0.05"),
            bar_index=0,
            trades=(),
        )
        assert not laddered.approved
        assert laddered.rejection is RiskReason.EXIT_LADDER_UNEXECUTABLE
        assert engine.rejections[RiskReason.EXIT_LADDER_UNEXECUTABLE] == 1

    def test_the_same_ladder_is_fine_on_a_larger_position(
        self, registry: InstrumentRegistry, account: AccountState, engine_factory: EngineFactory
    ) -> None:
        # 1.00 lot with a 5% smallest close is 0.05 lots, comfortably tradable.
        # Executability is a property of the pair, not of the ladder alone.
        decision = engine_factory().evaluate(
            signal_with_stop_points(registry, "EURUSD", points=50),
            account=account,
            stop_reference=NEVER_BINDING,
            smallest_exit_fraction=Decimal("0.05"),
            bar_index=0,
            trades=(),
        )
        assert decision.approved
        assert decision.size == Decimal("1.00")

    def test_a_fraction_outside_the_unit_interval_is_a_wiring_error(
        self, registry: InstrumentRegistry, account: AccountState, engine_factory: EngineFactory
    ) -> None:
        # Not a refused signal: no market condition produces this, only a
        # caller passing the wrong thing.
        with pytest.raises(ValueError, match="fraction in"):
            engine_factory().evaluate(
                signal_with_stop_points(registry, "EURUSD", points=50),
                account=account,
                stop_reference=NEVER_BINDING,
                smallest_exit_fraction=Decimal("1.5"),
                bar_index=0,
                trades=(),
            )


class TestMissingFxRate:
    """Question (a): no rate means no trade, never a rate of one."""

    def test_an_unavailable_rate_refuses_rather_than_assuming_parity(
        self, registry: InstrumentRegistry, account: AccountState
    ) -> None:
        # A converter that knows nothing. On GBPJPY, assuming parity would size
        # the position 150 times too large and look entirely plausible.
        engine = RiskEngine(
            instruments=registry,
            sizing=FixedFractional(0.005),
            converter=StaticFxConverter({}),
            config=RiskEngineConfig(stop_buffer=StopBufferConfig(spread_multiple=0.0)),
        )
        decision = engine.evaluate(
            signal_with_stop_points(registry, "GBPJPY", points=50),
            account=account,
            stop_reference=NEVER_BINDING,
            smallest_exit_fraction=WHOLE,
            bar_index=0,
            trades=(),
        )
        assert not decision.approved
        assert decision.rejection is RiskReason.FX_RATE_UNAVAILABLE
        assert engine.rejections[RiskReason.FX_RATE_UNAVAILABLE] == 1

    def test_a_usd_quoted_instrument_is_unaffected_by_the_same_gap(
        self, registry: InstrumentRegistry, account: AccountState
    ) -> None:
        # The refusal is per signal, not per run: a missing USDJPY must not
        # stop EURUSD from trading in the same backtest.
        engine = RiskEngine(
            instruments=registry,
            sizing=FixedFractional(0.005),
            converter=StaticFxConverter({}),
            config=RiskEngineConfig(stop_buffer=StopBufferConfig(spread_multiple=0.0)),
        )
        decision = engine.evaluate(
            signal_with_stop_points(registry, "EURUSD", points=50),
            account=account,
            stop_reference=NEVER_BINDING,
            smallest_exit_fraction=WHOLE,
            bar_index=0,
            trades=(),
        )
        assert decision.approved


class TestPreArithmeticRefusals:
    def test_an_unknown_symbol_is_refused(
        self, account: AccountState, engine_factory: EngineFactory
    ) -> None:
        signal = EntrySignal(
            strategy_id="s",
            symbol="NOTREAL",
            bar_close_ts=NOW,
            side=Side.BUY,
            reference_price=Price(100.0),
            invalidation_price=Price(99.0),
            quality=0.8,
        )
        decision = engine_factory().evaluate(
            signal,
            account=account,
            stop_reference=NEVER_BINDING,
            smallest_exit_fraction=WHOLE,
            bar_index=0,
            trades=(),
        )
        assert not decision.approved
        assert decision.rejection is RiskReason.UNKNOWN_INSTRUMENT

    def test_a_blown_account_is_refused_rather_than_being_unrepresentable(
        self, registry: InstrumentRegistry, engine_factory: EngineFactory
    ) -> None:
        # AccountState accepts non-positive equity: a blown account is a real
        # state. It is the engine that declines to size against it.
        dead = AccountState(
            currency="USD", balance=Decimal("-50"), equity=Decimal("-50"), as_of=NOW
        )
        decision = engine_factory().evaluate(
            signal_with_stop_points(registry, "EURUSD", points=50),
            account=dead,
            stop_reference=NEVER_BINDING,
            smallest_exit_fraction=WHOLE,
            bar_index=0,
            trades=(),
        )
        assert not decision.approved
        assert decision.rejection is RiskReason.NON_POSITIVE_EQUITY

    def test_a_missing_atr_refuses_instead_of_silently_using_zero(
        self, registry: InstrumentRegistry, account: AccountState, engine_factory: EngineFactory
    ) -> None:
        # An ATR-referenced stop with no ATR would otherwise become a zero
        # distance -- a tighter stop and therefore a larger position.
        decision = engine_factory().evaluate(
            signal_with_stop_points(registry, "EURUSD", points=50),
            account=account,
            stop_reference=AtrStop(multiple=2.0),
            smallest_exit_fraction=WHOLE,
            bar_index=0,
            trades=(),
            atr_price=None,
        )
        assert not decision.approved
        assert decision.rejection is RiskReason.ATR_UNAVAILABLE


class TestRiskCap:
    def test_a_flat_stake_larger_than_the_cap_is_capped(
        self, registry: InstrumentRegistry, engine_factory: EngineFactory
    ) -> None:
        # FixedAmount does not consult equity, so on a small account it is the
        # engine's cap that stops it: 2% of 10 000 = 200 USD, not the 5 000
        # the method asked for. 200 / 500 per lot = 0.40 lots.
        account = AccountState(
            currency="USD", balance=Decimal("10000"), equity=Decimal("10000"), as_of=NOW
        )
        decision = engine_factory(FixedAmount(Decimal("5000")), max_risk_pct=0.02).evaluate(
            signal_with_stop_points(registry, "EURUSD", points=50),
            account=account,
            stop_reference=NEVER_BINDING,
            smallest_exit_fraction=WHOLE,
            bar_index=0,
            trades=(),
        )
        assert decision.approved
        assert decision.risk_amount == Decimal("200")
        assert decision.size == Decimal("0.40")
        assert any(RiskReason.RISK_CAPPED.value in line for line in decision.reasons)

    def test_volatility_targeting_is_capped_like_everything_else(
        self, registry: InstrumentRegistry, account: AccountState, engine_factory: EngineFactory
    ) -> None:
        # A tiny ATR makes this method ask for an enormous size; the cap is
        # applied to the money it asked for, in one place, for every method.
        decision = engine_factory(VolatilityTargeting(0.01), max_risk_pct=0.02).evaluate(
            signal_with_stop_points(registry, "EURUSD", points=50),
            account=account,
            stop_reference=NEVER_BINDING,
            smallest_exit_fraction=WHOLE,
            bar_index=0,
            trades=(),
            atr_price=0.00005,
        )
        assert decision.approved
        assert decision.risk_amount <= Decimal("2000")


class TestReportedRiskIsMoney:
    """``risk_amount`` is an amount of the account's currency, to the cent.

    Found by the portfolio property test, which produced an open risk of
    ``5000.000000000000000000000001`` against a ceiling of ``5000.00000``. The
    residue is decimal division: ``requested / stop_value_per_lot`` is exact
    only while the quotient fits 28 significant digits, and the last of those
    digits rounding up is enough to make ``size * stop_value_per_lot`` exceed
    what was asked for by a septillionth of a dollar.

    The fix is not a tolerance in the comparison — a risk cap asserted with an
    epsilon is a risk cap that can be argued with — and not rounding the
    division down either, which costs a whole lot step whenever an FX rate is
    inexact. It is that the *reported* figure is quantised to the smallest
    amount the account can move by, downwards, so the comparisons above stay
    exact ``<=`` on a number that means something.
    """

    def test_the_report_carries_no_digits_the_account_could_hold(
        self, registry: InstrumentRegistry, account: AccountState, engine_factory: EngineFactory
    ) -> None:
        """GBPJPY: the one instrument whose sizing runs through an inexact rate."""
        decision = engine_factory().evaluate(
            signal_with_stop_points(registry, "GBPJPY", points=37),
            account=account,
            stop_reference=NEVER_BINDING,
            smallest_exit_fraction=WHOLE,
            bar_index=0,
            trades=(),
        )

        assert decision.approved
        assert decision.risk_amount == decision.risk_amount.quantize(RISK_AMOUNT_STEP)
        assert decision.risk_amount <= account.equity * Decimal("0.005")

    def test_the_quantisation_goes_down_and_therefore_never_over_the_cap(
        self, registry: InstrumentRegistry, engine_factory: EngineFactory
    ) -> None:
        """A cap that is not a whole number of cents still binds exactly.

        2.077% of 9 999.99 is 207.699… — the cap itself has more decimal places
        than money does, which is the case where rounding to nearest would
        report a risk above it.
        """
        account = AccountState(
            currency="USD", balance=Decimal("9999.99"), equity=Decimal("9999.99"), as_of=NOW
        )
        cap = account.equity * Decimal("0.02077")
        decision = engine_factory(FixedAmount(Decimal("5000")), max_risk_pct=0.02077).evaluate(
            signal_with_stop_points(registry, "GBPJPY", points=37),
            account=account,
            stop_reference=NEVER_BINDING,
            smallest_exit_fraction=WHOLE,
            bar_index=0,
            trades=(),
        )

        assert decision.approved
        assert decision.risk_amount <= cap
        assert decision.risk_amount == decision.risk_amount.quantize(RISK_AMOUNT_STEP)


class TestRiskNeverExceedsTheCap:
    """DoD property: ``risk_amount`` never exceeds ``max_risk_pct * equity``.

    Across every sizing method, every instrument in the registry, every account
    size and every stop distance. The claim is exact rather than approximate:
    the reported figure is recomputed from the quantised size, quantisation
    only ever rounds down, and the figure is then rounded down again to
    :data:`~trading_system.risk.engine.RISK_AMOUNT_STEP` so that no residue of
    decimal division survives into it. See :class:`TestReportedRiskIsMoney`.
    """

    @settings(max_examples=300, deadline=None)
    @given(
        symbol=st.sampled_from(["EURUSD", "GBPJPY", "XAUUSD", "NAS100", "US30", "BTCUSD"]),
        equity=st.decimals(min_value=1, max_value=10_000_000, places=2),
        stop_points=st.floats(min_value=1.0, max_value=5_000.0, allow_nan=False),
        quality=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
        max_risk_pct=st.floats(min_value=0.0001, max_value=0.5, allow_nan=False),
        method_index=st.integers(min_value=0, max_value=3),
        smallest_fraction=st.sampled_from(
            [Decimal("1"), Decimal("0.5"), Decimal("0.25"), Decimal("0.1")]
        ),
        atr_price=st.floats(min_value=0.0001, max_value=100.0, allow_nan=False),
        side=st.sampled_from([Side.BUY, Side.SELL]),
    )
    def test_risk_amount_never_exceeds_the_cap(
        self,
        symbol: str,
        equity: Decimal,
        stop_points: float,
        quality: float,
        max_risk_pct: float,
        method_index: int,
        smallest_fraction: Decimal,
        atr_price: float,
        side: Side,
    ) -> None:
        methods: list[SizingMethod] = [
            FixedFractional(0.9),
            FixedAmount(Decimal("1000000")),
            VolatilityTargeting(0.9),
            QualityScaled(min_risk_pct=0.5, max_risk_pct=1.0, quality_floor=0.0),
        ]
        # Built here rather than from the fixture: hypothesis does not reset
        # function-scoped fixtures between examples, and an engine carries
        # refusal counters. Everything it depends on is stateless.
        registry = load_instruments(REGISTRY_PATH)
        # Every method is configured to ask for far more than any plausible cap,
        # so the cap is the thing under test rather than the method's own bound.
        engine = RiskEngine(
            instruments=registry,
            sizing=methods[method_index],
            converter=StaticFxConverter({("USD", "JPY"): USDJPY}),
            config=RiskEngineConfig(
                max_risk_pct=max_risk_pct,
                stop_buffer=StopBufferConfig(spread_multiple=0.0),
            ),
        )
        account = AccountState(currency="USD", balance=equity, equity=equity, as_of=NOW)
        decision = engine.evaluate(
            signal_with_stop_points(
                registry, symbol, points=stop_points, side=side, quality=quality
            ),
            account=account,
            stop_reference=NEVER_BINDING,
            smallest_exit_fraction=smallest_fraction,
            bar_index=0,
            trades=(),
            atr_price=atr_price,
        )
        ceiling = equity * Decimal(str(max_risk_pct))
        assert decision.risk_amount <= ceiling
        assert Decimal(str(decision.risk_pct)) <= Decimal(str(max_risk_pct)) + Decimal("1e-12")
        if not decision.approved:
            assert decision.risk_amount == 0
            assert decision.size == 0
