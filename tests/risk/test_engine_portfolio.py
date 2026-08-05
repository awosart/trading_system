"""The stage-2 orchestration: breakers, limits, trimming, and what binds.

The property test at the end is the one that matters: no sequence of signals,
whatever their instruments, sizes or order, can put more at risk than the
portfolio ceiling allows — provided each approval is fed back through
``with_opened``, which is the contract the orchestrator carries.
"""

from dataclasses import replace
from datetime import timedelta
from decimal import Decimal

from hypothesis import given, settings
from hypothesis import strategies as st

from tests.risk.conftest import NOW, EngineFactory, signal_with_stop_points
from trading_system.core.instruments import InstrumentRegistry
from trading_system.core.types import Side
from trading_system.risk.circuit_breakers import (
    CircuitBreakerConfig,
    CircuitBreakers,
    ClosedTrade,
)
from trading_system.risk.correlation import CorrelationProvider
from trading_system.risk.models import AccountState, RiskReason
from trading_system.risk.portfolio_risk import PortfolioLimitsConfig, PortfolioRisk
from trading_system.strategies.schema import FixedPipsStop, StopReference

NEVER_BINDING: StopReference = FixedPipsStop(pips=0.001)
WHOLE = Decimal("1")

#: The five USD-quoted instruments the registry carries, treated as one bet.
ONE_BET = {"one_bet": ["EURUSD", "XAUUSD", "NAS100", "US30", "BTCUSD"]}


def evaluate(engine, registry, symbol, account, **kwargs):  # type: ignore[no-untyped-def]
    """Evaluate a 50-point signal on ``symbol``, with stage-2 arguments filled."""
    return engine.evaluate(
        signal_with_stop_points(registry, symbol, points=50, **kwargs.pop("signal", {})),
        account=account,
        stop_reference=NEVER_BINDING,
        smallest_exit_fraction=WHOLE,
        bar_index=kwargs.pop("bar_index", 0),
        trades=kwargs.pop("trades", ()),
    )


class TestCorrelatedSignalsDoNotMultiplyRisk:
    """DoD: five correlated signals are trimmed by the cluster limit."""

    def test_five_correlated_signals_do_not_give_five_times_the_risk(
        self, registry: InstrumentRegistry, engine_factory: EngineFactory
    ) -> None:
        # Each signal alone would risk 0.5% of 100k = 500. Five of them
        # independently would be 2 500. They are one bet, and the cluster
        # ceiling is 1.2% = 1 200, so that is what the five together may reach.
        portfolio = PortfolioRisk(
            PortfolioLimitsConfig(
                max_portfolio_risk_pct=1.0,
                max_instrument_risk_pct=1.0,
                max_cluster_risk_pct=0.012,
                max_strategy_risk_pct=1.0,
                max_direction_risk_pct=1.0,
                groups=ONE_BET,
            )
        )
        engine = engine_factory(portfolio=portfolio)
        state = AccountState(
            currency="USD",
            balance=Decimal("100000"),
            equity=Decimal("100000"),
            as_of=NOW,
        )

        approved = 0
        for symbol in ("EURUSD", "XAUUSD", "NAS100", "US30", "BTCUSD"):
            decision = evaluate(engine, registry, symbol, state)
            if decision.approved:
                approved += 1
                state = state.with_opened(symbol, "test-strategy", Side.BUY, decision.risk_amount)

        assert approved >= 2, "the cluster limit should permit more than one position"
        assert approved < 5, "five uncapped positions is the failure being guarded against"
        assert state.open_risk_amount <= Decimal("1200")
        # And decisively less than the 2 500 five unconstrained signals would be.
        assert state.open_risk_amount < Decimal("2500")

    def test_the_same_five_uncorrelated_are_not_trimmed(
        self, registry: InstrumentRegistry, engine_factory: EngineFactory
    ) -> None:
        # The control. With no group and no matrix, each is its own cluster and
        # the cluster limit binds none of them, so the trimming above is caused
        # by the clustering rather than by the limit merely existing.
        portfolio = PortfolioRisk(
            PortfolioLimitsConfig(
                max_portfolio_risk_pct=1.0,
                max_instrument_risk_pct=1.0,
                max_cluster_risk_pct=0.012,
                max_strategy_risk_pct=1.0,
                max_direction_risk_pct=1.0,
            )
        )
        engine = engine_factory(portfolio=portfolio)
        state = AccountState(
            currency="USD",
            balance=Decimal("100000"),
            equity=Decimal("100000"),
            as_of=NOW,
        )
        for symbol in ("EURUSD", "XAUUSD", "NAS100", "US30", "BTCUSD"):
            decision = evaluate(engine, registry, symbol, state)
            assert decision.approved
            state = state.with_opened(symbol, "test-strategy", Side.BUY, decision.risk_amount)
        assert state.open_risk_amount == Decimal("2500")

    def test_a_signal_is_trimmed_rather_than_refused(
        self, registry: InstrumentRegistry, engine_factory: EngineFactory
    ) -> None:
        # Cutting size to fit a risk budget is this layer doing its job.
        portfolio = PortfolioRisk(
            PortfolioLimitsConfig(
                max_portfolio_risk_pct=0.006,
                max_instrument_risk_pct=1.0,
                max_cluster_risk_pct=1.0,
                max_strategy_risk_pct=1.0,
                max_direction_risk_pct=1.0,
            )
        )
        state = AccountState(
            currency="USD",
            balance=Decimal("100000"),
            equity=Decimal("100000"),
            as_of=NOW,
            open_risks=(),
        ).with_opened("XAUUSD", "other", Side.BUY, Decimal("300"))

        decision = evaluate(engine_factory(portfolio=portfolio), registry, "EURUSD", state)
        assert decision.approved
        # 600 of portfolio ceiling, 300 used, so 300 of headroom: 0.6 lots.
        assert decision.risk_amount <= Decimal("300")
        assert decision.size == Decimal("0.60")
        assert any(RiskReason.TRIMMED_TO_LIMIT.value in line for line in decision.reasons)

    def test_no_headroom_at_all_is_a_refusal_naming_the_limit(
        self, registry: InstrumentRegistry, engine_factory: EngineFactory
    ) -> None:
        portfolio = PortfolioRisk(
            PortfolioLimitsConfig(
                max_portfolio_risk_pct=1.0,
                max_instrument_risk_pct=0.002,
                max_cluster_risk_pct=1.0,
                max_strategy_risk_pct=1.0,
                max_direction_risk_pct=1.0,
            )
        )
        state = AccountState(
            currency="USD", balance=Decimal("100000"), equity=Decimal("100000"), as_of=NOW
        ).with_opened("EURUSD", "other", Side.BUY, Decimal("200"))

        engine = engine_factory(portfolio=portfolio)
        decision = evaluate(engine, registry, "EURUSD", state)
        assert not decision.approved
        assert decision.rejection is RiskReason.INSTRUMENT_LIMIT_EXCEEDED
        assert engine.rejections[RiskReason.INSTRUMENT_LIMIT_EXCEEDED] == 1

    def test_a_trim_below_the_minimum_lot_becomes_a_refusal(
        self, registry: InstrumentRegistry, engine_factory: EngineFactory
    ) -> None:
        # Trimming has a floor: below min_lot there is no trade to place, and
        # rounding back up would defeat the limit that caused the trim.
        portfolio = PortfolioRisk(
            PortfolioLimitsConfig(
                max_portfolio_risk_pct=0.0001,
                max_instrument_risk_pct=1.0,
                max_cluster_risk_pct=1.0,
                max_strategy_risk_pct=1.0,
                max_direction_risk_pct=1.0,
            )
        )
        state = AccountState(
            currency="USD", balance=Decimal("1000"), equity=Decimal("1000"), as_of=NOW
        )
        decision = evaluate(engine_factory(portfolio=portfolio), registry, "EURUSD", state)
        assert not decision.approved
        assert decision.rejection is RiskReason.BELOW_MIN_LOT


class TestCorrelationDegradation:
    """DoD: too little history falls back to the groups and says so."""

    def test_an_unmeasured_pair_is_recorded_and_still_traded(
        self, registry: InstrumentRegistry, engine_factory: EngineFactory
    ) -> None:
        # A provider with no series at all measures nothing. The signal is
        # sized on the manual prior rather than refused, and the fact is
        # counted -- not as a rejection, because the trade happened.
        engine = engine_factory(correlations=CorrelationProvider({}))
        state = AccountState(
            currency="USD", balance=Decimal("100000"), equity=Decimal("100000"), as_of=NOW
        ).with_opened("XAUUSD", "other", Side.BUY, Decimal("100"))

        decision = evaluate(engine, registry, "EURUSD", state)
        assert decision.approved
        assert engine.degradations[RiskReason.CORRELATION_UNAVAILABLE] == 1
        # It is a degradation, not a refusal, and the two counters are separate
        # vocabularies: this reason has no entry in rejections at all, because
        # folding it in would make the refusal count look inflated and the
        # degradation look like a no-trade.
        assert RiskReason.CORRELATION_UNAVAILABLE not in engine.rejections
        assert engine.refused == 0
        assert any(RiskReason.CORRELATION_UNAVAILABLE.value in line for line in decision.reasons)

    def test_the_manual_groups_still_bind_without_any_matrix(
        self, registry: InstrumentRegistry, engine_factory: EngineFactory
    ) -> None:
        # The point of the fallback: losing the measurement loses the extra
        # tightening, never the baseline. A missing correlation must not be
        # more permissive than a measured one.
        portfolio = PortfolioRisk(
            PortfolioLimitsConfig(
                max_portfolio_risk_pct=1.0,
                max_instrument_risk_pct=1.0,
                max_cluster_risk_pct=0.004,
                max_strategy_risk_pct=1.0,
                max_direction_risk_pct=1.0,
                groups=ONE_BET,
            )
        )
        engine = engine_factory(portfolio=portfolio, correlations=CorrelationProvider({}))
        state = AccountState(
            currency="USD", balance=Decimal("100000"), equity=Decimal("100000"), as_of=NOW
        ).with_opened("XAUUSD", "other", Side.BUY, Decimal("400"))

        decision = evaluate(engine, registry, "EURUSD", state)
        assert not decision.approved
        assert decision.rejection is RiskReason.CLUSTER_LIMIT_EXCEEDED

    def test_no_provider_at_all_is_not_a_degradation(
        self, registry: InstrumentRegistry, engine_factory: EngineFactory
    ) -> None:
        # Configuring no provider is a statement that the run has no return
        # history to measure, which is a choice rather than a shortfall.
        engine = engine_factory(correlations=None)
        state = AccountState(
            currency="USD", balance=Decimal("100000"), equity=Decimal("100000"), as_of=NOW
        ).with_opened("XAUUSD", "other", Side.BUY, Decimal("100"))
        evaluate(engine, registry, "EURUSD", state)
        assert engine.degradations[RiskReason.CORRELATION_UNAVAILABLE] == 0


class TestBreakersComeFirst:
    def test_a_tripped_breaker_refuses_before_anything_is_sized(
        self, registry: InstrumentRegistry, engine_factory: EngineFactory
    ) -> None:
        breakers = CircuitBreakers(
            CircuitBreakerConfig(
                max_daily_loss_pct=0.01, max_weekly_loss_pct=None, max_monthly_loss_pct=None
            )
        )
        engine = engine_factory(breakers=breakers)
        state = AccountState(
            currency="USD", balance=Decimal("100000"), equity=Decimal("100000"), as_of=NOW
        )
        trades = [ClosedTrade(closed_at=NOW, pnl=Decimal("-2000"))]

        decision = evaluate(engine, registry, "EURUSD", state, trades=trades)
        assert not decision.approved
        assert decision.rejection is RiskReason.DAILY_LOSS_LIMIT
        assert decision.size == 0
        assert engine.rejections[RiskReason.DAILY_LOSS_LIMIT] == 1

    def test_and_lets_signals_through_once_the_day_rolls(
        self, registry: InstrumentRegistry, engine_factory: EngineFactory
    ) -> None:
        # DoD: after the reset, signals pass again. Same ledger, later instant.
        breakers = CircuitBreakers(
            CircuitBreakerConfig(
                max_daily_loss_pct=0.01, max_weekly_loss_pct=None, max_monthly_loss_pct=None
            )
        )
        engine = engine_factory(breakers=breakers)
        trades = [ClosedTrade(closed_at=NOW, pnl=Decimal("-2000"))]
        state = AccountState(
            currency="USD", balance=Decimal("100000"), equity=Decimal("100000"), as_of=NOW
        )
        assert not evaluate(engine, registry, "EURUSD", state, trades=trades).approved

        # The same signal a day later. Rebuilt rather than mutated: an
        # EntrySignal is frozen, and the point of the test is that nothing but
        # the instant changed.
        today = signal_with_stop_points(registry, "EURUSD", points=50)
        tomorrow = replace(today, bar_close_ts=NOW + timedelta(days=1))
        decision = engine.evaluate(
            tomorrow,
            account=state,
            stop_reference=NEVER_BINDING,
            smallest_exit_fraction=WHOLE,
            bar_index=1,
            trades=trades,
        )
        assert decision.approved

    def test_reset_clears_both_counters_and_pauses(self, engine_factory: EngineFactory) -> None:
        engine = engine_factory()
        engine.reset()
        assert engine.refused == 0
        assert engine.degradations[RiskReason.CORRELATION_UNAVAILABLE] == 0


class TestPortfolioRiskNeverExceedsItsLimit:
    """DoD property: total open risk stays under the ceiling, for any sequence."""

    @settings(max_examples=200, deadline=None)
    @given(
        symbols=st.lists(
            st.sampled_from(["EURUSD", "XAUUSD", "NAS100", "US30", "BTCUSD", "GBPJPY"]),
            min_size=1,
            max_size=12,
        ),
        equity=st.decimals(min_value=5_000, max_value=1_000_000, places=2),
        max_portfolio_pct=st.floats(min_value=0.005, max_value=0.2, allow_nan=False),
        grouped=st.booleans(),
        sides=st.lists(st.sampled_from([Side.BUY, Side.SELL]), min_size=1, max_size=12),
    )
    def test_total_open_risk_stays_within_the_portfolio_ceiling(
        self,
        symbols: list[str],
        equity: Decimal,
        max_portfolio_pct: float,
        grouped: bool,
        sides: list[Side],
    ) -> None:
        from tests.risk.conftest import NO_BREAKERS, REGISTRY_PATH, USDJPY
        from trading_system.core.instruments import load_instruments
        from trading_system.risk.conversion import StaticFxConverter
        from trading_system.risk.engine import RiskEngine, RiskEngineConfig
        from trading_system.risk.sizing.methods import FixedFractional
        from trading_system.risk.stop_calculator import StopBufferConfig

        registry = load_instruments(REGISTRY_PATH)
        engine = RiskEngine(
            instruments=registry,
            # Deliberately greedy: each signal alone wants 10% of equity, so the
            # portfolio ceiling is the only thing standing between the run and
            # a wildly overexposed account.
            sizing=FixedFractional(0.10),
            converter=StaticFxConverter({("USD", "JPY"): USDJPY}),
            config=RiskEngineConfig(
                max_risk_pct=0.10, stop_buffer=StopBufferConfig(spread_multiple=0.0)
            ),
            portfolio=PortfolioRisk(
                PortfolioLimitsConfig(
                    max_portfolio_risk_pct=max_portfolio_pct,
                    max_instrument_risk_pct=1.0,
                    max_cluster_risk_pct=1.0,
                    max_strategy_risk_pct=1.0,
                    max_direction_risk_pct=1.0,
                    groups=ONE_BET if grouped else {},
                )
            ),
            breakers=CircuitBreakers(NO_BREAKERS),
        )
        state = AccountState(currency="USD", balance=equity, equity=equity, as_of=NOW)
        ceiling = equity * Decimal(str(max_portfolio_pct))

        for index, symbol in enumerate(symbols):
            decision = engine.evaluate(
                signal_with_stop_points(
                    registry, symbol, points=50, side=sides[index % len(sides)]
                ),
                account=state,
                stop_reference=NEVER_BINDING,
                smallest_exit_fraction=WHOLE,
                bar_index=index,
                trades=(),
            )
            if decision.approved:
                state = state.with_opened(
                    symbol, "test-strategy", sides[index % len(sides)], decision.risk_amount
                )
            # The invariant, asserted after every single decision rather than
            # once at the end: a breach that is later offset would otherwise
            # pass unnoticed.
            assert state.open_risk_amount <= ceiling
