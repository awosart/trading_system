"""SortinoTimesSqrtTrades: scores a real run, and raises rather than papering over a bad one."""

import pytest

from tests.backtest.conftest import EURUSD_H4, ema_pullback, harness_inputs, swing_series
from trading_system.backtest.orchestrator import BacktestResult, StrategyBinding
from trading_system.core.instruments import InstrumentRegistry
from trading_system.exit.library import ExitLibrarySpec
from trading_system.validation.objective import SortinoTimesSqrtTrades


@pytest.fixture(scope="module")
def real_result(registry: InstrumentRegistry, library: ExitLibrarySpec) -> BacktestResult:
    spec = ema_pullback()
    preset = next(item for item in library.presets if item.id == spec.exit_ref)
    base = harness_inputs(
        registry,
        streams={EURUSD_H4: swing_series(2000)},
        bindings=[StrategyBinding(spec=spec, exit_preset=preset, keys=(EURUSD_H4,))],
    )
    return base.run()


class TestSortinoTimesSqrtTrades:
    def test_scores_a_real_run(self, real_result: BacktestResult) -> None:
        assert real_result.trades, "fixture must actually trade, or this test proves nothing"
        score = SortinoTimesSqrtTrades().score(real_result)
        assert isinstance(score, float)

    def test_matches_the_hand_computed_formula(self, real_result: BacktestResult) -> None:
        from trading_system.analytics.metrics import daily_curve, sortino_daily

        expected = (
            sortino_daily(daily_curve(real_result.curve)).value * len(real_result.trades) ** 0.5
        )
        assert SortinoTimesSqrtTrades().score(real_result) == pytest.approx(expected)

    def test_zero_trades_is_rejected(self) -> None:
        empty = BacktestResult(
            curve=(),
            trades=(),
            instants=0,
            rejections={},
            degradations={},
            exit_drops={},
            entry_drops={},
            signal_drops={},
            expired_orders=0,
            cost_degradations={},
            atr_ratio_coverage={},
            fills=0,
            fx_fallback_marks=0,
            open_at_end=0,
        )
        with pytest.raises(ValueError, match="zero trades"):
            SortinoTimesSqrtTrades().score(empty)

    def test_more_trades_scale_the_score_by_sqrt_of_the_ratio(
        self, real_result: BacktestResult
    ) -> None:
        """Doubling the trade count, holding Sortino fixed, scales the score by sqrt(2)."""
        from dataclasses import replace as dc_replace

        doubled = dc_replace(real_result, trades=list(real_result.trades) * 2)
        single = SortinoTimesSqrtTrades().score(real_result)
        double = SortinoTimesSqrtTrades().score(doubled)
        assert double == pytest.approx(single * (2**0.5))

    def test_is_picklable(self, real_result: BacktestResult) -> None:
        """Required for a process-pool worker to receive it inside a CalibrationIterationSpec."""
        import pickle

        objective = SortinoTimesSqrtTrades(risk_free_rate=0.01, mar=0.001)
        restored = pickle.loads(pickle.dumps(objective))
        assert restored == objective
        assert restored.score(real_result) == objective.score(real_result)
