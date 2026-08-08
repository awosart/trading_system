"""Permutation: same bar shapes, different bars — and no autocorrelation left to find."""

import statistics
from datetime import timedelta
from decimal import Decimal
from pathlib import Path

import pytest

from tests.backtest.conftest import EURUSD_H4, ema_pullback, harness_inputs, swing_series
from trading_system.backtest.clock import StreamKey
from trading_system.backtest.orchestrator import StrategyBinding
from trading_system.core.instruments import InstrumentRegistry
from trading_system.core.types import Timeframe
from trading_system.data.models import OHLCVFrame
from trading_system.data.resample import FX_DAY_ORIGIN
from trading_system.exit.library import ExitLibrarySpec
from trading_system.validation.nulls.permutation import (
    PermutationConfig,
    permutation_order,
    permute_frame,
    permute_run_streams,
)
from trading_system.validation.splitting import WalkForwardMode, WalkForwardSplitter
from trading_system.validation.walkforward import IdentitySelector, WalkForwardRunner

LENGTH = 1500


def _shapes(frame: OHLCVFrame) -> tuple[list[float], list[float], list[float], list[float]]:
    """Sorted gaps, bodies, high-ratios, low-ratios — the multiset a permutation must preserve."""
    table = frame.df
    o, h, low, c = (table[col].to_list() for col in ("open", "high", "low", "close"))
    n = len(table)
    gaps = sorted(o[i] / c[i - 1] for i in range(1, n))
    bodies = sorted(c[i] / o[i] for i in range(1, n))
    high_ratios = sorted(h[i] / max(o[i], c[i]) for i in range(1, n))
    low_ratios = sorted(low[i] / min(o[i], c[i]) for i in range(1, n))
    return gaps, bodies, high_ratios, low_ratios


def _lag1_autocorrelation(closes: list[float]) -> float:
    returns = [closes[i] / closes[i - 1] - 1 for i in range(1, len(closes))]
    mean = statistics.mean(returns)
    num = sum((returns[i] - mean) * (returns[i - 1] - mean) for i in range(1, len(returns)))
    den = sum((r - mean) ** 2 for r in returns)
    return num / den


def _variance_ratio(closes: list[float], q: int) -> float:
    """VR(q): var of q-period log returns over q * var of 1-period log returns. ~1 under IID."""
    import math

    log_prices = [math.log(c) for c in closes]
    one_period = [log_prices[i] - log_prices[i - 1] for i in range(1, len(log_prices))]
    q_period = [log_prices[i] - log_prices[i - q] for i in range(q, len(log_prices), q)]
    var_one = statistics.variance(one_period)
    var_q = statistics.variance(q_period)
    return var_q / (q * var_one)


class TestPermutationOrder:
    def test_too_few_bars_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="at least 2"):
            permutation_order(1, seed=1)

    def test_returns_a_permutation_of_1_to_n_minus_1(self) -> None:
        order = permutation_order(50, seed=3)
        assert sorted(order) == list(range(1, 50))

    def test_deterministic_on_seed(self) -> None:
        assert permutation_order(200, seed=11) == permutation_order(200, seed=11)

    def test_different_seeds_generally_differ(self) -> None:
        assert permutation_order(200, seed=11) != permutation_order(200, seed=12)


class TestPermuteFrame:
    def test_rejects_a_non_permutation(self) -> None:
        frame = swing_series(20)
        with pytest.raises(ValueError, match="permutation"):
            permute_frame(frame, list(range(1, 19)))  # too short

    def test_timestamps_are_unchanged(self) -> None:
        frame = swing_series(LENGTH)
        order = permutation_order(len(frame), seed=5)
        permuted = permute_frame(frame, order)
        assert permuted.df["timestamp"].to_list() == frame.df["timestamp"].to_list()

    def test_bar_zero_is_unchanged(self) -> None:
        frame = swing_series(LENGTH)
        order = permutation_order(len(frame), seed=5)
        permuted = permute_frame(frame, order)
        for col in ("open", "high", "low", "close", "volume"):
            assert permuted.df[col].item(0) == frame.df[col].item(0)

    def test_preserves_the_multiset_of_bar_shapes(self) -> None:
        """A permutation, not a resampling: pointwise-identical distribution."""
        frame = swing_series(LENGTH)
        order = permutation_order(len(frame), seed=17)
        permuted = permute_frame(frame, order)

        original = _shapes(frame)
        shuffled = _shapes(permuted)
        for orig_values, new_values in zip(original, shuffled, strict=True):
            assert orig_values == pytest.approx(new_values, rel=1e-9)

    def test_every_bar_is_valid_ohlc(self) -> None:
        frame = swing_series(LENGTH)
        order = permutation_order(len(frame), seed=23)
        permuted = permute_frame(frame, order)
        table = permuted.df
        o, h, low, c = (table[col].to_list() for col in ("open", "high", "low", "close"))
        for i in range(len(table)):
            assert h[i] >= max(o[i], c[i])
            assert low[i] <= min(o[i], c[i])

    def test_autocorrelation_is_near_zero(self) -> None:
        frame = swing_series(LENGTH)  # a strongly autocorrelated (sinusoidal) source series
        order = permutation_order(len(frame), seed=29)
        permuted = permute_frame(frame, order)
        assert abs(_lag1_autocorrelation(frame.df["close"].to_list())) > 0.05, (
            "the source series must actually be autocorrelated, or this test proves nothing"
        )
        assert abs(_lag1_autocorrelation(permuted.df["close"].to_list())) < 0.1

    def test_variance_ratio_is_near_one(self) -> None:
        frame = swing_series(LENGTH)
        order = permutation_order(len(frame), seed=31)
        permuted = permute_frame(frame, order)
        assert abs(_variance_ratio(permuted.df["close"].to_list(), q=2) - 1.0) < 0.25


class TestPermuteRunStreams:
    def test_no_stream_at_finest_is_rejected(self) -> None:
        streams = {StreamKey("EURUSD", Timeframe.H4): swing_series(LENGTH, timeframe=Timeframe.H4)}
        config = PermutationConfig(finest=Timeframe.H1, day_origin=FX_DAY_ORIGIN, seed=1)
        with pytest.raises(ValueError, match="no stream"):
            permute_run_streams(streams, config)

    def test_finest_streams_disagreeing_on_length_are_rejected(self) -> None:
        streams = {
            StreamKey("EURUSD", Timeframe.H1): swing_series(500, timeframe=Timeframe.H1),
            StreamKey("XAUUSD", Timeframe.H1): swing_series(
                400, symbol="XAUUSD", timeframe=Timeframe.H1
            ),
        }
        config = PermutationConfig(finest=Timeframe.H1, day_origin=FX_DAY_ORIGIN, seed=1)
        with pytest.raises(ValueError, match="disagree"):
            permute_run_streams(streams, config)

    def test_two_symbols_share_one_order(self) -> None:
        """Cross-symbol correlation is preserved: the same shuffle order for both."""
        eurusd = swing_series(LENGTH, timeframe=Timeframe.H1)
        xauusd = swing_series(
            LENGTH, symbol="XAUUSD", base=2000.0, scale=2000.0, phase=1.6, timeframe=Timeframe.H1
        )
        streams = {
            StreamKey("EURUSD", Timeframe.H1): eurusd,
            StreamKey("XAUUSD", Timeframe.H1): xauusd,
        }
        config = PermutationConfig(finest=Timeframe.H1, day_origin=FX_DAY_ORIGIN, seed=41)
        permuted = permute_run_streams(streams, config)

        order = permutation_order(len(eurusd), seed=41)
        expected_eurusd = permute_frame(eurusd, order)
        expected_xauusd = permute_frame(xauusd, order)
        assert permuted[StreamKey("EURUSD", Timeframe.H1)].df.equals(expected_eurusd.df)
        assert permuted[StreamKey("XAUUSD", Timeframe.H1)].df.equals(expected_xauusd.df)

    def test_coarser_stream_is_reaggregated_from_the_permuted_finest(self) -> None:
        h1 = swing_series(2400, timeframe=Timeframe.H1)
        streams = {
            StreamKey("EURUSD", Timeframe.H1): h1,
            StreamKey("EURUSD", Timeframe.H4): swing_series(600, timeframe=Timeframe.H4),
        }
        config = PermutationConfig(finest=Timeframe.H1, day_origin=FX_DAY_ORIGIN, seed=7)
        permuted = permute_run_streams(streams, config)

        from trading_system.data.resample import resample

        permuted_h1 = permuted[StreamKey("EURUSD", Timeframe.H1)]
        expected_h4 = resample(permuted_h1, Timeframe.H4, origin=None)
        assert permuted[StreamKey("EURUSD", Timeframe.H4)].df.equals(expected_h4.df)

    def test_coarser_stream_with_no_matching_symbol_at_finest_is_rejected(self) -> None:
        streams = {
            StreamKey("EURUSD", Timeframe.H1): swing_series(500, timeframe=Timeframe.H1),
            StreamKey("XAUUSD", Timeframe.H4): swing_series(
                100, symbol="XAUUSD", timeframe=Timeframe.H4
            ),
        }
        config = PermutationConfig(finest=Timeframe.H1, day_origin=FX_DAY_ORIGIN, seed=1)
        with pytest.raises(ValueError, match="XAUUSD"):
            permute_run_streams(streams, config)


class TestFoldSequenceAcceptance:
    """The acceptance test itself: walk-forward over permuted, zero-cost data scores at zero.

    Runs the real walk-forward machine (:mod:`trading_system.validation.walkforward`)
    over data permuted once per iteration, on a strictly zero-cost config —
    per CLAUDE.md P15 stage 1.5, this is the one place the acceptance
    criterion is stated as "indistinguishable from zero", which is only true
    without turnover cost. If this fails, the defect is in the fold sequence
    or the evaluation-window mechanics, and the fix is not a looser threshold.
    """

    def test_median_oos_expectancy_across_permuted_iterations_covers_zero(
        self, registry: InstrumentRegistry, library: ExitLibrarySpec, tmp_path: Path
    ) -> None:
        spec = ema_pullback()
        preset = next(item for item in library.presets if item.id == spec.exit_ref)
        zero_cost_registry = InstrumentRegistry(
            {
                **{s: registry[s] for s in registry.symbols},
                "EURUSD": registry["EURUSD"].model_copy(
                    update={"typical_spread_points": 0.0, "commission_per_lot": Decimal(0)}
                ),
            }
        )
        base = harness_inputs(
            zero_cost_registry,
            streams={EURUSD_H4: swing_series(2000)},
            bindings=[StrategyBinding(spec=spec, exit_preset=preset, keys=(EURUSD_H4,))],
        )
        from trading_system.execution.config import CostConfig

        base = base.__class__(
            **{
                **{field: getattr(base, field) for field in base.__dataclass_fields__},
                "costs": CostConfig(run_seed=base.costs.run_seed),
            }
        )

        splitter = WalkForwardSplitter(
            mode=WalkForwardMode.ROLLING,
            is_span=timedelta(days=60),
            oos_span=timedelta(days=20),
            step=timedelta(days=20),
            embargo=timedelta(days=2),
            warmup=timedelta(days=10),
        )
        from trading_system.validation.report import build_report

        medians = []
        n_iterations = 8
        for iteration in range(n_iterations):
            config = PermutationConfig(
                finest=Timeframe.H4, day_origin=FX_DAY_ORIGIN, seed=iteration
            )
            permuted_streams = permute_run_streams(base.streams, config)
            permuted_base = base.with_streams(permuted_streams)
            runner = WalkForwardRunner(
                base=permuted_base,
                splitter=splitter,
                selector=IdentitySelector(permuted_base),
                store_root=tmp_path / f"iter-{iteration}",
                max_drain_bars=20,
            )
            result = runner.run()
            report = build_report(result, min_trades_per_fold=1)
            oos_values = [v for v in report.per_fold_oos_expectancy_r if v is not None]
            if oos_values:
                medians.append(statistics.median(oos_values))

        assert len(medians) >= 4, "too many iterations produced no scoreable OOS fold"
        overall_median = statistics.median(medians)
        # A wide, honest band rather than a tuned threshold: proves "not
        # obviously biased away from zero" on a small iteration count, not
        # "statistically indistinguishable" to a precision this n cannot support.
        assert abs(overall_median) < 1.0
