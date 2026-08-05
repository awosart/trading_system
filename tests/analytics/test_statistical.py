"""Verification of the inferential layer against closed-form and textbook values.

Two tricks make the Deflated/Probabilistic Sharpe and MinTRL formulas hand
verifiable despite involving skewness, kurtosis and the normal CDF/quantile:

1. A **symmetric** return sample (``[-0.03, -0.01, 0.01, 0.03]``) has mean
   zero, hence per-period Sharpe exactly zero, which collapses the Mertens
   standard-error term ``1 - skew*SR + ((kurt-1)/4)*SR**2`` to exactly ``1``
   *regardless of the sample's actual skew and kurtosis* — the formula's
   skew/kurtosis-dependent code path still runs, its contribution is just
   provably zero. That turns the standard error into the textbook
   ``sqrt(1 / (T - 1))``.
2. Choosing the benchmark Sharpe to make the resulting z-score land on a
   well-known standard-normal table value (``Φ(1.0) = 0.8413447460``,
   ``Φ⁻¹(0.95) = 1.6448536270``) keeps the final answer a citable constant
   instead of a number only ``scipy`` can produce.

Where a genuine textbook constant is not available (t-distribution
p-values), the test calls ``scipy.stats`` directly as an independent oracle
— it is verifying that this module threads the right ``t``/``df`` into
``scipy``, not re-deriving what ``scipy`` itself computes.
"""

import math
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest
from scipy import stats

from tests.analytics.conftest import curve_from_returns, trade
from trading_system.analytics.statistical import (
    MIN_TRADES_FOR_INFERENCE,
    assess_sample_adequacy,
    bootstrap_ci,
    deflated_sharpe_ratio,
    minimum_track_record_length,
    probabilistic_sharpe_ratio,
    t_test_mean_r,
)
from trading_system.backtest.portfolio import TradeRecord

#: Symmetric about zero -> per-period Sharpe is exactly 0, which collapses
#: the Mertens standard-error term to 1 regardless of skew/kurtosis (see
#: module docstring). Shared across the PSR/DSR/MinTRL tests below.
ZERO_SHARPE_RETURNS = [Decimal("-0.03"), Decimal("-0.01"), Decimal("0.01"), Decimal("0.03")]

#: With se_term == 1, choosing benchmark_sr = -1/sqrt(3) makes
#: (observed_sr - benchmark_sr) / se == 1.0 exactly (se == sqrt(1/3)),
#: landing the z-score on the textbook constant Phi(1.0) = 0.8413447460.
BENCHMARK_FOR_Z_EQUALS_ONE = -1 / math.sqrt(3)


class TestProbabilisticSharpeRatio:
    def test_it_is_exactly_one_half_when_observed_equals_benchmark(self) -> None:
        # z = (SR - SR*) / SE = 0 whenever SR* == SR, for any SE -> Phi(0) = 0.5,
        # regardless of skewness or kurtosis.
        daily = curve_from_returns(ZERO_SHARPE_RETURNS)
        result = probabilistic_sharpe_ratio(daily, benchmark_sr=0.0)
        assert result.observed_sr == pytest.approx(0.0, abs=1e-12)
        assert result.value == pytest.approx(0.5, abs=1e-9)
        assert result.n_periods == 4

    def test_it_matches_the_textbook_phi_of_one(self) -> None:
        # se = sqrt(se_term / (n - 1)) = sqrt(1 / 3); z = (0 - benchmark) / se
        # = 1.0 by construction of BENCHMARK_FOR_Z_EQUALS_ONE; Phi(1.0) is the
        # standard normal table constant 0.8413447460.
        daily = curve_from_returns(ZERO_SHARPE_RETURNS)
        result = probabilistic_sharpe_ratio(daily, benchmark_sr=BENCHMARK_FOR_Z_EQUALS_ONE)
        assert result.value == pytest.approx(0.8413447460685429, rel=1e-9)

    def test_fewer_than_three_returns_is_rejected(self) -> None:
        daily = curve_from_returns([Decimal("0.01"), Decimal("0.02")])
        with pytest.raises(ValueError, match="at least three"):
            probabilistic_sharpe_ratio(daily)


class TestDeflatedSharpeRatio:
    def test_one_trial_reduces_exactly_to_psr_against_zero(self) -> None:
        daily = curve_from_returns(ZERO_SHARPE_RETURNS)
        dsr = deflated_sharpe_ratio(daily, n_trials=1)
        psr = probabilistic_sharpe_ratio(daily, benchmark_sr=0.0)
        assert dsr.expected_max_sr == 0.0
        assert dsr.value == pytest.approx(psr.value, abs=1e-12)

    def test_it_is_monotonically_non_increasing_in_the_trial_count(self) -> None:
        # A realistic, mildly positive-drift, asymmetric-ish return series —
        # deliberately not the symmetric zero-Sharpe fixture, so this test
        # exercises the formula on genuinely nonzero skew/kurtosis/Sharpe.
        returns = [
            Decimal("0.012"),
            Decimal("-0.004"),
            Decimal("0.018"),
            Decimal("0.006"),
            Decimal("-0.010"),
            Decimal("0.022"),
            Decimal("0.003"),
            Decimal("-0.006"),
            Decimal("0.015"),
            Decimal("0.009"),
            Decimal("-0.012"),
            Decimal("0.020"),
        ]
        daily = curve_from_returns(returns)
        trial_counts = (1, 10, 50, 200, 1000)
        values = [deflated_sharpe_ratio(daily, n_trials=n).value for n in trial_counts]
        assert values == sorted(values, reverse=True)
        assert values[0] > values[-1]

    def test_n_trials_below_one_is_rejected(self) -> None:
        daily = curve_from_returns(ZERO_SHARPE_RETURNS)
        with pytest.raises(ValueError, match="n_trials"):
            deflated_sharpe_ratio(daily, n_trials=0)


class TestMinimumTrackRecordLength:
    def test_it_matches_the_hand_derived_period_count(self) -> None:
        # se_term == 1 (zero-Sharpe fixture, see module docstring).
        # sharpe - benchmark_sr == 1/sqrt(3) by construction.
        # z = Phi^-1(0.95) = 1.6448536270 (standard normal table constant).
        # periods_required = 1 + 1 * (z / (1/sqrt(3))) ** 2
        #                   = 1 + 3 * z ** 2
        #                   = 1 + 3 * 1.6448536270 ** 2
        #                   ~= 9.1166  ->  ceil -> 10.
        daily = curve_from_returns(ZERO_SHARPE_RETURNS)
        result = minimum_track_record_length(
            daily, benchmark_sr=BENCHMARK_FOR_Z_EQUALS_ONE, confidence=0.95
        )
        z = 1.6448536269514722
        expected_continuous = 1 + 3 * z**2
        assert expected_continuous == pytest.approx(9.1166, abs=1e-3)
        assert result.periods_required == math.ceil(expected_continuous)
        assert result.periods_required == 10
        assert result.n_periods == 4

    def test_observed_sharpe_equal_to_benchmark_is_rejected(self) -> None:
        daily = curve_from_returns(ZERO_SHARPE_RETURNS)
        with pytest.raises(ValueError, match="undefined"):
            minimum_track_record_length(daily, benchmark_sr=0.0)


class TestTTestMeanR:
    def _trades(self, values: list[float]) -> list[TradeRecord]:
        base = datetime(2024, 1, 1, tzinfo=UTC)
        return [
            trade(
                position_id=f"t{i}",
                opened_at=base + timedelta(days=i),
                closed_at=base + timedelta(days=i, hours=1),
                net=100.0,
                realized_r=r,
            )
            for i, r in enumerate(values)
        ]

    def test_lag_zero_matches_the_hand_derived_naive_variance(self) -> None:
        # values = [2.0, -1.0, 3.0, -2.0, 1.0], mean = 0.6.
        # deviations: 1.4, -1.6, 2.4, -2.6, 0.4; squares sum = 17.20.
        # lag=0 leaves no autocovariance terms, so Var(mean) is the
        # population variance over n: (17.20 / 5) / 5 = 0.688.
        trades = self._trades([2.0, -1.0, 3.0, -2.0, 1.0])
        result = t_test_mean_r(trades, lag=0)
        assert result.mean_r == pytest.approx(0.6)
        assert result.standard_error == pytest.approx(math.sqrt(0.688), rel=1e-9)
        assert result.lag == 0
        assert result.n_trades == 5

        expected_t = 0.6 / math.sqrt(0.688)
        assert result.t_statistic == pytest.approx(expected_t, rel=1e-9)
        # p-value against a t-distribution is not a hand constant; scipy is
        # used directly here as an oracle for what this module should be
        # calling internally, with the same (t, df) this test derived above.
        expected_p = float(2 * stats.t.sf(abs(expected_t), df=4))
        assert result.p_value == pytest.approx(expected_p, rel=1e-9)

    def test_the_default_lag_follows_the_newey_west_plug_in_rule(self) -> None:
        # floor(4 * (n / 100) ** (2 / 9)) with n=100 is floor(4 * 1) = 4.
        values = [1.0 if i % 2 == 0 else -1.0 for i in range(100)]
        result = t_test_mean_r(self._trades(values))
        assert result.lag == 4

    def test_an_explicit_lag_larger_than_n_minus_one_is_clamped(self) -> None:
        trades = self._trades([1.0, -1.0, 2.0, -2.0, 3.0])
        result = t_test_mean_r(trades, lag=10)
        assert result.lag == 4  # n - 1

    def test_fewer_than_three_trades_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="at least three"):
            t_test_mean_r(self._trades([1.0, -1.0]))


class TestBootstrapCi:
    def test_the_point_estimate_is_the_statistic_on_the_original_sample(self) -> None:
        values = [1.0, 2.0, 3.0, 4.0, 5.0]

        def mean(sample: Sequence[float]) -> float:
            return sum(sample) / len(sample)

        result = bootstrap_ci(values, mean, seed=7, n_resamples=2000)
        assert result.point_estimate == pytest.approx(3.0)
        assert result.n == 5
        assert result.n_resamples == 2000
        assert result.confidence == 0.95
        assert result.seed == 7

    def test_the_interval_never_exceeds_the_range_of_the_original_data(self) -> None:
        # Every bootstrap resample draws with replacement from the original
        # values, so the mean of any resample is a weighted average of them
        # and cannot leave [min(values), max(values)].
        values = [10.0, 12.0, 11.0, 9.0, 15.0, 8.0]

        def mean(sample: Sequence[float]) -> float:
            return sum(sample) / len(sample)

        result = bootstrap_ci(values, mean, seed=3, n_resamples=5000)
        assert min(values) <= result.lower <= result.upper <= max(values)

    def test_the_same_seed_reproduces_the_same_interval(self) -> None:
        values = [1.0, 5.0, 2.0, 8.0, 3.0, 9.0, 4.0]

        def mean(sample: Sequence[float]) -> float:
            return sum(sample) / len(sample)

        first = bootstrap_ci(values, mean, seed=42, n_resamples=1000)
        second = bootstrap_ci(values, mean, seed=42, n_resamples=1000)
        assert first == second

    def test_a_different_seed_gives_a_different_interval(self) -> None:
        values = [1.0, 5.0, 2.0, 8.0, 3.0, 9.0, 4.0]

        def mean(sample: Sequence[float]) -> float:
            return sum(sample) / len(sample)

        first = bootstrap_ci(values, mean, seed=42, n_resamples=1000)
        second = bootstrap_ci(values, mean, seed=99, n_resamples=1000)
        assert (first.lower, first.upper) != (second.lower, second.upper)

    def test_fewer_than_two_values_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="at least two"):
            bootstrap_ci([1.0], lambda sample: sample[0], seed=1)


class TestSampleAdequacy:
    def test_it_matches_the_dod_floor_of_one_hundred(self) -> None:
        assert MIN_TRADES_FOR_INFERENCE == 100
        assert assess_sample_adequacy(19).adequate is False
        assert assess_sample_adequacy(19).minimum_required == 100
        assert assess_sample_adequacy(100).adequate is True
        assert assess_sample_adequacy(150).adequate is True

    def test_a_custom_floor_is_honoured(self) -> None:
        assert assess_sample_adequacy(19, minimum=10).adequate is True

    def test_a_negative_trade_count_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="negative"):
            assess_sample_adequacy(-1)
