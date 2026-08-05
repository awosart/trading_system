"""Whether what happened is distinguishable from noise.

:mod:`trading_system.analytics.metrics` is deliberately silent on this
question — it always computes, and always attaches a sample size, but never
decides "reliable: yes/no". That decision lives here, formally, so it is
answered once rather than guessed at with an ad hoc threshold wherever a
number is about to be shown.

**Deflated and Probabilistic Sharpe operate on the *per-period* Sharpe ratio,
not the annualised one.** :func:`~trading_system.analytics.metrics.sharpe_daily`
returns a figure already multiplied by ``sqrt(periods_per_year)`` for
comparability across runs — exactly the wrong input for the Bailey &
López de Prado standard-error formula, which assumes the Sharpe estimate and
its estimation variance come from the same ``T`` observations at the same
frequency. This module recomputes the per-period estimate itself from the
same daily-return sample rather than reusing the annualised result, so the
two figures never get silently mixed.

**The Deflated Sharpe Ratio's benchmark is an asymptotic approximation, not
exact for very small trial counts.** The expected maximum of ``N`` correlated
Sharpe-ratio draws is estimated via the Gumbel (extreme-value) approximation
from Bailey & López de Prado (2014); it is well-behaved and monotonically
increasing in ``N`` from ``N=1`` onward for realistic inputs (verified
numerically in the test suite across ``N`` spanning 1 to 1000), but it is an
asymptotic formula and the literature applies it in practice mainly for
``N`` in the tens or more. ``n_trials`` is taken as given — this module does
not attempt to estimate an *effective* (correlation-adjusted) trial count
from a set of correlated strategy variants; a caller who knows their trials
are correlated is responsible for discounting ``n_trials`` accordingly before
calling in.

**The t-test on mean R uses Newey-West (HAC) standard errors**, not the
naive ``stdev / sqrt(n)``, because consecutive trades from one strategy are
not independent draws — a losing streak can be a shared regime, not ``n``
unrelated coin flips. The naive standard error is smaller than the HAC one
whenever returns are positively autocorrelated, which means it methodically
overstates significance in exactly the scenario ("this system's had a rough
month") an honest evaluation most needs to catch.
"""

import math
import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from scipy import stats

from trading_system.analytics.metrics import (
    DailyCurve,
    field_meta,
    measured_periods_per_year,
    simple_returns,
)
from trading_system.backtest.portfolio import TradeRecord

#: Euler-Mascheroni constant, used by the expected-maximum-Sharpe approximation.
EULER_MASCHERONI = 0.5772156649015329

#: The DoD's blunt floor: below this many trades, every inferential result in
#: this module should be read as illustrative, not decisive. Named once here
#: rather than repeated as a magic number at every call site.
MIN_TRADES_FOR_INFERENCE = 100


def _moments(values: Sequence[float]) -> tuple[float, float, float]:
    """Mean, population (Pearson) skewness, and population (non-excess) kurtosis.

    Args:
        values: The sample.

    Returns:
        ``(mean, skewness, kurtosis)``. ``kurtosis`` is raw, not excess — a
        normal distribution reports 3, matching the ``(kurtosis - 1) / 4``
        term in the Sharpe standard-error formula, where a normal sample
        reduces it to the textbook ``sqrt(1 / (T - 1))``.

    Raises:
        ValueError: If the sample has zero variance, which leaves skewness
            and kurtosis undefined (division by zero).
    """
    n = len(values)
    mean = sum(values) / n
    m2 = sum((v - mean) ** 2 for v in values) / n
    if m2 == 0:
        raise ValueError("moments are undefined: sample has zero variance")
    m3 = sum((v - mean) ** 3 for v in values) / n
    m4 = sum((v - mean) ** 4 for v in values) / n
    return mean, m3 / m2**1.5, m4 / m2**2


def _per_period_sharpe(returns: Sequence[float], rf_period: float) -> tuple[float, int]:
    """Non-annualised Sharpe ratio of a return sample, and its sample size.

    Args:
        returns: Per-period returns.
        rf_period: Risk-free rate at the same period frequency.

    Returns:
        ``(sharpe, n)``.

    Raises:
        ValueError: If there are fewer than two returns, or their standard
            deviation is zero.
    """
    if len(returns) < 2:
        raise ValueError("a per-period sharpe ratio needs at least two returns")
    excess = [r - rf_period for r in returns]
    n = len(excess)
    mean = sum(excess) / n
    variance = sum((e - mean) ** 2 for e in excess) / (n - 1)
    stdev = math.sqrt(variance)
    if stdev == 0:
        raise ValueError("a per-period sharpe ratio is undefined: returns have zero variance")
    return mean / stdev, n


def _sharpe_se_term(skewness: float, kurtosis: float, sharpe: float) -> float:
    """The variance term inside the Mertens/López de Prado Sharpe standard error.

    Args:
        skewness: Sample skewness of the returns the Sharpe was estimated from.
        kurtosis: Sample (non-excess) kurtosis of the same returns.
        sharpe: The per-period Sharpe estimate.

    Returns:
        ``1 - skewness * sharpe + ((kurtosis - 1) / 4) * sharpe ** 2``.

    Raises:
        ValueError: If the term is not positive — the standard error would
            require a square root of a non-positive number.
    """
    term = 1 - skewness * sharpe + ((kurtosis - 1) / 4) * sharpe**2
    if term <= 0:
        raise ValueError(
            "the Sharpe standard-error term is non-positive; skewness and kurtosis of "
            "this sample make the standard Mertens approximation break down"
        )
    return term


def _rf_period(risk_free_rate: float, periods_per_year: float) -> float:
    """Per-period risk-free rate implied by an annual one."""
    return float((1 + risk_free_rate) ** (1 / periods_per_year)) - 1


# ---------------------------------------------------------------------------
# Probabilistic and Deflated Sharpe Ratio
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProbabilisticSharpeResult:
    """Probability that the true (per-period) Sharpe ratio exceeds a benchmark.

    Attributes:
        value: The probability, in ``[0, 1]``.
        observed_sr: The per-period (non-annualised) Sharpe ratio estimated
            from the sample.
        benchmark_sr: The per-period Sharpe the observed one is tested
            against. Zero by default: "is the true Sharpe positive at all."
        skewness: Sample skewness of the excess returns.
        kurtosis: Sample (non-excess) kurtosis of the excess returns.
        n_periods: Number of daily returns the estimate was computed over.
    """

    value: float = field(metadata=field_meta("value", n_field="n_periods"))
    observed_sr: float = field(metadata=field_meta("fact"))
    benchmark_sr: float = field(metadata=field_meta("fact"))
    skewness: float = field(metadata=field_meta("fact"))
    kurtosis: float = field(metadata=field_meta("fact"))
    n_periods: int = field(metadata=field_meta("n"))


def probabilistic_sharpe_ratio(
    daily: DailyCurve,
    *,
    benchmark_sr: float = 0.0,
    risk_free_rate: float = 0.0,
    periods_per_year: float | None = None,
) -> ProbabilisticSharpeResult:
    """Probability that the true per-period Sharpe ratio exceeds ``benchmark_sr``.

    Bailey & López de Prado's PSR: ``Φ((SR̂ - SR*) * sqrt(T - 1) / SE[SR̂])``,
    with ``SE[SR̂]`` from the Mertens (2002) approximation accounting for
    skewness and kurtosis of the return sample.

    Args:
        daily: The daily curve.
        benchmark_sr: Per-period Sharpe to test against. Zero tests "is the
            true Sharpe positive."
        risk_free_rate: Annual risk-free rate, converted to the same
            per-period frequency the Sharpe estimate uses.
        periods_per_year: Annualisation factor, only used to convert
            ``risk_free_rate`` to a per-period rate. Measured from ``daily``
            when ``None``.

    Returns:
        The probability and everything it was computed from.

    Raises:
        ValueError: If there are fewer than three daily returns, the return
            sample has zero variance, or the standard-error term is
            non-positive.
    """
    returns = simple_returns(daily)
    if len(returns) < 3:
        raise ValueError("probabilistic_sharpe_ratio needs at least three daily returns")
    factor = periods_per_year if periods_per_year is not None else measured_periods_per_year(daily)
    rf_period = _rf_period(risk_free_rate, factor)
    sharpe, n = _per_period_sharpe(returns, rf_period)
    _, skewness, kurtosis = _moments([r - rf_period for r in returns])
    se = math.sqrt(_sharpe_se_term(skewness, kurtosis, sharpe) / (n - 1))
    z = (sharpe - benchmark_sr) / se
    return ProbabilisticSharpeResult(
        value=float(stats.norm.cdf(z)),
        observed_sr=sharpe,
        benchmark_sr=benchmark_sr,
        skewness=skewness,
        kurtosis=kurtosis,
        n_periods=n,
    )


@dataclass(frozen=True)
class DeflatedSharpeResult:
    """Probabilistic Sharpe Ratio benchmarked against the best of ``n_trials`` variants.

    Attributes:
        value: The deflated probability, in ``[0, 1]``. Monotonically
            non-increasing in ``n_trials`` for fixed sample statistics.
        observed_sr: The per-period Sharpe ratio of the reported strategy.
        expected_max_sr: The benchmark this run's Sharpe is tested against —
            the expected maximum per-period Sharpe across ``n_trials``
            independent draws with the same estimation variance. Zero when
            ``n_trials <= 1``: testing exactly one strategy earns no
            multiple-testing penalty, and DSR reduces exactly to PSR(0).
        n_trials: Number of strategy variants the deflation accounts for, as
            given by the caller.
        skewness: Sample skewness of the excess returns.
        kurtosis: Sample (non-excess) kurtosis of the excess returns.
        n_periods: Number of daily returns the estimate was computed over.
    """

    value: float = field(metadata=field_meta("value", n_field="n_periods"))
    observed_sr: float = field(metadata=field_meta("fact"))
    expected_max_sr: float = field(metadata=field_meta("fact"))
    n_trials: int = field(metadata=field_meta("fact"))
    skewness: float = field(metadata=field_meta("fact"))
    kurtosis: float = field(metadata=field_meta("fact"))
    n_periods: int = field(metadata=field_meta("n"))


def deflated_sharpe_ratio(
    daily: DailyCurve,
    *,
    n_trials: int,
    risk_free_rate: float = 0.0,
    periods_per_year: float | None = None,
) -> DeflatedSharpeResult:
    """Probabilistic Sharpe Ratio deflated for having tested ``n_trials`` variants.

    Args:
        daily: The daily curve of the reported (typically best-performing)
            variant.
        n_trials: How many independent variants were tried before this one
            was reported. ``1`` means no multiple-testing correction.
        risk_free_rate: Annual risk-free rate.
        periods_per_year: Annualisation factor for converting
            ``risk_free_rate``. Measured from ``daily`` when ``None``.

    Returns:
        The deflated probability and everything it was computed from.

    Raises:
        ValueError: If ``n_trials`` is less than 1, there are fewer than
            three daily returns, or the standard-error term is non-positive.
    """
    if n_trials < 1:
        raise ValueError(f"n_trials must be at least 1, got {n_trials}")
    returns = simple_returns(daily)
    if len(returns) < 3:
        raise ValueError("deflated_sharpe_ratio needs at least three daily returns")
    factor = periods_per_year if periods_per_year is not None else measured_periods_per_year(daily)
    rf_period = _rf_period(risk_free_rate, factor)
    sharpe, n = _per_period_sharpe(returns, rf_period)
    _, skewness, kurtosis = _moments([r - rf_period for r in returns])
    se = math.sqrt(_sharpe_se_term(skewness, kurtosis, sharpe) / (n - 1))

    if n_trials <= 1:
        expected_max_sr = 0.0
    else:
        expected_max_sr = se * (
            (1 - EULER_MASCHERONI) * float(stats.norm.ppf(1 - 1 / n_trials))
            + EULER_MASCHERONI * float(stats.norm.ppf(1 - 1 / (n_trials * math.e)))
        )

    z = (sharpe - expected_max_sr) / se
    return DeflatedSharpeResult(
        value=float(stats.norm.cdf(z)),
        observed_sr=sharpe,
        expected_max_sr=expected_max_sr,
        n_trials=n_trials,
        skewness=skewness,
        kurtosis=kurtosis,
        n_periods=n,
    )


@dataclass(frozen=True)
class MinimumTrackRecordResult:
    """How many daily observations this strategy would need for a target confidence.

    Attributes:
        periods_required: Minimum number of *daily observations* (the same
            frequency ``observed_sr`` was estimated at — not a trade count)
            for the Probabilistic Sharpe Ratio against ``benchmark_sr`` to
            reach ``confidence``, given the sample's current skewness and
            kurtosis held fixed.
        observed_sr: The per-period Sharpe ratio the estimate is based on.
        benchmark_sr: The per-period Sharpe being tested against.
        confidence: Target confidence level.
        skewness: Sample skewness of the excess returns.
        kurtosis: Sample (non-excess) kurtosis of the excess returns.
        n_periods: Number of daily returns the current estimate is based on.
    """

    periods_required: int = field(metadata=field_meta("value", n_field="n_periods"))
    observed_sr: float = field(metadata=field_meta("fact"))
    benchmark_sr: float = field(metadata=field_meta("fact"))
    confidence: float = field(metadata=field_meta("fact"))
    skewness: float = field(metadata=field_meta("fact"))
    kurtosis: float = field(metadata=field_meta("fact"))
    n_periods: int = field(metadata=field_meta("n"))


def minimum_track_record_length(
    daily: DailyCurve,
    *,
    benchmark_sr: float = 0.0,
    confidence: float = 0.95,
    risk_free_rate: float = 0.0,
    periods_per_year: float | None = None,
) -> MinimumTrackRecordResult:
    """Minimum daily-observation count for statistical significance at ``confidence``.

    Args:
        daily: The daily curve.
        benchmark_sr: Per-period Sharpe being tested against.
        confidence: Target confidence level, e.g. ``0.95``.
        risk_free_rate: Annual risk-free rate.
        periods_per_year: Annualisation factor for converting
            ``risk_free_rate``. Measured from ``daily`` when ``None``.

    Returns:
        The minimum period count and everything it was computed from.

    Raises:
        ValueError: If there are fewer than three daily returns, the
            observed Sharpe equals ``benchmark_sr`` (division by zero — no
            finite track record distinguishes them), or the standard-error
            term is non-positive.
    """
    returns = simple_returns(daily)
    if len(returns) < 3:
        raise ValueError("minimum_track_record_length needs at least three daily returns")
    factor = periods_per_year if periods_per_year is not None else measured_periods_per_year(daily)
    rf_period = _rf_period(risk_free_rate, factor)
    sharpe, n = _per_period_sharpe(returns, rf_period)
    if sharpe == benchmark_sr:
        raise ValueError(
            "minimum_track_record_length is undefined: observed Sharpe equals the benchmark"
        )
    _, skewness, kurtosis = _moments([r - rf_period for r in returns])
    term = _sharpe_se_term(skewness, kurtosis, sharpe)
    z = float(stats.norm.ppf(confidence))
    required = 1 + term * (z / (sharpe - benchmark_sr)) ** 2
    return MinimumTrackRecordResult(
        periods_required=math.ceil(required),
        observed_sr=sharpe,
        benchmark_sr=benchmark_sr,
        confidence=confidence,
        skewness=skewness,
        kurtosis=kurtosis,
        n_periods=n,
    )


# ---------------------------------------------------------------------------
# t-test on mean R, autocorrelation-corrected
# ---------------------------------------------------------------------------


def _newey_west_variance(values: Sequence[float], *, lag: int) -> float:
    """Newey-West (Bartlett-kernel) HAC estimate of the sample mean's variance.

    Args:
        values: The sample, in the order autocorrelation is measured over —
            chronological, for a trade sequence.
        lag: Maximum autocorrelation lag to include.

    Returns:
        The estimated variance of the sample mean.
    """
    n = len(values)
    mean = sum(values) / n
    centered = [v - mean for v in values]
    long_run_variance = sum(c * c for c in centered) / n
    for lag_index in range(1, lag + 1):
        weight = 1 - lag_index / (lag + 1)
        autocovariance = sum(centered[t] * centered[t - lag_index] for t in range(lag_index, n)) / n
        long_run_variance += 2 * weight * autocovariance
    return long_run_variance / n


@dataclass(frozen=True)
class TTestResult:
    """One-sample t-test of mean realised R against zero, HAC-corrected.

    Attributes:
        mean_r: Mean realised R across trades.
        standard_error: Newey-West standard error of ``mean_r``, wider than
            the naive ``stdev / sqrt(n)`` whenever consecutive trades are
            positively autocorrelated.
        t_statistic: ``mean_r / standard_error``.
        p_value: Two-sided p-value against a t-distribution with
            ``n_trades - 1`` degrees of freedom.
        lag: Newey-West lag used.
        n_trades: Number of trades the test was computed over.
    """

    mean_r: float = field(metadata=field_meta("value", n_field="n_trades"))
    standard_error: float = field(metadata=field_meta("value", n_field="n_trades"))
    t_statistic: float = field(metadata=field_meta("value", n_field="n_trades"))
    p_value: float = field(metadata=field_meta("value", n_field="n_trades"))
    lag: int = field(metadata=field_meta("fact"))
    n_trades: int = field(metadata=field_meta("n"))


def t_test_mean_r(trades: Sequence[TradeRecord], *, lag: int | None = None) -> TTestResult:
    """Test whether mean realised R is distinguishable from zero.

    Args:
        trades: Completed trades, in any order — sorted internally by
            ``closed_at``, since autocorrelation is a property of trade
            sequence.
        lag: Newey-West lag. The Newey-West (1994) plug-in default,
            ``floor(4 * (n / 100) ** (2 / 9))`` clamped to at least 1, when
            ``None``.

    Returns:
        The test result.

    Raises:
        ValueError: If there are fewer than three trades, or the Newey-West
            variance estimate is non-positive.
    """
    if len(trades) < 3:
        raise ValueError("t_test_mean_r needs at least three trades")
    ordered = sorted(trades, key=lambda trade: trade.closed_at)
    values = [trade.realized_r for trade in ordered]
    n = len(values)
    effective_lag = lag if lag is not None else max(1, math.floor(4 * (n / 100) ** (2 / 9)))
    effective_lag = min(effective_lag, n - 1)
    variance = _newey_west_variance(values, lag=effective_lag)
    if variance <= 0:
        raise ValueError("t_test_mean_r is undefined: Newey-West variance is non-positive")
    standard_error = math.sqrt(variance)
    mean_r = sum(values) / n
    t_statistic = mean_r / standard_error
    p_value = float(2 * stats.t.sf(abs(t_statistic), df=n - 1))
    return TTestResult(
        mean_r=mean_r,
        standard_error=standard_error,
        t_statistic=t_statistic,
        p_value=p_value,
        lag=effective_lag,
        n_trades=n,
    )


# ---------------------------------------------------------------------------
# Bootstrap confidence intervals
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BootstrapResult:
    """Percentile bootstrap confidence interval for an arbitrary statistic.

    Attributes:
        point_estimate: The statistic computed on the original sample.
        lower: Lower bound of the confidence interval.
        upper: Upper bound of the confidence interval.
        confidence: Confidence level used, e.g. ``0.95``.
        n_resamples: Number of bootstrap resamples drawn.
        n: Size of the original sample.
        seed: Seed the resampling was drawn with.
    """

    point_estimate: float = field(metadata=field_meta("value", n_field="n"))
    lower: float = field(metadata=field_meta("value", n_field="n"))
    upper: float = field(metadata=field_meta("value", n_field="n"))
    confidence: float = field(metadata=field_meta("fact"))
    n_resamples: int = field(metadata=field_meta("fact"))
    n: int = field(metadata=field_meta("n"))
    seed: int = field(metadata=field_meta("fact"))


def bootstrap_ci(
    values: Sequence[float],
    statistic: Callable[[Sequence[float]], float],
    *,
    seed: int,
    n_resamples: int = 10_000,
    confidence: float = 0.95,
) -> BootstrapResult:
    """Percentile bootstrap confidence interval for any statistic of a sample.

    Generic by design: any metric this project reports as a single float can
    have a confidence interval put around it by passing its computation in as
    ``statistic``, rather than this module growing one bespoke bootstrap
    function per metric.

    Args:
        values: The sample to resample from.
        statistic: A function computing the statistic of interest from a
            sample of the same shape as ``values``.
        seed: Seed for the resampling draw. Required, no default — the same
            discipline P12's fill RNG uses: a value this consequential to
            reproducibility does not get a silently-shared default.
        n_resamples: Number of bootstrap resamples.
        confidence: Confidence level, e.g. ``0.95``.

    Returns:
        The point estimate and the interval.

    Raises:
        ValueError: If ``values`` has fewer than two observations.
    """
    if len(values) < 2:
        raise ValueError("bootstrap_ci needs at least two observations")
    rng = random.Random(seed)
    n = len(values)
    point_estimate = statistic(values)
    resampled = sorted(
        statistic([values[rng.randrange(n)] for _ in range(n)]) for _ in range(n_resamples)
    )
    lower_index = int((1 - confidence) / 2 * n_resamples)
    upper_index = min(int((1 + confidence) / 2 * n_resamples), n_resamples - 1)
    return BootstrapResult(
        point_estimate=point_estimate,
        lower=resampled[lower_index],
        upper=resampled[upper_index],
        confidence=confidence,
        n_resamples=n_resamples,
        n=n,
        seed=seed,
    )


# ---------------------------------------------------------------------------
# Sample adequacy
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SampleAdequacy:
    """Whether a trade count clears the blunt floor for trusting inference at all.

    Deliberately cruder than :class:`MinimumTrackRecordResult`: that one
    computes a strategy-specific requirement from its own Sharpe, skewness
    and kurtosis; this one is the fixed, always-applicable floor the DoD
    names outright. Both exist because they answer different questions — one
    "is this specific result's own math trustworthy," the other "are we even
    in the range where any of this analysis is worth reading."

    Attributes:
        n_trades: The trade count assessed.
        minimum_required: The floor it was compared against.
        adequate: Whether ``n_trades >= minimum_required``.
    """

    n_trades: int = field(metadata=field_meta("n"))
    minimum_required: int = field(metadata=field_meta("fact"))
    adequate: bool = field(metadata=field_meta("value", n_field="n_trades"))


def assess_sample_adequacy(
    n_trades: int, *, minimum: int = MIN_TRADES_FOR_INFERENCE
) -> SampleAdequacy:
    """Compare a trade count against the floor for trusting inference on it.

    Args:
        n_trades: Number of completed trades.
        minimum: The floor. :data:`MIN_TRADES_FOR_INFERENCE` by default.

    Returns:
        The assessment.

    Raises:
        ValueError: If ``n_trades`` is negative.
    """
    if n_trades < 0:
        raise ValueError(f"n_trades cannot be negative, got {n_trades}")
    return SampleAdequacy(n_trades=n_trades, minimum_required=minimum, adequate=n_trades >= minimum)
