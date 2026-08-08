"""Does the result survive being asked slightly differently.

**Perturbations are of the data and the clock, never of the trade list.** Every
check here changes what the strategy is fed and re-walks it, because each of
these questions — a different start bar, a noisier tape, a synthetic tape —
changes *which trades exist*, and no rearrangement of a finished trade list can
express that. That is the dividing line against
:mod:`trading_system.validation.monte_carlo`, which resamples outcomes and
therefore costs microseconds; everything here costs complete backtests, and its
N is small on purpose.

**What "same volatility" means in the synthetic test, and why calibrating it on
the tested window is not a leak.** The question the synthetic test answers is
"does this strategy make money on a tape with no exploitable structure?", and
the null it tests is *returns are serially independent, zero drift, at this
instrument's realised volatility*. Calibrating sigma on the very window being
tested is what **defines** that null: calibrate it somewhere else and the
comparison starts confounding "no structure" with "a quieter or noisier regime",
so a strategy could fail the test merely because the synthetic tape was calmer,
which says nothing about overfitting.

The lookahead worry belongs to a different object than one might expect. The
strategy cannot see sigma: it reads bars through
:class:`~trading_system.entry.context.BarContext`, whose accessors take only
``lookback`` (proven by P06's equivalence test), and
:class:`~trading_system.backtest.engine.BarStore`, which raises
``LookaheadError`` past its cursor. The generator's parameters are not
reachable from anywhere the strategy can stand — there is no path in the object
graph, so there is nothing to guard.

What genuinely needed handling is a different mismatch: a single global sigma
produces a tape with **no volatility clustering**, so the null would differ
from reality in more than predictability — and this system's costs are
volatility-sensitive (the spread's volatility multiplier, ATR-scaled slippage,
ATR-derived stops), meaning the comparison would silently include a difference
in *costs*. So sigma is piecewise constant **per trading day**, estimated from
the real series' own daily realised volatility, on the same ``trading_day``
boundary VWAP and the daily loss limit already use. Time-varying volatility
survives; direction does not, which is the only thing being nulled.

**Drift is the one moment deliberately not matched: it is exactly zero.** Match
the drift too and buy-and-hold would "earn" on the synthetic tape, which would
make this test and the trending-series fixture used to prove a genuinely robust
strategy mutually incoherent. A trending synthetic series is a *different*
fixture, built on purpose.

**The synthetic tape does not reuse real bar shapes.** That is precisely what
:mod:`trading_system.validation.nulls.permutation` already does, and two names
for one test would be worse than one. Bars here are built by simulating
sub-steps of a driftless walk and taking the running max and min, so the
intrabar range is generated rather than borrowed. The two nulls are
complementary rather than redundant: permutation preserves the marginal return
distribution exactly, fat tails included, and destroys only ordering; a
Gaussian walk destroys both. A strategy whose apparent edge is really a
tail-risk premium passes the first and fails the second, so both are reported
and a disagreement between them is information, not a contradiction.
"""

import math
import statistics
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from random import Random
from typing import Any

import polars as pl

from trading_system.analytics.metrics import (
    daily_curve,
    drawdown_stats,
    sharpe_daily,
    sortino_daily,
)
from trading_system.backtest.clock import StreamKey
from trading_system.backtest.portfolio import TradeRecord
from trading_system.backtest.spec import RunInputs
from trading_system.core.logging import get_logger
from trading_system.data.models import OHLCVFrame
from trading_system.data.resample import DayOrigin, resample, trading_day

logger = get_logger(__name__)

#: Sub-steps simulated inside each bar when building a synthetic tape. Four is
#: enough for the running max and min to be a meaningful intrabar range rather
#: than ``max(open, close)``, and small enough that the generator stays cheap on
#: a 45k-bar series. Not a tuning knob: it is recorded in the result so a reader
#: can see what the range was built from.
SYNTHETIC_SUBSTEPS = 4


@dataclass(frozen=True)
class RobustnessRun:
    """One perturbed run, reduced to what a comparison needs.

    Attributes:
        label: What the perturbation was.
        n_trades: Trades closed.
        expectancy_r: Mean realised R, or ``None`` with no trades.
        sharpe: Annualised Sharpe of the curve, or ``None``.
        sortino: Annualised Sortino, or ``None``.
        max_drawdown: Deepest peak-to-trough fall, as a fraction, or ``None``.
    """

    label: str
    n_trades: int
    expectancy_r: float | None
    sharpe: float | None
    sortino: float | None
    max_drawdown: float | None

    def to_dict(self) -> dict[str, Any]:
        """Serialise to JSON-able data."""
        return {
            "label": self.label,
            "n_trades": self.n_trades,
            "expectancy_r": self.expectancy_r,
            "sharpe": self.sharpe,
            "sortino": self.sortino,
            "max_drawdown": self.max_drawdown,
        }


def evaluate_run(inputs: RunInputs, label: str) -> RobustnessRun:
    """Walk one run and reduce it to a :class:`RobustnessRun`.

    Args:
        inputs: The run.
        label: What to call it.

    Returns:
        Its summary. Metrics the sample cannot support come back as ``None``
        rather than as an exception, because a perturbation that stops the
        strategy trading altogether is a *result* here, not an error.
    """
    result = inputs.run()
    if not result.trades:
        return RobustnessRun(label, 0, None, None, None, None)
    curve = daily_curve(result.curve)

    def safe(fn: Callable[[], float]) -> float | None:
        try:
            return fn()
        except ValueError:
            return None

    return RobustnessRun(
        label=label,
        n_trades=len(result.trades),
        expectancy_r=statistics.fmean(trade.realized_r for trade in result.trades),
        sharpe=safe(lambda: sharpe_daily(curve).value),
        sortino=safe(lambda: sortino_daily(curve).value),
        max_drawdown=safe(lambda: drawdown_stats(curve).max_drawdown_pct),
    )


# ---------------------------------------------------------------------------
# Start shift
# ---------------------------------------------------------------------------


def shift_start(streams: Mapping[StreamKey, OHLCVFrame], bars: int) -> dict[StreamKey, OHLCVFrame]:
    """Drop the first ``bars`` bars of the finest stream, and the same span from the rest.

    Shifted by a *bar count on the finest stream*, converted to a calendar span
    so that coarser streams lose the same period rather than the same number of
    their own (much longer) bars — the alternative would shift a D1 stream by
    ``bars`` days while shifting its H1 partner by ``bars`` hours, which is not
    one perturbation but two.

    Args:
        streams: The run's bars.
        bars: How many bars of the finest stream to drop.

    Returns:
        The shifted streams.

    Raises:
        ValueError: If ``bars`` is negative, or ``streams`` is empty.
    """
    if bars < 0:
        raise ValueError(f"bars must be non-negative, got {bars}")
    if not streams:
        raise ValueError("cannot shift the start of zero streams")
    finest = min(streams, key=lambda key: key.timeframe.duration)
    span = finest.timeframe.duration * bars
    starts = [frame.start for frame in streams.values() if frame.start is not None]
    if not starts:
        raise ValueError("every stream is empty; nothing to shift")
    new_start = min(starts) + span
    return {key: frame.slice(new_start, None) for key, frame in streams.items()}


def run_start_shift(
    base: RunInputs,
    *,
    shifts: Sequence[int],
    evaluate: Callable[[RunInputs, str], RobustnessRun] = evaluate_run,
) -> tuple[RobustnessRun, ...]:
    """Re-run with the series starting a few bars later each time.

    A result that moves materially when the first bar changes is a result whose
    fold anchors, warmup and therefore whole trade sequence were an accident of
    where the data happened to begin.

    Args:
        base: The run to shift.
        shifts: Bar counts to drop, on the finest stream. ``0`` is the
            unshifted control and is worth including.
        evaluate: How to walk one run. Swappable so a caller can spend a full
            walk-forward per shift or a single backtest, which is a cost policy
            this module should not decide.

    Returns:
        One summary per shift, in the order given.
    """
    return tuple(
        evaluate(base.with_streams(shift_start(base.streams, count)), f"start_shift_{count}")
        for count in shifts
    )


# ---------------------------------------------------------------------------
# Period consistency
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PeriodResult:
    """One equal-length slice of the run's calendar span.

    Attributes:
        index: Position in the sequence, earliest first.
        start: Inclusive lower bound.
        end: Exclusive upper bound.
        n_trades: Trades that closed inside it.
        expectancy_r: Mean realised R inside it, or ``None`` with no trades.
    """

    index: int
    start: datetime
    end: datetime
    n_trades: int
    expectancy_r: float | None

    def to_dict(self) -> dict[str, Any]:
        """Serialise to JSON-able data."""
        return {
            "index": self.index,
            "start": self.start.isoformat(),
            "end": self.end.isoformat(),
            "n_trades": self.n_trades,
            "expectancy_r": self.expectancy_r,
        }


@dataclass(frozen=True)
class PeriodConsistency:
    """How evenly the result was spread across equal slices of time.

    Attributes:
        periods: The slices, earliest first.
        n_periods: How many.
        n_profitable: How many had positive expectancy.
        n_empty: How many closed no trades at all.
        dispersion: Standard deviation of the scored periods' expectancy, or
            ``None`` with fewer than two. A strategy whose whole result came
            from one period shows up here and in ``n_profitable``, not in any
            aggregate.
    """

    periods: tuple[PeriodResult, ...]
    n_periods: int
    n_profitable: int
    n_empty: int
    dispersion: float | None

    def to_dict(self) -> dict[str, Any]:
        """Serialise to JSON-able data."""
        return {
            "periods": [period.to_dict() for period in self.periods],
            "n_periods": self.n_periods,
            "n_profitable": self.n_profitable,
            "n_empty": self.n_empty,
            "dispersion": self.dispersion,
        }


def period_consistency(
    trades: Sequence[TradeRecord], *, start: datetime, end: datetime, n_periods: int = 4
) -> PeriodConsistency:
    """Split ``[start, end)`` into equal spans and report each one's own result.

    Equal spans by arithmetic from the two ends, never boundaries chosen to
    make the picture look better — that is the whole point of the check, and it
    is why the split takes the run's own coverage rather than a list of dates.

    Args:
        trades: The run's trades. Attributed by ``closed_at``.
        start: Lower bound of the span to split.
        end: Upper bound.
        n_periods: How many equal slices.

    Returns:
        The per-period breakdown.

    Raises:
        ValueError: If ``n_periods`` is below two or the span is not positive.
    """
    if n_periods < 2:
        raise ValueError(f"n_periods must be at least 2, got {n_periods}")
    if start >= end:
        raise ValueError(f"span must be positive, got {start!r}..{end!r}")

    span = end - start
    edges = [start + (span * index) // n_periods for index in range(n_periods)] + [end]
    periods: list[PeriodResult] = []
    for index in range(n_periods):
        low, high = edges[index], edges[index + 1]
        inside = [trade for trade in trades if low <= trade.closed_at < high]
        periods.append(
            PeriodResult(
                index=index,
                start=low,
                end=high,
                n_trades=len(inside),
                expectancy_r=statistics.fmean(t.realized_r for t in inside) if inside else None,
            )
        )
    scored = [p.expectancy_r for p in periods if p.expectancy_r is not None]
    return PeriodConsistency(
        periods=tuple(periods),
        n_periods=n_periods,
        n_profitable=sum(1 for value in scored if value > 0),
        n_empty=sum(1 for p in periods if p.n_trades == 0),
        dispersion=statistics.stdev(scored) if len(scored) > 1 else None,
    )


# ---------------------------------------------------------------------------
# Noise
# ---------------------------------------------------------------------------


def add_price_noise(frame: OHLCVFrame, *, relative_sigma: float, seed: int) -> OHLCVFrame:
    """Shift every bar's four prices by a common lognormal factor.

    All four prices of a bar move together, which keeps ``low <= open, close <=
    high`` true by construction rather than by repair — perturbing the four
    independently would need a re-sort, and a re-sorted bar is a bar whose shape
    the noise invented rather than jittered. What the perturbation does change
    is each bar's level *relative to its neighbours*, which is what every
    indicator, swing level and stop distance in the system is actually computed
    from.

    Args:
        frame: The series to perturb.
        relative_sigma: Standard deviation of the log shift, as a fraction of
            price.
        seed: Fixes the noise.

    Returns:
        The perturbed series, same timestamps and volumes.

    Raises:
        ValueError: If ``relative_sigma`` is negative.
    """
    if relative_sigma < 0:
        raise ValueError(f"relative_sigma must be non-negative, got {relative_sigma}")
    rng = Random(seed)
    table = frame.df
    factors = [math.exp(rng.gauss(0.0, relative_sigma)) for _ in range(table.height)]
    scaled = table.with_columns(
        *[
            (pl.col(column) * pl.Series(f"_noise_{column}", factors)).alias(column)
            for column in ("open", "high", "low", "close")
        ]
    )
    return frame.with_df(scaled)


def run_noise_test(
    base: RunInputs,
    *,
    sigmas: Sequence[float] = (0.0002, 0.0005, 0.001),
    seed: int = 0,
    evaluate: Callable[[RunInputs, str], RobustnessRun] = evaluate_run,
) -> tuple[RobustnessRun, ...]:
    """Re-run under increasing price noise and watch the result decay.

    Graceful decay is the expected shape. A result that is unchanged under
    noise is suspicious in its own right — it suggests the strategy is not
    actually reading the prices being perturbed — and one that collapses at the
    smallest sigma was resting on price coincidences finer than the instrument's
    own tick.

    Args:
        base: The run to perturb.
        sigmas: Log-noise levels, as fractions of price.
        seed: Base seed; each stream and level derives its own stream so that
            adding a level does not renumber another's draws.
        evaluate: How to walk one run.

    Returns:
        One summary per sigma, in the order given.
    """
    runs = []
    for index, sigma in enumerate(sigmas):
        noisy = {
            key: add_price_noise(
                frame, relative_sigma=sigma, seed=seed + 31 * (index + 1) + hash(str(key)) % 1000
            )
            for key, frame in base.streams.items()
        }
        runs.append(evaluate(base.with_streams(noisy), f"noise_{sigma}"))
    return tuple(runs)


# ---------------------------------------------------------------------------
# Synthetic random walk
# ---------------------------------------------------------------------------


def daily_sigma(frame: OHLCVFrame, day_origin: DayOrigin) -> dict[Any, float]:
    """Per-trading-day standard deviation of log close-to-close bar returns.

    Cut on :func:`~trading_system.data.resample.trading_day`, the one day
    boundary this system has — the same one VWAP resets on and the daily loss
    limit measures against — so a synthetic tape's volatility schedule and the
    run's own day accounting cannot disagree.

    Args:
        frame: The real series to measure.
        day_origin: Where the trading day starts.

    Returns:
        Day label to sigma. A day with fewer than two returns is omitted;
        callers fall back to the pooled figure.
    """
    table = frame.df
    closes = table["close"].to_list()
    stamps = table["timestamp"].to_list()
    buckets: dict[Any, list[float]] = {}
    for index in range(1, len(closes)):
        if closes[index - 1] <= 0 or closes[index] <= 0:
            continue
        label = trading_day(stamps[index], day_origin)
        buckets.setdefault(label, []).append(math.log(closes[index] / closes[index - 1]))
    return {label: statistics.stdev(values) for label, values in buckets.items() if len(values) > 1}


def synthetic_frame(
    frame: OHLCVFrame,
    *,
    day_origin: DayOrigin,
    seed: int,
    substeps: int = SYNTHETIC_SUBSTEPS,
) -> OHLCVFrame:
    """A driftless random walk on ``frame``'s own time grid and volatility schedule.

    Bars are *generated*, not borrowed: each bar simulates ``substeps``
    increments of a driftless walk and takes the running maximum and minimum as
    its high and low, so the intrabar range comes from the simulation. Volume is
    carried across unchanged — it is not part of the price null, and a strategy
    with a liquidity filter needs plausible volume or the test would measure the
    filter rather than the tape.

    Args:
        frame: The real series, for its timestamps, starting price and daily
            volatility.
        day_origin: Day boundary the volatility schedule is cut on.
        seed: Fixes the walk.
        substeps: Sub-increments simulated per bar.

    Returns:
        A synthetic series with the same symbol, timeframe and timestamps.

    Raises:
        ValueError: If the frame is empty or ``substeps`` is below one.
    """
    if substeps < 1:
        raise ValueError(f"substeps must be at least 1, got {substeps}")
    table = frame.df
    if table.height == 0:
        raise ValueError("cannot build a synthetic series from an empty frame")

    per_day = daily_sigma(frame, day_origin)
    pooled = statistics.fmean(per_day.values()) if per_day else 0.0
    stamps = table["timestamp"].to_list()
    volumes = table["volume"].to_list()
    closes = table["close"].to_list()

    rng = Random(seed)
    level = closes[0]
    opens = [closes[0]]
    highs = [table["high"].to_list()[0]]
    lows = [table["low"].to_list()[0]]
    new_closes = [closes[0]]

    step_scale = 1.0 / math.sqrt(substeps)
    for index in range(1, len(stamps)):
        sigma = per_day.get(trading_day(stamps[index], day_origin), pooled)
        opening = level
        running = level
        top = level
        bottom = level
        for _ in range(substeps):
            running *= math.exp(rng.gauss(0.0, sigma * step_scale))
            top = max(top, running)
            bottom = min(bottom, running)
        level = running
        opens.append(opening)
        new_closes.append(running)
        highs.append(top)
        lows.append(bottom)

    built = table.with_columns(
        pl.Series("open", opens),
        pl.Series("high", highs),
        pl.Series("low", lows),
        pl.Series("close", new_closes),
        pl.Series("volume", volumes),
    )
    return frame.with_df(built)


def synthetic_streams(
    streams: Mapping[StreamKey, OHLCVFrame], *, day_origin: DayOrigin, seed: int
) -> dict[StreamKey, OHLCVFrame]:
    """Synthesise the finest stream per symbol and re-aggregate the coarser ones.

    The same rule :mod:`trading_system.validation.nulls.permutation` follows: a
    coarser timeframe is never generated independently, it is rebuilt from the
    synthetic finest series with
    :func:`~trading_system.data.resample.resample`, so an H4 stream can never
    disagree with the H1 stream it is supposed to summarise.

    Args:
        streams: The real bars.
        day_origin: Day boundary for the volatility schedule.
        seed: Base seed; each symbol derives its own so that adding a symbol
            does not redraw another's tape.

    Returns:
        The synthetic streams.

    Raises:
        ValueError: If a symbol has a coarse stream but no finest one to
            rebuild it from.
    """
    by_symbol: dict[str, list[StreamKey]] = {}
    for key in streams:
        by_symbol.setdefault(key.symbol, []).append(key)

    built: dict[StreamKey, OHLCVFrame] = {}
    for offset, (_symbol, keys) in enumerate(sorted(by_symbol.items())):
        finest = min(keys, key=lambda key: key.timeframe.duration)
        base_frame = synthetic_frame(
            streams[finest], day_origin=day_origin, seed=seed + 613 * (offset + 1)
        )
        built[finest] = base_frame
        for key in keys:
            if key == finest:
                continue
            built[key] = resample(base_frame, key.timeframe, origin=day_origin)
    return built


def run_synthetic_test(
    base: RunInputs,
    *,
    n_iterations: int = 20,
    seed: int = 0,
    evaluate: Callable[[RunInputs, str], RobustnessRun] = evaluate_run,
) -> tuple[RobustnessRun, ...]:
    """Re-run on ``n_iterations`` driftless random-walk tapes matched on volatility.

    A strategy that earns here has found something in the *shape* of a tape with
    no exploitable structure, which is the definition of an artifact.

    Args:
        base: The run to re-tape.
        n_iterations: How many synthetic tapes.
        seed: Base seed.
        evaluate: How to walk one run.

    Returns:
        One summary per tape.

    Raises:
        ValueError: If ``n_iterations`` is below one.
    """
    if n_iterations < 1:
        raise ValueError(f"n_iterations must be at least 1, got {n_iterations}")
    origin = base.config.day_origin
    runs = []
    for index in range(n_iterations):
        tape = synthetic_streams(base.streams, day_origin=origin, seed=seed + 9_973 * (index + 1))
        runs.append(evaluate(base.with_streams(tape), f"synthetic_{index}"))
    return tuple(runs)


@dataclass(frozen=True)
class SyntheticSummary:
    """What the synthetic tapes produced, against the real run.

    Attributes:
        n_iterations: How many tapes.
        real_expectancy_r: The real run's mean realised R.
        median_expectancy_r: Median across tapes.
        fraction_profitable: Share of tapes with positive expectancy.
        real_percentile: Where the real run falls among the tapes, in
            ``[0, 100]``. This is the figure a verdict reads: a real run that
            is not clearly above its own no-structure null has not shown an
            edge, whatever its absolute return.
        median_trade_count: Median trades per tape — reported because a tape
            that produced almost no trades makes its expectancy meaningless,
            and that must be visible rather than averaged away.
    """

    n_iterations: int
    real_expectancy_r: float | None
    median_expectancy_r: float | None
    fraction_profitable: float
    real_percentile: float | None
    median_trade_count: float

    def to_dict(self) -> dict[str, Any]:
        """Serialise to JSON-able data."""
        return {
            "n_iterations": self.n_iterations,
            "real_expectancy_r": self.real_expectancy_r,
            "median_expectancy_r": self.median_expectancy_r,
            "fraction_profitable": self.fraction_profitable,
            "real_percentile": self.real_percentile,
            "median_trade_count": self.median_trade_count,
        }


def summarise_synthetic(
    runs: Sequence[RobustnessRun], *, real_expectancy_r: float | None
) -> SyntheticSummary:
    """Reduce synthetic tapes plus the real result into a comparable summary.

    Args:
        runs: What each tape produced.
        real_expectancy_r: The real run's mean realised R.

    Returns:
        The summary.

    Raises:
        ValueError: If ``runs`` is empty.
    """
    if not runs:
        raise ValueError("cannot summarise zero synthetic runs")
    scored = [run.expectancy_r for run in runs if run.expectancy_r is not None]
    percentile: float | None = None
    if scored and real_expectancy_r is not None:
        below = sum(1 for value in scored if value < real_expectancy_r)
        equal = sum(1 for value in scored if value == real_expectancy_r)
        percentile = 100.0 * (below + 0.5 * equal) / len(scored)
    return SyntheticSummary(
        n_iterations=len(runs),
        real_expectancy_r=real_expectancy_r,
        median_expectancy_r=statistics.median(scored) if scored else None,
        fraction_profitable=(sum(1 for value in scored if value > 0) / len(scored))
        if scored
        else 0.0,
        real_percentile=percentile,
        median_trade_count=statistics.median([float(run.n_trades) for run in runs]),
    )


@dataclass(frozen=True)
class RobustnessReport:
    """Every perturbation this module runs.

    Attributes:
        start_shift: Results under a shifted first bar.
        noise: Results under increasing price noise.
        synthetic: Summary of the driftless random-walk tapes.
        period: How evenly the real result was spread across equal time slices.
        start_shift_dispersion: Standard deviation of the shifted runs'
            expectancy, or ``None`` with fewer than two scored — the single
            number a verdict reads from ``start_shift``.
        noise_retention: Expectancy at the largest noise level over expectancy
            at the smallest, or ``None`` when either is undefined or the
            denominator is not positive. Undefined rather than clamped: a ratio
            through zero would be a number with no meaning, the same reason
            :mod:`trading_system.validation.report` refuses an IS/OOS ratio.
    """

    start_shift: tuple[RobustnessRun, ...]
    noise: tuple[RobustnessRun, ...]
    synthetic: SyntheticSummary
    period: PeriodConsistency
    start_shift_dispersion: float | None
    noise_retention: float | None

    def to_dict(self) -> dict[str, Any]:
        """Serialise to JSON-able data."""
        return {
            "start_shift": [run.to_dict() for run in self.start_shift],
            "noise": [run.to_dict() for run in self.noise],
            "synthetic": self.synthetic.to_dict(),
            "period": self.period.to_dict(),
            "start_shift_dispersion": self.start_shift_dispersion,
            "noise_retention": self.noise_retention,
        }


def build_robustness_report(
    start_shift: Sequence[RobustnessRun],
    noise: Sequence[RobustnessRun],
    synthetic: SyntheticSummary,
    period: PeriodConsistency,
) -> RobustnessReport:
    """Assemble the report and derive the two scalars a verdict reads.

    Args:
        start_shift: Shifted-start results.
        noise: Noise results, ordered from smallest sigma to largest.
        synthetic: Synthetic-tape summary.
        period: Period breakdown.

    Returns:
        The report.
    """
    shifted = [run.expectancy_r for run in start_shift if run.expectancy_r is not None]
    dispersion = statistics.stdev(shifted) if len(shifted) > 1 else None

    retention: float | None = None
    if len(noise) > 1:
        first, last = noise[0].expectancy_r, noise[-1].expectancy_r
        if first is not None and last is not None and first > 0:
            retention = last / first

    return RobustnessReport(
        start_shift=tuple(start_shift),
        noise=tuple(noise),
        synthetic=synthetic,
        period=period,
        start_shift_dispersion=dispersion,
        noise_retention=retention,
    )


def run_all(
    base: RunInputs,
    *,
    trades: Sequence[TradeRecord],
    coverage: tuple[datetime, datetime],
    shifts: Sequence[int] = (0, 1, 5, 13, 29),
    noise_sigmas: Sequence[float] = (0.0002, 0.0005, 0.001),
    synthetic_iterations: int = 20,
    n_periods: int = 4,
    seed: int = 0,
) -> RobustnessReport:
    """Run every perturbation over one base run.

    Args:
        base: The run to perturb.
        trades: The real run's trades, for the period breakdown.
        coverage: ``(start, end)`` of the span to split into periods.
        shifts: Bar counts for the start shift. Prime-ish and uneven on
            purpose: a shift of exactly one bar per step would mostly re-test
            the same fold anchors.
        noise_sigmas: Noise levels, smallest first.
        synthetic_iterations: How many synthetic tapes.
        n_periods: Equal time slices for the period breakdown.
        seed: Base seed.

    Returns:
        The report.
    """
    real_expectancy = statistics.fmean(trade.realized_r for trade in trades) if trades else None
    return build_robustness_report(
        start_shift=run_start_shift(base, shifts=shifts),
        noise=run_noise_test(base, sigmas=noise_sigmas, seed=seed),
        synthetic=summarise_synthetic(
            run_synthetic_test(base, n_iterations=synthetic_iterations, seed=seed),
            real_expectancy_r=real_expectancy,
        ),
        period=period_consistency(trades, start=coverage[0], end=coverage[1], n_periods=n_periods),
    )


__all__ = [
    "SYNTHETIC_SUBSTEPS",
    "PeriodConsistency",
    "PeriodResult",
    "RobustnessReport",
    "RobustnessRun",
    "SyntheticSummary",
    "add_price_noise",
    "build_robustness_report",
    "daily_sigma",
    "evaluate_run",
    "period_consistency",
    "run_all",
    "run_noise_test",
    "run_start_shift",
    "run_synthetic_test",
    "shift_start",
    "summarise_synthetic",
    "synthetic_frame",
    "synthetic_streams",
]
