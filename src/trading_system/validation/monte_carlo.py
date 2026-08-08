"""How much of what happened was the order it happened in.

**Everything is measured in R, never in account currency, and the equity path
is rebuilt by compounding.** Sizing is a fraction of *equity*
(:class:`~trading_system.risk.sizing.methods.FixedFractional`), so a trade's
money result depends on when in the run it occurred while its R does not.
Reshuffling booked currency amounts would therefore mix scale with order: a
trade that made 2000 at an equity of 100k lands as the same 2000 when moved to
a point where equity was 80k, and the resulting drawdown distribution measures
the equity path the run happened to take rather than the ordering. Rebuilding
instead as ``E[n+1] = E[n] * (1 + risk_pct * R[n])`` keeps each trade worth the
same *fraction* wherever it lands, which is what the account would actually
have done. This matters more than the choice of resampling unit below and is
easier to get wrong, because currency-space shuffling looks perfectly correct.

**The resampling unit is a block: trades are shuffled WITHIN a fold, and the
fold order is preserved.** A walk-forward's folds do not share parameters — the
optimiser chose a different set for each — and they do not share a market
regime either. Pooling every trade and shuffling freely lets fold 0's trades
land at the end of the run, producing sequences that the system could not have
generated, and the resulting max-drawdown distribution answers a question about
no procedure that exists. Block resampling keeps every trade inside the fold
whose parameters produced it and whose period it belongs to, and randomises
only what is genuinely arbitrary: the order in which one fold's trades arrived.

Two other units are computed and reported, deliberately not used for the
verdict, because each answers a different question and collapsing them into one
number is the confusion this module exists to avoid:

* :class:`PooledPermutation` shuffles across folds. That is not meaningless —
  the artifact that gets deployed is the *walk-forward procedure* ("re-optimise
  every 270 days and trade the result"), so parameter change over time is part
  of the system rather than contamination of the sample. It is reported under
  its own name and kept out of the verdict.
* Per-fold runs (:func:`per_fold_monte_carlo`) satisfy exchangeability
  perfectly but on this system's fold sizes — 13 to 29 trades — a 20-trade
  sequence's maximum drawdown is decided by its worst two or three trades
  whatever the order, so the distribution is close to degenerate. Honest, and
  nearly uninformative; the trade count is reported beside it so a reader can
  see that for themselves rather than being told.

**A permutation cannot change the final equity, only the path — and that is
arithmetic, not a bug.** Compounded equity is ``prod(1 + risk_pct * R[i])``, a
product, and multiplication commutes, so every reordering of the same multiset
ends at exactly the same place. On a real run this shows up as a total-return
"distribution" whose 5th and 95th percentiles both equal the observed value,
which looks broken and is not. The consequence is worth stating rather than
discovering: **permutation isolates the drawdown question and says nothing
about return**, so the confidence interval on return has to come from
:class:`BlockBootstrap` (which changes the multiset) or :class:`BlockDeletion`
(which shrinks it). The two families are not redundant — each answers the
question the other structurally cannot.

**Cost randomisation is not resampling and cannot be.** Changing spread or
slippage changes *which trades exist*: stop distances move, so sizing moves;
fills move, so a resting LIMIT order's TTL expiry moves; a trade that filled may
not fill at all. So it is N complete backtests rather than N draws from a
finished list, which is why its N is in the hundreds while the trade-resampling
families run ten thousand. The perturbation range is not invented here — it
comes from the two sources P12 already established: ``run_seed``, which P12
already makes every fill's jitter depend on, and multiplicative scaling of the
declared :class:`~trading_system.execution.config.SlippageParams` and spread
fields, with :class:`~trading_system.execution.config.SlippageConfig`'s own
``stop >= market >= limit`` validation re-run on every draw so an invalid
combination fails loudly instead of yielding a flattering backtest.
"""

import math
import statistics
from collections.abc import Sequence
from dataclasses import dataclass, replace
from decimal import Decimal
from pathlib import Path
from random import Random
from typing import Any, Protocol

from trading_system.backtest.portfolio import TradeRecord
from trading_system.backtest.reproducibility import read_run
from trading_system.backtest.spec import RunInputs
from trading_system.core.logging import get_logger
from trading_system.execution.config import CostConfig, SlippageParams
from trading_system.validation.walkforward import WalkForwardResult

logger = get_logger(__name__)

#: Equity fraction at or below which an account is treated as ruined. Not zero:
#: a prop account is closed out by its drawdown limit long before the balance
#: reaches nothing, and a ruin probability computed against zero would describe
#: an account nobody is allowed to keep trading.
DEFAULT_RUIN_FRACTION = 0.5


# ---------------------------------------------------------------------------
# The sample
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FoldTrades:
    """One fold's realised R multiples, in the order they closed.

    Attributes:
        fold_index: Which fold produced them.
        r_multiples: ``TradeRecord.realized_r`` per trade, chronologically.
    """

    fold_index: int
    r_multiples: tuple[float, ...]

    def __len__(self) -> int:
        """How many trades the fold closed."""
        return len(self.r_multiples)


@dataclass(frozen=True)
class TradeSample:
    """Every trade of a walk-forward, grouped by the fold that opened it.

    Attributes:
        folds: Per-fold R multiples, in fold order.
        risk_pct: Equity fraction risked per trade, used to turn R back into an
            equity path. The run's own sizing parameter, not a choice made here.
        starting_equity: Where the rebuilt path begins.
    """

    folds: tuple[FoldTrades, ...]
    risk_pct: float
    starting_equity: Decimal = Decimal(100_000)

    def __post_init__(self) -> None:
        """Validate the sizing fraction.

        Raises:
            ValueError: If ``risk_pct`` is not in ``(0, 1]`` — outside it the
                compounding below stops describing any account that could
                exist.
        """
        if not 0 < self.risk_pct <= 1:
            raise ValueError(f"risk_pct must be in (0, 1], got {self.risk_pct}")

    @property
    def n_trades(self) -> int:
        """Trades across every fold."""
        return sum(len(fold) for fold in self.folds)

    @property
    def observed(self) -> tuple[float, ...]:
        """Every R multiple in the order it actually happened."""
        return tuple(r for fold in self.folds for r in fold.r_multiples)

    @property
    def fold_sizes(self) -> tuple[int, ...]:
        """Trades per fold, in fold order."""
        return tuple(len(fold) for fold in self.folds)


def sample_from_walkforward(
    result: WalkForwardResult, *, risk_pct: float, starting_equity: Decimal = Decimal(100_000)
) -> TradeSample:
    """Read every fold's out-of-sample trades off disk into a :class:`TradeSample`.

    Only out-of-sample trades: an in-sample trade is one the parameters were
    chosen on, and including it would let the choice score itself.

    Args:
        result: A finished walk-forward.
        risk_pct: The run's own per-trade equity fraction.
        starting_equity: Where rebuilt paths begin.

    Returns:
        The sample.
    """
    folds = []
    for fold_run in result.folds:
        stored = read_run(fold_run.oos_run.path)
        folds.append(
            FoldTrades(
                fold_index=fold_run.fold.index,
                r_multiples=tuple(trade.realized_r for trade in stored.result.trades),
            )
        )
    return TradeSample(folds=tuple(folds), risk_pct=risk_pct, starting_equity=starting_equity)


def sample_from_trades(
    trades: Sequence[TradeRecord], *, risk_pct: float, starting_equity: Decimal = Decimal(100_000)
) -> TradeSample:
    """A one-fold sample from a flat trade list, for a run that is not a walk-forward."""
    return TradeSample(
        folds=(FoldTrades(fold_index=0, r_multiples=tuple(t.realized_r for t in trades)),),
        risk_pct=risk_pct,
        starting_equity=starting_equity,
    )


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------


def equity_path(r_multiples: Sequence[float], *, risk_pct: float) -> list[float]:
    """Compound a sequence of R multiples into an equity path, starting at ``1.0``.

    Args:
        r_multiples: Results in multiples of the risk taken.
        risk_pct: Equity fraction risked per trade.

    Returns:
        ``len(r_multiples) + 1`` points, the first being ``1.0``. Clamped at
        zero: an account cannot go below nothing, and a negative multiplier
        would make every later step meaningless rather than merely bad.
    """
    path = [1.0]
    equity = 1.0
    for r in r_multiples:
        equity = max(0.0, equity * (1.0 + risk_pct * r))
        path.append(equity)
    return path


def max_drawdown(path: Sequence[float]) -> float:
    """Deepest peak-to-trough fall of a path, as a fraction in ``[0, 1]``."""
    peak = path[0] if path else 1.0
    worst = 0.0
    for value in path:
        peak = max(peak, value)
        if peak > 0:
            worst = max(worst, (peak - value) / peak)
    return worst


# ---------------------------------------------------------------------------
# Resamplers
# ---------------------------------------------------------------------------


class Resampler(Protocol):
    """Draws one alternative ordering (or subset) of a sample's trades."""

    @property
    def name(self) -> str:
        """Identifies the resampler in reports."""
        ...

    def draw(self, sample: TradeSample, rng: Random) -> list[float]:
        """One resampled sequence of R multiples."""
        ...


@dataclass(frozen=True)
class BlockPermutation:
    """Shuffle within each fold, keep the fold order — the verdict's resampler.

    Every trade stays in the fold whose parameters produced it and whose market
    period it belongs to; only the arrival order inside a fold is randomised.
    """

    @property
    def name(self) -> str:
        """``"block_permutation"``."""
        return "block_permutation"

    def draw(self, sample: TradeSample, rng: Random) -> list[float]:
        """One block-shuffled sequence."""
        drawn: list[float] = []
        for fold in sample.folds:
            block = list(fold.r_multiples)
            rng.shuffle(block)
            drawn.extend(block)
        return drawn


@dataclass(frozen=True)
class PooledPermutation:
    """Shuffle across every fold. Reported, never used for the verdict.

    Answers "given the whole procedure's realised outcome distribution, what
    equity paths are consistent with it" — a question about the deployed
    procedure rather than about any one parameter set. See the module docstring.
    """

    @property
    def name(self) -> str:
        """``"pooled_permutation"``."""
        return "pooled_permutation"

    def draw(self, sample: TradeSample, rng: Random) -> list[float]:
        """One fully pooled shuffle."""
        drawn = list(sample.observed)
        rng.shuffle(drawn)
        return drawn


@dataclass(frozen=True)
class BlockBootstrap:
    """Resample each fold with replacement, keeping that fold's own trade count.

    Fold sizes are preserved rather than resampled: letting the sizes vary too
    would mix "which outcomes recurred" with "how much each fold contributed",
    and a fold that happened to draw few trades would move the result for a
    reason unrelated to the outcomes being tested.
    """

    @property
    def name(self) -> str:
        """``"block_bootstrap"``."""
        return "block_bootstrap"

    def draw(self, sample: TradeSample, rng: Random) -> list[float]:
        """One with-replacement draw, fold sizes held fixed."""
        drawn: list[float] = []
        for fold in sample.folds:
            if not fold.r_multiples:
                continue
            drawn.extend(rng.choices(fold.r_multiples, k=len(fold)))
        return drawn


@dataclass(frozen=True)
class BlockDeletion:
    """Drop a fraction of each fold's trades, keeping the rest in order.

    Per fold, not from the pool: deleting 10% of a pooled list can take
    everything from a small fold or nothing from it, so the measured robustness
    would partly record which fold lost the draw.

    Attributes:
        fraction: Share of each fold's trades to remove.
    """

    fraction: float = 0.1

    def __post_init__(self) -> None:
        """Validate the share.

        Raises:
            ValueError: If ``fraction`` is not in ``(0, 1)``.
        """
        if not 0 < self.fraction < 1:
            raise ValueError(f"fraction must be in (0, 1), got {self.fraction}")

    @property
    def name(self) -> str:
        """``"block_deletion_{percent}"``."""
        return f"block_deletion_{int(round(self.fraction * 100))}pct"

    def draw(self, sample: TradeSample, rng: Random) -> list[float]:
        """One draw with a share of each fold removed, order otherwise intact."""
        drawn: list[float] = []
        for fold in sample.folds:
            count = len(fold)
            if count == 0:
                continue
            keep = count - int(math.floor(count * self.fraction))
            kept = sorted(rng.sample(range(count), k=max(1, min(keep, count))))
            drawn.extend(fold.r_multiples[index] for index in kept)
        return drawn


# ---------------------------------------------------------------------------
# Running it
# ---------------------------------------------------------------------------


def _percentile(sorted_values: Sequence[float], q: float) -> float:
    """Linear-interpolated percentile of an already-sorted sequence."""
    if not sorted_values:
        raise ValueError("cannot take a percentile of an empty distribution")
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = q / 100.0 * (len(sorted_values) - 1)
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return sorted_values[low]
    return sorted_values[low] + (sorted_values[high] - sorted_values[low]) * (position - low)


@dataclass(frozen=True)
class Distribution:
    """A simulated distribution, reduced to what a report shows.

    Attributes:
        n: How many draws.
        observed: The value the real, unresampled run produced.
        mean: Mean of the draws.
        median: Median of the draws.
        p05: 5th percentile.
        p25: 25th percentile.
        p75: 75th percentile.
        p95: 95th percentile.
        worst: Minimum draw.
        best: Maximum draw.
        observed_percentile: Where ``observed`` falls in the draws, in
            ``[0, 100]``. For a max-drawdown distribution this is the number
            that matters: an observed drawdown at the 10th percentile means the
            realised path was luckier than nine tenths of its own reorderings,
            and the deployed system should be sized for the rest.
    """

    n: int
    observed: float
    mean: float
    median: float
    p05: float
    p25: float
    p75: float
    p95: float
    worst: float
    best: float
    observed_percentile: float

    def to_dict(self) -> dict[str, Any]:
        """Serialise to JSON-able data."""
        return {
            "n": self.n,
            "observed": self.observed,
            "mean": self.mean,
            "median": self.median,
            "p05": self.p05,
            "p25": self.p25,
            "p75": self.p75,
            "p95": self.p95,
            "worst": self.worst,
            "best": self.best,
            "observed_percentile": self.observed_percentile,
        }


def _summarise(values: Sequence[float], observed: float) -> Distribution:
    """Reduce raw draws plus the observed value into a :class:`Distribution`."""
    ordered = sorted(values)
    below = sum(1 for value in ordered if value < observed)
    equal = sum(1 for value in ordered if value == observed)
    return Distribution(
        n=len(ordered),
        observed=observed,
        mean=statistics.fmean(ordered),
        median=statistics.median(ordered),
        p05=_percentile(ordered, 5),
        p25=_percentile(ordered, 25),
        p75=_percentile(ordered, 75),
        p95=_percentile(ordered, 95),
        worst=ordered[0],
        best=ordered[-1],
        observed_percentile=100.0 * (below + 0.5 * equal) / len(ordered),
    )


@dataclass(frozen=True)
class MonteCarloResult:
    """One resampler's simulated distributions.

    Attributes:
        resampler: :attr:`Resampler.name`.
        n_iterations: How many draws.
        n_trades: Trades in the sample the draws came from.
        fold_sizes: Trades per fold, so a reader can judge whether the
            resampling unit had anything to work with.
        max_drawdown: Distribution of deepest peak-to-trough fall, as a
            fraction.
        total_return: Distribution of final equity multiple minus one.
        final_equity: Distribution of final equity multiple.
    """

    resampler: str
    n_iterations: int
    n_trades: int
    fold_sizes: tuple[int, ...]
    max_drawdown: Distribution
    total_return: Distribution
    final_equity: Distribution

    def to_dict(self) -> dict[str, Any]:
        """Serialise to JSON-able data."""
        return {
            "resampler": self.resampler,
            "n_iterations": self.n_iterations,
            "n_trades": self.n_trades,
            "fold_sizes": list(self.fold_sizes),
            "max_drawdown": self.max_drawdown.to_dict(),
            "total_return": self.total_return.to_dict(),
            "final_equity": self.final_equity.to_dict(),
        }


def run_monte_carlo(
    sample: TradeSample, resampler: Resampler, *, n_iterations: int, seed: int = 0
) -> MonteCarloResult:
    """Draw ``n_iterations`` alternative histories and summarise what they produced.

    Args:
        sample: The trades to resample.
        resampler: How to resample them.
        n_iterations: How many draws.
        seed: Fixes the whole simulation.

    Returns:
        The distributions.

    Raises:
        ValueError: If ``n_iterations`` is below one, or the sample holds no
            trades — a distribution over nothing would be reported as a number
            rather than as the absence it is.
    """
    if n_iterations < 1:
        raise ValueError(f"n_iterations must be at least 1, got {n_iterations}")
    if sample.n_trades == 0:
        raise ValueError("cannot simulate a sample with no trades")

    rng = Random(seed)
    risk = sample.risk_pct
    drawdowns: list[float] = []
    finals: list[float] = []
    for _ in range(n_iterations):
        path = equity_path(resampler.draw(sample, rng), risk_pct=risk)
        drawdowns.append(max_drawdown(path))
        finals.append(path[-1])

    observed_path = equity_path(sample.observed, risk_pct=risk)
    observed_dd = max_drawdown(observed_path)
    observed_final = observed_path[-1]
    return MonteCarloResult(
        resampler=resampler.name,
        n_iterations=n_iterations,
        n_trades=sample.n_trades,
        fold_sizes=sample.fold_sizes,
        max_drawdown=_summarise(drawdowns, observed_dd),
        total_return=_summarise([value - 1.0 for value in finals], observed_final - 1.0),
        final_equity=_summarise(finals, observed_final),
    )


def per_fold_monte_carlo(
    sample: TradeSample, *, n_iterations: int, seed: int = 0
) -> tuple[MonteCarloResult, ...]:
    """One simulation per fold, each over that fold alone.

    Exchangeability holds cleanly inside a fold and this is the only unit for
    which that is true without qualification. On short folds the resulting
    distribution is nearly degenerate — see the module docstring — so every
    result carries its own ``n_trades`` rather than being filtered out here.

    Args:
        sample: The trades.
        n_iterations: Draws per fold.
        seed: Base seed; each fold's own stream is derived from it and the fold
            index, so adding a fold does not renumber the others' draws — the
            same discipline P12 applies to fill seeds.

    Returns:
        One result per fold that closed at least one trade, in fold order.
    """
    results = []
    for fold in sample.folds:
        if not fold.r_multiples:
            continue
        single = TradeSample(
            folds=(fold,), risk_pct=sample.risk_pct, starting_equity=sample.starting_equity
        )
        results.append(
            run_monte_carlo(
                single,
                BlockPermutation(),
                n_iterations=n_iterations,
                seed=seed + 1_000_003 * (fold.fold_index + 1),
            )
        )
    return tuple(results)


# ---------------------------------------------------------------------------
# Risk of ruin
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RuinEstimate:
    """Probability of being wiped out at one risk level.

    Attributes:
        risk_pct: The per-trade equity fraction simulated.
        ruin_fraction: Equity multiple at or below which the account counts as
            ruined.
        probability: Share of draws that touched it at any point — measured on
            the *path*, not on the final equity, because an account that hits
            the limit and recovers has still been closed.
        median_final_equity: Median ending equity multiple at this risk level.
        median_max_drawdown: Median deepest fall at this risk level.
    """

    risk_pct: float
    ruin_fraction: float
    probability: float
    median_final_equity: float
    median_max_drawdown: float

    def to_dict(self) -> dict[str, Any]:
        """Serialise to JSON-able data."""
        return {
            "risk_pct": self.risk_pct,
            "ruin_fraction": self.ruin_fraction,
            "probability": self.probability,
            "median_final_equity": self.median_final_equity,
            "median_max_drawdown": self.median_max_drawdown,
        }


def risk_of_ruin(
    sample: TradeSample,
    *,
    risk_levels: Sequence[float],
    n_iterations: int,
    ruin_fraction: float = DEFAULT_RUIN_FRACTION,
    seed: int = 0,
    resampler: Resampler | None = None,
) -> tuple[RuinEstimate, ...]:
    """Simulate ruin probability across a grid of per-trade risk levels.

    The R multiples are the sample's own; only the fraction of equity put at
    stake per trade varies. That is the honest form of the question — "what
    would this same edge have done at 2% instead of 1%" — and it is also the
    reason the result inherits the sample's central assumption: that future
    trades are drawn from the observed distribution, which is exactly what the
    nulls in :mod:`trading_system.validation.nulls` exist to doubt. A ruin
    probability is a statement conditional on the edge being real.

    Args:
        sample: The trades.
        risk_levels: Per-trade equity fractions to simulate.
        n_iterations: Draws per level.
        ruin_fraction: Equity multiple counting as ruin.
        seed: Base seed; each level derives its own stream.
        resampler: How to resample. Block permutation by default.

    Returns:
        One estimate per level, in the order given.

    Raises:
        ValueError: If any level is outside ``(0, 1]``, ``ruin_fraction`` is
            outside ``(0, 1)``, or the sample is empty.
    """
    if not 0 < ruin_fraction < 1:
        raise ValueError(f"ruin_fraction must be in (0, 1), got {ruin_fraction}")
    if sample.n_trades == 0:
        raise ValueError("cannot simulate ruin for a sample with no trades")
    draw_with = resampler if resampler is not None else BlockPermutation()

    estimates = []
    for index, level in enumerate(risk_levels):
        if not 0 < level <= 1:
            raise ValueError(f"risk level must be in (0, 1], got {level}")
        rng = Random(seed + 7_919 * (index + 1))
        ruined = 0
        finals: list[float] = []
        drawdowns: list[float] = []
        for _ in range(n_iterations):
            path = equity_path(draw_with.draw(sample, rng), risk_pct=level)
            if min(path) <= ruin_fraction:
                ruined += 1
            finals.append(path[-1])
            drawdowns.append(max_drawdown(path))
        estimates.append(
            RuinEstimate(
                risk_pct=level,
                ruin_fraction=ruin_fraction,
                probability=ruined / n_iterations,
                median_final_equity=statistics.median(finals),
                median_max_drawdown=statistics.median(drawdowns),
            )
        )
    return tuple(estimates)


# ---------------------------------------------------------------------------
# Cost randomisation
# ---------------------------------------------------------------------------


def _scaled_slippage(params: SlippageParams, factor: float) -> SlippageParams:
    """One :class:`~trading_system.execution.config.SlippageParams` with its magnitudes scaled."""
    return params.model_copy(
        update={
            "base_points": params.base_points * factor,
            "atr_coefficient": params.atr_coefficient * factor,
            "jitter_points": params.jitter_points * factor,
        }
    )


def perturb_costs(base: CostConfig, *, factor: float, run_seed: int) -> CostConfig:
    """One cost draw: every declared magnitude scaled, and a fresh fill seed.

    The scale is applied uniformly to spread and to all three slippage
    profiles, which preserves
    :class:`~trading_system.execution.config.SlippageConfig`'s
    ``stop >= market >= limit`` ordering by construction rather than by luck —
    a draw that inverted it would be the single most flattering error available
    in this module (breakout strategies would become cheap to run), and P12
    already refuses to construct such a config. Re-seeding is a perturbation
    P12 already models: a fill's jitter is drawn from
    ``(run_seed, symbol, ts, order_id)``, so a new seed is a different, equally
    legitimate realisation of the same declared cost model rather than a
    different model.

    Args:
        base: The run's own cost configuration.
        factor: Multiplier on every declared magnitude.
        run_seed: Seed for this draw's fills.

    Returns:
        The perturbed configuration.

    Raises:
        ValueError: If ``factor`` is negative.
    """
    if factor < 0:
        raise ValueError(f"factor must be non-negative, got {factor}")
    spread = base.spread.model_copy(
        update={"volatility_beta": base.spread.volatility_beta * factor}
    )
    slippage = base.slippage.model_copy(
        update={
            "market": _scaled_slippage(base.slippage.market, factor),
            "stop": _scaled_slippage(base.slippage.stop, factor),
            "limit": _scaled_slippage(base.slippage.limit, factor),
        }
    )
    gap = base.gap.model_copy(
        update={
            "penalty_base_points": base.gap.penalty_base_points * factor,
            "penalty_fraction": base.gap.penalty_fraction * factor,
        }
    )
    return base.model_copy(
        update={"spread": spread, "slippage": slippage, "gap": gap, "run_seed": run_seed}
    )


def cost_draws(
    base: CostConfig, *, n: int, low: float = 0.5, high: float = 2.0, seed: int = 0
) -> list[CostConfig]:
    """``n`` cost configurations spanning a multiplicative band around the declared one.

    Args:
        base: The run's own cost configuration.
        n: How many draws.
        low: Smallest multiplier.
        high: Largest multiplier.
        seed: Fixes the draws.

    Returns:
        The configurations, deterministic in ``seed``.

    Raises:
        ValueError: If ``n`` is below one or the band is inverted or negative.
    """
    if n < 1:
        raise ValueError(f"n must be at least 1, got {n}")
    if low < 0 or high < low:
        raise ValueError(f"cost band must satisfy 0 <= low <= high, got low={low}, high={high}")
    rng = Random(seed)
    return [
        perturb_costs(base, factor=rng.uniform(low, high), run_seed=seed + 104_729 * (index + 1))
        for index in range(n)
    ]


@dataclass(frozen=True)
class CostSensitivity:
    """What varying the cost model did to the result.

    Attributes:
        n_draws: How many complete backtests were walked.
        band: ``(low, high)`` multiplier applied to declared cost magnitudes.
        baseline_expectancy_r: The unperturbed run's mean realised R.
        expectancy_r: Distribution of mean realised R across draws.
        trade_count: Distribution of how many trades each draw produced —
            reported because costs change *which* trades exist, not only what
            they earned, and a draw that traded half as often is a different
            statement than one that traded the same and earned less.
        fraction_profitable: Share of draws with positive mean R.
    """

    n_draws: int
    band: tuple[float, float]
    baseline_expectancy_r: float
    expectancy_r: Distribution
    trade_count: Distribution
    fraction_profitable: float

    def to_dict(self) -> dict[str, Any]:
        """Serialise to JSON-able data."""
        return {
            "n_draws": self.n_draws,
            "band": list(self.band),
            "baseline_expectancy_r": self.baseline_expectancy_r,
            "expectancy_r": self.expectancy_r.to_dict(),
            "trade_count": self.trade_count.to_dict(),
            "fraction_profitable": self.fraction_profitable,
        }


def run_cost_sensitivity(
    base: RunInputs,
    *,
    n_draws: int = 100,
    low: float = 0.5,
    high: float = 2.0,
    seed: int = 0,
    store_root: Path | None = None,
    workers: int | None = None,
) -> CostSensitivity:
    """Walk ``n_draws`` complete backtests under perturbed cost models.

    Complete backtests, not resamples: see the module docstring on why cost
    perturbation cannot be done over a finished trade list.

    Args:
        base: The run to perturb.
        n_draws: How many cost draws.
        low: Smallest cost multiplier.
        high: Largest cost multiplier.
        seed: Fixes the draws.
        store_root: Where to store the draws' runs. Kept in memory when
            ``None``, which is what a sensitivity sweep normally wants — the
            individual runs are not the deliverable, the spread is.
        workers: Worker count when storing; ignored in the in-memory path.

    Returns:
        The sensitivity.

    Raises:
        ValueError: If the unperturbed run produced no trades, leaving nothing
            to be sensitive about.
    """
    from trading_system.backtest.parallel import run_parallel_to_store

    baseline = base.run()
    if not baseline.trades:
        raise ValueError("the unperturbed run closed no trades; cost sensitivity is undefined")
    baseline_expectancy = statistics.fmean(trade.realized_r for trade in baseline.trades)

    specs = [
        replace(base, costs=costs)
        for costs in cost_draws(base.costs, n=n_draws, low=low, high=high, seed=seed)
    ]
    expectancies: list[float] = []
    counts: list[float] = []
    if store_root is not None:
        for stored in run_parallel_to_store(specs, store_root, workers=workers):
            result = read_run(stored.path).result
            counts.append(float(len(result.trades)))
            expectancies.append(
                statistics.fmean(t.realized_r for t in result.trades) if result.trades else 0.0
            )
    else:
        for spec in specs:
            result = spec.run()
            counts.append(float(len(result.trades)))
            expectancies.append(
                statistics.fmean(t.realized_r for t in result.trades) if result.trades else 0.0
            )

    return CostSensitivity(
        n_draws=n_draws,
        band=(low, high),
        baseline_expectancy_r=baseline_expectancy,
        expectancy_r=_summarise(expectancies, baseline_expectancy),
        trade_count=_summarise(counts, float(len(baseline.trades))),
        fraction_profitable=sum(1 for value in expectancies if value > 0) / len(expectancies),
    )


@dataclass(frozen=True)
class MonteCarloReport:
    """Every simulation this module runs, side by side.

    Attributes:
        verdict_basis: The block-permutation result — the one a verdict reads.
        pooled: The fully pooled result, reported and deliberately unused for
            the verdict.
        bootstrap: With-replacement result, block unit.
        deletion: Trade-deletion result, block unit.
        per_fold: One result per fold, each over that fold alone.
        ruin: Ruin probability across a grid of risk levels.
        cost_sensitivity: What varying the cost model did, or ``None`` when
            that sweep was not run — it costs complete backtests, so it is
            optional in a way the resampling families are not.
    """

    verdict_basis: MonteCarloResult
    pooled: MonteCarloResult
    bootstrap: MonteCarloResult
    deletion: MonteCarloResult
    per_fold: tuple[MonteCarloResult, ...]
    ruin: tuple[RuinEstimate, ...]
    cost_sensitivity: CostSensitivity | None

    def to_dict(self) -> dict[str, Any]:
        """Serialise to JSON-able data."""
        return {
            "verdict_basis": self.verdict_basis.to_dict(),
            "pooled": self.pooled.to_dict(),
            "bootstrap": self.bootstrap.to_dict(),
            "deletion": self.deletion.to_dict(),
            "per_fold": [result.to_dict() for result in self.per_fold],
            "ruin": [estimate.to_dict() for estimate in self.ruin],
            "cost_sensitivity": None
            if self.cost_sensitivity is None
            else self.cost_sensitivity.to_dict(),
        }


#: Per-trade risk levels the ruin grid is evaluated over by default. Spans the
#: range a prop account is plausibly run at, so that the answer to "what if we
#: sized up" is in the report rather than left to be guessed at.
DEFAULT_RISK_LEVELS: tuple[float, ...] = (0.0025, 0.005, 0.01, 0.02, 0.03, 0.05)


def run_all(
    sample: TradeSample,
    *,
    n_iterations: int = 10_000,
    seed: int = 0,
    risk_levels: Sequence[float] = DEFAULT_RISK_LEVELS,
    ruin_iterations: int = 2_000,
    cost_sensitivity: CostSensitivity | None = None,
) -> MonteCarloReport:
    """Run every resampling family plus the ruin grid over one sample.

    Args:
        sample: The trades.
        n_iterations: Draws per resampling family.
        seed: Base seed; each family derives its own stream so that adding one
            does not renumber another's draws.
        risk_levels: Per-trade risk fractions for the ruin grid.
        ruin_iterations: Draws per risk level. Lower than ``n_iterations`` by
            default because the grid multiplies the cost by its own length.
        cost_sensitivity: Result of :func:`run_cost_sensitivity`, if it was
            run. Passed in rather than run here: it needs a
            :class:`~trading_system.backtest.spec.RunInputs`, which a trade
            sample does not carry.

    Returns:
        The full report.
    """
    return MonteCarloReport(
        verdict_basis=run_monte_carlo(
            sample, BlockPermutation(), n_iterations=n_iterations, seed=seed
        ),
        pooled=run_monte_carlo(
            sample, PooledPermutation(), n_iterations=n_iterations, seed=seed + 11
        ),
        bootstrap=run_monte_carlo(
            sample, BlockBootstrap(), n_iterations=n_iterations, seed=seed + 22
        ),
        deletion=run_monte_carlo(
            sample, BlockDeletion(0.1), n_iterations=n_iterations, seed=seed + 33
        ),
        per_fold=per_fold_monte_carlo(sample, n_iterations=n_iterations, seed=seed + 44),
        ruin=risk_of_ruin(
            sample, risk_levels=risk_levels, n_iterations=ruin_iterations, seed=seed + 55
        ),
        cost_sensitivity=cost_sensitivity,
    )


__all__ = [
    "DEFAULT_RISK_LEVELS",
    "DEFAULT_RUIN_FRACTION",
    "BlockBootstrap",
    "BlockDeletion",
    "BlockPermutation",
    "CostSensitivity",
    "Distribution",
    "FoldTrades",
    "MonteCarloReport",
    "MonteCarloResult",
    "PooledPermutation",
    "Resampler",
    "RuinEstimate",
    "TradeSample",
    "cost_draws",
    "equity_path",
    "max_drawdown",
    "per_fold_monte_carlo",
    "perturb_costs",
    "risk_of_ruin",
    "run_all",
    "run_cost_sensitivity",
    "run_monte_carlo",
    "sample_from_trades",
    "sample_from_walkforward",
]
