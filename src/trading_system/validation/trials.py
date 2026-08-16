"""How many independent tests a screen is really worth.

A screen of eight hundred strategies over eleven markets is sixteen thousand
comparisons, and the best of sixteen thousand is positive whether or not any of
them has an edge. Discounting for that requires a number, and the raw count is
the wrong one: the trials are heavily correlated — variants of one idea from one
source, and markets that move together — so treating them as independent
overstates the penalty and produces a figure nobody can defend. This is the same
refusal P15 stage 2 made about feeding a raw ``n_trials`` into the Deflated
Sharpe Ratio, and this module is the discount that decision deferred.

**The estimate factorises, and the factorisation is an assumption with a
direction.** A sixteen-thousand-square correlation matrix is neither estimable
nor necessary, because the correlation has known structure: two strategies on
one market are related through the ideas they share, and one strategy on two
markets is related through how the markets move. So

    N_eff ≈ S_eff × M_eff

with each factor taken from the eigenvalue spectrum of a matrix small enough to
estimate. This assumes the strategy-correlation is the same on every market — a
Kronecker structure. Where it is not, because a family only works on one market,
the true correlation is higher than assumed and ``N_eff`` comes out too large.
Too large means the discount is too *small*, so a candidate that survives it
would also survive the correct one: the assumption errs towards rejecting rather
than towards admitting.

**A floor from the corpus itself.** The normalised corpus knows how many
distinct ideas it holds — the number of distinct ``(family, indicator set)``
signatures. An estimate of independence larger than the number of distinguishable
ideas is describing noise in the return estimates, not variety in the shelf, so
``S_eff`` is capped there.
"""

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class SpectrumSummary:
    """What an eigenvalue spectrum came to.

    Attributes:
        n: Size of the matrix.
        effective: Participation ratio of the spectrum.
        top_eigenvalues: The largest few, for a reader to see concentration.
        mean_abs_correlation: Mean absolute off-diagonal correlation.
    """

    n: int
    effective: float
    top_eigenvalues: tuple[float, ...]
    mean_abs_correlation: float

    def as_record(self) -> dict[str, object]:
        """The summary as a manifest entry."""
        return {
            "n": self.n,
            "effective": round(self.effective, 3),
            "top_eigenvalues": [round(value, 4) for value in self.top_eigenvalues],
            "mean_abs_correlation": round(self.mean_abs_correlation, 4),
        }


@dataclass(frozen=True)
class EffectiveTrials:
    """The discount a screen is worth, and how it was arrived at.

    Attributes:
        n_trials_raw: Comparisons actually made.
        n_trials_effective: Independent comparisons they are worth.
        strategies: Spectrum of the sampled strategies' returns.
        markets: Spectrum of the instruments' daily returns.
        strategies_effective: ``S_eff`` after the signature floor.
        markets_effective: ``M_eff``.
        n_strategies: Strategies screened.
        n_markets: Markets screened.
        sample_size: Strategies whose returns were used for ``S_eff``.
        signature_floor: Distinct ideas on the shelf, the cap on ``S_eff``.
        method: What was computed, in one line.
    """

    n_trials_raw: int
    n_trials_effective: float
    strategies: SpectrumSummary | None
    markets: SpectrumSummary | None
    strategies_effective: float
    markets_effective: float
    n_strategies: int
    n_markets: int
    sample_size: int
    signature_floor: int | None
    method: str

    def as_record(self) -> dict[str, object]:
        """The estimate as a manifest entry."""
        return {
            "n_trials_raw": self.n_trials_raw,
            "n_trials_effective": round(self.n_trials_effective, 2),
            "strategies_effective": round(self.strategies_effective, 3),
            "markets_effective": round(self.markets_effective, 3),
            "n_strategies": self.n_strategies,
            "n_markets": self.n_markets,
            "sample_size": self.sample_size,
            "signature_floor": self.signature_floor,
            "method": self.method,
            "strategy_spectrum": self.strategies.as_record() if self.strategies else None,
            "market_spectrum": self.markets.as_record() if self.markets else None,
        }


def correlation_matrix(vectors: Sequence[Sequence[float]]) -> list[list[float]]:
    """Pearson correlations between equal-length vectors.

    Args:
        vectors: One series per variable. Shorter series are compared over the
            leading overlap, which is what makes a ragged set of daily returns
            usable at all.

    Returns:
        The matrix. A pair with no variance in either leg correlates 0 — no
        relationship measurable rather than a perfect one.
    """
    size = len(vectors)
    matrix = [[0.0] * size for _ in range(size)]
    for index in range(size):
        matrix[index][index] = 1.0
    for left in range(size):
        for right in range(left + 1, size):
            value = _pearson(vectors[left], vectors[right])
            matrix[left][right] = value
            matrix[right][left] = value
    return matrix


def _pearson(left: Sequence[float], right: Sequence[float]) -> float:
    """Pearson correlation over the leading overlap of two series."""
    size = min(len(left), len(right))
    if size < 3:
        return 0.0
    a = left[:size]
    b = right[:size]
    mean_a = sum(a) / size
    mean_b = sum(b) / size
    cov = sum((x - mean_a) * (y - mean_b) for x, y in zip(a, b, strict=True))
    var_a = sum((x - mean_a) ** 2 for x in a)
    var_b = sum((y - mean_b) ** 2 for y in b)
    if var_a <= 0 or var_b <= 0:
        return 0.0
    return cov / math.sqrt(var_a * var_b)


def _eigenvalues(matrix: Sequence[Sequence[float]]) -> list[float]:
    """Eigenvalues of a symmetric matrix, by the cyclic Jacobi method.

    Written out rather than taken from numpy because numpy is not a dependency
    of this project and a correlation matrix of a few hundred rows is small
    enough that the classical algorithm is fast and exact enough. The matrix is
    symmetric by construction, which is what makes Jacobi applicable.

    Args:
        matrix: A symmetric matrix.

    Returns:
        The eigenvalues, descending.
    """
    size = len(matrix)
    work = [list(row) for row in matrix]
    for _sweep in range(60):
        off = sum(work[i][j] ** 2 for i in range(size) for j in range(size) if i != j)
        if off < 1e-12:
            break
        for p in range(size - 1):
            for q in range(p + 1, size):
                if abs(work[p][q]) < 1e-15:
                    continue
                theta = (work[q][q] - work[p][p]) / (2.0 * work[p][q])
                sign = 1.0 if theta >= 0 else -1.0
                t = sign / (abs(theta) + math.sqrt(theta * theta + 1.0))
                c = 1.0 / math.sqrt(t * t + 1.0)
                s = t * c
                for k in range(size):
                    akp, akq = work[k][p], work[k][q]
                    work[k][p] = c * akp - s * akq
                    work[k][q] = s * akp + c * akq
                for k in range(size):
                    apk, aqk = work[p][k], work[q][k]
                    work[p][k] = c * apk - s * aqk
                    work[q][k] = s * apk + c * aqk
    return sorted((work[i][i] for i in range(size)), reverse=True)


def participation_ratio(matrix: Sequence[Sequence[float]]) -> SpectrumSummary:
    """How many independent directions a correlation matrix really has.

    ``(Σλ)² / Σλ²``: one when every variable is the same variable, ``n`` when
    they are mutually uncorrelated, and between the two in the way one would
    want — a matrix whose variance sits in three big eigenvalues scores about
    three however many rows it has.

    Args:
        matrix: A correlation matrix.

    Returns:
        The summary.

    Raises:
        ValueError: If the matrix is empty.
    """
    size = len(matrix)
    if size == 0:
        raise ValueError("cannot summarise an empty matrix")
    values = _eigenvalues(matrix)
    positive = [value for value in values if value > 1e-12]
    total = sum(positive)
    squared = sum(value * value for value in positive)
    effective = (total * total / squared) if squared > 0 else 1.0
    off = [abs(matrix[i][j]) for i in range(size) for j in range(size) if i != j]
    return SpectrumSummary(
        n=size,
        effective=min(max(effective, 1.0), float(size)),
        top_eigenvalues=tuple(values[:5]),
        mean_abs_correlation=(sum(off) / len(off)) if off else 0.0,
    )


def extrapolate(sample_effective: float, sample_size: int, population: int) -> float:
    """Scale an effective count measured on a sample up to the whole population.

    Linear in the population, because the participation ratio of a block-like
    correlation structure grows about linearly with how many members are drawn
    from the same blocks. This is the crude step of the estimate and is named as
    such; it is bounded below by 1 and above by the population, so it cannot
    produce a nonsense discount in either direction.

    Args:
        sample_effective: Effective count on the sample.
        sample_size: Members of the sample.
        population: Members in total.

    Returns:
        The extrapolated effective count.
    """
    if sample_size <= 0:
        return float(max(population, 1))
    scaled = sample_effective * (population / sample_size)
    return min(max(scaled, 1.0), float(max(population, 1)))


def estimate(
    *,
    strategy_returns: Mapping[str, Sequence[float]],
    market_returns: Mapping[str, Sequence[float]],
    n_strategies: int,
    n_markets: int,
    n_trials_raw: int,
    signature_floor: int | None = None,
) -> EffectiveTrials:
    """Estimate how many independent trials a screen amounts to.

    Args:
        strategy_returns: Daily returns of the sampled strategies, by task key.
        market_returns: Daily returns of each market, by symbol.
        n_strategies: Strategies screened in total.
        n_markets: Markets screened in total.
        n_trials_raw: Comparisons actually made.
        signature_floor: Distinct ideas on the shelf. ``S_eff`` is capped here:
            an independence estimate above the number of distinguishable ideas
            is describing estimation noise, not variety.

    Returns:
        The estimate, carrying both factors and both spectra so a reader can
        disagree with either.
    """
    strategies: SpectrumSummary | None = None
    strategies_effective = float(max(n_strategies, 1))
    sample = [list(values) for values in strategy_returns.values() if len(values) >= 3]
    if len(sample) >= 3:
        strategies = participation_ratio(correlation_matrix(sample))
        strategies_effective = extrapolate(strategies.effective, len(sample), n_strategies)
        strategies_effective = min(strategies_effective, float(max(n_strategies, 1)))

    markets: SpectrumSummary | None = None
    markets_effective = float(max(n_markets, 1))
    market_sample = [list(values) for values in market_returns.values() if len(values) >= 3]
    if len(market_sample) >= 2:
        markets = participation_ratio(correlation_matrix(market_sample))
        # A factor can never exceed the count it is a factor of. Caught on the
        # first real screen: the market matrix had been built from the whole
        # store while `n_markets` counted the markets actually run, and the
        # estimate reported 9.75 effective markets out of 7.
        markets_effective = min(markets.effective, float(max(n_markets, 1)))

    floored = strategies_effective
    if signature_floor is not None and signature_floor > 0:
        floored = min(strategies_effective, float(signature_floor))

    effective = min(floored * markets_effective, float(max(n_trials_raw, 1)))
    return EffectiveTrials(
        n_trials_raw=n_trials_raw,
        n_trials_effective=effective,
        strategies=strategies,
        markets=markets,
        strategies_effective=floored,
        markets_effective=markets_effective,
        n_strategies=n_strategies,
        n_markets=n_markets,
        sample_size=len(sample),
        signature_floor=signature_floor,
        method=(
            "N_eff = S_eff x M_eff; each factor is the participation ratio "
            "(sum(l)^2 / sum(l^2)) of a correlation matrix of daily returns. S_eff is measured "
            "on a sample and scaled linearly to the shelf, then capped at the number of "
            "distinct (family, indicator set) signatures. The factorisation assumes strategy "
            "correlation is the same on every market, which overstates N_eff where it is not "
            "- an error in the direction of a smaller discount, so a survivor of this "
            "correction would survive the correct one."
        ),
    )
