"""Resampling a finished run: the right unit, the right space, and no invented variance."""

import statistics
import time
from decimal import Decimal

import pytest

from trading_system.validation.monte_carlo import (
    BlockBootstrap,
    BlockDeletion,
    BlockPermutation,
    FoldTrades,
    PooledPermutation,
    TradeSample,
    equity_path,
    max_drawdown,
    per_fold_monte_carlo,
    risk_of_ruin,
    run_monte_carlo,
)


def _sample(*folds: tuple[float, ...], risk_pct: float = 0.01) -> TradeSample:
    """A sample from explicit per-fold R multiples."""
    return TradeSample(
        folds=tuple(FoldTrades(fold_index=index, r_multiples=rs) for index, rs in enumerate(folds)),
        risk_pct=risk_pct,
        starting_equity=Decimal(100_000),
    )


class TestEquityIsRebuiltByCompounding:
    """The R-space decision: a trade is worth the same fraction wherever it lands."""

    def test_a_path_compounds_rather_than_adding(self) -> None:
        path = equity_path([1.0, 1.0], risk_pct=0.1)
        assert path == pytest.approx([1.0, 1.1, 1.21])

    def test_a_losing_run_cannot_take_equity_below_zero(self) -> None:
        path = equity_path([-5.0, -5.0], risk_pct=0.5)
        assert min(path) >= 0.0

    def test_max_drawdown_is_the_deepest_peak_to_trough_fall(self) -> None:
        assert max_drawdown([1.0, 1.5, 0.75, 1.2]) == pytest.approx(0.5)
        assert max_drawdown([1.0, 1.1, 1.2]) == pytest.approx(0.0)


class TestPermutationCannotChangeFinalEquity:
    """Arithmetic, not a bug — and the reason bootstrap exists alongside it.

    Compounded equity is a product and multiplication commutes, so every
    reordering of one multiset ends at the same place. A reader seeing a
    zero-width return "distribution" must be able to find this pinned down
    rather than conclude the simulation is broken.
    """

    def test_every_reordering_ends_at_the_same_equity(self) -> None:
        sample = _sample((1.0, -1.0, 2.5), (-0.5, 3.0))
        result = run_monte_carlo(sample, BlockPermutation(), n_iterations=200, seed=1)
        assert result.final_equity.worst == pytest.approx(result.final_equity.best)
        assert result.final_equity.observed == pytest.approx(result.final_equity.best)

    def test_but_the_drawdown_distribution_is_genuinely_wide(self) -> None:
        sample = _sample((3.0, -1.0, -1.0, -1.0, 3.0, -1.0, -1.0, -1.0))
        result = run_monte_carlo(sample, BlockPermutation(), n_iterations=500, seed=1)
        assert result.max_drawdown.p95 > result.max_drawdown.p05

    def test_bootstrap_does_move_the_final_equity(self) -> None:
        sample = _sample((3.0, -1.0, -1.0, -1.0, 3.0, -1.0, -1.0, -1.0))
        result = run_monte_carlo(sample, BlockBootstrap(), n_iterations=500, seed=1)
        assert result.final_equity.p95 > result.final_equity.p05, (
            "the confidence interval on return has to come from a resampler that "
            "changes the multiset; permutation structurally cannot provide one"
        )


class TestTheBlockUnitKeepsTradesInTheirOwnFold:
    def test_block_permutation_never_moves_a_trade_between_folds(self) -> None:
        sample = _sample((1.0, 2.0, 3.0), (-1.0, -2.0))
        rng = __import__("random").Random(0)
        for _ in range(50):
            drawn = BlockPermutation().draw(sample, rng)
            assert sorted(drawn[:3]) == [1.0, 2.0, 3.0]
            assert sorted(drawn[3:]) == [-2.0, -1.0]

    def test_pooled_permutation_does_move_them(self) -> None:
        sample = _sample((1.0, 2.0, 3.0), (-1.0, -2.0))
        rng = __import__("random").Random(0)
        crossed = False
        for _ in range(50):
            drawn = PooledPermutation().draw(sample, rng)
            if sorted(drawn[:3]) != [1.0, 2.0, 3.0]:
                crossed = True
                break
        assert crossed, "the pooled resampler must actually cross fold boundaries"

    def test_the_two_units_disagree_on_a_sample_built_to_separate_them(self) -> None:
        # One fold of wins followed by one fold of losses. Block permutation can
        # never interleave them, so the drawdown is always the full losing tail;
        # pooling can spread the losses through the wins and soften it.
        sample = _sample((1.0,) * 10, (-1.0,) * 10, risk_pct=0.05)
        block = run_monte_carlo(sample, BlockPermutation(), n_iterations=400, seed=2)
        pooled = run_monte_carlo(sample, PooledPermutation(), n_iterations=400, seed=2)
        assert block.max_drawdown.median > pooled.max_drawdown.median

    def test_bootstrap_preserves_each_fold_s_trade_count(self) -> None:
        sample = _sample((1.0, 2.0, 3.0), (-1.0, -2.0))
        rng = __import__("random").Random(0)
        for _ in range(20):
            assert len(BlockBootstrap().draw(sample, rng)) == 5

    def test_deletion_removes_a_share_of_every_fold_not_of_the_pool(self) -> None:
        sample = _sample(tuple(float(i) for i in range(10)), tuple(float(i) for i in range(10)))
        rng = __import__("random").Random(0)
        drawn = BlockDeletion(0.1).draw(sample, rng)
        assert len(drawn) == 18, "nine kept from each fold, not eighteen from the pool"

    def test_deletion_never_empties_a_small_fold(self) -> None:
        sample = _sample((1.0,), (2.0, 3.0))
        rng = __import__("random").Random(0)
        for _ in range(20):
            assert len(BlockDeletion(0.9).draw(sample, rng)) >= 2


class TestSimulationIsDeterministic:
    def test_the_same_seed_gives_the_same_distribution(self) -> None:
        sample = _sample((1.0, -1.0, 2.0), (-0.5, 1.5, -1.0))
        first = run_monte_carlo(sample, BlockPermutation(), n_iterations=300, seed=7)
        second = run_monte_carlo(sample, BlockPermutation(), n_iterations=300, seed=7)
        assert first.to_dict() == second.to_dict()

    def test_a_different_seed_gives_a_different_one(self) -> None:
        # Deliberately more than a handful of trades: six trades admit only 36
        # block orderings, so two seeds can land on the same median by exhausting
        # the space rather than by agreeing about anything.
        rng = __import__("random").Random(11)
        folds = tuple(tuple(rng.gauss(0.0, 1.0) for _ in range(20)) for _ in range(3))
        sample = _sample(*folds, risk_pct=0.02)
        first = run_monte_carlo(sample, BlockPermutation(), n_iterations=300, seed=7)
        second = run_monte_carlo(sample, BlockPermutation(), n_iterations=300, seed=8)
        assert first.max_drawdown.median != second.max_drawdown.median

    def test_adding_a_fold_does_not_renumber_an_earlier_fold_s_draws(self) -> None:
        one = _sample((1.0, -1.0, 2.0), (-0.5, 1.5, -1.0))
        two = _sample((1.0, -1.0, 2.0), (-0.5, 1.5, -1.0), (2.0, -2.0, 1.0))
        first = per_fold_monte_carlo(one, n_iterations=200, seed=3)
        second = per_fold_monte_carlo(two, n_iterations=200, seed=3)
        assert first[0].to_dict() == second[0].to_dict()
        assert first[1].to_dict() == second[1].to_dict()


class TestRiskOfRuin:
    def test_ruin_probability_rises_with_the_risk_taken(self) -> None:
        sample = _sample((1.0, -1.0, -1.0, 1.0, -1.0, 1.0, -1.0, -1.0))
        estimates = risk_of_ruin(sample, risk_levels=(0.005, 0.05, 0.25), n_iterations=500, seed=0)
        probabilities = [estimate.probability for estimate in estimates]
        assert probabilities == sorted(probabilities)
        assert probabilities[0] < probabilities[-1]

    def test_ruin_is_measured_on_the_path_not_the_ending(self) -> None:
        # Dips to 0.4, then recovers above the limit: an account that touched it
        # and came back has still been closed. One trade per fold, because block
        # permutation would otherwise reorder them and the loss would land second
        # half the time — which is the resampler working, not the check failing.
        sample = _sample((-3.0,), (4.286,))
        estimates = risk_of_ruin(
            sample, risk_levels=(0.2,), n_iterations=50, ruin_fraction=0.5, seed=0
        )
        assert estimates[0].probability == 1.0

    def test_an_impossible_level_is_refused(self) -> None:
        sample = _sample((1.0, -1.0))
        with pytest.raises(ValueError, match="risk level must be in"):
            risk_of_ruin(sample, risk_levels=(1.5,), n_iterations=10)


class TestEmptyAndDegenerateSamples:
    def test_a_sample_with_no_trades_is_refused_rather_than_summarised(self) -> None:
        with pytest.raises(ValueError, match="no trades"):
            run_monte_carlo(_sample(()), BlockPermutation(), n_iterations=10)

    def test_an_impossible_risk_fraction_is_refused_at_construction(self) -> None:
        with pytest.raises(ValueError, match="risk_pct must be in"):
            _sample((1.0,), risk_pct=0.0)

    def test_per_fold_skips_folds_that_closed_nothing(self) -> None:
        sample = _sample((1.0, -1.0), (), (2.0, -2.0))
        results = per_fold_monte_carlo(sample, n_iterations=50, seed=0)
        assert len(results) == 2


class TestPerformanceBudget:
    """The DoD budget: ten thousand iterations of a realistic sample well under 30s."""

    def test_ten_thousand_iterations_of_a_realistic_sample_are_fast(self) -> None:
        rng = __import__("random").Random(0)
        folds = tuple(
            tuple(rng.gauss(-0.05, 1.2) for _ in range(size))
            for size in (11, 26, 6, 9, 9, 3, 45, 38)
        )
        sample = _sample(*folds)
        started = time.perf_counter()
        result = run_monte_carlo(sample, BlockPermutation(), n_iterations=10_000, seed=0)
        elapsed = time.perf_counter() - started
        assert result.n_iterations == 10_000
        assert elapsed < 30.0, f"took {elapsed:.1f}s, budget is 30s"


class TestObservedPercentileIsTheNumberAVerdictReads:
    def test_a_luckier_than_typical_path_lands_low_in_its_own_distribution(self) -> None:
        # Losses first, then wins: the realised path takes its whole drawdown at
        # the start, which is the worst ordering, so it should sit HIGH.
        sample = _sample((-1.0,) * 5 + (1.0,) * 5, risk_pct=0.05)
        result = run_monte_carlo(sample, BlockPermutation(), n_iterations=1000, seed=0)
        assert result.max_drawdown.observed_percentile > 80

    def test_and_the_reverse_ordering_lands_high(self) -> None:
        sample = _sample((1.0,) * 5 + (-1.0,) * 5, risk_pct=0.05)
        result = run_monte_carlo(sample, BlockPermutation(), n_iterations=1000, seed=0)
        assert result.max_drawdown.observed_percentile > 50


class TestDistributionSummary:
    def test_percentiles_are_ordered(self) -> None:
        sample = _sample(tuple(statistics.NormalDist(0, 1).inv_cdf(p / 21) for p in range(1, 21)))
        result = run_monte_carlo(sample, BlockBootstrap(), n_iterations=500, seed=0)
        dd = result.max_drawdown
        assert dd.worst <= dd.p05 <= dd.p25 <= dd.median <= dd.p75 <= dd.p95 <= dd.best
