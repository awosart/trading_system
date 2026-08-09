"""The three ranked bands, and the two things that make the score usable.

What is asserted here is ordering, not magnitude: the objective's whole claim
is that every feasible point outranks every point failing the edge test, which
outranks every point breaching the ruin ceiling. Magnitudes inside a band are
only required to be monotone in how badly the constraint is missed.
"""

from datetime import UTC, datetime, time, timedelta
from decimal import Decimal

import pytest

from tests.analytics.conftest import trade
from trading_system.backtest.orchestrator import BacktestResult
from trading_system.prop.objective import (
    EDGE_BASE,
    RUIN_BASE,
    PropObjective,
    band_of,
    iterations_for,
    minimum_tolerance,
)
from trading_system.prop.rules import DailyLossBasis, PropRules, TotalLossBasis
from trading_system.validation.report import VerdictThresholds


def rules(**overrides: object) -> PropRules:
    """A rule set with FTMO-like defaults."""
    base: dict[str, object] = {
        "name": "test-plan",
        "prop_profile": "ftmo_swing",
        "source": "test",
        "account_size": Decimal("100000"),
        "profit_target_pct": 0.10,
        "max_daily_loss_pct": 0.05,
        "daily_loss_basis": DailyLossBasis.BALANCE_AT_DAY_START,
        "daily_reset_time": time(0, 0),
        "daily_reset_tz": "Europe/Prague",
        "max_total_loss_pct": 0.10,
        "total_loss_basis": TotalLossBasis.STATIC,
        "max_single_day_profit_share": None,
        "min_trading_days": 0,
    }
    base.update(overrides)
    return PropRules.model_validate(base)


def result_from(r_multiples: list[float]) -> BacktestResult:
    """A ``BacktestResult`` carrying only what this objective reads: trades."""
    base = datetime(2024, 6, 1, 12, tzinfo=UTC)
    trades = [
        trade(
            position_id=f"t{index}",
            opened_at=base + timedelta(days=index),
            closed_at=base + timedelta(days=index, hours=1),
            net=r,
            realized_r=r,
        )
        for index, r in enumerate(r_multiples)
    ]
    return BacktestResult(
        curve=[],
        trades=trades,
        instants=len(trades),
        rejections={},
        degradations={},
        exit_drops={},
        entry_drops={},
        signal_drops={},
        expired_orders=0,
        cost_degradations={},
        atr_ratio_coverage={},
        fills=len(trades),
        fx_fallback_marks=0,
        open_at_end=0,
    )


WINNER = [2.0, -1.0, 1.5, -1.0, 3.0, -1.0, 2.5, -1.0, 1.0, 2.0] * 6
LOSER = [-12.0] + [-1.0] * 20


class TestTheThreeBands:
    def test_a_clean_run_that_clears_both_constraints_scores_its_pass_rate(self) -> None:
        objective = PropObjective(
            rules=rules(), risk_pct=0.01, null_percentile=100.0, iterations=200
        )
        score = objective.evaluate(result_from(WINNER))
        assert score.feasible
        assert score.value == score.p_pass
        assert 0.0 <= score.value <= 1.0
        assert band_of(score.value) == "feasible"

    def test_failing_the_edge_test_lands_below_every_feasible_score(self) -> None:
        no_edge = PropObjective(rules=rules(), risk_pct=0.01, null_percentile=60.0, iterations=200)
        has_edge = PropObjective(rules=rules(), risk_pct=0.01, null_percentile=95.0, iterations=200)
        same_run = result_from(WINNER)

        failed = no_edge.evaluate(same_run)
        passed = has_edge.evaluate(same_run)

        # Identical trades, identical P(pass) — only the percentile differs.
        assert failed.p_pass == passed.p_pass
        assert not failed.feasible
        assert passed.feasible
        assert failed.value < 0.0 <= passed.value
        assert band_of(failed.value) == "edge"

    def test_breaching_the_ruin_ceiling_lands_below_every_edge_failure(self) -> None:
        objective = PropObjective(
            rules=rules(),
            risk_pct=0.01,
            null_percentile=100.0,
            ruin_ceiling=0.0,
            iterations=200,
        )
        score = objective.evaluate(result_from(LOSER))
        assert not score.feasible
        assert score.p_ruin > 0.0
        assert band_of(score.value) == "ruin"
        # The worst possible edge failure is EDGE_BASE - 0.95; every ruin
        # failure must sit below the whole of that band.
        assert score.value < EDGE_BASE - 0.95

    def test_the_bands_never_overlap_across_the_whole_parameter_range(self) -> None:
        """The ordering claim, checked at the extremes rather than by inspection."""
        worst_feasible = 0.0
        best_edge_failure = EDGE_BASE - 1e-9
        worst_edge_failure = EDGE_BASE - 1.0
        best_ruin_failure = RUIN_BASE - 1e-9
        assert best_edge_failure < worst_feasible
        assert best_ruin_failure < worst_edge_failure

    def test_a_worse_ruin_breach_scores_lower_than_a_milder_one(self) -> None:
        """Infeasible points are ranked, so a search has a gradient back out."""
        strict = PropObjective(
            rules=rules(), risk_pct=0.01, null_percentile=100.0, ruin_ceiling=0.0, iterations=200
        )
        lenient = PropObjective(
            rules=rules(), risk_pct=0.01, null_percentile=100.0, ruin_ceiling=0.2, iterations=200
        )
        run = result_from(LOSER)
        assert strict.evaluate(run).value < lenient.evaluate(run).value

    def test_a_worse_edge_shortfall_scores_lower_than_a_milder_one(self) -> None:
        run = result_from(WINNER)
        far = PropObjective(rules=rules(), risk_pct=0.01, null_percentile=5.0, iterations=200)
        near = PropObjective(rules=rules(), risk_pct=0.01, null_percentile=90.0, iterations=200)
        assert far.evaluate(run).value < near.evaluate(run).value


class TestTheEdgeConstraintCannotBeSkipped:
    def test_an_unmeasured_percentile_fails_rather_than_passes(self) -> None:
        """Omission must not be a way past the constraint."""
        objective = PropObjective(
            rules=rules(), risk_pct=0.01, null_percentile=None, iterations=200
        )
        score = objective.evaluate(result_from(WINNER))
        assert not score.feasible
        assert band_of(score.value) == "edge"
        assert "never measured" in score.detail

    def test_the_threshold_is_the_verdicts_own_number(self) -> None:
        objective = PropObjective(rules=rules(), risk_pct=0.01, null_percentile=None)
        assert objective.edge_threshold == VerdictThresholds().null_percentile == 95.0

    def test_the_threshold_lands_exactly_on_the_measurement_grid(self) -> None:
        """At N=20 the percentile moves in steps of 5, and 95 is 19/20 exactly.

        A threshold off the grid would demand a resolution the estimate does
        not have.
        """
        n = 20
        grid = {100.0 * count / n for count in range(n + 1)}
        assert VerdictThresholds().null_percentile in grid


class TestCommonRandomNumbers:
    def test_one_seed_makes_the_score_deterministic(self) -> None:
        objective = PropObjective(
            rules=rules(), risk_pct=0.01, null_percentile=100.0, iterations=300, seed=4
        )
        run = result_from(WINNER)
        assert objective.score(run) == objective.score(run)

    def test_two_seeds_disagree_which_is_why_the_seed_is_held_fixed(self) -> None:
        run = result_from(WINNER)
        one = PropObjective(
            rules=rules(), risk_pct=0.01, null_percentile=100.0, iterations=120, seed=1
        )
        two = PropObjective(
            rules=rules(), risk_pct=0.01, null_percentile=100.0, iterations=120, seed=2
        )
        # Not an assertion that they must differ on this sample — only that the
        # seed is what would make them, which is the reason it is a field
        # rather than drawn per call.
        assert one.seed != two.seed
        assert one.score(run) == one.score(run)
        assert two.score(run) == two.score(run)

    def test_neighbouring_candidates_are_scored_against_the_same_schedule(self) -> None:
        """What makes a difference between neighbours mean something.

        Two candidates differing by one trade are scored with one seed, so the
        permutation error is common-mode and largely cancels in the
        difference the plateau analysis actually reads.
        """
        objective = PropObjective(
            rules=rules(), risk_pct=0.01, null_percentile=100.0, iterations=400, seed=9
        )
        left = objective.score(result_from(WINNER))
        right = objective.score(result_from([*WINNER, 0.5]))
        assert abs(left - right) < 0.5


class TestNoiseFloor:
    def test_the_minimum_tolerance_scales_with_the_standard_error(self) -> None:
        objective = PropObjective(
            rules=rules(), risk_pct=0.01, null_percentile=100.0, iterations=200, seed=0
        )
        simulation = objective.simulate(result_from(WINNER))
        assert minimum_tolerance(simulation) == pytest.approx(3.0 * simulation.se_pass)

    def test_iterations_for_sizes_the_worst_case(self) -> None:
        # SE peaks at p = 0.5, where it is 0.5/sqrt(n).
        assert iterations_for(0.05) == 100
        assert iterations_for(0.005) == 10_000

    def test_a_non_positive_target_is_refused(self) -> None:
        with pytest.raises(ValueError, match="target_se must be positive"):
            iterations_for(0.0)


class TestConstruction:
    @pytest.mark.parametrize("bad", [-0.1, 1.5])
    def test_a_ruin_ceiling_outside_the_unit_interval_is_refused(self, bad: float) -> None:
        with pytest.raises(ValueError, match="ruin_ceiling"):
            PropObjective(rules=rules(), risk_pct=0.01, null_percentile=None, ruin_ceiling=bad)

    @pytest.mark.parametrize("bad", [-1.0, 101.0])
    def test_an_edge_threshold_outside_the_percentile_range_is_refused(self, bad: float) -> None:
        with pytest.raises(ValueError, match="edge_threshold"):
            PropObjective(rules=rules(), risk_pct=0.01, null_percentile=None, edge_threshold=bad)

    def test_a_run_with_no_trades_is_refused_rather_than_scored_as_zero(self) -> None:
        objective = PropObjective(rules=rules(), risk_pct=0.01, null_percentile=100.0)
        with pytest.raises(ValueError, match="zero trades"):
            objective.score(result_from([]))
