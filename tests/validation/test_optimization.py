"""Searching a parameter space: determinism, structural in-sample isolation, plateau shape."""

import math
from dataclasses import fields, is_dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest

from tests.backtest.conftest import ema_pullback, harness_inputs, swing_series
from trading_system.backtest.clock import StreamKey
from trading_system.backtest.orchestrator import StrategyBinding
from trading_system.backtest.spec import RunInputs
from trading_system.core.instruments import InstrumentRegistry
from trading_system.core.types import Timeframe
from trading_system.data.models import OHLCVFrame
from trading_system.data.resample import FX_DAY_ORIGIN
from trading_system.exit.library import ExitLibrarySpec
from trading_system.strategies.schema import StrategySpec
from trading_system.validation.objective import (
    ScoredPoint,
    SortinoTimesSqrtTrades,
    analyse_plateau,
    roughness,
)
from trading_system.validation.optimization import (
    GridSearch,
    ISWindowView,
    OptunaSearch,
    ParameterAxis,
    ParamSet,
    RandomSearch,
    SearchSpace,
    TrialLedger,
    TrialOutcome,
    TrialRunner,
)
from trading_system.validation.splitting import (
    FoldWindow,
    WalkForwardMode,
    WalkForwardSplitter,
)
from trading_system.validation.walkforward import (
    IdentitySelector,
    OptimizingSelector,
    WalkForwardRunner,
)

EURUSD_H4 = StreamKey("EURUSD", Timeframe.H4)


def _space() -> SearchSpace:
    """A two-axis space over ``ema_pullback``, small enough to grid exhaustively."""
    return SearchSpace(
        axes=(
            ParameterAxis(
                name="ema_slow",
                paths=(
                    "/entries/0/trigger/conditions/0/right/params/period",
                    "/entries/0/invalidation/price_level/params/period",
                ),
                values=(30, 50, 80),
            ),
            ParameterAxis(
                name="ema_fast",
                paths=("/entries/0/confirmation/0/right/params/period",),
                values=(10, 20),
            ),
        ),
        constraints=(),
    )


def _base(registry: InstrumentRegistry, library: ExitLibrarySpec, length: int) -> RunInputs:
    """An ``ema_pullback``/``swing_series`` run long enough for several folds."""
    spec = ema_pullback()
    preset = next(item for item in library.presets if item.id == spec.exit_ref)
    return harness_inputs(
        registry,
        streams={EURUSD_H4: swing_series(length)},
        bindings=[StrategyBinding(spec=spec, exit_preset=preset, keys=(EURUSD_H4,))],
    )


def _analytic(surface: dict[tuple[int, ...], float]):
    """An evaluate callable over a lookup table, so a search can be driven without bars."""

    def evaluate(params: ParamSet) -> TrialOutcome:
        score = surface.get(params.coords)
        return TrialOutcome(
            score=score,
            piece_scores=(score,),
            dispersion=None,
            n_trades=1,
            returns=(),
            unscoreable=None if score is not None else "not in surface",
        )

    return evaluate


class TestSearchSpaceDescribesOneParameterEvenWhenItLivesInSeveralPlaces:
    """The coupled-pointer requirement, which is where a silent wrong strategy would come from."""

    def test_both_pointers_of_one_axis_receive_the_same_value(self) -> None:
        space = _space()
        spec = ema_pullback()
        varied = space.apply(spec, space.point((2, 0)))
        document = varied.model_dump(mode="json")
        trigger = document["entries"][0]["trigger"]["conditions"][0]["right"]["params"]["period"]
        invalidation = document["entries"][0]["invalidation"]["price_level"]["params"]["period"]
        assert trigger == 80
        assert invalidation == 80, (
            "the slow EMA lives in both the trigger and the invalidation; writing only one "
            "would give a strategy filtered on one period and invalidated by another"
        )

    def test_an_integer_axis_stays_integral_through_the_round_trip(self) -> None:
        space = _space()
        varied = space.apply(ema_pullback(), space.point((1, 1)))
        period = varied.model_dump(mode="json")["entries"][0]["confirmation"][0]["right"]["params"][
            "period"
        ]
        assert isinstance(period, int)

    def test_a_pointer_that_does_not_resolve_raises_rather_than_creating_a_key(self) -> None:
        space = SearchSpace(
            axes=(ParameterAxis(name="nope", paths=("/entries/0/no_such_field",), values=(1, 2)),)
        )
        with pytest.raises(ValueError, match="does not resolve"):
            space.apply(ema_pullback(), space.point((0,)))

    def test_a_range_expands_to_the_same_discrete_axis_a_list_would_give(self) -> None:
        listed = ParameterAxis(name="a", paths=("/x",), values=(10, 15, 20, 25))
        ranged = ParameterAxis(name="a", paths=("/x",), low=10, high=25, step=5)
        assert ranged.values == listed.values
        assert all(isinstance(value, int) for value in ranged.values)

    def test_a_constraint_naming_an_unknown_axis_is_refused_at_construction(self) -> None:
        with pytest.raises(ValueError, match="names no axis"):
            SearchSpace(
                axes=(ParameterAxis(name="a", paths=("/x",), values=(1, 2)),),
                constraints=({"less": "a", "greater": "typo"},),  # type: ignore[arg-type]
            )

    def test_constraints_remove_points_from_the_space_rather_than_failing_them(self) -> None:
        space = SearchSpace(
            axes=(
                ParameterAxis(name="fast", paths=("/x",), values=(10, 20, 30)),
                ParameterAxis(name="slow", paths=("/y",), values=(10, 20, 30)),
            ),
            constraints=({"less": "fast", "greater": "slow"},),  # type: ignore[arg-type]
        )
        assert space.grid_size == 9
        assert space.feasible_size() == 3
        for params in space.enumerate():
            values = params.as_dict()
            assert values["fast"] < values["slow"]


class TestGridSearchIsDeterministic:
    """DoD: the same space gives the same order of proposals on every run."""

    def test_two_runs_propose_the_same_points_in_the_same_order(self) -> None:
        space = _space()
        first = [params.coords for params in GridSearch().suggest(space, budget=6)]
        second = [params.coords for params in GridSearch().suggest(space, budget=6)]
        assert first == second
        assert len(first) == space.feasible_size()

    def test_a_budget_that_cannot_cover_the_grid_is_refused_rather_than_truncated(self) -> None:
        space = _space()
        with pytest.raises(ValueError, match="at least 6"):
            list(GridSearch().suggest(space, budget=5))


class TestRandomSearchIsSeeded:
    def test_the_same_seed_samples_the_same_points_in_the_same_order(self) -> None:
        space = _space()
        first = [p.coords for p in RandomSearch(seed=7).suggest(space, budget=4)]
        second = [p.coords for p in RandomSearch(seed=7).suggest(space, budget=4)]
        assert first == second

    def test_a_different_seed_samples_differently(self) -> None:
        space = _space()
        first = [p.coords for p in RandomSearch(seed=1).suggest(space, budget=4)]
        second = [p.coords for p in RandomSearch(seed=2).suggest(space, budget=4)]
        assert first != second

    def test_sampling_is_without_replacement(self) -> None:
        space = _space()
        drawn = [p.coords for p in RandomSearch(seed=3).suggest(space, budget=6)]
        assert len(set(drawn)) == len(drawn) == space.feasible_size()


class TestOptunaIsReproducible:
    """DoD: TPE with a fixed seed replays exactly, given the same deterministic surface."""

    def test_two_studies_with_the_same_seed_visit_the_same_points(self) -> None:
        pytest.importorskip("optuna")
        space = _space()
        surface = {p.coords: float(sum(p.coords)) for p in space.enumerate()}
        first = OptunaSearch(seed=11, n_startup_trials=3).run(space, _analytic(surface), 8)
        second = OptunaSearch(seed=11, n_startup_trials=3).run(space, _analytic(surface), 8)
        assert [r.params.coords for r in first] == [r.params.coords for r in second]
        assert [r.outcome.score for r in first] == [r.outcome.score for r in second]

    def test_a_different_seed_gives_a_different_sequence(self) -> None:
        pytest.importorskip("optuna")
        space = _space()
        surface = {p.coords: float(sum(p.coords)) for p in space.enumerate()}
        first = OptunaSearch(seed=11, n_startup_trials=3).run(space, _analytic(surface), 6)
        second = OptunaSearch(seed=12, n_startup_trials=3).run(space, _analytic(surface), 6)
        assert [r.params.coords for r in first] != [r.params.coords for r in second]


class TestScoreDoesNotDependOnSuggestionOrder:
    """The invariant: a point's in-sample score is a property of the point, not of the visit order.

    Both searches are driven over one analytic surface; if either carried state
    that leaked between trials, the same coordinates would score differently
    depending on when they were reached.
    """

    def test_grid_and_random_agree_on_every_point_they_both_visited(self) -> None:
        space = _space()
        surface = {p.coords: math.sin(p.coords[0]) + p.coords[1] for p in space.enumerate()}
        grid = {
            r.params.coords: r.outcome.score for r in GridSearch().run(space, _analytic(surface), 6)
        }
        shuffled = {
            r.params.coords: r.outcome.score
            for r in RandomSearch(seed=5).run(space, _analytic(surface), 6)
        }
        assert grid == shuffled


# ---------------------------------------------------------------------------
# Structural in-sample isolation
# ---------------------------------------------------------------------------


def _reachable_frames(root: Any, seen: set[int] | None = None) -> list[OHLCVFrame]:
    """Every :class:`OHLCVFrame` reachable from ``root`` through the object graph.

    Reflective rather than a list of places to look: the point of the test that
    uses it is that *no* path leads to an out-of-sample bar, which a hand-written
    list of attributes could only ever check for the paths its author thought of.
    """
    seen = set() if seen is None else seen
    if id(root) in seen:
        return []
    seen.add(id(root))
    if isinstance(root, OHLCVFrame):
        return [root]
    found: list[OHLCVFrame] = []
    if isinstance(root, (str, bytes, int, float, bool)) or root is None:
        return found
    if isinstance(root, dict):
        for key, value in root.items():
            found.extend(_reachable_frames(key, seen))
            found.extend(_reachable_frames(value, seen))
        return found
    if isinstance(root, (list, tuple, set, frozenset)):
        for item in root:
            found.extend(_reachable_frames(item, seen))
        return found
    if is_dataclass(root) and not isinstance(root, type):
        for spec in fields(root):
            found.extend(_reachable_frames(getattr(root, spec.name, None), seen))
        return found
    for value in getattr(root, "__dict__", {}).values():
        found.extend(_reachable_frames(value, seen))
    return found


class TestTheSearchCannotReachAnOutOfSampleBar:
    """DoD: not a mock and not a promise — two constructors that refuse to exist."""

    def test_a_view_holding_a_bar_at_the_boundary_cannot_be_constructed(
        self, registry: InstrumentRegistry, library: ExitLibrarySpec
    ) -> None:
        base = _base(registry, library, length=600)
        frame = base.streams[EURUSD_H4]
        assert frame.start is not None and frame.end is not None
        window = FoldWindow(
            data_start=frame.start,
            trade_start=frame.start + timedelta(days=5),
            trade_end=frame.start + timedelta(days=20),
        )
        with pytest.raises(ValueError, match="at or after the in-sample"):
            ISWindowView(
                streams={EURUSD_H4: frame},  # the full, unsliced stream
                data_start=window.data_start,
                trade_start=window.trade_start,
                trade_end=window.trade_end,
            )

    def test_a_trial_runner_refuses_a_template_carrying_full_coverage_streams(
        self, registry: InstrumentRegistry, library: ExitLibrarySpec
    ) -> None:
        base = _base(registry, library, length=600)
        frame = base.streams[EURUSD_H4]
        assert frame.start is not None
        window = FoldWindow(
            data_start=frame.start,
            trade_start=frame.start + timedelta(days=5),
            trade_end=frame.start + timedelta(days=20),
        )
        view = ISWindowView.build(base.streams, window)
        with pytest.raises(ValueError, match="exactly the ISWindowView's streams"):
            TrialRunner(
                view=view,
                template=base,  # full coverage: the mistake this guard exists for
                space=_space(),
                objective=SortinoTimesSqrtTrades(),
            )

    def test_no_object_reachable_from_the_trial_runner_holds_a_later_bar(
        self, registry: InstrumentRegistry, library: ExitLibrarySpec
    ) -> None:
        base = _base(registry, library, length=900)
        splitter = WalkForwardSplitter(
            mode=WalkForwardMode.ROLLING,
            is_span=timedelta(days=60),
            oos_span=timedelta(days=20),
            step=timedelta(days=20),
            embargo=timedelta(days=2),
            warmup=timedelta(days=10),
        )
        frame = base.streams[EURUSD_H4]
        assert frame.start is not None and frame.end is not None
        folds = splitter.split((frame.start, frame.end), day_origin=FX_DAY_ORIGIN)
        fold = folds[0]

        view = ISWindowView.build(base.streams, fold.is_window)
        from dataclasses import replace

        runner = TrialRunner(
            view=view,
            template=replace(base, streams=view.streams),
            space=_space(),
            objective=SortinoTimesSqrtTrades(),
        )

        frames = _reachable_frames(runner)
        assert frames, "the walk must actually find the bars, or it proves nothing"
        for found in frames:
            assert found.end is not None
            assert found.end < fold.is_window.trade_end
        assert fold.oos_window.trade_end > fold.is_window.trade_end, (
            "the fold must genuinely have later data for this to be a real check"
        )

    def test_every_window_a_trial_is_evaluated_over_stops_at_the_in_sample_boundary(
        self, registry: InstrumentRegistry, library: ExitLibrarySpec
    ) -> None:
        base = _base(registry, library, length=600)
        frame = base.streams[EURUSD_H4]
        assert frame.start is not None
        window = FoldWindow(
            data_start=frame.start,
            trade_start=frame.start + timedelta(days=5),
            trade_end=frame.start + timedelta(days=20),
        )
        view = ISWindowView.build(base.streams, window)
        for start, end in view.sub_windows(None):
            assert start >= window.trade_start
            assert end <= window.trade_end
        with pytest.raises(ValueError, match="falls outside the in-sample window"):
            view.sub_windows([(window.trade_start, window.trade_end + timedelta(days=1))])


# ---------------------------------------------------------------------------
# Plateau versus spike
# ---------------------------------------------------------------------------


def _grid_points(scores: dict[tuple[int, int], float]) -> list[ScoredPoint]:
    """Scored points on a 2-D integer grid."""
    return [ScoredPoint(coords=coords, score=score) for coords, score in sorted(scores.items())]


class TestAnOverfittedPeakIsDistinguishableFromABroadPlateau:
    """DoD: a narrow spike tuned to one fold must score low on plateau width.

    Both surfaces below reach the same maximum by construction, so nothing about
    the *height* of the optimum separates them — only the shape of the ground
    around it, which is the whole claim.
    """

    def test_a_spike_and_a_plateau_with_the_same_maximum_differ_in_measured_width(self) -> None:
        spike = {(x, y): 0.0 for x in range(9) for y in range(9)}
        spike[(4, 4)] = 10.0
        broad = {(x, y): 0.0 for x in range(9) for y in range(9)}
        for x in range(2, 7):
            for y in range(2, 7):
                broad[(x, y)] = 10.0

        spike_analysis = analyse_plateau(_grid_points(spike), penalty_weight=0.0)
        broad_analysis = analyse_plateau(_grid_points(broad), penalty_weight=0.0)

        assert spike_analysis.plateau_size == 1
        assert broad_analysis.plateau_size == 25
        assert spike_analysis.plateau_fraction < broad_analysis.plateau_fraction
        assert spike_analysis.axis_extent == (1, 1)
        assert broad_analysis.axis_extent == (5, 5)

    def test_the_penalty_makes_a_slightly_lower_broad_optimum_win_the_selection(self) -> None:
        surface = {(x, y): 0.0 for x in range(9) for y in range(9)}
        surface[(1, 1)] = 10.0  # the overfitted spike: highest raw score anywhere
        for x in range(5, 8):
            for y in range(5, 8):
                surface[(x, y)] = 9.0  # a genuinely broad, slightly lower region

        points = _grid_points(surface)
        raw = analyse_plateau(points, penalty_weight=0.0)
        penalised = analyse_plateau(points, penalty_weight=0.5)

        assert points[raw.best_index].coords == (1, 1), "without a penalty, argmax takes the spike"
        assert points[penalised.best_index].coords != (1, 1), (
            "with a penalty the spike must lose to the broad region, which is the point "
            "of penalising steepness rather than height"
        )
        assert points[penalised.selected_index].coords == (6, 6), (
            "and the selection must land at the plateau's centre, not on its edge"
        )

    def test_roughness_is_zero_in_the_middle_of_a_plateau_and_large_on_a_spike(self) -> None:
        surface = {(x, y): 0.0 for x in range(7) for y in range(7)}
        surface[(1, 1)] = 10.0
        for x in range(3, 7):
            for y in range(3, 7):
                surface[(x, y)] = 10.0
        points = _grid_points(surface)
        index = {point.coords: number for number, point in enumerate(points)}

        assert roughness(points, index[(1, 1)]) == pytest.approx(10.0)
        assert roughness(points, index[(5, 5)]) == pytest.approx(0.0)

    def test_a_point_with_no_neighbours_reports_an_absent_roughness_not_a_zero_one(self) -> None:
        points = [ScoredPoint(coords=(0, 0), score=1.0), ScoredPoint(coords=(5, 5), score=2.0)]
        assert roughness(points, 0) is None
        assert analyse_plateau(points).n_without_neighbours == 2

    def test_the_selected_point_is_always_one_that_was_actually_evaluated(self) -> None:
        surface = {(x, 0): 0.0 for x in range(9)}
        for x in (2, 4, 6):
            surface[(x, 0)] = 5.0
        points = _grid_points(surface)
        analysis = analyse_plateau(points, tolerance_sigmas=3.0)
        assert 0 <= analysis.selected_index < len(points)
        assert points[analysis.selected_index] in points


# ---------------------------------------------------------------------------
# Budget accounting
# ---------------------------------------------------------------------------


class TestBudgetIsPerFoldAndNeverCarried:
    def test_the_ledger_keeps_folds_apart_and_refuses_a_double_entry(self) -> None:
        ledger = TrialLedger()
        ledger.record(0, trials=10, runs=30)
        ledger.record(1, trials=10, runs=30)
        assert ledger.per_fold == {0: 10, 1: 10}
        assert ledger.total_trials == 20
        assert ledger.total_runs == 60
        with pytest.raises(ValueError, match="already recorded"):
            ledger.record(0, trials=5, runs=15)

    def test_every_fold_of_a_real_walk_forward_spends_the_same_budget(
        self, registry: InstrumentRegistry, library: ExitLibrarySpec, tmp_path: Path
    ) -> None:
        base = _base(registry, library, length=900)
        splitter = WalkForwardSplitter(
            mode=WalkForwardMode.ROLLING,
            is_span=timedelta(days=60),
            oos_span=timedelta(days=20),
            step=timedelta(days=20),
            embargo=timedelta(days=2),
            warmup=timedelta(days=10),
        )
        frame = base.streams[EURUSD_H4]
        assert frame.start is not None and frame.end is not None
        folds = splitter.split((frame.start, frame.end), day_origin=FX_DAY_ORIGIN)
        selector = OptimizingSelector(
            base=base,
            space=_space(),
            search=GridSearch(),
            objective=SortinoTimesSqrtTrades(),
            trial_budget=6,
            store_root=tmp_path,
        )
        for fold in folds[:2]:
            selector.select(fold, is_result=None)  # type: ignore[arg-type]
        assert set(selector.ledger.per_fold.values()) == {6}, (
            "a fold must not inherit budget from an earlier one"
        )


class TestASelectionIsResumedRatherThanRepeated:
    def test_a_second_pass_reads_the_stored_selection_instead_of_re_searching(
        self, registry: InstrumentRegistry, library: ExitLibrarySpec, tmp_path: Path
    ) -> None:
        base = _base(registry, library, length=900)
        splitter = WalkForwardSplitter(
            mode=WalkForwardMode.ROLLING,
            is_span=timedelta(days=60),
            oos_span=timedelta(days=20),
            step=timedelta(days=20),
            embargo=timedelta(days=2),
            warmup=timedelta(days=10),
        )
        frame = base.streams[EURUSD_H4]
        assert frame.start is not None and frame.end is not None
        fold = splitter.split((frame.start, frame.end), day_origin=FX_DAY_ORIGIN)[0]

        def build() -> OptimizingSelector:
            return OptimizingSelector(
                base=base,
                space=_space(),
                search=GridSearch(),
                objective=SortinoTimesSqrtTrades(),
                trial_budget=6,
                store_root=tmp_path,
            )

        first = build()
        chosen = first.select(fold, is_result=None)  # type: ignore[arg-type]
        assert first.outcomes[0].n_trials == 6

        second = build()
        resumed = second.select(fold, is_result=None)  # type: ignore[arg-type]
        assert second.outcomes == {}, "a resumed fold must not have re-run the search"
        assert second.ledger.per_fold == {0: 6}, "but its cost must still be on the ledger"
        assert _selected_periods(chosen) == _selected_periods(resumed)


def _selected_periods(inputs: RunInputs) -> tuple[int, int]:
    """The two EMA periods a run's single binding ended up with."""
    document: dict[str, Any] = inputs.bindings[0].spec.model_dump(mode="json")
    return (
        document["entries"][0]["trigger"]["conditions"][0]["right"]["params"]["period"],
        document["entries"][0]["confirmation"][0]["right"]["params"]["period"],
    )


class TestTheSelectorReturnsFullCoverageForTheRunnerToSlice:
    """The returned inputs legitimately carry every bar; only the *search* is confined."""

    def test_the_returned_run_still_has_the_whole_history(
        self, registry: InstrumentRegistry, library: ExitLibrarySpec, tmp_path: Path
    ) -> None:
        base = _base(registry, library, length=900)
        splitter = WalkForwardSplitter(
            mode=WalkForwardMode.ROLLING,
            is_span=timedelta(days=60),
            oos_span=timedelta(days=20),
            step=timedelta(days=20),
            embargo=timedelta(days=2),
            warmup=timedelta(days=10),
        )
        frame = base.streams[EURUSD_H4]
        assert frame.start is not None and frame.end is not None
        fold = splitter.split((frame.start, frame.end), day_origin=FX_DAY_ORIGIN)[0]
        selector = OptimizingSelector(
            base=base,
            space=_space(),
            search=GridSearch(),
            objective=SortinoTimesSqrtTrades(),
            trial_budget=6,
            store_root=tmp_path,
        )
        chosen = selector.select(fold, is_result=None)  # type: ignore[arg-type]
        assert chosen.streams[EURUSD_H4].end == frame.end, (
            "WalkForwardRunner slices the OOS window out of what select() returns, so the "
            "returned object must still contain it"
        )
        assert isinstance(chosen.bindings[0].spec, StrategySpec)


class TestTwoSelectorsCannotShareAWalkForwardId:
    """The defect this stage found: ``wf_id`` did not depend on the selector.

    With only ``IdentitySelector`` in existence the selector was a constant and
    left no fingerprint in the id. The moment a second one existed, an
    optimising and a non-optimising walk-forward over the same history and the
    same fold geometry landed on the same ``wf_id`` — and since
    :meth:`WalkForwardRunner.run` is idempotent on that id, the second silently
    returned the first's folds without running anything, then reported on them
    as if they were its own.
    """

    def _runner(self, base: RunInputs, selector: Any, root: Path) -> WalkForwardRunner:
        return WalkForwardRunner(
            base=base,
            splitter=WalkForwardSplitter(
                mode=WalkForwardMode.ROLLING,
                is_span=timedelta(days=60),
                oos_span=timedelta(days=20),
                step=timedelta(days=20),
                embargo=timedelta(days=2),
                warmup=timedelta(days=10),
            ),
            selector=selector,
            store_root=root,
            max_drain_bars=20,
        )

    def test_an_optimising_run_and_a_baseline_run_get_different_ids(
        self, registry: InstrumentRegistry, library: ExitLibrarySpec, tmp_path: Path
    ) -> None:
        base = _base(registry, library, length=900)
        baseline = self._runner(base, IdentitySelector(base), tmp_path).run()
        optimising = self._runner(
            base,
            OptimizingSelector(
                base=base,
                space=_space(),
                search=GridSearch(),
                objective=SortinoTimesSqrtTrades(),
                trial_budget=6,
                store_root=tmp_path,
            ),
            tmp_path,
        ).run()
        assert baseline.wf_id != optimising.wf_id

    def test_two_differently_configured_optimisers_get_different_ids(
        self, registry: InstrumentRegistry, library: ExitLibrarySpec, tmp_path: Path
    ) -> None:
        base = _base(registry, library, length=900)

        def optimiser(penalty: float) -> OptimizingSelector:
            return OptimizingSelector(
                base=base,
                space=_space(),
                search=GridSearch(),
                objective=SortinoTimesSqrtTrades(),
                trial_budget=6,
                store_root=tmp_path,
                penalty_weight=penalty,
            )

        first = self._runner(base, optimiser(0.0), tmp_path).run()
        second = self._runner(base, optimiser(0.9), tmp_path).run()
        assert first.wf_id != second.wf_id, (
            "two optimisers that can choose differently must not share an id"
        )

    def test_the_same_configuration_still_reproduces_one_id(
        self, registry: InstrumentRegistry, library: ExitLibrarySpec, tmp_path: Path
    ) -> None:
        base = _base(registry, library, length=900)
        first = self._runner(base, IdentitySelector(base), tmp_path).run()
        second = self._runner(base, IdentitySelector(base), tmp_path).run()
        assert first.wf_id == second.wf_id, "idempotence must survive the fix"


class TestTheSameSpaceScoresAPointIdenticallyThroughARealEngine:
    """The order-independence invariant, driven through real backtests rather than a surface.

    The analytic version above proves the searches carry no state. This one
    proves the *evaluation* carries none either: the same parameter set, run
    first and run last, must produce the identical score, or a plateau built
    from the table would be describing evaluation order rather than the
    parameter space.
    """

    def test_a_point_scores_the_same_whether_it_is_evaluated_first_or_last(
        self, registry: InstrumentRegistry, library: ExitLibrarySpec
    ) -> None:
        from dataclasses import replace

        base = _base(registry, library, length=900)
        frame = base.streams[EURUSD_H4]
        assert frame.start is not None
        window = FoldWindow(
            data_start=frame.start,
            trade_start=frame.start + timedelta(days=15),
            trade_end=frame.start + timedelta(days=90),
        )
        view = ISWindowView.build(base.streams, window)
        space = _space()
        runner = TrialRunner(
            view=view,
            template=replace(base, streams=view.streams),
            space=space,
            objective=SortinoTimesSqrtTrades(),
        )
        points = list(space.enumerate())
        forward = {p.coords: runner.evaluate(p).score for p in points}
        backward = {p.coords: runner.evaluate(p).score for p in reversed(points)}
        assert forward == backward
        assert any(value is not None for value in forward.values()), (
            "the window must actually produce scoreable runs, or this proves nothing"
        )


class TestRandomSearchDoesNotEnumerateLargeSpaces:
    """Drawing k points must cost O(k x axes), not O(size of the space).

    The old sampler materialised the whole space and shuffled it. On an
    eleven-axis space from the real shelf that was 3 906 250 ``ParamSet``
    objects, 108.3 seconds and 1.14 GB of resident memory — to return 48
    points. These tests fix the four properties that make the replacement
    equivalent rather than merely faster.
    """

    def _space(self, widths: list[int], constraints: list[dict[str, str]] | None = None):
        from trading_system.validation.optimization import SearchSpace

        return SearchSpace.model_validate(
            {
                "axes": [
                    {
                        "name": f"a{index}",
                        "paths": [f"/entries/0/p{index}"],
                        "values": list(range(width)),
                    }
                    for index, width in enumerate(widths)
                ],
                "constraints": constraints or [],
            }
        )

    def test_the_small_path_draws_exactly_what_the_old_sampler_drew(self) -> None:
        # Below the limit the algorithm is unchanged, which is what lets every
        # stored run whose space was small enough keep its digest.
        import random as _random

        from trading_system.validation.optimization import RandomSearch

        space = self._space([5, 5, 5, 2])

        def old(budget: int, seed: int) -> list[tuple[int, ...]]:
            population = list(space.enumerate())
            rng = _random.Random(seed)
            rng.shuffle(population)
            return [point.coords for point in population[:budget]]

        for seed in range(5):
            for budget in (1, 7, 48, 200):
                drawn = [p.coords for p in RandomSearch(seed=seed).suggest(space, budget)]
                assert drawn == old(budget, seed)

    def test_a_large_space_is_never_enumerated(self) -> None:
        import time

        from trading_system.validation.optimization import RandomSearch

        # Ten axes of five values: 9 765 625 points. Enumerating would take
        # tens of seconds; the assertion is on the clock because the claim is
        # about cost, and a cost claim that is not timed is an opinion.
        space = self._space([5] * 10)
        started = time.monotonic()
        drawn = list(RandomSearch(seed=0).suggest(space, 48))
        assert time.monotonic() - started < 1.0
        assert len(drawn) == 48
        assert len({point.coords for point in drawn}) == 48

    def test_the_two_algorithms_agree_in_distribution(self) -> None:
        # A space small enough to enumerate, sampled many times by both paths;
        # the empirical frequency of each point must agree. Equivalence of
        # distribution is the claim that makes the swap safe, so it is measured
        # rather than argued.
        import random as _random
        from collections import Counter

        from trading_system.validation.optimization import RandomSearch

        space = self._space([4, 4])
        draws = 4000
        new: Counter[tuple[int, ...]] = Counter()
        old: Counter[tuple[int, ...]] = Counter()
        import trading_system.validation.optimization as optimization

        original = optimization.ENUMERATION_LIMIT
        try:
            # Force the rejection path onto a space small enough to also walk
            # with the old algorithm: the two are only comparable side by side.
            optimization.ENUMERATION_LIMIT = 0
            for seed in range(draws):
                new.update(point.coords for point in RandomSearch(seed=seed).suggest(space, 2))
        finally:
            optimization.ENUMERATION_LIMIT = original
        for seed in range(draws):
            population = list(space.enumerate())
            _random.Random(seed).shuffle(population)
            old.update(point.coords for point in population[:2])

        total = space.size
        for coords in (point.coords for point in space.enumerate()):
            expected = 2 * draws / total
            assert abs(new[coords] - expected) < 0.35 * expected
            assert abs(new[coords] - old[coords]) < 0.4 * expected

    def test_fewer_feasible_points_than_the_budget_returns_them_all(self) -> None:
        from trading_system.validation.optimization import RandomSearch

        space = self._space([3, 2])
        drawn = list(RandomSearch(seed=0).suggest(space, 100))
        assert len(drawn) == space.size

    def test_constraints_that_leave_almost_nothing_feasible_fail_loudly(self) -> None:
        import pytest

        from trading_system.validation.optimization import RandomSearch, SearchSpace

        # 125 000 points, past the enumeration limit, of which *none* satisfy
        # the ordering: a0 is drawn from values that are all above a1's. A short
        # sample here would be a fold reporting a search it did not run.
        space = SearchSpace.model_validate(
            {
                "axes": [
                    {"name": "a0", "paths": ["/entries/0/p0"], "values": list(range(100, 150))},
                    {"name": "a1", "paths": ["/entries/0/p1"], "values": list(range(50))},
                    {"name": "a2", "paths": ["/entries/0/p2"], "values": list(range(50))},
                ],
                "constraints": [{"less": "a0", "greater": "a1"}],
            }
        )
        assert space.size > 100_000
        with pytest.raises(ValueError, match="too few feasible points"):
            list(RandomSearch(seed=0).suggest(space, 48))


class TestFeasibleSizeIsNotCountedByEnumeration:
    """Counting a product by walking it took 9.2 seconds on a real space."""

    def _space(self, widths: list[int], constraints: list[dict[str, str]] | None = None):
        from trading_system.validation.optimization import SearchSpace

        return SearchSpace.model_validate(
            {
                "axes": [
                    {
                        "name": f"a{index}",
                        "paths": [f"/entries/0/p{index}"],
                        "values": list(range(width)),
                    }
                    for index, width in enumerate(widths)
                ],
                "constraints": constraints or [],
            }
        )

    def test_an_unconstrained_space_is_counted_by_arithmetic(self) -> None:
        import time

        space = self._space([5] * 10)
        started = time.monotonic()
        assert space.feasible_size() == 9_765_625 == space.size
        assert time.monotonic() - started < 0.1

    def test_a_constrained_space_within_the_limit_is_still_counted_exactly(self) -> None:
        space = self._space([4, 4], [{"less": "a0", "greater": "a1"}])
        counted = sum(1 for _ in space.enumerate())
        assert space.feasible_size() == counted
        assert counted < space.size

    def test_a_large_constrained_space_refuses_rather_than_spending_the_time(self) -> None:
        import pytest

        space = self._space([100, 100, 100], [{"less": "a0", "greater": "a1"}])
        with pytest.raises(ValueError, match="past the limit"):
            space.feasible_size()
        # The upper bound is always available and always cheap.
        assert space.size == 1_000_000
