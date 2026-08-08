"""Results bound to the spec and the data that produced them."""

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import polars as pl
import pytest

from trading_system.backtest.clock import StreamKey
from trading_system.backtest.orchestrator import StrategyBinding
from trading_system.backtest.reproducibility import RunManifest
from trading_system.core.exceptions import ValidationError
from trading_system.core.instruments import InstrumentRegistry, load_instruments
from trading_system.core.types import Timeframe
from trading_system.data.models import OHLCVFrame
from trading_system.strategies import schema as schema_module
from trading_system.strategies.repository import Status, StrategyRepository, spec_digest
from trading_system.strategies.results_link import (
    RUN_KIND_FLAT,
    RUN_KIND_WALKFORWARD,
    ResultsLink,
    approve_from_result,
    build_record,
    dataset_hash,
    overlap_fraction,
)
from trading_system.strategies.schema import StrategySpec

EXAMPLES_DIR = Path(schema_module.__file__).parent / "examples"
CONFIGS = Path(__file__).resolve().parents[2] / "configs"


@pytest.fixture
def spec() -> StrategySpec:
    """The shipped ema_pullback example."""
    return StrategySpec.model_validate_json(
        (EXAMPLES_DIR / "ema_pullback.json").read_text(encoding="utf-8")
    )


@pytest.fixture
def instruments() -> InstrumentRegistry:
    """The project's instrument registry."""
    return load_instruments(CONFIGS / "instruments.yaml")


def bars(symbol: str, start: datetime, count: int) -> OHLCVFrame:
    """A deterministic H1 frame of ``count`` bars from ``start``."""
    stamps = [start + timedelta(hours=index) for index in range(count)]
    prices = [1.10 + index * 1e-4 for index in range(count)]
    return OHLCVFrame(
        pl.DataFrame(
            {
                "timestamp": pl.Series(stamps, dtype=pl.Datetime("us", "UTC")),
                "open": prices,
                "high": [price + 5e-4 for price in prices],
                "low": [price - 5e-4 for price in prices],
                "close": prices,
                "volume": [100.0] * count,
            }
        ),
        symbol,
        Timeframe.H1,
    )


def manifest_for(
    spec: StrategySpec,
    streams: dict[StreamKey, OHLCVFrame],
    instruments: InstrumentRegistry,
) -> RunManifest:
    """A manifest over ``streams``, binding ``spec`` to all of them."""
    return RunManifest.build(
        config={"kind": "test"},
        streams=streams,
        bindings=[StrategyBinding(spec=spec, exit_preset="structure_trail", keys=tuple(streams))],
        instruments=instruments,
        seed=7,
        components={"costs": "none"},
    )


@pytest.fixture
def streams() -> dict[StreamKey, OHLCVFrame]:
    """One EURUSD H1 stream of 500 bars."""
    key = StreamKey(symbol="EURUSD", timeframe=Timeframe.H1)
    return {key: bars("EURUSD", datetime(2024, 1, 1, tzinfo=UTC), 500)}


@pytest.fixture
def link(tmp_path: Path) -> ResultsLink:
    """An empty result log."""
    return ResultsLink(tmp_path)


def record_for(
    spec: StrategySpec,
    streams: dict[StreamKey, OHLCVFrame],
    instruments: InstrumentRegistry,
    **overrides: Any,
) -> Any:
    """A result record over ``streams``, with ``overrides`` applied."""
    kwargs: dict[str, Any] = {
        "spec": spec,
        "manifest": manifest_for(spec, streams, instruments),
        "streams": streams,
        "run_id": "run-1",
        "run_kind": RUN_KIND_FLAT,
        "selector_key": "identity",
        "metrics": {"sharpe": 1.4, "expectancy_r": 0.12},
        "created_at": datetime(2026, 1, 1, tzinfo=UTC),
    }
    kwargs.update(overrides)
    return build_record(**kwargs)


class TestBinding:
    def test_the_spec_digest_is_the_libraries_number(
        self, spec: StrategySpec, streams: Any, instruments: InstrumentRegistry
    ) -> None:
        # If these two differ, "is this result still valid for what I hold?"
        # compares nothing at all.
        assert record_for(spec, streams, instruments).spec_digest == spec_digest(spec)

    def test_the_binding_digest_is_kept_separately(
        self, spec: StrategySpec, streams: Any, instruments: InstrumentRegistry
    ) -> None:
        record = record_for(spec, streams, instruments)
        assert record.binding_digest != record.spec_digest
        assert record.binding_digest == manifest_for(spec, streams, instruments).strategies[spec.id]

    def test_a_changed_exit_ref_moves_the_binding_but_not_the_spec_digest_of_the_old_spec(
        self, spec: StrategySpec, streams: Any, instruments: InstrumentRegistry
    ) -> None:
        # Collapsing the two would make a changed exit preset read as "same spec".
        other = manifest_for(spec, streams, instruments)
        swapped = RunManifest.build(
            config={"kind": "test"},
            streams=streams,
            bindings=[
                StrategyBinding(spec=spec, exit_preset="conservative_2r", keys=tuple(streams))
            ],
            instruments=instruments,
            seed=7,
            components={"costs": "none"},
        )
        assert other.strategies[spec.id] != swapped.strategies[spec.id]

    def test_a_run_that_never_bound_this_strategy_is_refused(
        self, spec: StrategySpec, streams: Any, instruments: InstrumentRegistry
    ) -> None:
        other = StrategySpec.model_validate(
            {**json.loads(spec.model_dump_json()), "id": "someone-else"}
        )
        with pytest.raises(ValidationError, match="no binding"):
            # The manifest is the one that ran; the spec offered is not in it.
            record_for(
                other, streams, instruments, manifest=manifest_for(spec, streams, instruments)
            )

    def test_the_selector_key_is_required(
        self, spec: StrategySpec, streams: Any, instruments: InstrumentRegistry
    ) -> None:
        # A result that cannot name its selector cannot say what it is evidence
        # about: an optimising run overwrote the spec's own parameters per fold.
        with pytest.raises(Exception, match="selector_key"):
            record_for(spec, streams, instruments, selector_key="")

    def test_an_unknown_run_kind_is_refused(
        self, spec: StrategySpec, streams: Any, instruments: InstrumentRegistry
    ) -> None:
        with pytest.raises(Exception, match="run_kind"):
            record_for(spec, streams, instruments, run_kind="sideways")

    def test_a_run_over_no_bars_has_no_coverage(
        self, spec: StrategySpec, instruments: InstrumentRegistry
    ) -> None:
        key = StreamKey(symbol="EURUSD", timeframe=Timeframe.H1)
        empty = {key: OHLCVFrame.empty("EURUSD", Timeframe.H1)}
        with pytest.raises(ValidationError, match="cannot cover no bars"):
            record_for(spec, empty, instruments)


class TestIdempotence:
    def test_two_runs_with_one_run_id_give_identical_metrics(
        self, link: ResultsLink, spec: StrategySpec, streams: Any, instruments: InstrumentRegistry
    ) -> None:
        first = record_for(spec, streams, instruments)
        link.record(first)
        # Same run id, same everything: recording again is a confirmed no-op,
        # not a second row.
        again = link.record(record_for(spec, streams, instruments))
        assert again.metrics == first.metrics
        assert len(link.records()) == 1

    def test_a_differing_result_under_the_same_run_id_is_refused(
        self, link: ResultsLink, spec: StrategySpec, streams: Any, instruments: InstrumentRegistry
    ) -> None:
        link.record(record_for(spec, streams, instruments))
        with pytest.raises(ValidationError, match="different result"):
            link.record(record_for(spec, streams, instruments, metrics={"sharpe": 9.9}))

    def test_the_stored_row_survives_a_refused_write(
        self, link: ResultsLink, spec: StrategySpec, streams: Any, instruments: InstrumentRegistry
    ) -> None:
        link.record(record_for(spec, streams, instruments))
        with pytest.raises(ValidationError):
            link.record(record_for(spec, streams, instruments, metrics={"sharpe": 9.9}))
        stored = link.get("run-1")
        assert stored is not None and stored.metric("sharpe") == 1.4

    def test_the_verdict_is_not_part_of_what_a_run_id_promises(
        self, spec: StrategySpec, streams: Any, instruments: InstrumentRegistry
    ) -> None:
        # "Recorded, then graded" and "recorded differently" are opposite
        # situations. Folding the verdict into content_digest would make the
        # first indistinguishable from the second.
        ungraded = record_for(spec, streams, instruments)
        graded = record_for(spec, streams, instruments, verdict="OVERFIT")
        assert ungraded.content_digest() == graded.content_digest()

    def test_a_record_round_trips_through_storage(
        self, link: ResultsLink, spec: StrategySpec, streams: Any, instruments: InstrumentRegistry
    ) -> None:
        original = record_for(spec, streams, instruments)
        link.record(original)
        assert link.get("run-1") == original


class TestGrading:
    def test_a_verdict_attaches_to_a_run_recorded_ungraded(
        self, link: ResultsLink, spec: StrategySpec, streams: Any, instruments: InstrumentRegistry
    ) -> None:
        link.record(record_for(spec, streams, instruments))
        assert link.get("run-1") is not None
        graded = link.grade("run-1", "OVERFIT")
        assert graded.verdict == "OVERFIT"

    def test_grading_persists(
        self, link: ResultsLink, spec: StrategySpec, streams: Any, instruments: InstrumentRegistry
    ) -> None:
        link.record(record_for(spec, streams, instruments))
        link.grade("run-1", "OVERFIT")
        reopened = ResultsLink(link.root)
        stored = reopened.get("run-1")
        assert stored is not None and stored.verdict == "OVERFIT"

    def test_grading_does_not_add_a_row(
        self, link: ResultsLink, spec: StrategySpec, streams: Any, instruments: InstrumentRegistry
    ) -> None:
        link.record(record_for(spec, streams, instruments))
        link.grade("run-1", "OVERFIT")
        assert len(link.records()) == 1

    def test_grading_leaves_the_metrics_untouched(
        self, link: ResultsLink, spec: StrategySpec, streams: Any, instruments: InstrumentRegistry
    ) -> None:
        original = link.record(record_for(spec, streams, instruments))
        graded = link.grade("run-1", "OVERFIT")
        assert graded.content_digest() == original.content_digest()

    def test_regrading_with_the_same_verdict_is_a_no_op(
        self, link: ResultsLink, spec: StrategySpec, streams: Any, instruments: InstrumentRegistry
    ) -> None:
        link.record(record_for(spec, streams, instruments))
        link.grade("run-1", "OVERFIT")
        assert link.grade("run-1", "OVERFIT").verdict == "OVERFIT"

    def test_a_verdict_is_attached_once_and_never_revised(
        self, link: ResultsLink, spec: StrategySpec, streams: Any, instruments: InstrumentRegistry
    ) -> None:
        # Re-grading means re-running: the nulls and perturbations that produced
        # the first grade were measured over these exact metrics.
        link.record(record_for(spec, streams, instruments))
        link.grade("run-1", "OVERFIT")
        with pytest.raises(ValidationError, match="already graded"):
            link.grade("run-1", "ROBUST")

    def test_grading_carries_the_numbers_the_grade_was_reached_from(
        self, link: ResultsLink, spec: StrategySpec, streams: Any, instruments: InstrumentRegistry
    ) -> None:
        # A reader who sees OVERFIT and cannot see how far from its nulls the
        # run fell cannot tell whether the call was close.
        link.record(record_for(spec, streams, instruments))
        graded = link.grade(
            "run-1",
            "OVERFIT",
            metrics={"permutation_percentile": 5.0, "synthetic_percentile": 20.0},
        )
        assert graded.metric("permutation_percentile") == 5.0
        assert graded.metric("synthetic_percentile") == 20.0

    def test_graded_metrics_join_the_ones_the_run_reported(
        self, link: ResultsLink, spec: StrategySpec, streams: Any, instruments: InstrumentRegistry
    ) -> None:
        link.record(record_for(spec, streams, instruments))
        graded = link.grade("run-1", "OVERFIT", metrics={"permutation_percentile": 5.0})
        assert graded.metric("sharpe") == 1.4, "the run's own metrics were lost"

    def test_graded_metrics_persist(
        self, link: ResultsLink, spec: StrategySpec, streams: Any, instruments: InstrumentRegistry
    ) -> None:
        link.record(record_for(spec, streams, instruments))
        link.grade("run-1", "OVERFIT", metrics={"permutation_percentile": 5.0})
        reopened = ResultsLink(link.root).get("run-1")
        assert reopened is not None and reopened.metric("permutation_percentile") == 5.0

    def test_grading_may_not_revise_a_metric_the_run_reported(
        self, link: ResultsLink, spec: StrategySpec, streams: Any, instruments: InstrumentRegistry
    ) -> None:
        # A grade adds measurements over the same run; it does not get to
        # restate what the run itself measured.
        link.record(record_for(spec, streams, instruments))
        with pytest.raises(ValidationError, match="does not revise"):
            link.grade("run-1", "OVERFIT", metrics={"sharpe": 9.9})

    def test_restating_a_metric_unchanged_is_allowed(
        self, link: ResultsLink, spec: StrategySpec, streams: Any, instruments: InstrumentRegistry
    ) -> None:
        link.record(record_for(spec, streams, instruments))
        assert link.grade("run-1", "OVERFIT", metrics={"sharpe": 1.4}).metric("sharpe") == 1.4

    def test_grading_without_metrics_still_works(
        self, link: ResultsLink, spec: StrategySpec, streams: Any, instruments: InstrumentRegistry
    ) -> None:
        link.record(record_for(spec, streams, instruments))
        assert link.grade("run-1", "OVERFIT").verdict == "OVERFIT"

    def test_grading_a_run_nobody_recorded_is_refused(self, link: ResultsLink) -> None:
        with pytest.raises(KeyError, match="record it before grading"):
            link.grade("never-happened", "ROBUST")

    def test_a_run_recorded_already_graded_still_refuses_a_different_grade(
        self, link: ResultsLink, spec: StrategySpec, streams: Any, instruments: InstrumentRegistry
    ) -> None:
        link.record(record_for(spec, streams, instruments, verdict="ROBUST"))
        with pytest.raises(ValidationError, match="already graded"):
            link.grade("run-1", "OVERFIT")


class TestDatasetIdentityAndCoverage:
    def test_different_windows_hash_differently(
        self, spec: StrategySpec, instruments: InstrumentRegistry
    ) -> None:
        key = StreamKey(symbol="EURUSD", timeframe=Timeframe.H1)
        short = {key: bars("EURUSD", datetime(2024, 1, 1, tzinfo=UTC), 300)}
        long = {key: bars("EURUSD", datetime(2024, 1, 1, tzinfo=UTC), 500)}
        assert dataset_hash(manifest_for(spec, short, instruments)) != dataset_hash(
            manifest_for(spec, long, instruments)
        )

    def test_the_same_bars_hash_the_same(
        self, spec: StrategySpec, streams: Any, instruments: InstrumentRegistry
    ) -> None:
        again = {key: bars("EURUSD", datetime(2024, 1, 1, tzinfo=UTC), 500) for key in streams}
        assert dataset_hash(manifest_for(spec, streams, instruments)) == dataset_hash(
            manifest_for(spec, again, instruments)
        )

    def test_the_dataset_hash_ignores_the_code_that_ran(
        self, spec: StrategySpec, streams: Any, instruments: InstrumentRegistry
    ) -> None:
        # run_id folds in code and configs and therefore cannot answer
        # "same data?"; this is why the column exists separately.
        manifest = manifest_for(spec, streams, instruments)
        other = RunManifest.build(
            config={"kind": "different"},
            streams=streams,
            bindings=[
                StrategyBinding(spec=spec, exit_preset="structure_trail", keys=tuple(streams))
            ],
            instruments=instruments,
            seed=99,
            components={"costs": "different"},
        )
        assert manifest.run_id != other.run_id
        assert dataset_hash(manifest) == dataset_hash(other)

    def test_coverage_reports_the_extent_a_hash_cannot(
        self, spec: StrategySpec, streams: Any, instruments: InstrumentRegistry
    ) -> None:
        record = record_for(spec, streams, instruments)
        assert record.symbols == ("EURUSD",)
        assert record.n_bars == 500
        assert record.period_start == datetime(2024, 1, 1, tzinfo=UTC)

    def test_overlap_is_measured_not_left_to_the_caller_to_reconstruct(
        self, spec: StrategySpec, instruments: InstrumentRegistry
    ) -> None:
        key = StreamKey(symbol="EURUSD", timeframe=Timeframe.H1)
        early = record_for(
            spec,
            {key: bars("EURUSD", datetime(2024, 1, 1, tzinfo=UTC), 400)},
            instruments,
            run_id="a",
        )
        overlapping = record_for(
            spec,
            {key: bars("EURUSD", datetime(2024, 1, 1, tzinfo=UTC), 800)},
            instruments,
            run_id="b",
        )
        assert not early.comparable_to(overlapping), "different bars are different datasets"
        assert overlap_fraction(early, overlapping) == pytest.approx(1.0)

    def test_disjoint_periods_do_not_overlap(
        self, spec: StrategySpec, instruments: InstrumentRegistry
    ) -> None:
        key = StreamKey(symbol="EURUSD", timeframe=Timeframe.H1)
        first = record_for(
            spec,
            {key: bars("EURUSD", datetime(2024, 1, 1, tzinfo=UTC), 100)},
            instruments,
            run_id="a",
        )
        second = record_for(
            spec,
            {key: bars("EURUSD", datetime(2025, 1, 1, tzinfo=UTC), 100)},
            instruments,
            run_id="b",
        )
        assert overlap_fraction(first, second) == 0.0


class TestQueries:
    @pytest.fixture
    def stocked(
        self, link: ResultsLink, spec: StrategySpec, instruments: InstrumentRegistry
    ) -> ResultsLink:
        key = StreamKey(symbol="EURUSD", timeframe=Timeframe.H1)
        short = {key: bars("EURUSD", datetime(2024, 1, 1, tzinfo=UTC), 300)}
        long = {key: bars("EURUSD", datetime(2024, 1, 1, tzinfo=UTC), 500)}
        link.record(record_for(spec, short, instruments, run_id="a", metrics={"sharpe": 0.4}))
        link.record(record_for(spec, short, instruments, run_id="b", metrics={"sharpe": 1.8}))
        link.record(
            record_for(
                spec,
                long,
                instruments,
                run_id="c",
                run_kind=RUN_KIND_WALKFORWARD,
                selector_key="optimize:grid",
                metrics={"sharpe": 1.1},
                verdict="OVERFIT",
            )
        )
        return link

    def test_best_returns_one_row_per_dataset_not_a_single_winner(
        self, stocked: ResultsLink, spec: StrategySpec
    ) -> None:
        # Two runs over different windows are not comparable and not
        # independent either; picking one silently would reward the kinder
        # period.
        best = stocked.best(spec.id)
        assert [record.run_id for record in best] == ["b", "c"]
        assert len({record.dataset_hash for record in best}) == 2

    def test_best_ranks_within_a_dataset(self, stocked: ResultsLink, spec: StrategySpec) -> None:
        same_dataset = [r for r in stocked.best(spec.id) if r.run_id in {"a", "b"}]
        assert [record.run_id for record in same_dataset] == ["b"], "1.8 beats 0.4"

    def test_a_metric_nobody_reported_yields_nothing(
        self, stocked: ResultsLink, spec: StrategySpec
    ) -> None:
        assert stocked.best(spec.id, metric="calmar") == []

    def test_an_unreported_metric_is_none_not_zero(
        self, stocked: ResultsLink, spec: StrategySpec
    ) -> None:
        assert stocked.for_strategy(spec.id)[0].metric("calmar") is None

    def test_find_answers_sharpe_above_one_on_oos(self, stocked: ResultsLink) -> None:
        found = stocked.find(metric="sharpe", minimum=1.0, run_kind=RUN_KIND_WALKFORWARD)
        assert [record.run_id for record in found] == ["c"]

    def test_find_across_run_kinds(self, stocked: ResultsLink) -> None:
        found = stocked.find(metric="sharpe", minimum=1.0)
        assert {record.run_id for record in found} == {"b", "c"}

    def test_show_all_runs_of_one_strategy(self, stocked: ResultsLink, spec: StrategySpec) -> None:
        assert {record.run_id for record in stocked.for_strategy(spec.id)} == {"a", "b", "c"}

    def test_filtering_runs_by_spec_digest_is_how_staleness_is_asked(
        self, stocked: ResultsLink, spec: StrategySpec
    ) -> None:
        assert len(stocked.for_strategy(spec.id, spec_digest=spec_digest(spec))) == 3
        assert stocked.for_strategy(spec.id, spec_digest="not-this-one") == []

    def test_the_log_is_queryable_as_sql(self, stocked: ResultsLink) -> None:
        rows = (
            stocked.index()
            .execute("SELECT run_id FROM backtest_results WHERE verdict = 'OVERFIT'")
            .fetchall()
        )
        assert rows == [("c",)]

    def test_an_empty_log_indexes_to_no_rows(self, link: ResultsLink) -> None:
        assert link.index().execute("SELECT count(*) FROM backtest_results").fetchone() == (0,)


class TestApprovalGate:
    @pytest.fixture
    def library(self, tmp_path: Path, spec: StrategySpec) -> StrategyRepository:
        repository = StrategyRepository(tmp_path / "library")
        repository.add(spec, name="EMA Pullback", author="ts")
        return repository

    def test_approval_requires_a_robust_verdict(
        self,
        library: StrategyRepository,
        link: ResultsLink,
        spec: StrategySpec,
        streams: Any,
        instruments: InstrumentRegistry,
    ) -> None:
        link.record(record_for(spec, streams, instruments, verdict="OVERFIT"))
        with pytest.raises(ValidationError, match="OVERFIT"):
            approve_from_result(library, link, "run-1")
        assert library.get(spec.id).status is Status.DRAFT

    def test_a_run_without_a_verdict_cannot_approve(
        self,
        library: StrategyRepository,
        link: ResultsLink,
        spec: StrategySpec,
        streams: Any,
        instruments: InstrumentRegistry,
    ) -> None:
        link.record(record_for(spec, streams, instruments))
        with pytest.raises(ValidationError, match="no verdict"):
            approve_from_result(library, link, "run-1")

    def test_a_robust_run_approves_and_carries_its_provenance(
        self,
        library: StrategyRepository,
        link: ResultsLink,
        spec: StrategySpec,
        streams: Any,
        instruments: InstrumentRegistry,
    ) -> None:
        link.record(
            record_for(
                spec,
                streams,
                instruments,
                verdict="ROBUST",
                selector_key="optimize:grid",
                run_kind=RUN_KIND_WALKFORWARD,
            )
        )
        record = approve_from_result(library, link, "run-1")
        approval = record.meta.approval
        assert record.status is Status.APPROVED
        assert approval is not None
        assert approval.run_id == "run-1"
        assert approval.selector_key == "optimize:grid", (
            "approval names the procedure, not the spec"
        )

    def test_a_spec_edited_since_the_run_cannot_be_approved_on_it(
        self,
        library: StrategyRepository,
        link: ResultsLink,
        spec: StrategySpec,
        streams: Any,
        instruments: InstrumentRegistry,
    ) -> None:
        link.record(record_for(spec, streams, instruments, verdict="ROBUST"))
        edited = StrategySpec.model_validate(
            {
                **json.loads(spec.model_dump_json()),
                "version": "1.1.0",
                "risk_profile": {
                    **json.loads(spec.risk_profile.model_dump_json()),
                    "base_quality": 0.99,
                },
            }
        )
        library.update(edited)
        with pytest.raises(ValidationError, match="has changed since"):
            approve_from_result(library, link, "run-1")

    def test_an_unknown_run_cannot_approve(
        self, library: StrategyRepository, link: ResultsLink
    ) -> None:
        with pytest.raises(KeyError, match="no result recorded"):
            approve_from_result(library, link, "never-happened")
