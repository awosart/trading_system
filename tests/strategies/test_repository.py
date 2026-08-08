"""The strategy library: two files per entry, append-only logs, a rebuilt index."""

import json
from pathlib import Path
from typing import Any

import pytest

from trading_system.core.exceptions import ValidationError
from trading_system.core.instruments import InstrumentClass
from trading_system.strategies import schema as schema_module
from trading_system.strategies.repository import (
    META_SUFFIX,
    LifecycleEvent,
    Status,
    StrategyMeta,
    StrategyRepository,
    spec_digest,
)
from trading_system.strategies.schema import Regime, StrategySpec, StrategyType

EXAMPLES_DIR = Path(schema_module.__file__).parent / "examples"


@pytest.fixture
def spec_dict() -> dict[str, Any]:
    """The shipped ema_pullback example, as raw data a test may edit."""
    return json.loads((EXAMPLES_DIR / "ema_pullback.json").read_text(encoding="utf-8"))


@pytest.fixture
def spec(spec_dict: dict[str, Any]) -> StrategySpec:
    """The shipped ema_pullback example, parsed."""
    return StrategySpec.model_validate(spec_dict)


@pytest.fixture
def repository(tmp_path: Path) -> StrategyRepository:
    """An empty library under a temporary root."""
    return StrategyRepository(tmp_path)


def bumped(spec_dict: dict[str, Any], version: str, **changes: Any) -> StrategySpec:
    """The example spec at a new version, with ``changes`` applied at the top level."""
    raw = {**spec_dict, "version": version, **changes}
    return StrategySpec.model_validate(raw)


class TestTheSpecNoLongerCarriesBookkeeping:
    """The removal is the point of the split, so it is asserted, not assumed."""

    @pytest.mark.parametrize("field", ["name", "author", "source", "status", "metadata"])
    def test_the_spec_rejects_a_bookkeeping_field(
        self, spec_dict: dict[str, Any], field: str
    ) -> None:
        # extra="forbid" is what makes this a removal rather than a deprecation:
        # a spec carrying the old field fails to load instead of loading with it
        # ignored, which is how the field would quietly come back.
        spec_dict[field] = "anything"
        with pytest.raises(Exception, match="extra_forbidden|Extra inputs"):
            StrategySpec.model_validate(spec_dict)

    def test_status_is_not_importable_from_the_schema(self) -> None:
        # Re-exporting Status for compatibility is exactly the path by which the
        # field returns to the spec. There must be one place to import it from.
        assert not hasattr(schema_module, "Status")

    def test_the_schema_module_defines_no_lifecycle_enum_under_another_name(self) -> None:
        members = {
            name
            for name in dir(schema_module)
            if isinstance(getattr(schema_module, name), type)
            and hasattr(getattr(schema_module, name), "__members__")
        }
        lifecycle_values = {"DRAFT", "TESTING", "APPROVED", "RETIRED"}
        for name in members:
            values = set(getattr(schema_module, name).__members__)
            assert values != lifecycle_values, f"{name} is Status wearing a different name"


class TestAddAndRead:
    def test_add_writes_two_files_side_by_side(
        self, repository: StrategyRepository, spec: StrategySpec
    ) -> None:
        record = repository.add(spec, name="EMA Pullback", author="ts")
        assert record.path.exists()
        assert record.path.with_name(f"{record.path.stem}{META_SUFFIX}").exists()

    def test_the_spec_file_holds_no_bookkeeping(
        self, repository: StrategyRepository, spec: StrategySpec
    ) -> None:
        record = repository.add(spec, name="EMA Pullback", author="ts", tags=("trend",))
        stored = json.loads(record.path.read_text(encoding="utf-8"))
        assert not {"name", "author", "source", "status", "metadata"} & set(stored)

    def test_a_new_entry_is_a_draft(
        self, repository: StrategyRepository, spec: StrategySpec
    ) -> None:
        assert repository.add(spec, name="x", author="ts").status is Status.DRAFT

    def test_adding_twice_is_refused(
        self, repository: StrategyRepository, spec: StrategySpec
    ) -> None:
        repository.add(spec, name="x", author="ts")
        with pytest.raises(ValidationError, match="already in the library"):
            repository.add(spec, name="x", author="ts")

    def test_the_digest_matches_the_one_a_run_manifest_would_record(
        self, repository: StrategyRepository, spec: StrategySpec
    ) -> None:
        # The library and the result log must compute the same number, or the
        # "is this result still valid?" comparison compares nothing.
        from trading_system.backtest.reproducibility import digest

        record = repository.add(spec, name="x", author="ts")
        assert record.digest == digest(spec) == spec_digest(spec)

    def test_a_missing_strategy_raises(self, repository: StrategyRepository) -> None:
        with pytest.raises(KeyError, match="no strategy"):
            repository.get("nothing-here")


class TestVersioning:
    def test_updating_creates_a_version_and_the_old_one_still_reads(
        self, repository: StrategyRepository, spec_dict: dict[str, Any], spec: StrategySpec
    ) -> None:
        repository.add(spec, name="x", author="ts")
        second = bumped(
            spec_dict, "1.1.0", risk_profile={**spec_dict["risk_profile"], "base_quality": 0.9}
        )
        repository.update(second, note="raised base quality")

        assert repository.get(spec.id).spec.version == "1.1.0"
        archived = repository.get(spec.id, "1.0.0")
        assert archived.spec.version == "1.0.0"
        assert archived.spec.risk_profile.base_quality == spec.risk_profile.base_quality

    def test_the_version_log_grows_and_never_rewrites(
        self, repository: StrategyRepository, spec_dict: dict[str, Any], spec: StrategySpec
    ) -> None:
        repository.add(spec, name="x", author="ts")
        first = repository.get(spec.id).meta.versions
        repository.update(bumped(spec_dict, "1.1.0"))
        repository.update(bumped(spec_dict, "1.2.0"))
        log = repository.get(spec.id).meta.versions

        assert [entry.version for entry in log] == ["1.0.0", "1.1.0", "1.2.0"]
        assert log[: len(first)] == first, "earlier entries were rewritten"

    def test_each_version_records_the_digest_of_that_spec(
        self, repository: StrategyRepository, spec_dict: dict[str, Any], spec: StrategySpec
    ) -> None:
        repository.add(spec, name="x", author="ts")
        second = bumped(
            spec_dict, "1.1.0", risk_profile={**spec_dict["risk_profile"], "base_quality": 0.9}
        )
        repository.update(second)
        digests = [entry.spec_digest for entry in repository.get(spec.id).meta.versions]
        assert digests == [spec_digest(spec), spec_digest(second)]
        assert len(set(digests)) == 2

    def test_updating_without_bumping_is_refused(
        self, repository: StrategyRepository, spec: StrategySpec
    ) -> None:
        repository.add(spec, name="x", author="ts")
        with pytest.raises(ValidationError, match="bump it to update"):
            repository.update(spec)

    def test_republishing_a_retired_version_number_is_refused(
        self, repository: StrategyRepository, spec_dict: dict[str, Any], spec: StrategySpec
    ) -> None:
        repository.add(spec, name="x", author="ts")
        repository.update(bumped(spec_dict, "1.1.0"))
        with pytest.raises(ValidationError, match="append-only"):
            repository.update(bumped(spec_dict, "1.0.0"))

    def test_changing_the_holding_class_is_a_different_strategy(
        self, repository: StrategyRepository, spec_dict: dict[str, Any], spec: StrategySpec
    ) -> None:
        repository.add(spec, name="x", author="ts")
        with pytest.raises(ValidationError, match="different strategy"):
            repository.update(bumped(spec_dict, "2.0.0", type="SCALP"))

    def test_an_unknown_version_names_what_it_has(
        self, repository: StrategyRepository, spec: StrategySpec
    ) -> None:
        repository.add(spec, name="x", author="ts")
        with pytest.raises(KeyError, match="1.0.0"):
            repository.get(spec.id, "9.9.9")


class TestLifecycle:
    def test_status_is_derived_from_the_log_not_stored(self) -> None:
        # There is no settable status field, so a stage cannot exist without the
        # record that produced it. This is the whole provenance guarantee.
        assert "status" not in StrategyMeta.model_fields
        assert StrategyMeta(id="x", name="x", author="t").status is Status.DRAFT

    def test_approval_without_provenance_cannot_be_constructed(self) -> None:
        with pytest.raises(Exception, match="requires"):
            LifecycleEvent(
                status=Status.APPROVED, at=_any_time(), spec_digest="abc", verdict="ROBUST"
            )

    @pytest.mark.parametrize("verdict", ["OVERFIT", "FRAGILE", "INSUFFICIENT", "robust", ""])
    def test_approval_requires_the_robust_verdict(self, verdict: str) -> None:
        with pytest.raises(Exception, match="ROBUST"):
            LifecycleEvent(
                status=Status.APPROVED,
                at=_any_time(),
                spec_digest="abc",
                run_id="r1",
                selector_key="identity",
                verdict=verdict,
            )

    def test_retirement_requires_a_reason(self) -> None:
        with pytest.raises(Exception, match="reason"):
            LifecycleEvent(status=Status.RETIRED, at=_any_time(), spec_digest="abc")

    def test_approve_refuses_a_non_robust_verdict(
        self, repository: StrategyRepository, spec: StrategySpec
    ) -> None:
        repository.add(spec, name="x", author="ts")
        with pytest.raises(ValidationError, match="ROBUST"):
            repository.approve(spec.id, run_id="r1", selector_key="identity", verdict="OVERFIT")
        assert repository.get(spec.id).status is Status.DRAFT

    def test_approve_records_its_provenance(
        self, repository: StrategyRepository, spec: StrategySpec
    ) -> None:
        repository.add(spec, name="x", author="ts")
        record = repository.approve(
            spec.id, run_id="wf-123", selector_key="optimize:grid", verdict="ROBUST"
        )
        approval = record.meta.approval
        assert approval is not None
        assert (approval.run_id, approval.selector_key) == ("wf-123", "optimize:grid")
        assert approval.spec_digest == record.digest

    def test_retiring_appends_rather_than_replacing(
        self, repository: StrategyRepository, spec: StrategySpec
    ) -> None:
        repository.add(spec, name="x", author="ts")
        repository.approve(spec.id, run_id="r1", selector_key="identity", verdict="ROBUST")
        record = repository.retire(spec.id, "superseded by v2")

        assert record.status is Status.RETIRED
        assert record.meta.approval is None, "a later stage supersedes rather than amends"
        stages = [event.status for event in record.meta.lifecycle]
        assert stages == [Status.APPROVED, Status.RETIRED], "the approval is still on the record"

    def test_retiring_without_a_reason_is_refused(
        self, repository: StrategyRepository, spec: StrategySpec
    ) -> None:
        repository.add(spec, name="x", author="ts")
        with pytest.raises(ValidationError, match="reason"):
            repository.retire(spec.id, "   ")

    def test_the_repository_exposes_no_way_to_shorten_either_log(self) -> None:
        # An append-only log with a truncating method is not append-only.
        surface = [name for name in dir(StrategyRepository) if not name.startswith("_")]
        for name in surface:
            assert not any(
                word in name for word in ("delete", "remove", "clear", "drop", "purge", "rewrite")
            ), f"{name} could shorten a log"


class TestMissingMetaFile:
    def test_a_spec_without_meta_still_reads(
        self, repository: StrategyRepository, spec: StrategySpec
    ) -> None:
        record = repository.add(spec, name="EMA", author="ts")
        repository.meta_path(spec.type, spec.id).unlink()

        recovered = repository.get(spec.id)
        assert recovered.spec == record.spec, "the spec alone is enough to trade"
        assert recovered.status is Status.DRAFT

    def test_losing_meta_drops_an_approval_rather_than_granting_one(
        self, repository: StrategyRepository, spec: StrategySpec
    ) -> None:
        repository.add(spec, name="EMA", author="ts")
        repository.approve(spec.id, run_id="r1", selector_key="identity", verdict="ROBUST")
        repository.meta_path(spec.type, spec.id).unlink()
        assert repository.get(spec.id).status is Status.DRAFT

    def test_lost_is_distinguishable_from_never_filled_in(
        self, repository: StrategyRepository, spec: StrategySpec
    ) -> None:
        # Both read as DRAFT, which is safe; if they also looked identical, a
        # deleted approval would pass for an honest draft.
        repository.add(spec, name="EMA", author="ts")
        assert repository.get(spec.id).meta_present is True
        repository.meta_path(spec.type, spec.id).unlink()
        assert repository.get(spec.id).meta_present is False

    def test_a_meta_naming_a_different_strategy_is_rejected(
        self, repository: StrategyRepository, spec: StrategySpec
    ) -> None:
        repository.add(spec, name="EMA", author="ts")
        path = repository.meta_path(spec.type, spec.id)
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["id"] = "someone-else"
        path.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(ValidationError, match="does not match spec id"):
            repository.get(spec.id)


class TestFilters:
    @pytest.fixture
    def stocked(self, repository: StrategyRepository) -> StrategyRepository:
        for path in sorted(EXAMPLES_DIR.glob("*.json")):
            spec = StrategySpec.model_validate_json(path.read_text(encoding="utf-8"))
            tags = {"ema-pullback": ("trend",), "london-breakout": ("breakout",)}.get(
                spec.id, ("mean-reversion",)
            )
            repository.add(spec, name=spec.id, author="ts", tags=tags)
        return repository

    def test_every_example_loads_and_is_found(self, stocked: StrategyRepository) -> None:
        assert {record.id for record in stocked.list()} == {
            "ema-pullback",
            "london-breakout",
            "rsi-mean-reversion",
        }

    def test_filter_by_type(self, stocked: StrategyRepository) -> None:
        found = stocked.list(type=StrategyType.SWING)
        assert found and all(record.spec.type is StrategyType.SWING for record in found)

    def test_filter_by_status(self, stocked: StrategyRepository) -> None:
        stocked.retire("london-breakout", "demo")
        assert [record.id for record in stocked.list(status=Status.RETIRED)] == ["london-breakout"]
        assert "london-breakout" not in {r.id for r in stocked.list(status=Status.DRAFT)}

    def test_filter_by_several_statuses(self, stocked: StrategyRepository) -> None:
        stocked.retire("london-breakout", "demo")
        found = stocked.list(status={Status.DRAFT, Status.RETIRED})
        assert len(found) == 3

    def test_filter_by_tag(self, stocked: StrategyRepository) -> None:
        assert [record.id for record in stocked.list(tag="breakout")] == ["london-breakout"]

    def test_filter_by_instrument(self, stocked: StrategyRepository) -> None:
        for record in stocked.list(instrument="EURUSD"):
            assert "EURUSD" not in record.spec.instruments.denied_symbols

    def test_an_unrestricted_regime_list_matches_every_regime(
        self, stocked: StrategyRepository
    ) -> None:
        # An empty market_regimes means unrestricted in the spec, so it must not
        # read as "permitted in no regime" here.
        unrestricted = [r.id for r in stocked.list() if not r.spec.market_regimes]
        for regime in Regime:
            found = {record.id for record in stocked.list(regime=regime)}
            assert set(unrestricted) <= found

    def test_filter_by_instrument_class(self, stocked: StrategyRepository) -> None:
        found = stocked.list(instrument_class=InstrumentClass.FX)
        assert found and all(
            InstrumentClass.FX in record.spec.instruments.allowed_classes for record in found
        )

    def test_filters_compose(self, stocked: StrategyRepository) -> None:
        assert stocked.list(tag="breakout", status=Status.RETIRED) == []


class TestIndex:
    def test_the_index_is_rebuilt_from_the_files(
        self, repository: StrategyRepository, spec: StrategySpec
    ) -> None:
        repository.add(spec, name="EMA", author="ts", tags=("trend",))
        connection = repository.index()
        rows = connection.execute(
            "SELECT id, status, meta_present FROM strategies WHERE id = ?", [spec.id]
        ).fetchall()
        assert rows == [(spec.id, "DRAFT", True)]

    def test_the_index_cannot_go_stale(
        self, repository: StrategyRepository, spec: StrategySpec
    ) -> None:
        repository.add(spec, name="EMA", author="ts")
        repository.retire(spec.id, "demo")
        status = repository.index().execute("SELECT status FROM strategies").fetchall()
        assert status == [("RETIRED",)]

    def test_an_empty_library_indexes_to_no_rows(self, repository: StrategyRepository) -> None:
        assert repository.index().execute("SELECT count(*) FROM strategies").fetchone() == (0,)

    def test_tags_are_queryable_in_sql(
        self, repository: StrategyRepository, spec: StrategySpec
    ) -> None:
        repository.add(spec, name="EMA", author="ts", tags=("trend", "pullback"))
        found = (
            repository.index()
            .execute("SELECT id FROM strategies WHERE list_contains(tags, 'pullback')")
            .fetchall()
        )
        assert found == [(spec.id,)]


class TestDiff:
    def test_a_changed_field_is_named_by_its_path(
        self, repository: StrategyRepository, spec_dict: dict[str, Any], spec: StrategySpec
    ) -> None:
        repository.add(spec, name="x", author="ts")
        repository.update(
            bumped(
                spec_dict, "1.1.0", risk_profile={**spec_dict["risk_profile"], "base_quality": 0.91}
            )
        )
        out = repository.diff(spec.id, "1.0.0", "1.1.0")
        assert "risk_profile.base_quality" in out
        assert "0.91" in out

    def test_a_nested_entry_change_names_its_index(
        self, repository: StrategyRepository, spec_dict: dict[str, Any], spec: StrategySpec
    ) -> None:
        entries = [dict(entry) for entry in spec_dict["entries"]]
        entries[0] = {**entries[0], "confirmation_window_bars": 9}
        repository.add(spec, name="x", author="ts")
        repository.update(bumped(spec_dict, "1.1.0", entries=entries))
        assert "entries[0].confirmation_window_bars" in repository.diff(spec.id, "1.0.0", "1.1.0")

    def test_identical_specs_say_so(
        self, repository: StrategyRepository, spec: StrategySpec
    ) -> None:
        repository.add(spec, name="x", author="ts")
        assert "identical" in repository.diff(spec.id, "1.0.0", "1.0.0")

    def test_diff_is_symmetric_in_the_paths_it_names(
        self, repository: StrategyRepository, spec_dict: dict[str, Any], spec: StrategySpec
    ) -> None:
        repository.add(spec, name="x", author="ts")
        repository.update(bumped(spec_dict, "1.1.0", exit_ref="conservative_2r"))
        forward = repository.diff(spec.id, "1.0.0", "1.1.0")
        backward = repository.diff(spec.id, "1.1.0", "1.0.0")
        assert "exit_ref" in forward and "exit_ref" in backward


def _any_time() -> Any:
    """A timestamp for constructing lifecycle events under test."""
    from datetime import UTC, datetime

    return datetime(2026, 1, 1, tzinfo=UTC)
