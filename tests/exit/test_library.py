"""The Exit DB: preset specs, the loader, and the eight bundled presets.

DoD items covered here: all eight presets load and validate; ``load_library``
raises a clear, preset-naming error on a broken one; the smallest-fraction hooks
answers correctly for every preset, ladder or not.
"""

from decimal import Decimal
from pathlib import Path

import pydantic
import pytest

from trading_system.core.exceptions import ValidationError
from trading_system.data.sessions import Session
from trading_system.exit.base import IntrabarPolicy
from trading_system.exit.library import (
    DEFAULT_LIBRARY_PATH,
    ExitLibrary,
    ExitLibrarySpec,
    ExitPresetSpec,
    PartialRungSpec,
    TimeExitSpec,
    build_plan,
    known_exit_ids,
    load_library,
)
from trading_system.exit.rules import (
    ATRStop,
    FixedRR,
    ProtectiveStop,
)

BUNDLED_PRESET_IDS = frozenset(
    {
        "conservative_2r",
        "atr_trail_aggressive",
        "scalp_quick",
        "swing_partial_ladder",
        "session_close",
        "breakeven_runner",
        "structure_trail",
        "time_boxed",
    }
)


def minimal_preset(**overrides: object) -> dict[str, object]:
    """The smallest preset dict satisfying every required field."""
    preset: dict[str, object] = {
        "id": "test-preset",
        "name": "Test Preset",
        "protective_stop": {"kind": "PROTECTIVE_STOP"},
    }
    preset.update(overrides)
    return preset


class TestProtectiveStopIsMandatory:
    def test_a_preset_without_a_protective_stop_does_not_parse(self) -> None:
        preset = minimal_preset()
        del preset["protective_stop"]
        with pytest.raises(pydantic.ValidationError, match="protective_stop"):
            ExitPresetSpec.model_validate(preset)

    def test_a_null_protective_stop_does_not_parse(self) -> None:
        with pytest.raises(pydantic.ValidationError):
            ExitPresetSpec.model_validate(minimal_preset(protective_stop=None))


class TestSpecsRejectUnknownFields:
    def test_an_unknown_field_on_a_preset_is_rejected(self) -> None:
        with pytest.raises(pydantic.ValidationError):
            ExitPresetSpec.model_validate(minimal_preset(unexpected="field"))

    def test_an_unknown_field_on_a_rule_is_rejected(self) -> None:
        with pytest.raises(pydantic.ValidationError):
            ExitPresetSpec.model_validate(
                minimal_preset(rules=[{"kind": "FIXED_RR", "r_multiple": 2.0, "extra": 1}])
            )

    def test_an_unknown_kind_discriminator_is_rejected(self) -> None:
        with pytest.raises(pydantic.ValidationError):
            ExitPresetSpec.model_validate(minimal_preset(rules=[{"kind": "NOT_A_RULE"}]))


class TestFractionsAreExactDecimals:
    def test_a_string_fraction_parses_to_an_exact_decimal(self) -> None:
        rung = PartialRungSpec.model_validate({"r_multiple": 1.0, "fraction": "0.1"})
        assert rung.fraction == Decimal("0.1")
        assert isinstance(rung.fraction, Decimal)

    def test_a_fraction_outside_the_open_unit_interval_is_rejected(self) -> None:
        for bad in ("0", "1", "1.5", "-0.1"):
            with pytest.raises(pydantic.ValidationError):
                PartialRungSpec.model_validate({"r_multiple": 1.0, "fraction": bad})


class TestTimeExitSpecModes:
    def test_max_bars_held_accepts_a_positive_count(self) -> None:
        spec = TimeExitSpec.model_validate({"mode": "MAX_BARS_HELD", "max_bars_held": 5})
        assert spec.max_bars_held == 5

    def test_session_close_accepts_a_session(self) -> None:
        spec = TimeExitSpec.model_validate({"mode": "SESSION_CLOSE", "session": "LONDON"})
        assert spec.session is Session.LONDON

    def test_an_unknown_session_name_is_rejected(self) -> None:
        with pytest.raises(pydantic.ValidationError):
            TimeExitSpec.model_validate({"mode": "SESSION_CLOSE", "session": "MOON"})

    def test_the_semantic_check_still_runs_at_build_time(self) -> None:
        # The spec layer alone can't know MAX_BARS_HELD needs max_bars_held —
        # that's TimeExit's own __init__ check, exercised through build_plan.
        preset = ExitPresetSpec.model_validate(
            minimal_preset(rules=[{"kind": "TIME_EXIT", "mode": "MAX_BARS_HELD"}])
        )
        with pytest.raises(ValidationError, match="test-preset"):
            build_plan(preset)


class TestBuildPlan:
    def test_a_minimal_preset_builds_a_plan_with_only_the_protective_stop(self) -> None:
        preset = ExitPresetSpec.model_validate(minimal_preset())
        plan = build_plan(preset)
        assert plan.exit_id == "test-preset"
        assert [rule.name for rule in plan.rules] == ["protective_stop"]
        assert plan.stop_modifiers == ()

    def test_rules_and_modifiers_resolve_to_the_right_runtime_types(self) -> None:
        preset = ExitPresetSpec.model_validate(
            minimal_preset(
                rules=[{"kind": "FIXED_RR", "r_multiple": 2.0}],
                stop_modifiers=[{"kind": "ATR_STOP", "period": 10, "multiple": 1.0}],
            )
        )
        plan = build_plan(preset)
        assert isinstance(plan.rules[0], ProtectiveStop)
        assert isinstance(plan.rules[1], FixedRR)
        assert isinstance(plan.stop_modifiers[0], ATRStop)

    def test_an_intrabar_policy_is_honoured(self) -> None:
        preset = ExitPresetSpec.model_validate(minimal_preset(intrabar_policy="OPTIMISTIC"))
        plan = build_plan(preset)
        assert plan.intrabar_policy is IntrabarPolicy.OPTIMISTIC

    def test_a_partial_ladder_summing_past_100_percent_fails_at_build_with_the_preset_id(
        self,
    ) -> None:
        # Each rung is individually valid (fraction in (0, 1)); only together
        # do they overcommit the position. PartialClose's own constructor
        # catches it, and build_plan must name the preset when re-raising.
        preset = ExitPresetSpec.model_validate(
            minimal_preset(
                rules=[
                    {
                        "kind": "PARTIAL_CLOSE",
                        "rungs": [
                            {"r_multiple": 1.0, "fraction": "0.6"},
                            {"r_multiple": 2.0, "fraction": "0.6"},
                        ],
                    }
                ]
            )
        )
        with pytest.raises(ValidationError, match="test-preset"):
            build_plan(preset)


class TestLoadLibrary:
    def test_the_bundled_library_loads(self) -> None:
        library = load_library()
        assert isinstance(library, ExitLibrary)
        assert len(library) == 8

    def test_every_bundled_preset_id_is_present(self) -> None:
        assert set(load_library()) == BUNDLED_PRESET_IDS

    def test_every_bundled_preset_carries_a_protective_stop(self) -> None:
        library = load_library()
        for exit_id in library:
            plan = library[exit_id]
            assert any(isinstance(rule, ProtectiveStop) for rule in plan.rules), exit_id

    def test_a_missing_file_raises_with_the_path_named(self) -> None:
        missing = Path("/nonexistent/exit/library.json")
        with pytest.raises(ValidationError, match=str(missing)):
            load_library(missing)

    def test_malformed_json_raises_clearly(self, tmp_path: Path) -> None:
        broken = tmp_path / "library.json"
        broken.write_text("{not valid json", encoding="utf-8")
        with pytest.raises(ValidationError, match="malformed exit library"):
            load_library(broken)

    def test_a_preset_missing_its_protective_stop_raises_clearly(self, tmp_path: Path) -> None:
        broken = tmp_path / "library.json"
        broken.write_text('{"presets": [{"id": "no-stop", "name": "No Stop"}]}', encoding="utf-8")
        with pytest.raises(ValidationError, match="malformed exit library"):
            load_library(broken)

    def test_a_semantically_broken_preset_names_itself_in_the_error(self, tmp_path: Path) -> None:
        broken = tmp_path / "library.json"
        broken.write_text(
            """
            {"presets": [{
                "id": "overcommitted",
                "name": "Overcommitted",
                "protective_stop": {"kind": "PROTECTIVE_STOP"},
                "rules": [{
                    "kind": "PARTIAL_CLOSE",
                    "rungs": [
                        {"r_multiple": 1.0, "fraction": "0.7"},
                        {"r_multiple": 2.0, "fraction": "0.7"}
                    ]
                }]
            }]}
            """,
            encoding="utf-8",
        )
        with pytest.raises(ValidationError, match="overcommitted"):
            load_library(broken)

    def test_duplicate_preset_ids_are_rejected(self, tmp_path: Path) -> None:
        one = minimal_preset(id="dupe")
        other = minimal_preset(id="dupe", name="Also Dupe")
        broken = tmp_path / "library.json"
        spec = ExitLibrarySpec.model_validate({"presets": [one, other]})
        broken.write_text(spec.model_dump_json(), encoding="utf-8")
        with pytest.raises(ValidationError, match="duplicate"):
            load_library(broken)

    def test_the_default_path_points_at_the_bundled_file(self) -> None:
        assert DEFAULT_LIBRARY_PATH.name == "library.json"
        assert DEFAULT_LIBRARY_PATH.exists()


class TestExitLibraryApi:
    def test_contains_getitem_iter_len(self) -> None:
        library = load_library()
        assert "conservative_2r" in library
        assert "no_such_preset" not in library
        assert len(library) == 8
        assert set(iter(library)) == BUNDLED_PRESET_IDS
        assert library["conservative_2r"].exit_id == "conservative_2r"

    def test_getitem_on_an_unknown_id_raises_with_it_named(self) -> None:
        library = load_library()
        with pytest.raises(ValidationError, match="no_such_preset"):
            library["no_such_preset"]

    def test_get_returns_none_for_an_unknown_id(self) -> None:
        library = load_library()
        assert library.get("no_such_preset") is None
        assert library.get("conservative_2r") is not None

    def test_ids_matches_known_exit_ids(self) -> None:
        library = load_library()
        assert library.ids == known_exit_ids()
        assert library.ids == BUNDLED_PRESET_IDS


class TestSmallestFractionPerPreset:
    """DoD: the smallest-fraction hooks are correct for every preset.

    ``smallest_closing_fraction()`` is exactly what the Risk Engine calls at
    position-open time to decide whether a preset survives the instrument's
    minimum lot.
    """

    @pytest.mark.parametrize(
        "exit_id",
        sorted(BUNDLED_PRESET_IDS - {"swing_partial_ladder"}),
    )
    def test_presets_without_a_ladder_request_no_partials(self, exit_id: str) -> None:
        library = load_library()
        assert library[exit_id].smallest_partial_fraction() is None
        assert library[exit_id].smallest_closing_fraction() == Decimal("1")

    def test_swing_partial_ladder_reports_its_smallest_rung(self) -> None:
        library = load_library()
        assert library["swing_partial_ladder"].smallest_partial_fraction() == Decimal("0.25")

    def test_swing_partial_ladders_residual_is_no_smaller_than_its_rungs(self) -> None:
        # 0.5 + 0.25 leaves 0.25, equal to the smallest rung, so the bundled
        # ladder is not the tail case -- but it is asserted rather than assumed.
        library = load_library()
        assert library["swing_partial_ladder"].smallest_closing_fraction() == Decimal("0.25")


class TestKnownExitIds:
    def test_matches_the_loaded_librarys_ids(self) -> None:
        assert known_exit_ids() == load_library().ids

    def test_a_broken_library_fails_loudly_rather_than_reporting_partial_ids(
        self, tmp_path: Path
    ) -> None:
        broken = tmp_path / "library.json"
        broken.write_text("{not valid json", encoding="utf-8")
        with pytest.raises(ValidationError):
            known_exit_ids(broken)
