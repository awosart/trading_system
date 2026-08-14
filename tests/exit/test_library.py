"""The Exit DB: preset specs, the loader, and the bundled presets.

DoD items covered here: every bundled preset loads and validates; ``load_library``
raises a clear, preset-naming error on a broken one; the smallest-fraction hooks
answers correctly for every preset, ladder or not.
"""

import json
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

#: The eight presets the Exit DB opened with: one shape each, no family.
SHAPE_PRESET_IDS = frozenset(
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

#: The risk/reward ladder. Ids rather than a parameter because a strategy names
#: its exit through ``exit_ref`` and has no way to pass a number to it, so a
#: target of 3R can only exist as something with a name.
LADDER_PRESET_IDS = frozenset(
    {
        "rr_1r",
        "rr_1_2r",
        "rr_1_25r",
        "rr_1_5r",
        "rr_2_5r",
        "rr_3r",
        "rr_4r",
        "rr_5r",
        "rr_6r",
        "rr_8r",
        "rr_10r",
        "rr_3r_breakeven",
        "rr_5r_breakeven",
        "rr_10r_breakeven",
        "rr_ladder_1r_3r",
        "rr_ladder_1r_10r",
    }
)

#: Presets that close part of the position before the final exit.
PARTIAL_PRESET_IDS = frozenset({"swing_partial_ladder", "rr_ladder_1r_3r", "rr_ladder_1r_10r"})

BUNDLED_PRESET_IDS = SHAPE_PRESET_IDS | LADDER_PRESET_IDS


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

    def test_with_no_override_the_presets_own_policy_governs(self) -> None:
        """The fallback a caller with no run gets — ``ExitLibrary``, this test."""
        preset = ExitPresetSpec.model_validate(minimal_preset(intrabar_policy="OPTIMISTIC"))
        plan = build_plan(preset, intrabar_policy=None)
        assert plan.intrabar_policy is IntrabarPolicy.OPTIMISTIC

    def test_an_explicit_override_beats_the_presets_own_policy(self) -> None:
        """What a run passes; see test_orchestrator.py for the end-to-end version."""
        preset = ExitPresetSpec.model_validate(minimal_preset(intrabar_policy="OPTIMISTIC"))
        plan = build_plan(preset, intrabar_policy=IntrabarPolicy.PESSIMISTIC)
        assert plan.intrabar_policy is IntrabarPolicy.PESSIMISTIC

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
        assert len(library) == len(BUNDLED_PRESET_IDS) == 24

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
        assert len(library) == len(BUNDLED_PRESET_IDS)
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

    @pytest.mark.parametrize("exit_id", sorted(BUNDLED_PRESET_IDS - PARTIAL_PRESET_IDS))
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

    def test_the_1r_3r_ladder_leaves_the_larger_half_to_run(self) -> None:
        library = load_library()
        plan = library["rr_ladder_1r_3r"]
        assert plan.smallest_partial_fraction() == Decimal("0.5")
        # One rung of a half leaves a half: the residual is the binding one only
        # when it is smaller than every rung, which here it is not.
        assert plan.smallest_closing_fraction() == Decimal("0.5")

    def test_the_1r_10r_ladders_residual_is_the_binding_fraction(self) -> None:
        # 0.5 + 0.25 leaves 0.25 -- the same size as the smallest rung, so this
        # ladder needs a quarter-lot to be executable, not a half.
        library = load_library()
        plan = library["rr_ladder_1r_10r"]
        assert plan.smallest_partial_fraction() == Decimal("0.25")
        assert plan.smallest_closing_fraction() == Decimal("0.25")


class TestRiskRewardLadder:
    """The 1R..10R family, and the two things a family can get wrong.

    A ladder of ids is how a fixed target gets expressed at all: ``exit_ref`` is
    a name, so "take profit at 3R" has to *be* a preset. That makes two failure
    modes possible which a library of one-off shapes did not have — a rung that
    silently duplicates another preset, and a gap nobody meant to leave.
    """

    #: Every rung, and the target it must resolve to.
    RUNGS = {
        "rr_1r": 1.0,
        "rr_1_2r": 1.2,
        "rr_1_25r": 1.25,
        "rr_1_5r": 1.5,
        "conservative_2r": 2.0,
        "rr_2_5r": 2.5,
        "rr_3r": 3.0,
        "rr_4r": 4.0,
        "rr_5r": 5.0,
        "rr_6r": 6.0,
        "rr_8r": 8.0,
        "rr_10r": 10.0,
    }

    @pytest.mark.parametrize(("exit_id", "r_multiple"), sorted(RUNGS.items()))
    def test_each_rung_takes_profit_where_its_name_says(
        self, exit_id: str, r_multiple: float
    ) -> None:
        plan = load_library()[exit_id]
        targets = [rule for rule in plan.rules if isinstance(rule, FixedRR)]
        assert [rule.r_multiple for rule in targets] == [r_multiple]

    def test_the_ladder_has_no_2r_rung_of_its_own(self) -> None:
        """2R is ``conservative_2r``, and a second id building it would be a duplicate.

        Asserted rather than left as an absence, so the gap reads as a decision
        instead of an oversight the next person "fixes" by adding ``rr_2r``.
        """
        assert "rr_2r" not in load_library().ids
        assert "conservative_2r" in self.RUNGS

    def test_no_two_bundled_presets_are_configured_alike(self) -> None:
        """Ids are free; two ids that build the same plan are not.

        Compared on the configuration rather than the built objects because that
        is the thing an author edits, and the thing a copy-paste rung would
        leave identical.
        """
        raw = json.loads(DEFAULT_LIBRARY_PATH.read_text(encoding="utf-8"))
        seen: dict[str, str] = {}
        for preset in raw["presets"]:
            behaviour = json.dumps(
                {k: v for k, v in preset.items() if k not in {"id", "name", "description"}},
                sort_keys=True,
            )
            assert behaviour not in seen, f"{preset['id']} duplicates {seen.get(behaviour)}"
            seen[behaviour] = preset["id"]

    def test_the_breakeven_variants_differ_from_their_bare_rung_by_one_modifier(self) -> None:
        """The pairing is what makes the comparison attributable.

        ``rr_10r`` against ``rr_10r_breakeven`` isolates the breakeven move: if
        anything else differed, the difference between their results would be
        explained by more than one thing.
        """
        library = load_library()
        for bare, protected in (("rr_3r", "rr_3r_breakeven"), ("rr_10r", "rr_10r_breakeven")):
            bare_targets = [r.r_multiple for r in library[bare].rules if isinstance(r, FixedRR)]
            protected_targets = [
                r.r_multiple for r in library[protected].rules if isinstance(r, FixedRR)
            ]
            assert bare_targets == protected_targets
            assert not library[bare].stop_modifiers
            assert len(library[protected].stop_modifiers) == 1


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
