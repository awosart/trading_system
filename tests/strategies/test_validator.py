"""Tests for semantic validation layered above the StrategySpec contract."""

import json
from pathlib import Path
from typing import Any

import pytest

from trading_system.strategies.schema import FeatureRef, StrategySpec
from trading_system.strategies.validator import (
    Severity,
    check_exit_ref,
    check_feature_reference,
    check_feature_references,
    check_regime_trigger_contradiction,
    check_timeframe_order,
    check_unique_ids,
    load_spec,
    validate_paths,
    validate_spec,
)

from .conftest import EXAMPLE_FILES, feature_ref, leaf


class TestCheckFeatureReference:
    """Checks a single :class:`FeatureRef` against the indicator registry.

    Covers the three failure modes the old ``"feature:<name>"`` string
    heuristic could not catch: an indicator name that was never registered, a
    parameter that indicator doesn't accept, and a missing channel on a
    multi-output indicator.
    """

    def test_rejects_unknown_indicator_name(self) -> None:
        # A naive migration from the old string convention might carry the
        # whole "ema_500" name into the new `indicator` field verbatim. The
        # old prefix heuristic would have silently matched this against
        # kind "ema"; the registry has no such key, so this must be rejected.
        ref = FeatureRef(indicator="ema_500")
        issues = check_feature_reference(ref)
        assert [issue.code for issue in issues] == ["unknown_indicator"]
        assert issues[0].severity is Severity.ERROR

    def test_accepts_a_valid_indicator_and_params(self) -> None:
        ref = FeatureRef(indicator="ema", params={"period": 500})
        assert check_feature_reference(ref) == []

    def test_rejects_nonexistent_parameter(self) -> None:
        ref = FeatureRef(indicator="ema", params={"period": 14, "bogus_param": 1})
        issues = check_feature_reference(ref)
        assert [issue.code for issue in issues] == ["invalid_indicator_params"]

    def test_rejects_out_of_bounds_parameter(self) -> None:
        ref = FeatureRef(indicator="ema", params={"period": -5})
        issues = check_feature_reference(ref)
        assert [issue.code for issue in issues] == ["invalid_indicator_params"]

    def test_rejects_multi_output_indicator_without_channel(self) -> None:
        ref = FeatureRef(indicator="macd")
        issues = check_feature_reference(ref)
        assert [issue.code for issue in issues] == ["missing_channel"]

    def test_rejects_unknown_channel(self) -> None:
        ref = FeatureRef(indicator="macd", channel="bogus_channel")
        issues = check_feature_reference(ref)
        assert [issue.code for issue in issues] == ["unknown_channel"]

    def test_accepts_multi_output_indicator_with_valid_channel(self) -> None:
        ref = FeatureRef(indicator="macd", channel="signal")
        assert check_feature_reference(ref) == []

    def test_rejects_single_output_indicator_with_a_channel(self) -> None:
        ref = FeatureRef(indicator="rsi", params={"period": 14}, channel="value")
        issues = check_feature_reference(ref)
        assert [issue.code for issue in issues] == ["unexpected_channel"]


class TestCheckFeatureReferences:
    def test_flags_unregistered_indicator_in_trigger(
        self, minimal_spec_dict: dict[str, Any]
    ) -> None:
        minimal_spec_dict["entry"]["trigger"]["right"] = feature_ref("not_a_real_indicator")
        spec = StrategySpec.model_validate(minimal_spec_dict)
        issues = check_feature_references(spec)
        assert [issue.code for issue in issues] == ["unknown_indicator"]

    def test_accepts_a_registered_indicator(self, minimal_spec_dict: dict[str, Any]) -> None:
        spec = StrategySpec.model_validate(minimal_spec_dict)  # references ema, period=50
        assert check_feature_references(spec) == []

    def test_flags_unregistered_indicator_in_volatility_range_filter(
        self, minimal_spec_dict: dict[str, Any]
    ) -> None:
        minimal_spec_dict["filters"] = [
            {
                "kind": "volatility_range",
                "feature": feature_ref("bogus_indicator"),
                "min_value": 0.0,
            }
        ]
        spec = StrategySpec.model_validate(minimal_spec_dict)
        issues = check_feature_references(spec)
        assert [issue.code for issue in issues] == ["unknown_indicator"]


class TestCheckExitRef:
    def test_skips_when_no_registry_supplied(self, minimal_spec_dict: dict[str, Any]) -> None:
        spec = StrategySpec.model_validate(minimal_spec_dict)
        assert check_exit_ref(spec, known_exit_ids=None) == []

    def test_flags_unregistered_exit_ref(self, minimal_spec_dict: dict[str, Any]) -> None:
        spec = StrategySpec.model_validate(minimal_spec_dict)
        issues = check_exit_ref(spec, known_exit_ids={"some-other-exit"})
        assert [issue.code for issue in issues] == ["unknown_exit_ref"]

    def test_accepts_registered_exit_ref(self, minimal_spec_dict: dict[str, Any]) -> None:
        spec = StrategySpec.model_validate(minimal_spec_dict)
        assert check_exit_ref(spec, known_exit_ids={spec.exit_ref}) == []


class TestCheckTimeframeOrder:
    def test_entry_tf_no_coarser_than_signal_tf_is_clean(
        self, minimal_spec_dict: dict[str, Any]
    ) -> None:
        minimal_spec_dict["timeframes"] = {
            "signal_tf": "H4",
            "htf_filter_tf": "D1",
            "entry_tf": "H1",
        }
        spec = StrategySpec.model_validate(minimal_spec_dict)
        assert check_timeframe_order(spec) == []

    def test_htf_filter_tf_not_strictly_coarser_is_flagged(
        self, minimal_spec_dict: dict[str, Any]
    ) -> None:
        minimal_spec_dict["timeframes"] = {
            "signal_tf": "H4",
            "htf_filter_tf": "H4",
            "entry_tf": "H1",
        }
        spec = StrategySpec.model_validate(minimal_spec_dict)
        issues = check_timeframe_order(spec)
        assert [issue.code for issue in issues] == ["timeframe_order"]

    def test_entry_tf_coarser_than_signal_tf_is_flagged(
        self, minimal_spec_dict: dict[str, Any]
    ) -> None:
        minimal_spec_dict["timeframes"] = {"signal_tf": "H4", "entry_tf": "D1"}
        spec = StrategySpec.model_validate(minimal_spec_dict)
        issues = check_timeframe_order(spec)
        assert [issue.code for issue in issues] == ["timeframe_order"]

    def test_absent_htf_filter_is_not_checked(self, minimal_spec_dict: dict[str, Any]) -> None:
        minimal_spec_dict["timeframes"] = {"signal_tf": "M15", "entry_tf": "M15"}
        spec = StrategySpec.model_validate(minimal_spec_dict)
        assert check_timeframe_order(spec) == []


class TestCheckRegimeTriggerContradiction:
    def test_warns_on_range_with_breakout_shaped_trigger(
        self, minimal_spec_dict: dict[str, Any]
    ) -> None:
        minimal_spec_dict["market_regimes"] = ["RANGE"]
        minimal_spec_dict["entry"]["trigger"] = leaf(
            "cross_above", "price:close", feature_ref("donchian", {"period": 20}, "upper")
        )
        spec = StrategySpec.model_validate(minimal_spec_dict)
        issues = check_regime_trigger_contradiction(spec)
        assert [issue.code for issue in issues] == ["range_breakout_conflict"]
        assert issues[0].severity is Severity.WARNING

    def test_silent_without_range_regime(self, minimal_spec_dict: dict[str, Any]) -> None:
        minimal_spec_dict["market_regimes"] = ["TREND_UP"]
        minimal_spec_dict["entry"]["trigger"] = leaf(
            "cross_above", "price:close", feature_ref("donchian", {"period": 20}, "upper")
        )
        spec = StrategySpec.model_validate(minimal_spec_dict)
        assert check_regime_trigger_contradiction(spec) == []

    def test_silent_for_range_without_breakout_shaped_trigger(
        self, minimal_spec_dict: dict[str, Any]
    ) -> None:
        minimal_spec_dict["market_regimes"] = ["RANGE"]
        spec = StrategySpec.model_validate(minimal_spec_dict)  # trigger uses "gt"
        assert check_regime_trigger_contradiction(spec) == []


class TestValidateSpec:
    def test_minimal_spec_is_clean(self, minimal_spec_dict: dict[str, Any]) -> None:
        spec = StrategySpec.model_validate(minimal_spec_dict)
        assert validate_spec(spec) == []

    @pytest.mark.parametrize("path", EXAMPLE_FILES, ids=lambda p: p.stem)
    def test_worked_examples_are_clean(self, path: Path) -> None:
        spec = StrategySpec.model_validate_json(path.read_text(encoding="utf-8"))
        assert validate_spec(spec) == []


class TestCheckUniqueIds:
    def test_flags_duplicate_ids(self, minimal_spec_dict: dict[str, Any]) -> None:
        spec_a = StrategySpec.model_validate(minimal_spec_dict)
        spec_b = StrategySpec.model_validate(minimal_spec_dict)
        issues = check_unique_ids([spec_a, spec_b])
        assert [issue.code for issue in issues] == ["duplicate_id"]

    def test_distinct_ids_are_clean(self, minimal_spec_dict: dict[str, Any]) -> None:
        spec_a = StrategySpec.model_validate(minimal_spec_dict)
        other = dict(minimal_spec_dict)
        other["id"] = "another-strategy"
        spec_b = StrategySpec.model_validate(other)
        assert check_unique_ids([spec_a, spec_b]) == []


class TestLoadSpec:
    def test_reports_readable_syntax_errors(self, tmp_path: Path) -> None:
        broken = tmp_path / "broken.json"
        broken.write_text(json.dumps({"id": "not a slug!", "name": "x"}), encoding="utf-8")
        parsed = load_spec(broken)
        assert parsed.spec is None
        assert parsed.issues
        assert all(issue.code == "schema_error" for issue in parsed.issues)
        assert all(issue.message for issue in parsed.issues)

    def test_reports_read_error_for_missing_file(self, tmp_path: Path) -> None:
        parsed = load_spec(tmp_path / "does_not_exist.json")
        assert parsed.spec is None
        assert parsed.issues[0].code == "read_error"

    def test_loads_a_well_formed_file(
        self, tmp_path: Path, minimal_spec_dict: dict[str, Any]
    ) -> None:
        path = tmp_path / "spec.json"
        path.write_text(json.dumps(minimal_spec_dict), encoding="utf-8")
        parsed = load_spec(path)
        assert parsed.issues == ()
        assert parsed.spec is not None
        assert parsed.spec.id == minimal_spec_dict["id"]


class TestValidatePaths:
    def test_flags_duplicate_ids_across_files(
        self, tmp_path: Path, minimal_spec_dict: dict[str, Any]
    ) -> None:
        path_a = tmp_path / "a.json"
        path_b = tmp_path / "b.json"
        path_a.write_text(json.dumps(minimal_spec_dict), encoding="utf-8")
        path_b.write_text(json.dumps(minimal_spec_dict), encoding="utf-8")

        results = validate_paths([path_a, path_b])

        assert any(issue.code == "duplicate_id" for issue in results[path_a])
        assert any(issue.code == "duplicate_id" for issue in results[path_b])

    def test_worked_examples_validate_without_errors(self) -> None:
        results = validate_paths(EXAMPLE_FILES)
        for path, issues in results.items():
            errors = [issue for issue in issues if issue.severity is Severity.ERROR]
            assert not errors, f"{path}: {errors}"
