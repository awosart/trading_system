"""Tests for the StrategySpec pydantic contract: construction, invariants, round trip."""

import json
from typing import Any

import pytest
from jsonschema import Draft202012Validator
from pydantic import ValidationError

from trading_system.strategies.schema import (
    SCHEMA_JSON_PATH,
    AllOf,
    AnyOf,
    ConditionOp,
    FeatureRef,
    LeafCondition,
    Not,
    StrategySpec,
    iter_leaf_conditions,
    operand_feature_ref,
    operand_price_field,
    strategy_json_schema,
)

from .conftest import EXAMPLE_FILES, feature_ref, leaf


class TestRoundTrip:
    def test_model_survives_json_round_trip(self, minimal_spec_dict: dict[str, Any]) -> None:
        spec = StrategySpec.model_validate(minimal_spec_dict)
        restored = StrategySpec.model_validate_json(spec.model_dump_json())
        assert restored == spec

    @pytest.mark.parametrize("path", EXAMPLE_FILES, ids=lambda p: p.stem)
    def test_example_survives_json_round_trip(self, path: Any) -> None:
        spec = StrategySpec.model_validate_json(path.read_text())
        restored = StrategySpec.model_validate_json(spec.model_dump_json())
        assert restored == spec


class TestIdAndVersion:
    @pytest.mark.parametrize(
        "bad_id", ["Bad_ID", "UPPER", "trailing-", "-leading", "double--dash", "has space"]
    )
    def test_rejects_non_slug_id(self, minimal_spec_dict: dict[str, Any], bad_id: str) -> None:
        minimal_spec_dict["id"] = bad_id
        with pytest.raises(ValidationError, match="slug"):
            StrategySpec.model_validate(minimal_spec_dict)

    @pytest.mark.parametrize("bad_version", ["1.0", "v1.0.0", "1.0.0-beta", "1.0.x"])
    def test_rejects_non_semver_version(
        self, minimal_spec_dict: dict[str, Any], bad_version: str
    ) -> None:
        minimal_spec_dict["version"] = bad_version
        with pytest.raises(ValidationError, match="semver"):
            StrategySpec.model_validate(minimal_spec_dict)


class TestOperands:
    @pytest.mark.parametrize("bad_ref", ["price:", "price:bogus", "just_a_string"])
    def test_rejects_malformed_price_ref(
        self, minimal_spec_dict: dict[str, Any], bad_ref: str
    ) -> None:
        minimal_spec_dict["entry"]["trigger"]["right"] = bad_ref
        with pytest.raises(ValidationError):
            StrategySpec.model_validate(minimal_spec_dict)

    def test_accepts_numeric_constant(self, minimal_spec_dict: dict[str, Any]) -> None:
        minimal_spec_dict["entry"]["trigger"] = leaf("gt", "price:close", 100.5)
        spec = StrategySpec.model_validate(minimal_spec_dict)
        assert spec.entry.trigger.right == 100.5  # type: ignore[union-attr]

    def test_accepts_structural_feature_ref(self, minimal_spec_dict: dict[str, Any]) -> None:
        minimal_spec_dict["entry"]["trigger"] = leaf(
            "gt", "price:close", feature_ref("rsi", {"period": 14})
        )
        spec = StrategySpec.model_validate(minimal_spec_dict)
        expected = FeatureRef(indicator="rsi", params={"period": 14})
        assert spec.entry.trigger.right == expected  # type: ignore[union-attr]

    def test_feature_ref_rejects_unknown_field(self) -> None:
        with pytest.raises(ValidationError, match="bogus"):
            FeatureRef.model_validate({"indicator": "rsi", "bogus": 1})

    def test_feature_ref_rejects_empty_channel(self) -> None:
        with pytest.raises(ValidationError):
            FeatureRef.model_validate({"indicator": "rsi", "channel": ""})

    def test_operand_feature_ref(self) -> None:
        ref = FeatureRef(indicator="rsi", params={"period": 14})
        assert operand_feature_ref(ref) is ref
        assert operand_feature_ref("price:close") is None
        assert operand_feature_ref(70.0) is None

    def test_operand_price_field(self) -> None:
        assert operand_price_field("price:close") == "close"
        assert operand_price_field(FeatureRef(indicator="rsi")) is None
        assert operand_price_field(70.0) is None


class TestConditionTree:
    def test_iter_leaf_conditions_walks_nested_tree(self) -> None:
        tree = AnyOf(
            conditions=[
                AllOf(
                    conditions=[
                        LeafCondition(op=ConditionOp.GT, left="price:close", right=1.0),
                        Not(
                            condition=LeafCondition(
                                op=ConditionOp.LT,
                                left=FeatureRef(indicator="rsi", params={"period": 14}),
                                right=30.0,
                            )
                        ),
                    ]
                ),
                LeafCondition(
                    op=ConditionOp.CROSS_ABOVE,
                    left="price:close",
                    right=FeatureRef(indicator="ema", params={"period": 50}),
                ),
            ]
        )
        ops = {found.op for found in iter_leaf_conditions(tree)}
        assert ops == {ConditionOp.GT, ConditionOp.LT, ConditionOp.CROSS_ABOVE}

    def test_leaf_requires_explicit_type_tag(self) -> None:
        with pytest.raises(ValidationError, match="type"):
            StrategySpec.model_validate(
                {"op": "gt", "left": "price:close", "right": 1.0}  # missing "type"
            )


class TestEntryConsistency:
    def test_confirmation_requires_positive_window(self, minimal_spec_dict: dict[str, Any]) -> None:
        minimal_spec_dict["entry"]["confirmation"] = [
            leaf("gt", feature_ref("rsi", {"period": 14}), 50.0)
        ]
        with pytest.raises(ValidationError, match="confirmation_window_bars"):
            StrategySpec.model_validate(minimal_spec_dict)

    def test_confirmation_window_without_conditions_rejected(
        self, minimal_spec_dict: dict[str, Any]
    ) -> None:
        minimal_spec_dict["entry"]["confirmation_window_bars"] = 3
        with pytest.raises(ValidationError, match="confirmation_window_bars"):
            StrategySpec.model_validate(minimal_spec_dict)

    def test_confirmation_with_matching_window_is_accepted(
        self, minimal_spec_dict: dict[str, Any]
    ) -> None:
        minimal_spec_dict["entry"]["confirmation"] = [
            leaf("gt", feature_ref("rsi", {"period": 14}), 50.0)
        ]
        minimal_spec_dict["entry"]["confirmation_window_bars"] = 3
        spec = StrategySpec.model_validate(minimal_spec_dict)
        assert spec.entry.confirmation_window_bars == 3

    def test_invalidation_requires_condition_or_price_level(
        self, minimal_spec_dict: dict[str, Any]
    ) -> None:
        minimal_spec_dict["entry"]["invalidation"] = {}
        with pytest.raises(ValidationError, match="invalidation"):
            StrategySpec.model_validate(minimal_spec_dict)


class TestInstrumentsAndFilters:
    def test_rejects_symbol_both_allowed_and_denied(
        self, minimal_spec_dict: dict[str, Any]
    ) -> None:
        minimal_spec_dict["instruments"]["allowed_symbols"] = ["EURUSD"]
        minimal_spec_dict["instruments"]["denied_symbols"] = ["EURUSD"]
        with pytest.raises(ValidationError, match="both allowed and denied"):
            StrategySpec.model_validate(minimal_spec_dict)

    def test_volatility_range_filter_requires_a_bound(
        self, minimal_spec_dict: dict[str, Any]
    ) -> None:
        minimal_spec_dict["filters"] = [
            {"kind": "volatility_range", "feature": feature_ref("atr", {"period": 14})}
        ]
        with pytest.raises(ValidationError, match="min_value"):
            StrategySpec.model_validate(minimal_spec_dict)


class TestStrictConfig:
    def test_rejects_unknown_top_level_field(self, minimal_spec_dict: dict[str, Any]) -> None:
        minimal_spec_dict["unexpected_field"] = "oops"
        with pytest.raises(ValidationError, match="unexpected_field"):
            StrategySpec.model_validate(minimal_spec_dict)

    def test_spec_is_frozen(self, minimal_spec_dict: dict[str, Any]) -> None:
        spec = StrategySpec.model_validate(minimal_spec_dict)
        with pytest.raises(ValidationError):
            spec.name = "changed"  # type: ignore[misc]


class TestJsonSchemaExport:
    def test_generated_schema_is_valid_draft_2020_12(self) -> None:
        Draft202012Validator.check_schema(strategy_json_schema())

    def test_committed_schema_matches_generated_schema(self) -> None:
        committed = json.loads(SCHEMA_JSON_PATH.read_text(encoding="utf-8"))
        assert committed == strategy_json_schema()

    @pytest.mark.parametrize("path", EXAMPLE_FILES, ids=lambda p: p.stem)
    def test_example_conforms_to_json_schema(self, path: Any) -> None:
        validator = Draft202012Validator(strategy_json_schema())
        validator.validate(json.loads(path.read_text(encoding="utf-8")))
