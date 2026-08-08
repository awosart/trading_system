"""Deriving a search space from a spec: nobody writes a JSON pointer."""

from pathlib import Path
from typing import Any

import pytest

from trading_system.core.exceptions import ValidationError
from trading_system.features.registry import build_indicator
from trading_system.strategies.schema import StrategySpec
from trading_system.validation.optimization import SearchSpace
from trading_system.validation.space_builder import (
    NO_LADDER,
    OUTPUT_RANGES,
    build_candidates,
    build_space_document,
    infer_constraints,
    multiple_ladder,
    period_ladder,
    prune,
    render,
    shift_ladder,
    threshold_ladder,
    verify,
)

ROOT = Path(__file__).resolve().parents[2]
CONFIGS = ROOT / "configs" / "strategies"


def load(name: str) -> StrategySpec:
    """Load a shipped strategy config by filename stem."""
    return StrategySpec.model_validate_json((CONFIGS / f"{name}.json").read_text(encoding="utf-8"))


def generated(spec: StrategySpec) -> dict[str, Any]:
    """The pruned, verified space document for ``spec``."""
    document, _ = prune(spec, build_space_document(spec))
    verify(spec, document)
    return document


def axis_named(document: dict[str, Any], name: str) -> dict[str, Any]:
    """One axis out of a space document, by name."""
    found = [item for item in document["axes"] if item["name"] == name]
    assert found, f"no axis {name!r}; have {[item['name'] for item in document['axes']]}"
    return found[0]


class TestPointersAreDerivedNotTyped:
    def test_channel_breakout_reproduces_the_hand_written_pointers(self) -> None:
        # The hand-written space for this strategy was the motivating case:
        # swing_lookback alone needed four pointers, two per leg, and a missed
        # one would have made the trigger and the invalidation disagree.
        #
        # Pinned here rather than read from configs/spaces/, because that file
        # is now generated: comparing the generator against its own output would
        # assert nothing. These are the pointers as a person actually typed them.
        hand_written = {
            "swing_lookback": {
                "/entries/0/trigger/conditions/0/right/params/lookback",
                "/entries/0/invalidation/price_level/params/lookback",
                "/entries/1/trigger/conditions/0/right/params/lookback",
                "/entries/1/invalidation/price_level/params/lookback",
            },
            "ema_period": {
                "/entries/0/trigger/conditions/1/right/params/period",
                "/entries/1/trigger/conditions/1/right/params/period",
            },
            "multiple": {"/risk_profile/stop_reference/multiple"},
        }
        document = generated(load("channel_breakout_h4"))
        for name, pointers in hand_written.items():
            assert set(axis_named(document, name)["paths"]) == pointers, (
                f"{name} does not cover the positions a person wrote by hand"
            )

    def test_it_finds_exactly_seven_pointers_across_those_three_axes(self) -> None:
        document = generated(load("channel_breakout_h4"))
        total = sum(
            len(axis_named(document, name)["paths"])
            for name in ("swing_lookback", "ema_period", "multiple")
        )
        assert total == 7

    def test_both_legs_of_a_two_sided_strategy_collapse_into_one_axis(self) -> None:
        # The whole point: a symmetric strategy carries the same value on both
        # sides, so one axis moves both and no one has to remember the mirror.
        document = generated(load("channel_breakout_h4"))
        paths = axis_named(document, "swing_lookback")["paths"]
        assert sum(1 for path in paths if path.startswith("/entries/0/")) == 2
        assert sum(1 for path in paths if path.startswith("/entries/1/")) == 2

    @pytest.mark.parametrize(
        "name",
        ["channel_breakout_h4", "bollinger_fade_h4", "volume_thrust_h4", "ema_pullback_h1"],
    )
    def test_no_position_is_claimed_by_two_axes(self, name: str) -> None:
        # num_std is both an indicator parameter and a name in MULTIPLE_NAMES,
        # so bollinger_fade produced two axes over the same pointer — a search
        # would then move one number along two independent dials.
        claimed: dict[str, str] = {}
        for axis in build_candidates(load(name)):
            for pointer in axis.pointers:
                assert pointer not in claimed, (
                    f"{pointer} claimed by both {claimed.get(pointer)} and {axis.name}"
                )
                claimed[pointer] = axis.name

    def test_every_pointer_is_a_valid_json_pointer(self) -> None:
        for name in ("channel_breakout_h4", "bollinger_fade_h4", "volume_thrust_h4"):
            for axis in generated(load(name))["axes"]:
                for path in axis["paths"]:
                    assert path.startswith("/"), path


class TestRolesSplitByCurrentValue:
    def test_two_periods_of_one_indicator_stay_separate(self) -> None:
        # ema_pullback carries ema.period=50 (filter and invalidation) and
        # ema.period=20 (confirmation). Fusing them would tune one dial where
        # the author wrote two.
        document = generated(load("ema_pullback_h1"))
        slow = axis_named(document, "ema_period_50")
        fast = axis_named(document, "ema_period_20")
        assert set(slow["paths"]).isdisjoint(fast["paths"])

    def test_the_slow_period_finds_both_of_its_occurrences(self) -> None:
        # The trigger and the invalidation are derived from the same EMA; a
        # space that moved only one would filter on EMA(65) and be invalidated
        # by EMA(50).
        document = generated(load("ema_pullback_h1"))
        assert len(axis_named(document, "ema_period_50")["paths"]) == 2

    def test_a_single_occurrence_keeps_an_unqualified_name(self) -> None:
        document = generated(load("channel_breakout_h4"))
        assert any(axis["name"] == "ema_period" for axis in document["axes"])


class TestLaddersStateTheirRule:
    def test_every_candidate_carries_a_rule(self) -> None:
        for axis in build_candidates(load("channel_breakout_h4")):
            assert axis.rule, f"{axis.name} proposes values with no stated rule"

    def test_the_rendered_draft_prints_the_rule(self) -> None:
        spec = load("channel_breakout_h4")
        text = render(spec, build_candidates(spec))
        assert "PROPOSALS" in text
        assert "period: 0.5x..2x" in text

    def test_a_period_ladder_is_multiplicative_and_whole(self) -> None:
        values, rule = period_ladder(20)
        assert values == (10, 14, 20, 28, 40)
        assert all(isinstance(value, int) for value in values)
        assert "0.5x..2x" in rule

    def test_a_multiple_ladder_is_additive_and_positive(self) -> None:
        values, rule = multiple_ladder(2.0)
        assert values == (1.0, 1.5, 2.0, 2.5, 3.0)
        assert "additive" in rule
        assert all(value > 0 for value in values)

    def test_a_shift_ladder_is_small_and_starts_at_zero(self) -> None:
        values, rule = shift_ladder(1)
        assert values == (0, 1, 2)
        assert "shift" in rule

    def test_a_parameter_kind_with_no_rule_says_so_instead_of_guessing(self) -> None:
        # A period below one has no multiplicative neighbourhood: every factor
        # lands on the same floor, so a "ladder" would be one number repeated.
        values, rule = period_ladder(0)
        assert values == (0,)
        assert rule == NO_LADDER


class TestThresholdsRespectTheOutputRange:
    def test_an_adx_threshold_is_clamped_to_its_output_range(self) -> None:
        values, rule = threshold_ladder(95.0, "adx")
        assert max(values) <= OUTPUT_RANGES["adx"][1]
        assert "clamped" in rule

    def test_a_low_threshold_does_not_go_below_the_range(self) -> None:
        values, _ = threshold_ladder(5.0, "rsi")
        assert min(values) >= OUTPUT_RANGES["rsi"][0]

    def test_a_negative_range_indicator_is_handled(self) -> None:
        values, _ = threshold_ladder(-33.3, "willr")
        low, high = OUTPUT_RANGES["willr"]
        assert all(low <= value <= high for value in values)

    def test_an_unknown_indicator_says_its_range_is_unknown(self) -> None:
        _, rule = threshold_ladder(100.0, "cci")
        assert "unknown" in rule

    def test_a_non_negative_indicator_never_gets_a_negative_threshold(self) -> None:
        # rvol is a ratio of volumes; -0.2 is not a stricter filter, it is a
        # meaningless one. The bound is a fact about the indicator, not a guess.
        values, _ = threshold_ladder(1.8, "rvol")
        assert min(values) > 0
        assert values == (0.8, 1.8, 2.8, 3.8)

    def test_an_open_upper_bound_reads_as_infinite_rather_than_a_number(self) -> None:
        _, rule = threshold_ladder(1.8, "rvol")
        assert "inf" in rule, "an invented upper bound would be a guess presented as a fact"


class TestLaddersStayInsideRegistryBounds:
    """Bounds come from ``build_indicator``, never from a table in the builder."""

    def test_no_proposed_indicator_parameter_is_refused_by_its_own_indicator(self) -> None:
        for name in (
            "channel_breakout_h4",
            "bollinger_fade_h4",
            "volume_thrust_h4",
            "ema_pullback_h1",
        ):
            spec = load(name)
            document = generated(spec)
            raw = spec.model_dump(mode="json")
            for axis in document["axes"]:
                for path in axis["paths"]:
                    parts = path.strip("/").split("/")
                    if "params" not in parts:
                        continue
                    node: Any = raw
                    for token in parts[: parts.index("params")]:
                        node = node[int(token)] if token.isdigit() else node[token]
                    param = parts[-1]
                    for value in axis["values"]:
                        build_indicator(node["indicator"], {**node["params"], param: value})

    def test_a_ladder_that_would_go_below_a_minimum_is_trimmed(self) -> None:
        # swing lookback of 1 builds; 0 does not. A ladder around a small
        # current value must not propose the illegal one.
        spec = load("channel_breakout_h4")
        document = generated(spec)
        values = axis_named(document, "swing_lookback")["values"]
        assert min(values) >= 1
        with pytest.raises(ValidationError):
            build_indicator("swing", {"lookback": 0})

    def test_the_bound_is_the_indicators_own_constructor(self) -> None:
        # Not a declared table: a second statement of the same bound could
        # drift from the first. Proven by asking the constructor directly.
        from trading_system.validation.space_builder import _bounded_by_registry

        kept = _bounded_by_registry([0, 1, 2], indicator="swing", params={}, param="lookback")
        assert kept == (1, 2)


class TestVerificationByApplication:
    def test_every_generated_space_applies_cleanly(self) -> None:
        for name in (
            "channel_breakout_h4",
            "bollinger_fade_h4",
            "volume_thrust_h4",
            "ema_pullback_h1",
        ):
            spec = load(name)
            verify(spec, generated(spec))

    def test_a_generated_space_parses_as_a_real_search_space(self) -> None:
        spec = load("channel_breakout_h4")
        space = SearchSpace.model_validate(generated(spec))
        assert space.feasible_size() > 0

    def test_a_broken_pointer_is_caught_rather_than_written_out(self) -> None:
        spec = load("channel_breakout_h4")
        document = generated(spec)
        document["axes"][0]["paths"] = ["/entries/0/nonsense/period"]
        with pytest.raises(ValidationError, match="does not apply"):
            verify(spec, document)

    def test_a_value_the_spec_refuses_is_pruned_not_shipped(self) -> None:
        # confirmation_window_bars is present and numeric on an entry with no
        # confirmation, but the schema pins it at zero there. Proposing a ladder
        # for it would produce a space whose every point is invalid.
        spec = load("channel_breakout_h4")
        _, notes = prune(spec, build_space_document(spec))
        assert any("confirmation_window_bars" in note for note in notes)

    def test_pruning_reports_what_it_dropped(self) -> None:
        spec = load("channel_breakout_h4")
        document, notes = prune(spec, build_space_document(spec))
        names = {axis["name"] for axis in document["axes"]}
        assert "confirmation_window_bars" not in names
        assert notes, "a silent prune is a prune nobody can audit"


class TestInferredConstraints:
    def test_a_fast_slow_pair_gets_an_order_constraint(self) -> None:
        document = generated(load("ema_pullback_h1"))
        assert {"less": "ema_period_20", "greater": "ema_period_50"} in document["constraints"]

    def test_a_single_axis_infers_nothing(self) -> None:
        document = generated(load("channel_breakout_h4"))
        assert "constraints" not in document

    def test_three_axes_on_one_role_infer_nothing(self) -> None:
        # Two is a fast/slow pair. Three is a structure this rule has no opinion
        # about, and guessing an ordering over it would be inventing intent.
        axes = build_candidates(load("ema_pullback_h1"))
        trio = [axis for axis in axes if axis.signature.owner == "ema"]
        assert len(trio) == 2, "fixture assumption: ema_pullback has exactly two ema axes"
        assert infer_constraints(trio + list(trio[:1])) == []


class TestSelectingAxes:
    def test_keep_narrows_the_draft(self) -> None:
        spec = load("channel_breakout_h4")
        document = build_space_document(spec, keep=["swing_lookback", "multiple"])
        assert {axis["name"] for axis in document["axes"]} == {"swing_lookback", "multiple"}

    def test_keeping_nothing_yields_no_axes(self) -> None:
        spec = load("channel_breakout_h4")
        assert build_space_document(spec, keep=[])["axes"] == []
