"""Categorical axes: what they refuse, and that a choice reaches every document it must.

The refusals matter more than the happy paths here. Every one of them guards a
configuration that would have run, produced numbers, and been wrong in a way
nothing downstream could see: a preset named in the spec but not executed, a
reward multiple pointed at an exit that has no target, a risk level the engine
silently trims to the same number as its neighbour.
"""

import pytest
from pydantic import ValidationError as PydanticValidationError

from tests.backtest.conftest import ema_pullback
from trading_system.core.exceptions import ValidationError
from trading_system.exit.library import ExitLibrarySpec
from trading_system.risk.sizing.config import (
    FixedFractionalConfig,
    QualityScaledConfig,
    load_sizing_methods,
)
from trading_system.validation.optimization import (
    AxisTarget,
    ParameterAxis,
    SearchSpace,
    VariationTargets,
)

SLOW_EMA = (
    "/entries/0/trigger/conditions/0/right/params/period",
    "/entries/0/invalidation/price_level/params/period",
)


def _numeric(name: str = "ema_slow") -> ParameterAxis:
    return ParameterAxis(name=name, paths=SLOW_EMA, values=(30, 50, 80))


def _exit_axis(*ids: str) -> ParameterAxis:
    return ParameterAxis(name="exit", target=AxisTarget.EXIT_PRESET, values=ids)


def _targets(library: ExitLibrarySpec, preset_id: str = "conservative_2r") -> VariationTargets:
    presets = {item.id: item for item in library.presets}
    return VariationTargets(
        spec=ema_pullback(),
        exit_preset=presets[preset_id],
        sizing=FixedFractionalConfig(risk_pct=0.01),
        exit_library=presets,
        sizing_library={
            "fixed_05": FixedFractionalConfig(risk_pct=0.005),
            "quality": QualityScaledConfig(
                min_risk_pct=0.004, max_risk_pct=0.012, quality_floor=0.5
            ),
        },
    )


class TestAnAxisIsEitherOrderedOrNamed:
    """Mixing the two would leave the axis ordered for some pairs and not others."""

    def test_values_may_not_mix_names_with_numbers(self) -> None:
        with pytest.raises(PydanticValidationError, match="mixes names with numbers"):
            ParameterAxis(name="x", paths=("/a",), values=(1, "two"))

    def test_a_string_valued_axis_is_categorical(self) -> None:
        assert ParameterAxis(name="x", paths=("/a",), values=("a", "b")).categorical

    def test_a_number_valued_axis_is_not(self) -> None:
        assert not _numeric().categorical

    def test_a_choice_target_is_categorical_whatever_its_values_look_like(self) -> None:
        assert _exit_axis("a", "b").categorical

    def test_a_range_may_not_be_declared_on_a_categorical_axis(self) -> None:
        with pytest.raises(PydanticValidationError, match="range bounds must be numeric"):
            ParameterAxis(name="x", paths=("/a",), low="a", high="b", step="c")


class TestPointersAreRequiredExactlyWhereTheyAreRead:
    """A pointer nothing reads is the same defect class as a condition that never fires."""

    def test_a_choice_axis_may_not_carry_a_pointer(self) -> None:
        with pytest.raises(PydanticValidationError, match="must not carry paths"):
            ParameterAxis(
                name="exit", target=AxisTarget.EXIT_PRESET, paths=("/id",), values=("a", "b")
            )

    def test_a_choice_axis_values_must_be_names(self) -> None:
        with pytest.raises(PydanticValidationError, match="ids naming entries"):
            ParameterAxis(name="exit", target=AxisTarget.EXIT_PRESET, values=(1, 2))

    @pytest.mark.parametrize("target", [AxisTarget.SPEC, AxisTarget.EXIT_PARAM, AxisTarget.RUN])
    def test_a_pointer_target_needs_at_least_one(self, target: AxisTarget) -> None:
        with pytest.raises(PydanticValidationError, match="needs at least one"):
            ParameterAxis(name="x", target=target, values=(1, 2))


class TestTheSpaceRefusesCombinationsThatDescribeNothing:
    """Products containing points that are not configurations of anything."""

    def test_choosing_an_exit_and_tuning_one_are_mutually_exclusive(self) -> None:
        with pytest.raises(PydanticValidationError, match="not both"):
            SearchSpace(
                axes=(
                    _exit_axis("conservative_2r", "scalp_quick"),
                    ParameterAxis(
                        name="rr",
                        target=AxisTarget.EXIT_PARAM,
                        paths=("/rules/0/r_multiple",),
                        values=(1.5, 3.0),
                    ),
                )
            )

    def test_choosing_a_sizing_method_and_tuning_one_are_mutually_exclusive(self) -> None:
        with pytest.raises(PydanticValidationError, match="not both"):
            SearchSpace(
                axes=(
                    ParameterAxis(
                        name="sizing", target=AxisTarget.SIZING_METHOD, values=("a", "b")
                    ),
                    ParameterAxis(
                        name="risk",
                        target=AxisTarget.RUN,
                        paths=("/sizing/risk_pct",),
                        values=(0.005, 0.01),
                    ),
                )
            )

    def test_an_ordering_constraint_may_not_name_a_categorical_axis(self) -> None:
        with pytest.raises(PydanticValidationError, match="names a categorical axis"):
            SearchSpace(
                axes=(_exit_axis("a", "b"), _numeric()),
                constraints=({"less": "exit", "greater": "ema_slow"},),
            )


class TestDisabledAxesAreCarriedAndNotSearched:
    """The generator proposes an exit axis switched off; nothing may search it by accident."""

    def test_a_disabled_axis_does_not_enter_the_grid(self) -> None:
        space = SearchSpace(axes=(_numeric(),), disabled_axes=(_exit_axis("a", "b", "c"),))
        assert space.grid_size == 3
        assert space.categorical_mask == (False,)
        assert len(next(space.enumerate()).coords) == 1

    def test_an_axis_may_not_be_enabled_and_disabled_at_once(self) -> None:
        with pytest.raises(PydanticValidationError, match="both enabled and disabled"):
            SearchSpace(axes=(_numeric(),), disabled_axes=(_numeric(),))


class TestApplyRefusesToSilentlyIgnoreAnAxis:
    """A spec returned from a space that also varies an exit would look fully varied."""

    def test_apply_refuses_a_space_carrying_a_non_spec_axis(self) -> None:
        space = SearchSpace(axes=(_numeric(), _exit_axis("conservative_2r", "scalp_quick")))
        with pytest.raises(ValueError, match="target another document"):
            space.apply(ema_pullback(), space.point((0, 0)))

    def test_apply_still_works_for_a_spec_only_space(self) -> None:
        space = SearchSpace(axes=(_numeric(),))
        varied = space.apply(ema_pullback(), space.point((2,)))
        assert varied.entries[0].trigger.conditions[0].right.params["period"] == 80


class TestChoosingAnExitRewritesBothPlacesThatNameIt:
    """The recorded strategy must never name one exit while the engine ran another."""

    def test_the_binding_and_the_spec_agree_after_every_choice(self, library) -> None:
        space = SearchSpace(axes=(_exit_axis("conservative_2r", "breakeven_runner"),))
        targets = _targets(library)
        for index, expected in enumerate(("conservative_2r", "breakeven_runner")):
            applied = space.apply_to(targets, space.point((index,)))
            assert applied.exit_preset.id == expected
            assert applied.spec.exit_ref == expected

    def test_an_unknown_preset_id_is_named_in_the_error(self, library) -> None:
        space = SearchSpace(axes=(_exit_axis("conservative_2r", "nonesuch"),))
        with pytest.raises(ValueError, match="nonesuch"):
            space.apply_to(_targets(library), space.point((1,)))

    def test_a_choice_without_a_library_refuses_rather_than_guessing(self, library) -> None:
        space = SearchSpace(axes=(_exit_axis("conservative_2r", "scalp_quick"),))
        bare = VariationTargets(
            spec=ema_pullback(),
            exit_preset={item.id: item for item in library.presets}["conservative_2r"],
        )
        with pytest.raises(ValueError, match="no exit library"):
            space.apply_to(bare, space.point((0,)))


class TestTuningAnExitGoesThroughTheExitsOwnSemantics:
    """Syntax is the model; semantics is build_plan — the same split P07 stage 3 uses."""

    def test_a_reward_multiple_reaches_the_preset(self, library) -> None:
        space = SearchSpace(
            axes=(
                ParameterAxis(
                    name="rr",
                    target=AxisTarget.EXIT_PARAM,
                    paths=("/rules/0/r_multiple",),
                    values=(1.5, 3.0),
                ),
            )
        )
        applied = space.apply_to(_targets(library), space.point((1,)))
        assert applied.exit_preset.rules[0].r_multiple == 3.0

    def test_a_ladder_summing_past_one_hundred_percent_fails_at_apply(self, library) -> None:
        space = SearchSpace(
            axes=(
                ParameterAxis(
                    name="first_rung",
                    target=AxisTarget.EXIT_PARAM,
                    paths=("/rules/0/rungs/0/fraction",),
                    values=("0.2", "0.99"),
                ),
            )
        )
        # The shipped ladder is 0.5 + 0.25. Moving the first rung to 0.99 keeps
        # every field individually legal and pushes the sum past 100%, which
        # only PartialClose's own constructor knows about — reached here via
        # build_plan, not at the first position a backtest would have opened.
        targets = _targets(library, preset_id="swing_partial_ladder")
        space.apply_to(targets, space.point((0,)))
        with pytest.raises(ValidationError, match="swing_partial_ladder"):
            space.apply_to(targets, space.point((1,)))


class TestRunTargetAxesVaryWhatIsNotInTheStrategyAtAll:
    """RunInputs holds a built sizing method; the description has to be passed in."""

    def test_a_run_axis_edits_the_sizing_description(self, library) -> None:
        space = SearchSpace(
            axes=(
                ParameterAxis(
                    name="risk",
                    target=AxisTarget.RUN,
                    paths=("/sizing/risk_pct",),
                    values=(0.0025, 0.005),
                ),
            )
        )
        applied = space.apply_to(_targets(library), space.point((0,)))
        assert applied.sizing is not None
        assert applied.sizing.risk_pct == 0.0025

    def test_a_run_axis_without_a_sizing_description_refuses(self, library) -> None:
        space = SearchSpace(
            axes=(
                ParameterAxis(
                    name="risk",
                    target=AxisTarget.RUN,
                    paths=("/sizing/risk_pct",),
                    values=(0.0025, 0.005),
                ),
            )
        )
        targets = _targets(library)
        bare = VariationTargets(spec=targets.spec, exit_preset=targets.exit_preset)
        with pytest.raises(ValueError, match="no sizing config"):
            space.apply_to(bare, space.point((0,)))

    def test_choosing_a_sizing_method_by_name_swaps_the_whole_block(self, library) -> None:
        space = SearchSpace(
            axes=(
                ParameterAxis(
                    name="sizing",
                    target=AxisTarget.SIZING_METHOD,
                    values=("fixed_05", "quality"),
                ),
            )
        )
        applied = space.apply_to(_targets(library), space.point((1,)))
        assert applied.sizing is not None
        assert applied.sizing.method == "QUALITY_SCALED"


class TestTheShippedSizingLibraryStaysUnderTheEnginesCap:
    """Every fraction on file must be reachable; one above the cap is a fake plateau."""

    def test_no_shipped_variant_declares_more_than_two_percent(self) -> None:
        from trading_system.risk.sizing.config import declared_risk_fraction

        methods = load_sizing_methods()
        assert methods, "the shipped sizing library should not be empty"
        for name, config in methods.items():
            fraction = declared_risk_fraction(config)
            assert fraction is None or fraction <= 0.02, name
