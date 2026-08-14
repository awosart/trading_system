"""The generated exit axis: proposed, described, and switched off.

An exit axis of eight presets multiplies every numeric trial by eight and turns
on a per-preset null calibration. Neither cost is visible in the axis itself, so
the generator offers it where the author reads the draft and leaves enabling it
to a visible edit.
"""

import pytest
from pydantic import ValidationError

from tests.backtest.conftest import ema_pullback
from trading_system.exit.library import ExitLibrarySpec
from trading_system.validation.optimization import AxisTarget, SearchSpace
from trading_system.validation.space_builder import (
    EXIT_AXIS_NAME,
    EXIT_AXIS_WARNING,
    build_candidates,
    build_space_document,
    exit_preset_axis,
    prune,
    render,
    verify,
)


def _ids(library: ExitLibrarySpec) -> list[str]:
    return [item.id for item in library.presets]


class TestTheExitAxisArrivesDisabled:
    """Enabling it means moving the entry, which shows up in a diff."""

    def test_it_lands_in_disabled_axes_not_axes(self, library: ExitLibrarySpec) -> None:
        document = build_space_document(ema_pullback(), exit_preset_ids=_ids(library))
        assert EXIT_AXIS_NAME not in {axis["name"] for axis in document["axes"]}
        assert [axis["name"] for axis in document["disabled_axes"]] == [EXIT_AXIS_NAME]

    def test_the_parsed_space_does_not_search_it(self, library: ExitLibrarySpec) -> None:
        document = build_space_document(ema_pullback(), exit_preset_ids=_ids(library))
        document, _ = prune(ema_pullback(), document)
        space = SearchSpace.model_validate(document)
        assert all(not axis.categorical for axis in space.axes)
        assert space.categorical_mask == (False,) * len(space.axes)

    def test_moving_it_into_axes_is_all_that_enabling_takes(self, library: ExitLibrarySpec) -> None:
        document = build_space_document(ema_pullback(), exit_preset_ids=_ids(library))
        document, _ = prune(ema_pullback(), document)
        moved = {
            "axes": [*document["axes"], *document["disabled_axes"]],
            "constraints": document.get("constraints", []),
        }
        space = SearchSpace.model_validate(moved)
        assert space.axes[-1].target is AxisTarget.EXIT_PRESET
        assert space.categorical_mask[-1] is True

    def test_no_exit_library_means_no_offer(self) -> None:
        document = build_space_document(ema_pullback(), exit_preset_ids=None)
        assert "disabled_axes" not in document

    def test_a_library_of_one_is_not_an_axis(self) -> None:
        document = build_space_document(ema_pullback(), exit_preset_ids=["only_one"])
        assert "disabled_axes" not in document
        with pytest.raises(ValueError, match="at least two presets"):
            exit_preset_axis(["only_one"])


class TestTheDraftSaysWhatEnablingItCosts:
    """An axis nobody read is an axis nobody meant to enable."""

    def test_the_rendered_draft_names_the_axis_and_its_warning(
        self, library: ExitLibrarySpec
    ) -> None:
        spec = ema_pullback()
        text = render(spec, build_candidates(spec), exit_preset_ids=_ids(library))
        assert EXIT_AXIS_NAME in text
        assert EXIT_AXIS_WARNING in text
        for preset_id in _ids(library):
            assert preset_id in text

    def test_a_draft_without_an_offer_does_not_mention_one(self) -> None:
        text = render(ema_pullback(), build_candidates(ema_pullback()))
        assert EXIT_AXIS_NAME not in text


class TestVerificationIgnoresWhatIsNotSearched:
    """A disabled axis has no pointers to resolve, and must not be asked for any."""

    def test_verify_passes_with_a_disabled_categorical_axis_present(
        self, library: ExitLibrarySpec
    ) -> None:
        spec = ema_pullback()
        document = build_space_document(spec, exit_preset_ids=_ids(library))
        document, _ = prune(spec, document)
        verify(spec, document)

    def test_the_document_round_trips_through_the_model(self, library: ExitLibrarySpec) -> None:
        spec = ema_pullback()
        document = build_space_document(spec, exit_preset_ids=_ids(library))
        document, _ = prune(spec, document)
        space = SearchSpace.model_validate(document)
        assert space.disabled_axes[0].values == tuple(_ids(library))
        try:
            SearchSpace.model_validate(space.model_dump(mode="json"))
        except ValidationError as error:  # pragma: no cover - a failure is the assertion
            pytest.fail(f"a generated space must survive its own serialisation: {error}")
