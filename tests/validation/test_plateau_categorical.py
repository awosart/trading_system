"""Plateau geometry with a categorical axis present.

The regression this file exists for: a categorical axis's grid index is the
position its value happened to occupy in a list somebody typed. If adjacency
used that index, reordering the list — changing nothing that was run — would
change the measured plateau width and, through it, the selected parameters.
"""

import itertools

import pytest

from trading_system.validation.objective import (
    ScoredPoint,
    analyse_plateau,
    neighbours,
    roughness,
)


def _surface(order: list[str], heights: dict[str, list[float]]) -> list[ScoredPoint]:
    """One row per categorical value, in the order given, five numeric columns."""
    points = []
    for category_index, name in enumerate(order):
        for numeric_index, score in enumerate(heights[name]):
            points.append(ScoredPoint(coords=(category_index, numeric_index), score=score))
    return points


#: A broad, lower plateau and a tall, narrow spike — the pair analyse_plateau
#: exists to tell apart, here separated onto two categorical values.
HEIGHTS = {
    "broad": [1.0, 1.0, 1.0, 1.0, 1.0],
    "spike": [0.0, 0.0, 3.0, 0.0, 0.0],
    "flat_low": [-1.0, -1.0, -1.0, -1.0, -1.0],
}
MASK = (True, False)


class TestIndexAdjacencyDoesNotCrossACategoricalValue:
    """Equality along a categorical axis, Chebyshev along the numeric ones."""

    def test_a_numeric_only_mask_leaks_into_the_next_category(self) -> None:
        points = _surface(["broad", "spike"], HEIGHTS)
        # Point 2 is ("broad", 2). Without a mask its neighbours include the
        # whole ("spike", 1..3) block simply because those rows were written
        # next in the file.
        assert set(neighbours(points, 2)) == {1, 3, 6, 7, 8}

    def test_the_mask_confines_neighbours_to_one_category(self) -> None:
        points = _surface(["broad", "spike"], HEIGHTS)
        assert set(neighbours(points, 2, MASK)) == {1, 3}

    def test_roughness_is_measured_within_the_category(self) -> None:
        points = _surface(["broad", "spike"], HEIGHTS)
        # ("spike", 2) sits at 3.0 with both numeric neighbours at 0.0.
        assert roughness(points, 7, MASK) == pytest.approx(3.0)

    def test_a_mask_that_does_not_cover_the_axes_is_refused(self) -> None:
        points = _surface(["broad", "spike"], HEIGHTS)
        with pytest.raises(ValueError, match="covers 1 axes"):
            neighbours(points, 0, (True,))


class TestReorderingTheValuesChangesNothing:
    """The property that makes a categorical axis safe to put in a config file."""

    @pytest.mark.parametrize(
        "order",
        [
            ["broad", "spike", "flat_low"],
            ["spike", "broad", "flat_low"],
            ["flat_low", "spike", "broad"],
            ["spike", "flat_low", "broad"],
        ],
    )
    def test_the_same_configuration_is_selected_whatever_order_the_values_are_listed_in(
        self, order: list[str]
    ) -> None:
        points = _surface(order, HEIGHTS)
        analysis = analyse_plateau(points, categorical=MASK, penalty_weight=2.0)
        chosen = points[analysis.selected_index]
        assert order[chosen.coords[0]] == "broad"
        assert chosen.coords[1] == 2
        assert analysis.plateau_size == 5

    def test_without_the_mask_the_answer_depends_on_the_order(self) -> None:
        """The defect, demonstrated rather than described."""
        selected = set()
        for order in itertools.permutations(["broad", "spike", "flat_low"]):
            points = _surface(list(order), HEIGHTS)
            analysis = analyse_plateau(points, penalty_weight=2.0)
            chosen = points[analysis.selected_index]
            selected.add((order[chosen.coords[0]], chosen.coords[1], analysis.plateau_size))
        assert len(selected) > 1, (
            "reordering the categorical values should change an unmasked analysis; "
            "if it stops doing so this test no longer demonstrates anything"
        )


class TestTheChoiceIsMadeByScoreAndTheCentringOnlyMovesNumbers:
    """Two steps that divide cleanly once adjacency stops crossing categories."""

    def test_the_penalised_maximum_still_ranges_over_every_category(self) -> None:
        points = _surface(["broad", "spike", "flat_low"], HEIGHTS)
        without_penalty = analyse_plateau(points, categorical=MASK, penalty_weight=0.0)
        assert points[without_penalty.best_index].coords == (1, 2)  # the spike wins on height

    def test_a_categorical_axis_never_spans_a_plateau(self) -> None:
        points = _surface(["broad", "spike", "flat_low"], HEIGHTS)
        analysis = analyse_plateau(points, categorical=MASK, penalty_weight=2.0)
        assert analysis.axis_extent[0] == 1
        assert analysis.axis_extent[1] == 5

    def test_two_identical_categories_do_not_merge_into_one_wide_plateau(self) -> None:
        """Equal-scoring twins are two plateaus of five, not one of ten."""
        points = _surface(["broad", "twin"], {"broad": HEIGHTS["broad"], "twin": HEIGHTS["broad"]})
        analysis = analyse_plateau(points, categorical=MASK, penalty_weight=0.5)
        assert analysis.plateau_size == 5
