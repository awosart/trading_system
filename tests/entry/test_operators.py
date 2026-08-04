"""Every operator against a hand-computed answer, plus the unknown algebra."""

import pytest

from trading_system.entry.operators import (
    Truth,
    and_all,
    between,
    cross_above,
    cross_below,
    falling,
    gt,
    gte,
    inside_range,
    lt,
    lte,
    negate,
    or_any,
    rising,
)


@pytest.mark.parametrize(
    ("left", "right", "expected"),
    [(2.0, 1.0, True), (1.0, 1.0, False), (1.0, 2.0, False)],
)
def test_gt(left: float, right: float, expected: bool) -> None:
    assert gt(left, right) is expected


@pytest.mark.parametrize(
    ("left", "right", "expected"),
    [(2.0, 1.0, True), (1.0, 1.0, True), (1.0, 2.0, False)],
)
def test_gte(left: float, right: float, expected: bool) -> None:
    assert gte(left, right) is expected


@pytest.mark.parametrize(
    ("left", "right", "expected"),
    [(2.0, 1.0, False), (1.0, 1.0, False), (1.0, 2.0, True)],
)
def test_lt(left: float, right: float, expected: bool) -> None:
    assert lt(left, right) is expected


@pytest.mark.parametrize(
    ("left", "right", "expected"),
    [(2.0, 1.0, False), (1.0, 1.0, True), (1.0, 2.0, True)],
)
def test_lte(left: float, right: float, expected: bool) -> None:
    assert lte(left, right) is expected


@pytest.mark.parametrize("operator", [gt, gte, lt, lte])
def test_comparisons_are_unknown_when_an_operand_is_missing(operator: object) -> None:
    compare = operator
    assert compare(None, 1.0) is None  # type: ignore[operator]
    assert compare(1.0, None) is None  # type: ignore[operator]
    assert compare(None, None) is None  # type: ignore[operator]


class TestCrossings:
    """``cross_above(a, b) == a[t] > b[t] and a[t-1] <= b[t-1]``."""

    def test_clean_cross_above(self) -> None:
        # a: 0.9 -> 1.1, b flat at 1.0.
        assert cross_above(1.1, 0.9, 1.0, 1.0) is True

    def test_already_above_is_not_a_cross(self) -> None:
        assert cross_above(1.2, 1.1, 1.0, 1.0) is False

    def test_touching_then_breaking_counts_as_a_cross(self) -> None:
        # a sat exactly on b last bar. Requiring a strict a[t-1] < b[t-1] would
        # drop precisely the touch-and-go breakout this is meant to catch.
        assert cross_above(1.1, 1.0, 1.0, 1.0) is True

    def test_equal_on_the_current_bar_is_not_yet_a_cross(self) -> None:
        assert cross_above(1.0, 0.9, 1.0, 1.0) is False

    def test_both_series_moving(self) -> None:
        # a: 1.0 -> 1.3, b: 1.1 -> 1.2. Was below, is above.
        assert cross_above(1.3, 1.0, 1.2, 1.1) is True

    def test_clean_cross_below(self) -> None:
        assert cross_below(0.9, 1.1, 1.0, 1.0) is True

    def test_already_below_is_not_a_cross_below(self) -> None:
        assert cross_below(0.8, 0.9, 1.0, 1.0) is False

    def test_touching_then_breaking_down_counts(self) -> None:
        assert cross_below(0.9, 1.0, 1.0, 1.0) is True

    @pytest.mark.parametrize("missing", range(4))
    def test_any_missing_value_makes_the_cross_unknown(self, missing: int) -> None:
        values: list[float | None] = [1.1, 0.9, 1.0, 1.0]
        values[missing] = None
        assert cross_above(*values) is None
        assert cross_below(*values) is None


@pytest.mark.parametrize(
    ("value", "expected"),
    [(29.9, False), (30.0, True), (50.0, True), (70.0, True), (70.1, False)],
)
def test_between_includes_its_bounds(value: float, expected: bool) -> None:
    assert between(value, 30.0, 70.0) is expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [(29.9, False), (30.0, False), (50.0, True), (70.0, False), (70.1, False)],
)
def test_inside_range_excludes_its_bounds(value: float, expected: bool) -> None:
    assert inside_range(value, 30.0, 70.0) is expected


def test_between_and_inside_range_differ_only_at_the_boundary() -> None:
    assert between(30.0, 30.0, 70.0) is True
    assert inside_range(30.0, 30.0, 70.0) is False
    assert between(50.0, 30.0, 70.0) == inside_range(50.0, 30.0, 70.0)


def test_range_operators_are_unknown_without_a_value() -> None:
    assert between(None, 30.0, 70.0) is None
    assert inside_range(None, 30.0, 70.0) is None


@pytest.mark.parametrize(
    ("now", "past", "expected"), [(2.0, 1.0, True), (1.0, 1.0, False), (1.0, 2.0, False)]
)
def test_rising(now: float, past: float, expected: bool) -> None:
    assert rising(now, past) is expected


@pytest.mark.parametrize(
    ("now", "past", "expected"), [(2.0, 1.0, False), (1.0, 1.0, False), (1.0, 2.0, True)]
)
def test_falling(now: float, past: float, expected: bool) -> None:
    assert falling(now, past) is expected


def test_slopes_compare_endpoints_not_monotonicity() -> None:
    # A series that dipped in between still counts as rising over the window;
    # the endpoint reading is the documented one.
    assert rising(2.0, 1.0) is True
    assert rising(None, 1.0) is None
    assert falling(1.0, None) is None


class TestKleeneLogic:
    """Unknown must not collapse into False, or Not would fire on warmup bars."""

    @pytest.mark.parametrize(
        ("values", "expected"),
        [
            ((True, True), True),
            ((True, False), False),
            ((False, None), False),
            ((None, False), False),
            ((True, None), None),
            ((None, None), None),
            ((), True),
        ],
    )
    def test_and_all(self, values: tuple[Truth, ...], expected: Truth) -> None:
        assert and_all(values) is expected

    @pytest.mark.parametrize(
        ("values", "expected"),
        [
            ((False, False), False),
            ((True, False), True),
            ((True, None), True),
            ((None, True), True),
            ((False, None), None),
            ((None, None), None),
            ((), False),
        ],
    )
    def test_or_any(self, values: tuple[Truth, ...], expected: Truth) -> None:
        assert or_any(values) is expected

    @pytest.mark.parametrize(("value", "expected"), [(True, False), (False, True), (None, None)])
    def test_negate(self, value: Truth, expected: Truth) -> None:
        assert negate(value) is expected

    def test_negating_unknown_does_not_manufacture_a_signal(self) -> None:
        # The bug this whole three-valued layer exists to prevent: an indicator
        # in warmup makes "rsi < 30" unknown, and a False here would make
        # "not (rsi < 30)" true on a bar where nothing was known at all.
        assert negate(lt(None, 30.0)) is None

    def test_and_all_short_circuits_on_false(self) -> None:
        evaluated: list[int] = []

        def values() -> object:
            for index, value in enumerate((True, False, True)):
                evaluated.append(index)
                yield value

        assert and_all(values()) is False  # type: ignore[arg-type]
        assert evaluated == [0, 1]

    def test_or_any_short_circuits_on_true(self) -> None:
        evaluated: list[int] = []

        def values() -> object:
            for index, value in enumerate((False, True, False)):
                evaluated.append(index)
                yield value

        assert or_any(values()) is True  # type: ignore[arg-type]
        assert evaluated == [0, 1]
