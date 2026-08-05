"""The sample-size invariant, proved by walking every dataclass field.

``metrics.py``'s docstring promises that a value derived from a sample never
appears without that sample's size on the same object. A promise like that
decays the moment someone adds a field under deadline pressure without
noticing it was ever made — so it is proved here by reflecting over every
dataclass ``metrics`` and ``statistical`` define, rather than trusted to
whoever writes the next metric.
"""

import dataclasses
import inspect
from types import ModuleType

import pytest

import trading_system.analytics.metrics as metrics
import trading_system.analytics.statistical as statistical

MODULES = (metrics, statistical)
VALID_KINDS = {"raw", "n", "value", "fact"}


def _local_dataclasses(module: ModuleType) -> list[type]:
    """Dataclasses defined in ``module`` itself, not merely imported into it."""
    return [
        obj
        for _, obj in inspect.getmembers(module, inspect.isclass)
        if dataclasses.is_dataclass(obj) and obj.__module__ == module.__name__
    ]


def _cases() -> list[tuple[type, dataclasses.Field]]:  # type: ignore[type-arg]
    return [
        (cls, f)
        for module in MODULES
        for cls in _local_dataclasses(module)
        for f in dataclasses.fields(cls)
    ]


def _classification_error(cls: type, f: dataclasses.Field) -> str | None:  # type: ignore[type-arg]
    """``None`` if ``f`` carries a recognised ``kind``, else why it does not."""
    kind = f.metadata.get("kind")
    if kind is None:
        return f"{cls.__name__}.{f.name} has no 'kind' metadata"
    if kind not in VALID_KINDS:
        return f"{cls.__name__}.{f.name} has unknown kind {kind!r}"
    return None


def _pairing_error(cls: type, f: dataclasses.Field) -> str | None:  # type: ignore[type-arg]
    """``None`` if a ``kind='value'`` field ``f`` names a valid sibling ``kind='n'`` field."""
    if f.metadata.get("kind") != "value":
        return None
    n_field_name = f.metadata.get("n_field")
    if n_field_name is None:
        return f"{cls.__name__}.{f.name} is kind='value' but names no n_field"
    siblings = {sibling.name: sibling for sibling in dataclasses.fields(cls)}
    if n_field_name not in siblings:
        return (
            f"{cls.__name__}.{f.name} names n_field={n_field_name!r}, "
            f"which is not a field of {cls.__name__}"
        )
    if siblings[n_field_name].metadata.get("kind") != "n":
        actual = siblings[n_field_name].metadata.get("kind")
        return (
            f"{cls.__name__}.{f.name} names n_field={n_field_name!r}, but "
            f"{cls.__name__}.{n_field_name} is kind={actual!r}, not 'n'"
        )
    return None


CASES = _cases()
CASE_IDS = [f"{cls.__name__}.{f.name}" for cls, f in CASES]


class TestEveryFieldOfEveryStatsDataclassIsClassified:
    """Nothing in ``metrics``/``statistical`` may skip the metadata contract."""

    def test_the_scan_actually_found_the_project_s_dataclasses(self) -> None:
        # A sanity floor, not the real count: proves the reflection walked
        # something rather than silently scanning zero classes because an
        # import path changed. metrics.py alone declares 66 tagged fields
        # across 12 dataclasses at the time this test was written.
        assert len(CASES) >= 90

    @pytest.mark.parametrize(("cls", "field"), CASES, ids=CASE_IDS)
    def test_a_field_carries_a_known_kind(
        self, cls: type, field: "dataclasses.Field[object]"
    ) -> None:
        error = _classification_error(cls, field)
        assert error is None, error


class TestEveryValueFieldNamesTheSiblingItWasComputedOver:
    """A ``kind='value'`` field is worthless without a real ``kind='n'`` neighbour."""

    @pytest.mark.parametrize(("cls", "field"), CASES, ids=CASE_IDS)
    def test_a_value_field_pairs_with_an_n_field_on_the_same_class(
        self, cls: type, field: "dataclasses.Field[object]"
    ) -> None:
        error = _pairing_error(cls, field)
        assert error is None, error


class TestTheCheckerItselfCatchesViolations:
    """The two checks above are exercised against deliberately broken classes.

    Without this, the parametrized tests above could pass vacuously forever
    if ``_classification_error``/``_pairing_error`` were themselves broken —
    this class is what proves they actually flag what they claim to.
    """

    def test_a_field_with_no_kind_at_all_is_flagged(self) -> None:
        @dataclasses.dataclass
        class Unclassified:
            x: int = dataclasses.field(default=0)

        (field,) = dataclasses.fields(Unclassified)
        assert _classification_error(Unclassified, field) is not None

    def test_a_field_with_an_unrecognised_kind_is_flagged(self) -> None:
        @dataclasses.dataclass
        class Mystery:
            x: int = dataclasses.field(default=0, metadata={"kind": "vibes"})

        (field,) = dataclasses.fields(Mystery)
        assert _classification_error(Mystery, field) is not None

    def test_a_value_field_naming_no_n_field_is_flagged(self) -> None:
        @dataclasses.dataclass
        class Loose:
            x: float = dataclasses.field(default=0.0, metadata={"kind": "value"})

        (field,) = dataclasses.fields(Loose)
        assert _pairing_error(Loose, field) is not None

    def test_a_value_field_naming_a_sibling_that_does_not_exist_is_flagged(self) -> None:
        @dataclasses.dataclass
        class Dangling:
            x: float = dataclasses.field(default=0.0, metadata={"kind": "value", "n_field": "nope"})

        (field,) = dataclasses.fields(Dangling)
        assert _pairing_error(Dangling, field) is not None

    def test_a_value_field_naming_a_sibling_that_is_not_kind_n_is_flagged(self) -> None:
        @dataclasses.dataclass
        class Mismatched:
            x: float = dataclasses.field(default=0.0, metadata={"kind": "value", "n_field": "y"})
            y: float = dataclasses.field(default=0.0, metadata={"kind": "fact"})

        fields = {f.name: f for f in dataclasses.fields(Mismatched)}
        assert _pairing_error(Mismatched, fields["x"]) is not None

    def test_a_correctly_paired_value_field_is_not_flagged(self) -> None:
        @dataclasses.dataclass
        class Good:
            x: float = dataclasses.field(default=0.0, metadata={"kind": "value", "n_field": "n"})
            n: int = dataclasses.field(default=0, metadata={"kind": "n"})

        fields = {f.name: f for f in dataclasses.fields(Good)}
        assert _pairing_error(Good, fields["x"]) is None

    def test_a_raw_or_fact_field_needs_no_pairing(self) -> None:
        @dataclasses.dataclass
        class Standalone:
            x: float = dataclasses.field(default=0.0, metadata={"kind": "fact"})
            y: tuple[float, ...] = dataclasses.field(
                default_factory=tuple, metadata={"kind": "raw"}
            )

        fields = {f.name: f for f in dataclasses.fields(Standalone)}
        assert _pairing_error(Standalone, fields["x"]) is None
        assert _pairing_error(Standalone, fields["y"]) is None
