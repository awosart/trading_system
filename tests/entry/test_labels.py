"""Categorical bar labels, and the operators that read them."""

from datetime import UTC, datetime

import pytest

from trading_system.entry.compiler import compile_entry
from trading_system.entry.context import BarSeries
from trading_system.entry.features import FeatureRegistry, required_label_categories
from trading_system.entry.labels import (
    PATTERN_WARMUP,
    LabelCategory,
    label_columns,
    pattern_labels,
    session_labels,
)
from trading_system.entry.operators import label_is
from trading_system.strategies.schema import Invalidation

from .conftest import frame_from_closes, labels, leaf, ref, series_from_features, strategy_spec


class TestLabelIs:
    """Intersection, and the unknown case that keeps Not honest."""

    def test_true_when_the_bar_carries_any_wanted_label(self) -> None:
        assert label_is(frozenset({"HAMMER", "DOJI"}), frozenset({"DOJI"})) is True

    def test_false_when_it_carries_none_of_them(self) -> None:
        assert label_is(frozenset({"HAMMER"}), frozenset({"DOJI"})) is False

    def test_an_empty_set_is_a_decided_false_not_unknown(self) -> None:
        # "Classified, and none apply" is an answer; the operator must not
        # confuse it with "could not classify".
        assert label_is(frozenset(), frozenset({"DOJI"})) is False

    def test_an_unclassifiable_bar_is_unknown(self) -> None:
        assert label_is(None, frozenset({"DOJI"})) is None


class TestPatternLabels:
    """Warmup is uniform across the vocabulary, as it is for indicators."""

    def test_leading_bars_are_unknown_not_empty(self) -> None:
        # A three-bar pattern cannot be evaluated on bar 0, and a label set has
        # nowhere to record "DOJI false, MORNING_STAR unknown". So the whole
        # category is unknown until every member of it can be decided.
        frame = frame_from_closes([1.10, 1.11, 1.12, 1.13, 1.14])
        column = pattern_labels(frame)
        assert PATTERN_WARMUP == 2
        assert column[:PATTERN_WARMUP] == [None, None]
        assert all(value is not None for value in column[PATTERN_WARMUP:])

    def test_a_doji_is_labelled(self) -> None:
        # Open == close with wicks on both sides, on a bar past the warmup.
        frame = frame_from_closes(
            [1.1000, 1.1010, 1.1020, 1.1020],
            lows=[1.0990, 1.1000, 1.1010, 1.1000],
            highs=[1.1010, 1.1020, 1.1030, 1.1040],
        )
        assert "DOJI" in (pattern_labels(frame)[3] or frozenset())

    def test_every_label_is_a_known_pattern_name(self) -> None:
        from trading_system.features.patterns import Pattern

        frame = frame_from_closes([1.10 + 0.001 * index for index in range(30)])
        known = {member.value for member in Pattern}
        for value in pattern_labels(frame):
            assert value is None or value <= known


class TestSessionLabels:
    """A bar belongs to the session it opened in."""

    def test_a_bar_carries_every_session_open_at_its_start(self) -> None:
        column = session_labels(frame_from_closes([1.10] * 4))
        assert all(value is not None for value in column)

    def test_sessions_overlap_and_the_derived_overlap_appears(self) -> None:
        # 13:00 UTC in January is inside both London and New York.
        import polars as pl

        from trading_system.core.types import Timeframe
        from trading_system.data.models import OHLCVFrame

        start = datetime(2024, 1, 3, 14, 0, tzinfo=UTC)
        frame = OHLCVFrame.from_raw(
            pl.DataFrame(
                {
                    "timestamp": [start],
                    "open": [1.1],
                    "high": [1.1],
                    "low": [1.1],
                    "close": [1.1],
                    "volume": [1.0],
                }
            ),
            "TESTFX",
            Timeframe.M15,
        )
        assert session_labels(frame)[0] == frozenset({"LONDON", "NEWYORK", "LONDON_NY_OVERLAP"})


class TestLabelColumns:
    """Only what was asked for gets built."""

    def test_nothing_is_built_for_an_empty_request(self) -> None:
        assert label_columns(frame_from_closes([1.1, 1.2])) == {}

    def test_only_the_requested_categories_are_built(self) -> None:
        columns = label_columns(frame_from_closes([1.1] * 5), [LabelCategory.SESSION])
        assert set(columns) == {"session"}

    def test_regime_cannot_be_built_yet(self) -> None:
        with pytest.raises(ValueError, match="no Regime module"):
            label_columns(frame_from_closes([1.1] * 5), [LabelCategory.REGIME])


class TestRequiredCategories:
    """A strategy pays for classification only where it asks a question."""

    def test_a_spec_with_no_label_operators_needs_none(self) -> None:
        spec = strategy_spec(trigger=leaf("gt", ref("rsi", period=14), 50.0))
        assert required_label_categories(spec) == ()

    def test_categories_are_collected_from_every_condition_slot(self) -> None:
        spec = strategy_spec(
            trigger=leaf("session_is", None, labels("LONDON")),
            confirmation=[leaf("pattern_is", None, labels("HAMMER"))],
            confirmation_window_bars=2,
        )
        assert required_label_categories(spec) == (LabelCategory.PATTERN, LabelCategory.SESSION)

    def test_a_category_asked_about_twice_is_listed_once(self) -> None:
        spec = strategy_spec(
            trigger=leaf("pattern_is", None, labels("HAMMER")),
            confirmation=[leaf("pattern_is", None, labels("DOJI"))],
            confirmation_window_bars=2,
        )
        assert required_label_categories(spec) == (LabelCategory.PATTERN,)


class TestCompiledLabelConditions:
    """End to end: a spec asking about candles fires on the candle."""

    def test_a_pattern_confirmation_fires_on_the_pattern_bar(self) -> None:
        rsi = ref("rsi", period=14)
        rsi_key = "rsi_14"
        spec = strategy_spec(
            trigger=leaf("gt", rsi, 50.0),
            confirmation=[leaf("pattern_is", None, labels("HAMMER", "DOJI"))],
            confirmation_window_bars=4,
        )
        series = series_from_features(
            [1.10] * 4,
            {rsi_key: [40.0, 60.0, 60.0, 60.0]},
            labels={
                "pattern": [
                    frozenset(),
                    frozenset({"INSIDE_BAR"}),
                    frozenset(),
                    frozenset({"DOJI"}),
                ]
            },
        )
        signals = compile_entry(spec, FeatureRegistry.from_strategy(spec)).run(series).signals
        assert [signal.context["signal_bar_index"] for signal in signals] == [3]

    def test_a_pattern_test_is_unknown_during_warmup_so_not_never_fires_there(self) -> None:
        rsi = ref("rsi", period=14)
        spec = strategy_spec(
            trigger=leaf("gt", rsi, 50.0),
            confirmation=[{"type": "not", "condition": leaf("pattern_is", None, labels("DOJI"))}],  # type: ignore[list-item]
            confirmation_window_bars=1,
        )
        series = series_from_features(
            [1.10] * 3,
            {"rsi_14": [60.0, 60.0, 60.0]},
            labels={"pattern": [None, None, frozenset({"HAMMER"})]},
        )
        signals = compile_entry(spec, FeatureRegistry.from_strategy(spec)).run(series).signals
        # Bars 0 and 1 cannot classify the candle, so "not a doji" is unknown
        # there and must not fire. Bar 2 knows, and does.
        assert [signal.context["signal_bar_index"] for signal in signals] == [2]

    def test_a_session_condition_reads_the_session_column(self) -> None:
        spec = strategy_spec(
            trigger=leaf("session_is", None, labels("LONDON")),
            invalidation=Invalidation(price_level=1.0),
        )
        series = series_from_features(
            [1.10] * 3,
            labels={
                "session": [frozenset({"TOKYO"}), frozenset({"LONDON"}), frozenset({"NEWYORK"})]
            },
        )
        signals = compile_entry(spec, FeatureRegistry.from_strategy(spec)).run(series).signals
        assert [signal.context["signal_bar_index"] for signal in signals] == [1]

    def test_from_frame_builds_exactly_the_categories_a_spec_needs(self) -> None:
        spec = strategy_spec(
            trigger=leaf("pattern_is", None, labels("DOJI")),
            invalidation=Invalidation(price_level=1.0),
        )
        frame = frame_from_closes([1.10 + 0.001 * index for index in range(30)])
        series = BarSeries.from_frame(frame, None, required_label_categories(spec))
        assert series.label_categories == ("pattern",)
        compile_entry(spec, FeatureRegistry.from_strategy(spec)).run(series)
