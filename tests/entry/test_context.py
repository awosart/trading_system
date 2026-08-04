"""BarContext: what it exposes, and the proof that none of it reaches forward."""

import inspect
from datetime import timedelta

import pytest

from trading_system.core.exceptions import ValidationError
from trading_system.entry.context import PRICE_FIELDS, BarContext, BarSeries
from trading_system.features.pipeline import FeaturePipeline, FeatureSpec

from .conftest import TIMEFRAME, frame_from_closes, series_from_features

#: The whole public surface of a bar context. Anything added to this set is a new
#: route to data and has to be argued for; anything that could name a bar after
#: ``t`` does not belong in it at all.
EXPECTED_API = frozenset(
    {"index", "symbol", "timeframe", "bar_close_ts", "price", "feature", "feature_snapshot"}
)


class TestNoLookahead:
    """The DoD check: reaching bar ``t+1`` is an absence, not a rule."""

    def test_public_api_is_exactly_the_history_only_surface(self) -> None:
        public = {name for name in dir(BarContext) if not name.startswith("_")}
        assert public == EXPECTED_API

    def test_there_is_no_forward_accessor_by_any_conventional_name(self) -> None:
        # Named explicitly rather than left to the set comparison above, so a
        # future reader sees which shapes were considered and rejected.
        for name in (
            "next",
            "next_bar",
            "peek",
            "future",
            "forward",
            "bar_at",
            "at",
            "series",
            "frame",
            "df",
            "__getitem__",
            "__iter__",
            "__len__",
            "__next__",
        ):
            assert not hasattr(BarContext, name), f"BarContext exposes {name!r}"

    @pytest.mark.parametrize("accessor", ["price", "feature"])
    def test_every_data_accessor_counts_backwards(self, accessor: str) -> None:
        parameters = list(inspect.signature(getattr(BarContext, accessor)).parameters)
        assert parameters[-1] == "lookback", (
            f"{accessor} must take a lookback; an offset that could be positive would be a "
            "forward reference"
        )

    def test_context_slots_prevent_attaching_extra_state(self) -> None:
        series = series_from_features([1.0, 2.0, 3.0])
        ctx = series.context(1)
        with pytest.raises(AttributeError):
            ctx.smuggled = series  # type: ignore[attr-defined]

    def test_a_negative_lookback_is_rejected_rather_than_wrapping(self) -> None:
        # Python would happily read index -1 as the last bar of the series, which
        # is the future. The API refuses instead.
        series = series_from_features([1.0, 2.0, 3.0], {"f": [10.0, 20.0, 30.0]})
        ctx = series.context(0)
        with pytest.raises(ValueError, match="non-negative"):
            ctx.price("close", -1)
        with pytest.raises(ValueError, match="non-negative"):
            ctx.feature("f", -1)

    def test_every_accessor_agrees_with_a_series_truncated_at_the_bar(self) -> None:
        # The real proof, by exhaustion over the API: if any accessor could read
        # past ``t``, deleting everything past ``t`` would change its answer.
        closes = [1.10 + 0.01 * index for index in range(20)]
        features = {
            "fast": [None, None, *[value + 0.5 for value in closes[2:]]],
            "slow": [*[None] * 5, *[value - 0.5 for value in closes[5:]]],
        }
        full = series_from_features(closes, features)

        for index in range(len(closes)):
            truncated = full.truncated(index + 1)
            here, there = full.context(index), truncated.context(index)
            assert here.index == there.index
            assert here.symbol == there.symbol
            assert here.timeframe == there.timeframe
            assert here.bar_close_ts == there.bar_close_ts
            assert here.feature_snapshot() == there.feature_snapshot()
            for lookback in range(6):
                for field in PRICE_FIELDS:
                    assert here.price(field, lookback) == there.price(field, lookback)
                for key in ("fast", "slow"):
                    assert here.feature(key, lookback) == there.feature(key, lookback)


class TestAccess:
    """Ordinary reads."""

    def test_lookback_zero_is_the_current_bar(self) -> None:
        series = series_from_features([1.0, 2.0, 3.0])
        assert series.context(2).price("close") == 3.0

    def test_lookback_walks_backwards_one_bar_at_a_time(self) -> None:
        series = series_from_features([1.0, 2.0, 3.0])
        ctx = series.context(2)
        assert [ctx.price("close", back) for back in range(3)] == [3.0, 2.0, 1.0]

    def test_reading_before_the_start_of_the_series_is_missing_not_an_error(self) -> None:
        series = series_from_features([1.0, 2.0])
        assert series.context(0).price("close", 1) is None

    def test_a_feature_null_and_a_missing_bar_are_both_reported_as_missing(self) -> None:
        # Both mean "this condition cannot be decided here"; distinguishing them
        # would give a caller something to branch on that it must not branch on.
        series = series_from_features([1.0, 2.0, 3.0], {"f": [None, 20.0, 30.0]})
        ctx = series.context(0)
        assert ctx.feature("f", 0) is None
        assert ctx.feature("f", 1) is None

    def test_bar_close_ts_is_open_plus_one_timeframe(self) -> None:
        series = series_from_features([1.0, 2.0, 3.0])
        first, second = series.context(0), series.context(1)
        assert second.bar_close_ts - first.bar_close_ts == TIMEFRAME.duration
        assert first.bar_close_ts == series.context(0).bar_close_ts

    def test_bar_close_ts_is_utc(self) -> None:
        moment = series_from_features([1.0]).context(0).bar_close_ts
        assert moment.tzinfo is not None
        assert moment.utcoffset() == timedelta(0)

    def test_feature_snapshot_reads_only_the_current_bar(self) -> None:
        series = series_from_features(
            [1.0, 2.0, 3.0], {"a": [1.0, 2.0, 3.0], "b": [None, 5.0, 6.0]}
        )
        assert series.context(1).feature_snapshot() == {"a": 2.0, "b": 5.0}

    def test_unknown_price_field_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="unknown price field"):
            series_from_features([1.0]).context(0).price("vwap")

    def test_unknown_feature_is_rejected_with_what_is_available(self) -> None:
        series = series_from_features([1.0], {"rsi_14": [50.0]})
        with pytest.raises(ValidationError, match="rsi_14"):
            series.context(0).feature("ema_20")

    def test_repr_names_the_symbol_and_bar(self) -> None:
        assert "TESTFX" in repr(series_from_features([1.0, 2.0]).context(1))


class TestBarSeries:
    """Construction, alignment and iteration."""

    def test_contexts_are_yielded_oldest_first(self) -> None:
        series = series_from_features([1.0, 2.0, 3.0])
        assert [ctx.index for ctx in series.contexts()] == [0, 1, 2]

    def test_index_outside_the_series_is_an_index_error(self) -> None:
        series = series_from_features([1.0, 2.0])
        with pytest.raises(IndexError):
            series.context(2)
        with pytest.raises(IndexError):
            series.context(-1)

    def test_from_frame_carries_every_feature_column(self) -> None:
        frame = frame_from_closes([1.0 + 0.01 * i for i in range(40)])
        pipeline = FeaturePipeline([FeatureSpec(name="sma_5", kind="sma", params={"period": 5})])
        series = BarSeries.from_frame(frame, pipeline.compute(frame))
        assert series.feature_keys == ("sma_5",)
        assert series.context(0).feature("sma_5") is None
        assert series.context(10).feature("sma_5") is not None

    def test_from_frame_without_features_carries_prices_only(self) -> None:
        series = BarSeries.from_frame(frame_from_closes([1.0, 2.0]))
        assert series.feature_keys == ()
        assert series.context(1).price("close") == 2.0

    def test_misaligned_features_are_rejected(self) -> None:
        frame = frame_from_closes([1.0 + 0.01 * i for i in range(40)])
        pipeline = FeaturePipeline([FeatureSpec(name="sma_5", kind="sma", params={"period": 5})])
        shorter = frame.last(20)
        with pytest.raises(ValidationError, match="bars"):
            BarSeries.from_frame(frame, pipeline.compute(shorter))

    def test_features_for_another_symbol_are_rejected(self) -> None:
        frame = frame_from_closes([1.0 + 0.01 * i for i in range(40)])
        other = frame_from_closes([1.0 + 0.01 * i for i in range(40)], symbol="OTHER")
        pipeline = FeaturePipeline([FeatureSpec(name="sma_5", kind="sma", params={"period": 5})])
        with pytest.raises(ValidationError, match="OTHER"):
            BarSeries.from_frame(frame, pipeline.compute(other))

    def test_a_column_of_the_wrong_length_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="rows"):
            series_from_features([1.0, 2.0, 3.0], {"f": [1.0, 2.0]})

    def test_truncated_keeps_the_leading_bars(self) -> None:
        series = series_from_features([1.0, 2.0, 3.0], {"f": [1.0, 2.0, 3.0]})
        shorter = series.truncated(2)
        assert len(shorter) == 2
        assert shorter.context(1).price("close") == 2.0
        assert shorter.feature_keys == ("f",)

    def test_truncated_rejects_a_length_beyond_the_series(self) -> None:
        with pytest.raises(ValueError, match="length must be"):
            series_from_features([1.0, 2.0]).truncated(3)
