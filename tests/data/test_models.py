"""OHLCVFrame structural guarantees."""

from datetime import UTC, datetime, timedelta, timezone

import polars as pl
import pytest

from trading_system.core.exceptions import DataError
from trading_system.core.types import Timeframe
from trading_system.data.models import OHLCVFrame, empty_ohlcv_dataframe

from .conftest import make_frame, timestamp_zone

START = datetime(2024, 1, 1, tzinfo=UTC)


def raw(**overrides: object) -> pl.DataFrame:
    """Build a small raw OHLCV DataFrame, with optional column overrides."""
    base: dict[str, object] = {
        "timestamp": [START, START + timedelta(minutes=1)],
        "open": [1.0, 2.0],
        "high": [1.5, 2.5],
        "low": [0.5, 1.5],
        "close": [1.2, 2.2],
        "volume": [10.0, 20.0],
    }
    base.update(overrides)
    return pl.DataFrame(base)


def test_accepts_well_formed_data() -> None:
    frame = OHLCVFrame.from_raw(raw(), "EURUSD", Timeframe.M1)
    assert len(frame) == 2
    assert frame.symbol == "EURUSD"
    assert frame.timeframe is Timeframe.M1
    assert frame.start == START
    assert frame.end == START + timedelta(minutes=1)


def test_rejects_missing_columns() -> None:
    incomplete = raw().drop("volume")
    with pytest.raises(DataError, match="missing required columns"):
        OHLCVFrame.from_raw(incomplete, "EURUSD", Timeframe.M1)


def test_rejects_naive_timestamps_without_assume_tz() -> None:
    naive = raw(timestamp=[datetime(2024, 1, 1), datetime(2024, 1, 1, 0, 1)])  # noqa: DTZ001
    with pytest.raises(DataError, match="tz-naive"):
        OHLCVFrame.from_raw(naive, "EURUSD", Timeframe.M1)


def test_assume_tz_localises_naive_timestamps() -> None:
    """A broker export in local time converts to the correct UTC instant."""
    naive = raw(timestamp=[datetime(2024, 1, 1, 10, 0), datetime(2024, 1, 1, 10, 1)])  # noqa: DTZ001
    frame = OHLCVFrame.from_raw(naive, "EURUSD", Timeframe.M1, assume_tz="Europe/Riga")
    # Riga is UTC+2 in January.
    assert frame.start == datetime(2024, 1, 1, 8, 0, tzinfo=UTC)


def test_non_utc_input_is_converted() -> None:
    tokyo = timezone(timedelta(hours=9))
    aware = raw(
        timestamp=[
            datetime(2024, 1, 1, 9, 0, tzinfo=tokyo),
            datetime(2024, 1, 1, 9, 1, tzinfo=tokyo),
        ]
    )
    frame = OHLCVFrame.from_raw(aware, "EURUSD", Timeframe.M1)
    assert frame.start == datetime(2024, 1, 1, 0, 0, tzinfo=UTC)
    assert timestamp_zone(frame.df) == "UTC"


def test_from_raw_sorts_and_deduplicates() -> None:
    """Unsorted input with a repeated timestamp normalises; the later row wins."""
    messy = pl.DataFrame(
        {
            "timestamp": [START + timedelta(minutes=1), START, START],
            "open": [2.0, 1.0, 9.0],
            "high": [2.5, 1.5, 9.5],
            "low": [1.5, 0.5, 8.5],
            "close": [2.2, 1.2, 9.2],
            "volume": [20.0, 10.0, 90.0],
        }
    ).with_columns(pl.col("timestamp").dt.replace_time_zone("UTC"))
    frame = OHLCVFrame.from_raw(messy, "EURUSD", Timeframe.M1)
    assert len(frame) == 2
    assert frame.df["close"].to_list() == [9.2, 2.2]  # de-duplicated to the last copy


def test_integer_columns_are_cast_to_float() -> None:
    integers = raw(open=[1, 2], high=[2, 3], low=[0, 1], close=[1, 2], volume=[10, 20])
    frame = OHLCVFrame.from_raw(integers, "EURUSD", Timeframe.M1)
    assert all(frame.df.schema[column] == pl.Float64 for column in ("open", "high", "low", "close"))


def test_constructor_rejects_unsorted_input() -> None:
    """The strict constructor refuses what from_raw would have repaired."""
    descending = raw(timestamp=[START + timedelta(minutes=1), START]).with_columns(
        pl.col("timestamp").dt.replace_time_zone("UTC")
    )
    with pytest.raises(DataError, match="strictly increasing"):
        OHLCVFrame(
            descending.select("timestamp", "open", "high", "low", "close", "volume"),
            "EURUSD",
            Timeframe.M1,
        )


def test_constructor_rejects_duplicate_timestamps() -> None:
    duplicated = raw(timestamp=[START, START]).with_columns(
        pl.col("timestamp").dt.replace_time_zone("UTC")
    )
    with pytest.raises(DataError, match="strictly increasing"):
        OHLCVFrame(duplicated, "EURUSD", Timeframe.M1)


def test_constructor_rejects_nulls() -> None:
    holed = raw(close=[1.2, None]).with_columns(pl.col("timestamp").dt.replace_time_zone("UTC"))
    with pytest.raises(DataError, match="null values"):
        OHLCVFrame(holed, "EURUSD", Timeframe.M1)


def test_constructor_rejects_wrong_dtype() -> None:
    typed = raw().with_columns(
        pl.col("timestamp").dt.replace_time_zone("UTC"), pl.col("volume").cast(pl.Int64)
    )
    with pytest.raises(DataError, match="must be Float64"):
        OHLCVFrame(typed, "EURUSD", Timeframe.M1)


def test_constructor_rejects_empty_symbol() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        OHLCVFrame(empty_ohlcv_dataframe(), "", Timeframe.M1)


def test_structurally_valid_but_nonsensical_bars_are_accepted() -> None:
    """A close outside [low, high] is a quality issue, not a structural one.

    The frame's job is to guarantee shape; grading plausibility belongs to the
    quality module, which reports severity instead of refusing to load.
    """
    absurd = raw(high=[1.0, 1.0], low=[2.0, 2.0]).with_columns(
        pl.col("timestamp").dt.replace_time_zone("UTC")
    )
    frame = OHLCVFrame(absurd, "EURUSD", Timeframe.M1)
    assert len(frame) == 2


class TestSlice:
    """Slicing uses a half-open range so adjacent slices tile without overlap."""

    def test_half_open_range(self) -> None:
        frame = make_frame(START, 10)
        sliced = frame.slice(START + timedelta(minutes=2), START + timedelta(minutes=5))
        assert len(sliced) == 3
        assert sliced.start == START + timedelta(minutes=2)
        assert sliced.end == START + timedelta(minutes=4)

    def test_adjacent_slices_tile_exactly(self) -> None:
        frame = make_frame(START, 10)
        middle = START + timedelta(minutes=5)
        first, second = frame.slice(end=middle), frame.slice(start=middle)
        assert len(first) + len(second) == len(frame)

    def test_open_ended_bounds(self) -> None:
        frame = make_frame(START, 10)
        assert len(frame.slice(start=START + timedelta(minutes=7))) == 3
        assert len(frame.slice(end=START + timedelta(minutes=3))) == 3
        assert len(frame.slice()) == 10

    def test_rejects_naive_bounds(self) -> None:
        frame = make_frame(START, 10)
        with pytest.raises(ValueError, match="tz-aware"):
            frame.slice(start=datetime(2024, 1, 1))  # noqa: DTZ001

    def test_rejects_inverted_range(self) -> None:
        frame = make_frame(START, 10)
        with pytest.raises(ValueError, match="is after end"):
            frame.slice(START + timedelta(minutes=5), START)

    def test_slice_preserves_identity(self) -> None:
        frame = make_frame(START, 10)
        sliced = frame.slice(start=START + timedelta(minutes=2))
        assert sliced.symbol == frame.symbol
        assert sliced.timeframe is frame.timeframe


class TestLast:
    def test_returns_most_recent_bars(self) -> None:
        frame = make_frame(START, 10)
        assert len(frame.last(3)) == 3
        assert frame.last(3).end == frame.end

    def test_more_than_available_returns_everything(self) -> None:
        frame = make_frame(START, 10)
        assert len(frame.last(99)) == 10

    def test_zero_returns_empty(self) -> None:
        assert make_frame(START, 10).last(0).is_empty

    def test_negative_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            make_frame(START, 10).last(-1)


def test_empty_frame_is_valid() -> None:
    frame = OHLCVFrame.empty("EURUSD", Timeframe.M1)
    assert frame.is_empty
    assert len(frame) == 0
    assert frame.start is None
    assert frame.end is None


def test_repr_summarises_the_range() -> None:
    text = repr(make_frame(START, 10))
    assert "EURUSD" in text
    assert "n=10" in text


def test_equals_compares_identity_and_rows() -> None:
    assert make_frame(START, 5).equals(make_frame(START, 5))
    assert not make_frame(START, 5).equals(make_frame(START, 6))
    assert not make_frame(START, 5).equals(make_frame(START, 5, symbol="GBPUSD"))
