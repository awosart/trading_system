"""Store behaviour, with write idempotency as the headline concern."""

from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl
import pytest

from trading_system.core.exceptions import DataError
from trading_system.core.types import Timeframe
from trading_system.data.models import OHLCVFrame
from trading_system.data.store import ParquetStore

from .conftest import make_frame, timestamp_zone

START = datetime(2024, 1, 1, tzinfo=UTC)


@pytest.fixture
def store(tmp_path: Path) -> ParquetStore:
    """A store rooted in a temporary directory."""
    return ParquetStore(tmp_path)


def hourly(start: datetime, count: int, symbol: str = "EURUSD") -> OHLCVFrame:
    """Build an hourly frame."""
    return make_frame(start, count, timeframe=Timeframe.H1, symbol=symbol)


class TestIdempotency:
    """Re-importing a period must correct it, never duplicate it."""

    def test_repeated_upsert_does_not_duplicate(self, store: ParquetStore) -> None:
        frame = hourly(START, 24)
        for _ in range(3):
            store.upsert(frame)
        coverage = store.coverage("EURUSD", Timeframe.H1)
        assert coverage is not None
        assert coverage.bars == 24
        assert len(store.get("EURUSD", Timeframe.H1)) == 24

    def test_overlapping_upsert_merges(self, store: ParquetStore) -> None:
        store.upsert(hourly(START, 24))
        store.upsert(hourly(START + timedelta(hours=12), 24))
        coverage = store.coverage("EURUSD", Timeframe.H1)
        assert coverage is not None
        assert coverage.bars == 36  # 24 + 24 with 12 overlapping
        assert coverage.start == START
        assert coverage.end == START + timedelta(hours=35)

    def test_incoming_bars_win_on_collision(self, store: ParquetStore) -> None:
        """A corrected re-issue of a bar replaces the original."""
        store.upsert(hourly(START, 10))
        original = store.get("EURUSD", Timeframe.H1)
        assert original.df["close"].item(0) == 1.0

        corrected = hourly(START, 3)
        corrected = corrected.with_df(corrected.df.with_columns(pl.col("close") + 1000.0))
        store.upsert(corrected)

        merged = store.get("EURUSD", Timeframe.H1)
        assert len(merged) == 10  # unchanged count
        assert merged.df["close"].to_list()[:3] == [1001.0, 1002.0, 1003.0]
        assert merged.df["close"].item(3) == 4.0  # untouched


class TestPartitioning:
    def test_data_splits_by_year(self, store: ParquetStore) -> None:
        frame = hourly(datetime(2023, 12, 31, 20, 0, tzinfo=UTC), 10)
        store.upsert(frame)
        names = [path.name for path in store.partitions("EURUSD", Timeframe.H1)]
        assert names == ["2023.parquet", "2024.parquet"]

    def test_layout_matches_the_documented_scheme(self, store: ParquetStore) -> None:
        store.upsert(hourly(START, 5))
        expected = store.root / "ohlcv" / "EURUSD" / "H1" / "2024.parquet"
        assert expected.is_file()

    def test_range_query_spans_partitions(self, store: ParquetStore) -> None:
        store.upsert(hourly(datetime(2023, 12, 31, 20, 0, tzinfo=UTC), 10))
        crossing = store.get(
            "EURUSD",
            Timeframe.H1,
            datetime(2023, 12, 31, 22, 0, tzinfo=UTC),
            datetime(2024, 1, 1, 2, 0, tzinfo=UTC),
        )
        assert len(crossing) == 4


class TestGet:
    def test_returns_half_open_range(self, store: ParquetStore) -> None:
        store.upsert(hourly(START, 24))
        window = store.get(
            "EURUSD", Timeframe.H1, START + timedelta(hours=2), START + timedelta(hours=5)
        )
        assert len(window) == 3
        assert window.start == START + timedelta(hours=2)
        assert window.end == START + timedelta(hours=4)

    def test_unknown_symbol_yields_empty_frame(self, store: ParquetStore) -> None:
        frame = store.get("NOPE", Timeframe.H1)
        assert frame.is_empty
        assert frame.symbol == "NOPE"

    def test_result_is_utc_and_sorted(self, store: ParquetStore) -> None:
        store.upsert(hourly(START, 24))
        frame = store.get("EURUSD", Timeframe.H1)
        assert timestamp_zone(frame.df) == "UTC"
        assert frame.df["timestamp"].is_sorted()

    def test_rejects_naive_bounds(self, store: ParquetStore) -> None:
        store.upsert(hourly(START, 5))
        with pytest.raises(ValueError, match="tz-aware"):
            store.get("EURUSD", Timeframe.H1, datetime(2024, 1, 1))  # noqa: DTZ001


class TestCoverage:
    def test_reports_range_and_count(self, store: ParquetStore) -> None:
        store.upsert(hourly(START, 24))
        coverage = store.coverage("EURUSD", Timeframe.H1)
        assert coverage is not None
        assert coverage.start == START
        assert coverage.end == START + timedelta(hours=23)
        assert coverage.bars == 24
        assert coverage.symbol == "EURUSD"

    def test_none_when_nothing_stored(self, store: ParquetStore) -> None:
        assert store.coverage("EURUSD", Timeframe.H1) is None


def test_empty_upsert_writes_nothing(store: ParquetStore) -> None:
    assert store.upsert(OHLCVFrame.empty("EURUSD", Timeframe.H1)) == 0
    assert store.partitions("EURUSD", Timeframe.H1) == []


def test_symbols_and_timeframes_are_listed(store: ParquetStore) -> None:
    store.upsert(hourly(START, 5))
    store.upsert(hourly(START, 5, symbol="GBPUSD"))
    store.upsert(make_frame(START, 5, timeframe=Timeframe.M1))
    assert store.symbols() == ["EURUSD", "GBPUSD"]
    assert set(store.timeframes("EURUSD")) == {Timeframe.H1, Timeframe.M1}


def test_empty_store_lists_nothing(store: ParquetStore) -> None:
    assert store.symbols() == []
    assert store.timeframes("EURUSD") == []


@pytest.mark.parametrize("symbol", ["../escape", "a/b", "sym bol", "sym;rm", ""])
def test_unsafe_symbols_are_rejected(store: ParquetStore, symbol: str) -> None:
    """A crafted symbol must not be able to write outside the store root."""
    with pytest.raises(DataError, match="unsafe symbol"):
        store.directory_for(symbol, Timeframe.H1)


def test_duckdb_query_layer_is_available(store: ParquetStore) -> None:
    """DuckDB remains usable for aggregate SQL over the partitions."""
    store.upsert(hourly(START, 24))
    rows = store.query("SELECT count(*) FROM read_parquet(?)", "EURUSD", Timeframe.H1)
    assert rows == [(24,)]


def test_roundtrip_preserves_values(store: ParquetStore) -> None:
    original = hourly(START, 50)
    store.upsert(original)
    assert store.get("EURUSD", Timeframe.H1).df.equals(original.df)
