"""Providers must hand back validated frames, never raw tables."""

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from trading_system.core.exceptions import DataError
from trading_system.core.types import Timeframe
from trading_system.data.models import OHLCVFrame
from trading_system.data.providers import (
    CSVProvider,
    CSVSchema,
    DataProvider,
    ParquetProvider,
)

from .conftest import make_frame, timestamp_zone

START = datetime(2024, 1, 1, tzinfo=UTC)

ISO_CSV = """timestamp,open,high,low,close,volume
2024-01-01T00:00:00,1.1000,1.1010,1.0990,1.1005,100
2024-01-01T00:01:00,1.1005,1.1015,1.1000,1.1010,120
2024-01-01T00:02:00,1.1010,1.1020,1.1005,1.1015,110
"""

SPLIT_DATETIME_CSV = """DATE,TIME,OPEN,HIGH,LOW,CLOSE,TICKVOL
2024.01.01,00:00,1.10000,1.10100,1.09900,1.10050,100
2024.01.01,00:01,1.10050,1.10150,1.10000,1.10100,120
"""

UNSORTED_CSV = """timestamp,open,high,low,close,volume
2024-01-01T00:02:00,1.1010,1.1020,1.1005,1.1015,110
2024-01-01T00:00:00,1.1000,1.1010,1.0990,1.1005,100
2024-01-01T00:01:00,1.1005,1.1015,1.1000,1.1010,120
"""


@pytest.fixture
def csv_dir(tmp_path: Path) -> Path:
    """A directory holding EURUSD.csv."""
    (tmp_path / "EURUSD.csv").write_text(ISO_CSV, encoding="utf-8")
    return tmp_path


class TestCSVProvider:
    def test_reads_iso_timestamps(self, csv_dir: Path) -> None:
        frame = CSVProvider(csv_dir).fetch("EURUSD", Timeframe.M1)
        assert isinstance(frame, OHLCVFrame)
        assert len(frame) == 3
        assert frame.start == START
        assert frame.df["close"].item(0) == pytest.approx(1.1005)

    def test_returns_a_validated_frame(self, csv_dir: Path) -> None:
        """The provider's contract is an OHLCVFrame, so invariants hold downstream."""
        frame = CSVProvider(csv_dir).fetch("EURUSD", Timeframe.M1)
        assert timestamp_zone(frame.df) == "UTC"
        assert frame.symbol == "EURUSD"
        assert frame.timeframe is Timeframe.M1

    def test_source_timezone_is_converted(self, tmp_path: Path) -> None:
        """A file recorded in broker time lands on the correct UTC instant."""
        (tmp_path / "EURUSD.csv").write_text(ISO_CSV, encoding="utf-8")
        frame = CSVProvider(tmp_path, CSVSchema(source_tz="Europe/Riga")).fetch(
            "EURUSD", Timeframe.M1
        )
        # Riga is UTC+2 in January, so 00:00 local is 22:00 the previous day.
        assert frame.start == datetime(2023, 12, 31, 22, 0, tzinfo=UTC)

    def test_split_date_and_time_columns(self, tmp_path: Path) -> None:
        (tmp_path / "EURUSD.csv").write_text(SPLIT_DATETIME_CSV, encoding="utf-8")
        schema = CSVSchema(
            column_map={
                "open": "open",
                "high": "high",
                "low": "low",
                "close": "close",
                "tickvol": "volume",
            },
            date_column="DATE",
            time_column="TIME",
            timestamp_format="%Y.%m.%d %H:%M",
        )
        frame = CSVProvider(tmp_path, schema).fetch("EURUSD", Timeframe.M1)
        assert len(frame) == 2
        assert frame.start == START
        assert frame.df["volume"].item(0) == 100.0

    def test_unsorted_input_is_normalised(self, tmp_path: Path) -> None:
        (tmp_path / "EURUSD.csv").write_text(UNSORTED_CSV, encoding="utf-8")
        frame = CSVProvider(tmp_path).fetch("EURUSD", Timeframe.M1)
        assert frame.df["timestamp"].is_sorted()
        assert frame.start == START

    def test_range_is_applied(self, csv_dir: Path) -> None:
        frame = CSVProvider(csv_dir).fetch(
            "EURUSD", Timeframe.M1, START + timedelta(minutes=1), START + timedelta(minutes=2)
        )
        assert len(frame) == 1
        assert frame.start == START + timedelta(minutes=1)

    def test_single_file_path(self, tmp_path: Path) -> None:
        path = tmp_path / "anything.csv"
        path.write_text(ISO_CSV, encoding="utf-8")
        assert len(CSVProvider(path).fetch("EURUSD", Timeframe.M1)) == 3

    def test_missing_file_is_reported(self, tmp_path: Path) -> None:
        with pytest.raises(DataError, match="no CSV for"):
            CSVProvider(tmp_path).fetch("EURUSD", Timeframe.M1)

    def test_unmappable_columns_are_reported(self, tmp_path: Path) -> None:
        (tmp_path / "EURUSD.csv").write_text("a,b,c\n1,2,3\n", encoding="utf-8")
        with pytest.raises(DataError, match="could not map columns"):
            CSVProvider(tmp_path).fetch("EURUSD", Timeframe.M1)

    def test_date_column_without_time_column_is_reported(self, csv_dir: Path) -> None:
        schema = CSVSchema(date_column="DATE")
        with pytest.raises(DataError, match="without time_column"):
            CSVProvider(csv_dir, schema).fetch("EURUSD", Timeframe.M1)


class TestParquetProvider:
    def test_roundtrip(self, tmp_path: Path) -> None:
        provider = ParquetProvider(tmp_path)
        original = make_frame(START, 20)
        provider.write(original)
        assert provider.fetch("EURUSD", Timeframe.M1).df.equals(original.df)

    def test_range_is_applied(self, tmp_path: Path) -> None:
        provider = ParquetProvider(tmp_path)
        provider.write(make_frame(START, 20))
        window = provider.fetch(
            "EURUSD", Timeframe.M1, START + timedelta(minutes=5), START + timedelta(minutes=8)
        )
        assert len(window) == 3

    def test_missing_data_is_reported(self, tmp_path: Path) -> None:
        with pytest.raises(DataError, match="no Parquet data"):
            ParquetProvider(tmp_path).fetch("EURUSD", Timeframe.M1)

    def test_write_creates_parent_directories(self, tmp_path: Path) -> None:
        destination = tmp_path / "nested" / "deeper" / "out.parquet"
        ParquetProvider(tmp_path).write(make_frame(START, 3), destination)
        assert destination.is_file()


def test_providers_satisfy_the_protocol(tmp_path: Path) -> None:
    """Both concrete providers structurally implement DataProvider."""
    assert isinstance(CSVProvider(tmp_path), DataProvider)
    assert isinstance(ParquetProvider(tmp_path), DataProvider)
