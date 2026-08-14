"""Providers must hand back validated frames, never raw tables."""

import subprocess
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl
import pytest

from trading_system.core.exceptions import DataError
from trading_system.core.types import Timeframe
from trading_system.data.models import OHLCVFrame
from trading_system.data.providers import (
    CSVProvider,
    CSVSchema,
    DataProvider,
    DukascopyProvider,
    ParquetProvider,
)
from trading_system.data.providers.dukascopy_provider import _merge_mid, _to_date_arg

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

#: Six names in the header, seven fields in every row: the last is a per-bar
#: spread the export never named. Tab-separated, as those exports tend to be.
RAGGED_TSV = (
    "Time\tOpen\tHigh\tLow\tClose\tVolume\n"
    "2024-01-01 00:00:00\t1.1000\t1.1010\t1.0990\t1.1005\t100\t3\n"
    "2024-01-01 00:01:00\t1.1005\t1.1015\t1.1000\t1.1010\t120\t2\n"
    "2024-01-01 00:02:00\t1.1010\t1.1020\t1.1005\t1.1015\t110\t4\n"
)

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


class TestVendorFilesWithUnnamedTrailingFields:
    """A six-name header over seven tab-separated fields.

    The shape a broker export takes when it appends a per-bar spread without
    naming it. The frame has nowhere to put that column, so the question is
    only whether the file is read at all — and whether it is read *silently*,
    which is the part that matters: a row wider than its header is the usual
    symptom of the separator being wrong, in which case every column is off by
    one and the bars are nonsense that a backtest will happily consume.
    """

    def write(self, tmp_path: Path) -> Path:
        (tmp_path / "EURUSD.csv").write_text(RAGGED_TSV, encoding="utf-8")
        return tmp_path

    def test_by_default_the_file_is_refused(self, tmp_path: Path) -> None:
        schema = CSVSchema(separator="\t", timestamp_format="%Y-%m-%d %H:%M:%S")
        with pytest.raises(DataError, match="drop_unnamed_fields"):
            CSVProvider(self.write(tmp_path), schema).fetch("EURUSD", Timeframe.M1)

    def test_the_surplus_is_dropped_only_when_the_caller_says_so(self, tmp_path: Path) -> None:
        schema = CSVSchema(
            separator="\t",
            timestamp_format="%Y-%m-%d %H:%M:%S",
            drop_unnamed_fields=True,
        )
        frame = CSVProvider(self.write(tmp_path), schema).fetch("EURUSD", Timeframe.M1)
        assert len(frame) == 3

    def test_the_columns_do_not_shift_by_one(self, tmp_path: Path) -> None:
        """The failure the refusal above exists to prevent, asserted positively.

        Dropping the trailing field must not move volume into close: the whole
        risk of accepting a ragged row is that the surplus is taken from the
        wrong end.
        """
        schema = CSVSchema(
            separator="\t",
            timestamp_format="%Y-%m-%d %H:%M:%S",
            drop_unnamed_fields=True,
        )
        frame = CSVProvider(self.write(tmp_path), schema).fetch("EURUSD", Timeframe.M1)
        assert frame.df["close"].to_list() == [1.1005, 1.1010, 1.1015]
        assert frame.df["volume"].to_list() == [100.0, 120.0, 110.0]

    def test_the_wrong_separator_is_still_refused(self, tmp_path: Path) -> None:
        """``drop_unnamed_fields`` opts into a known surplus, not into guessing.

        Read with the default comma, every line is one field and no OHLCV
        column maps. That has to stay an error even with the flag set, or the
        flag would be a way to turn a mis-parsed file into a frame of nulls.
        """
        schema = CSVSchema(timestamp_format="%Y-%m-%d %H:%M:%S", drop_unnamed_fields=True)
        with pytest.raises(DataError, match="could not map columns"):
            CSVProvider(self.write(tmp_path), schema).fetch("EURUSD", Timeframe.M1)


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
    assert isinstance(DukascopyProvider(), DataProvider)


BID_H1_CSV = """timestamp,open,high,low,close,volume
1704067200000,1.1000,1.1010,1.0990,1.1005,100
1704070800000,1.1005,1.1015,1.1000,1.1010,120
"""

ASK_H1_CSV = """timestamp,open,high,low,close,volume
1704067200000,1.1002,1.1012,1.0992,1.1007,140
1704070800000,1.1007,1.1017,1.1002,1.1012,160
"""

EMPTY_H1_CSV = "timestamp,open,high,low,close,volume\n"


class FakeCLIRunner:
    """Stands in for the ``dukascopy-node`` subprocess.

    Writes the canned CSV for the requested price side to the ``-dir``/``-fn``
    path the provider asked for, instead of touching the network, and records
    every invocation so tests can assert on the argv built for it.
    """

    def __init__(
        self,
        csv_by_price_type: dict[str, str],
        *,
        returncode: int = 0,
        stderr: str = "",
        write_file: bool = True,
    ) -> None:
        """Store the canned responses this fake will hand back."""
        self.csv_by_price_type = csv_by_price_type
        self.returncode = returncode
        self.stderr = stderr
        self.write_file = write_file
        self.calls: list[list[str]] = []

    def __call__(self, argv: Sequence[str]) -> "subprocess.CompletedProcess[str]":
        args = list(argv)
        self.calls.append(args)
        if self.returncode == 0 and self.write_file:
            price_type = args[args.index("-p") + 1]
            out_dir = Path(args[args.index("-dir") + 1])
            file_name = args[args.index("-fn") + 1]
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / f"{file_name}.csv").write_text(
                self.csv_by_price_type[price_type], encoding="utf-8"
            )
        return subprocess.CompletedProcess(
            args=args, returncode=self.returncode, stdout="", stderr=self.stderr
        )


class TestDukascopyProvider:
    def test_merges_bid_ask_into_mid(self) -> None:
        runner = FakeCLIRunner({"bid": BID_H1_CSV, "ask": ASK_H1_CSV})
        provider = DukascopyProvider(runner=runner)
        frame = provider.fetch(
            "EURUSD",
            Timeframe.H1,
            datetime(2024, 1, 1, tzinfo=UTC),
            datetime(2024, 1, 1, 2, tzinfo=UTC),
        )
        assert isinstance(frame, OHLCVFrame)
        assert frame.symbol == "EURUSD"
        assert frame.timeframe is Timeframe.H1
        assert len(frame) == 2
        first = frame.df.row(0, named=True)
        assert first["open"] == pytest.approx(1.1001)
        assert first["high"] == pytest.approx(1.1011)
        assert first["low"] == pytest.approx(1.0991)
        assert first["close"] == pytest.approx(1.1006)
        assert first["volume"] == pytest.approx(120.0)

    def test_invokes_cli_once_per_side_with_correct_flags(self) -> None:
        runner = FakeCLIRunner({"bid": BID_H1_CSV, "ask": ASK_H1_CSV})
        provider = DukascopyProvider(runner=runner)
        provider.fetch(
            "EURUSD",
            Timeframe.H1,
            datetime(2024, 1, 1, tzinfo=UTC),
            datetime(2024, 1, 2, tzinfo=UTC),
        )
        assert len(runner.calls) == 2
        price_types = [call[call.index("-p") + 1] for call in runner.calls]
        assert price_types == ["bid", "ask"]
        for call in runner.calls:
            assert call[call.index("-i") + 1] == "eurusd"
            assert call[call.index("-t") + 1] == "h1"
            assert call[call.index("-from") + 1] == "2024-01-01"

    def test_requires_explicit_start(self) -> None:
        provider = DukascopyProvider(runner=FakeCLIRunner({}))
        with pytest.raises(DataError, match="requires an explicit start"):
            provider.fetch("EURUSD", Timeframe.H1)

    def test_naive_datetimes_are_rejected(self) -> None:
        provider = DukascopyProvider(runner=FakeCLIRunner({}))
        with pytest.raises(DataError, match="tz-aware"):
            provider.fetch("EURUSD", Timeframe.H1, datetime(2024, 1, 1))

    def test_nonzero_exit_is_reported(self) -> None:
        runner = FakeCLIRunner({}, returncode=1, stderr="instrument not found")
        provider = DukascopyProvider(runner=runner)
        with pytest.raises(DataError, match="instrument not found"):
            provider.fetch("EURUSD", Timeframe.H1, datetime(2024, 1, 1, tzinfo=UTC))

    def test_missing_output_file_is_reported(self) -> None:
        runner = FakeCLIRunner({}, write_file=False)
        provider = DukascopyProvider(runner=runner)
        with pytest.raises(DataError, match="wrote no output"):
            provider.fetch("EURUSD", Timeframe.H1, datetime(2024, 1, 1, tzinfo=UTC))

    def test_empty_result_yields_empty_frame(self) -> None:
        runner = FakeCLIRunner({"bid": EMPTY_H1_CSV, "ask": EMPTY_H1_CSV})
        provider = DukascopyProvider(runner=runner)
        frame = provider.fetch("EURUSD", Timeframe.H1, datetime(2024, 1, 1, tzinfo=UTC))
        assert frame.is_empty
        assert frame.symbol == "EURUSD"
        assert frame.timeframe is Timeframe.H1

    def test_range_is_applied_after_merge(self) -> None:
        runner = FakeCLIRunner({"bid": BID_H1_CSV, "ask": ASK_H1_CSV})
        provider = DukascopyProvider(runner=runner)
        frame = provider.fetch(
            "EURUSD",
            Timeframe.H1,
            datetime(2024, 1, 1, 1, tzinfo=UTC),
            datetime(2024, 1, 1, 2, tzinfo=UTC),
        )
        assert len(frame) == 1
        assert frame.start == datetime(2024, 1, 1, 1, tzinfo=UTC)


class TestMergeMid:
    def test_only_common_timestamps_survive(self) -> None:
        bid = pl.DataFrame(
            {
                "timestamp": [
                    datetime(2024, 1, 1, tzinfo=UTC),
                    datetime(2024, 1, 1, 1, tzinfo=UTC),
                ],
                "open": [1.10, 1.11],
                "high": [1.11, 1.12],
                "low": [1.09, 1.10],
                "close": [1.105, 1.115],
                "volume": [100.0, 110.0],
            }
        ).with_columns(pl.col("timestamp").dt.convert_time_zone("UTC"))
        ask = pl.DataFrame(
            {
                "timestamp": [datetime(2024, 1, 1, tzinfo=UTC)],
                "open": [1.102],
                "high": [1.112],
                "low": [1.092],
                "close": [1.107],
                "volume": [140.0],
            }
        ).with_columns(pl.col("timestamp").dt.convert_time_zone("UTC"))

        merged, corrected = _merge_mid(bid, ask)
        assert merged.height == 1
        assert merged["close"].item(0) == pytest.approx((1.105 + 1.107) / 2)
        assert corrected == 0

    def test_empty_side_yields_empty_result(self) -> None:
        bid = pl.DataFrame(
            {
                "timestamp": [datetime(2024, 1, 1, tzinfo=UTC)],
                "open": [1.10],
                "high": [1.11],
                "low": [1.09],
                "close": [1.105],
                "volume": [100.0],
            }
        ).with_columns(pl.col("timestamp").dt.convert_time_zone("UTC"))
        ask = bid.clear()

        merged, corrected = _merge_mid(bid, ask)
        assert merged.is_empty()
        assert merged.columns == ["timestamp", "open", "high", "low", "close", "volume"]
        assert corrected == 0

    def test_low_wider_than_close_is_widened(self) -> None:
        """Reproduces the real artifact found in the EURUSD 2024-10-15 16:00 bar.

        bid low/close 1.08894/1.08893, ask low/close 1.08896/1.08896 average to
        mid_low=1.08895 > mid_close=1.088945 — the mid low sits above the mid
        close, which is impossible for a real bar.
        """
        bid = pl.DataFrame(
            {
                "timestamp": [datetime(2024, 1, 1, tzinfo=UTC)],
                "open": [1.08990],
                "high": [1.09023],
                "low": [1.08894],
                "close": [1.08893],
                "volume": [100.0],
            }
        ).with_columns(pl.col("timestamp").dt.convert_time_zone("UTC"))
        ask = pl.DataFrame(
            {
                "timestamp": [datetime(2024, 1, 1, tzinfo=UTC)],
                "open": [1.08993],
                "high": [1.09025],
                "low": [1.08896],
                "close": [1.08896],
                "volume": [100.0],
            }
        ).with_columns(pl.col("timestamp").dt.convert_time_zone("UTC"))

        merged, corrected = _merge_mid(bid, ask)
        assert corrected == 1
        row = merged.row(0, named=True)
        assert row["low"] == pytest.approx(row["close"])
        assert row["low"] <= row["open"]
        assert row["low"] <= row["close"]
        assert row["high"] >= row["open"]
        assert row["high"] >= row["close"]
        # high was already consistent; only low needed widening.
        assert row["high"] == pytest.approx((1.09023 + 1.09025) / 2)


class TestToDateArg:
    def test_midnight_is_used_as_is(self) -> None:
        assert _to_date_arg(datetime(2026, 8, 5, tzinfo=UTC)) == "2026-08-05"

    def test_nonzero_time_of_day_rolls_forward(self) -> None:
        assert _to_date_arg(datetime(2026, 8, 5, 12, 30, tzinfo=UTC)) == "2026-08-06"
