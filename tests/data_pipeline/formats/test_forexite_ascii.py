"""Tests for the Forexite ASCII reader.

The sample is the literal first twenty lines of ``data/EURGBP.txt``, CRLF and
all, embedded rather than read from disk: ``/data/`` is gitignored, so a test
that opened the real export would pass on this machine and vanish on a clean
clone. :class:`TestTheEmbeddedSampleStillMatchesTheRealExport` keeps the copy
honest by diffing it against the real file whenever that file happens to exist.
"""

from pathlib import Path

import polars as pl
import pytest

from trading_system.core.exceptions import DataError
from trading_system.data_pipeline.formats.forexite_ascii import (
    COLUMNS,
    DuplicateStamp,
    detect,
    parse,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
REAL_EXPORT = REPO_ROOT / "data" / "EURGBP.txt"

HEADER = "<TICKER>,<DTYYYYMMDD>,<TIME>,<OPEN>,<HIGH>,<LOW>,<CLOSE>,<VOL>"

# Verbatim, including the vendor's CRLF line endings.
SAMPLE_ROWS = (
    "EURGBP,20010102,230100,0.6328,0.6328,0.6328,0.6328,4",
    "EURGBP,20010102,230200,0.6328,0.6328,0.6327,0.6327,4",
    "EURGBP,20010102,230300,0.6327,0.6327,0.6327,0.6327,4",
    "EURGBP,20010102,230400,0.6327,0.6328,0.6327,0.6328,4",
    "EURGBP,20010102,230500,0.6328,0.6328,0.6328,0.6328,4",
    "EURGBP,20010102,230600,0.6327,0.6327,0.6327,0.6327,4",
    "EURGBP,20010102,230700,0.6328,0.6329,0.6328,0.6329,4",
    "EURGBP,20010102,230800,0.6329,0.6329,0.6329,0.6329,4",
    "EURGBP,20010102,230900,0.6329,0.6329,0.6329,0.6329,4",
    "EURGBP,20010102,231000,0.6329,0.6329,0.6329,0.6329,4",
    "EURGBP,20010102,231100,0.6329,0.6329,0.6329,0.6329,4",
    "EURGBP,20010102,231200,0.6329,0.6329,0.6329,0.6329,4",
    "EURGBP,20010102,231300,0.6329,0.6329,0.6329,0.6329,4",
    "EURGBP,20010102,231400,0.6329,0.6330,0.6329,0.6330,4",
    "EURGBP,20010102,231500,0.6330,0.6330,0.6330,0.6330,4",
    "EURGBP,20010102,231600,0.6330,0.6330,0.6330,0.6330,4",
    "EURGBP,20010102,232000,0.6330,0.6330,0.6330,0.6330,4",
    "EURGBP,20010102,232100,0.6330,0.6330,0.6330,0.6330,4",
    "EURGBP,20010102,233000,0.6329,0.6329,0.6329,0.6329,4",
)


def write(path: Path, lines: tuple[str, ...], *, header: str | None = HEADER) -> Path:
    """Write a CRLF file with the vendor's line endings and return its path."""
    body = list(lines) if header is None else [header, *lines]
    path.write_bytes(("\r\n".join(body) + "\r\n").encode("ascii"))
    return path


@pytest.fixture
def sample(tmp_path: Path) -> Path:
    return write(tmp_path / "EURGBP.txt", SAMPLE_ROWS)


class TestDetect:
    def test_the_real_header_is_recognised(self, sample: Path) -> None:
        assert detect(sample) is True

    def test_only_the_first_line_counts(self, tmp_path: Path) -> None:
        # The marker further down does not make it this format; a reader that
        # scanned the whole file would accept a document that merely quotes it.
        path = tmp_path / "prose.txt"
        path.write_bytes(f"some other header\r\n{HEADER}\r\n".encode("ascii"))
        assert detect(path) is False

    def test_a_headerless_file_is_rejected(self, tmp_path: Path) -> None:
        assert detect(write(tmp_path / "raw.txt", SAMPLE_ROWS, header=None)) is False

    def test_an_empty_file_is_rejected(self, tmp_path: Path) -> None:
        path = tmp_path / "empty.txt"
        path.write_bytes(b"")
        assert detect(path) is False

    def test_a_missing_file_raises_rather_than_answering_no(self, tmp_path: Path) -> None:
        # "There is no such file" is not "that file is not Forexite".
        with pytest.raises(OSError):
            detect(tmp_path / "absent.txt")


class TestParsingTheSample:
    def test_the_columns_are_exactly_the_declared_ones_in_order(self, sample: Path) -> None:
        assert tuple(parse(sample, "EURGBP").frame.columns) == COLUMNS

    def test_the_volume_column_keeps_its_vendor_name(self, sample: Path) -> None:
        # Renaming it to `volume` here would hand normalisation a decision about
        # units that nobody has made yet.
        frame = parse(sample, "EURGBP").frame
        assert "source_vol" in frame.columns
        assert "volume" not in frame.columns

    def test_every_row_is_read(self, sample: Path) -> None:
        assert parse(sample, "EURGBP").frame.height == len(SAMPLE_ROWS)

    def test_stamps_are_text_and_prices_are_floats(self, sample: Path) -> None:
        schema = parse(sample, "EURGBP").frame.schema
        assert schema["date_str"] == pl.String
        assert schema["time_str"] == pl.String
        assert schema["open"] == pl.Float64
        assert schema["source_vol"] == pl.Int64

    def test_the_first_row_survives_intact(self, sample: Path) -> None:
        row = parse(sample, "EURGBP").frame.row(0, named=True)
        assert row == {
            "ticker": "EURGBP",
            "date_str": "20010102",
            "time_str": "230100",
            "open": 0.6328,
            "high": 0.6328,
            "low": 0.6328,
            "close": 0.6328,
            "source_vol": 4,
        }

    def test_the_crlf_endings_do_not_leak_into_the_last_column(self, sample: Path) -> None:
        # A reader that split on "\n" alone would leave "\r" on <VOL> and the
        # integer cast would fail — or worse, the column would stay a string.
        assert parse(sample, "EURGBP").frame["source_vol"].to_list() == [4] * len(SAMPLE_ROWS)

    def test_a_clean_file_reports_no_duplicates(self, sample: Path) -> None:
        assert parse(sample, "EURGBP").duplicates == ()


class TestLeadingZeros:
    """The reason both stamp fields are text rather than integers."""

    def test_an_early_hour_keeps_its_leading_zero(self, tmp_path: Path) -> None:
        path = write(
            tmp_path / "EURGBP.txt",
            ("EURGBP,20010103,030100,0.6319,0.6322,0.6319,0.6321,4",),
        )
        assert parse(path, "EURGBP").frame["time_str"].to_list() == ["030100"]

    def test_an_unpadded_time_is_padded_rather_than_misread(self, tmp_path: Path) -> None:
        # 30100 means 03:01:00. Read as an integer it is five digits, and
        # %H%M%S against it is either an error or the wrong instant.
        path = write(
            tmp_path / "EURGBP.txt",
            ("EURGBP,20010103,30100,0.6319,0.6322,0.6319,0.6321,4",),
        )
        assert parse(path, "EURGBP").frame["time_str"].to_list() == ["030100"]

    def test_midnight_pads_to_all_zeros(self, tmp_path: Path) -> None:
        path = write(
            tmp_path / "EURGBP.txt",
            ("EURGBP,20010103,0,0.6319,0.6322,0.6319,0.6321,4",),
        )
        assert parse(path, "EURGBP").frame["time_str"].to_list() == ["000000"]


class TestHardFailures:
    def test_an_impossible_time_is_rejected(self, tmp_path: Path) -> None:
        path = write(
            tmp_path / "EURGBP.txt",
            ("EURGBP,20010102,246000,0.6328,0.6328,0.6328,0.6328,4",),
        )
        with pytest.raises(DataError, match="246000"):
            parse(path, "EURGBP")

    def test_an_impossible_date_is_rejected(self, tmp_path: Path) -> None:
        path = write(
            tmp_path / "EURGBP.txt",
            ("EURGBP,20010230,230100,0.6328,0.6328,0.6328,0.6328,4",),
        )
        with pytest.raises(DataError, match="20010230"):
            parse(path, "EURGBP")

    def test_one_bad_row_among_good_ones_still_fails(self, tmp_path: Path) -> None:
        # The defect must not be diluted by the rows around it.
        path = write(
            tmp_path / "EURGBP.txt",
            (*SAMPLE_ROWS, "EURGBP,20010102,246000,0.6328,0.6328,0.6328,0.6328,4"),
        )
        with pytest.raises(DataError, match="246000"):
            parse(path, "EURGBP")

    def test_a_ticker_mismatch_names_both_symbols(self, sample: Path) -> None:
        with pytest.raises(DataError, match="EURUSD.*EURGBP"):
            parse(sample, "EURUSD")

    def test_the_ticker_comparison_is_case_sensitive(self, sample: Path) -> None:
        # Silently case-folding would make a file named after one symbol and
        # filled with another pass whenever the difference was only in case.
        with pytest.raises(DataError):
            parse(sample, "eurgbp")

    def test_a_file_without_the_header_is_rejected(self, tmp_path: Path) -> None:
        path = write(tmp_path / "EURGBP.txt", SAMPLE_ROWS, header=None)
        with pytest.raises(DataError, match="<TICKER>"):
            parse(path, "EURGBP")

    def test_a_file_over_the_size_ceiling_is_refused(self, sample: Path) -> None:
        with pytest.raises(DataError, match="max_bytes"):
            parse(sample, "EURGBP", max_bytes=10)


class TestDuplicateStamps:
    """Reported, kept, and never quietly resolved."""

    @pytest.fixture
    def doubled(self, tmp_path: Path) -> Path:
        return write(
            tmp_path / "EURGBP.txt",
            (
                *SAMPLE_ROWS,
                # Same stamp as the first row, different prices: a correction,
                # or two bars stamped alike. The reader cannot tell which.
                "EURGBP,20010102,230100,0.6400,0.6400,0.6400,0.6400,9",
                "EURGBP,20010102,230200,0.6401,0.6401,0.6401,0.6401,9",
            ),
        )

    def test_the_repeated_stamps_come_back(self, doubled: Path) -> None:
        assert parse(doubled, "EURGBP").duplicates == (
            DuplicateStamp(date_str="20010102", time_str="230100", count=2),
            DuplicateStamp(date_str="20010102", time_str="230200", count=2),
        )

    def test_the_rows_are_still_in_the_frame(self, doubled: Path) -> None:
        # Not dropped: which of the two to keep is the caller's decision, and a
        # reader that already made it would hide that it had been made.
        result = parse(doubled, "EURGBP")
        assert result.frame.height == len(SAMPLE_ROWS) + 2
        at_stamp = result.frame.filter(
            (pl.col("date_str") == "20010102") & (pl.col("time_str") == "230100")
        )
        assert sorted(at_stamp["close"].to_list()) == [0.6328, 0.6400]

    def test_duplicates_do_not_raise(self, doubled: Path) -> None:
        # A handful of repeated stamps must not make a multi-year export
        # unreadable; that is the difference between this and a bad time.
        assert parse(doubled, "EURGBP").frame.height > 0


@pytest.mark.skipif(not REAL_EXPORT.exists(), reason="data/EURGBP.txt is gitignored")
class TestTheEmbeddedSampleStillMatchesTheRealExport:
    """Guards the copy above against drifting from the file it was taken from."""

    def test_the_header_and_first_rows_are_byte_identical(self) -> None:
        with REAL_EXPORT.open("rb") as handle:
            actual = [handle.readline() for _ in range(len(SAMPLE_ROWS) + 1)]
        expected = [f"{line}\r\n".encode("ascii") for line in (HEADER, *SAMPLE_ROWS)]
        assert actual == expected

    def test_the_real_export_is_detected(self) -> None:
        assert detect(REAL_EXPORT) is True
