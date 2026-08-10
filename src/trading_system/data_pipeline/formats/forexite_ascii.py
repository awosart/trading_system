"""Forexite's ASCII M1 export, read exactly as the vendor wrote it.

This module is a *reader*, not an interpreter. It turns one vendor file into a
polars frame whose columns still carry the vendor's own spellings and units, and
it refuses to guess at anything it cannot verify. Turning that into an
:class:`~trading_system.data.models.OHLCVFrame` — a real UTC timestamp, a
``volume`` column whose meaning has been decided, de-duplication policy — is the
next stage's job and deliberately not done here.

**Dates and times are strings, and that is the whole point.** The vendor writes
them as bare integers with no separators::

    EURGBP,20010102,230100,0.6328,0.6328,0.6328,0.6328,4

Read ``<TIME>`` as an integer and ``030000`` becomes ``30000``: the leading zero
is gone, and every downstream parse of ``%H%M%S`` is now working on a five-digit
string that either fails or, worse, succeeds against a different clock time.
So both fields are read as text and left-padded explicitly — ``date_str`` to
eight characters, ``time_str`` to six — before anything else touches them. The
shipped files are already padded; the padding is here so that a file that is not
produces a correct value rather than a plausible wrong one.

**``source_vol``, not ``volume``.** The vendor's volume column is a
vendor-defined quantity whose meaning this module has no way to establish, and
in these exports it carries no information at all: measured over EURGBP, GBPJPY,
XAGUSD and USDJPY — 29.2 million rows — it is the constant ``4``, one distinct
value per file. Naming it ``volume`` would hand the next stage a decision nobody
has made, and would let a column that says nothing flow into indicators that
weight by it. The rename, and whatever unit it turns out to be in, belong to
normalisation, which is also where the question of dropping it has to be
answered.

**Duplicates are returned, never dropped.** A repeated ``(date, time)`` is a
real defect in vendor data and the caller has to decide what it means — a
correction re-sent, or two genuinely different bars stamped alike. Dropping the
second silently would make the file look clean; raising would make an otherwise
usable multi-year export unreadable over a handful of rows. So the rows stay in
the frame and the offending stamps come back beside it. That is why
:func:`parse` returns a :class:`ForexiteFile` and not a bare frame: a lone
``DataFrame`` has nowhere to put the finding, and a side channel the caller has
to remember to read is the same as no finding at all.

Everything the file *cannot* be read as is a hard failure instead: a missing
header, a ticker that is not the one asked for, a date or time that is not a
real instant.
"""

from dataclasses import dataclass
from pathlib import Path

import polars as pl

from trading_system.core.exceptions import DataError

#: Literal text the first line must contain for the file to be this format.
HEADER_MARKER = "<TICKER>"

#: Output column names, in order. ``source_vol`` is deliberately not ``volume``.
COLUMNS: tuple[str, ...] = (
    "ticker",
    "date_str",
    "time_str",
    "open",
    "high",
    "low",
    "close",
    "source_vol",
)

#: Column dtypes, applied positionally so that renaming cannot misalign them.
_DTYPES: list[pl.DataType] = [
    pl.String(),
    pl.String(),
    pl.String(),
    pl.Float64(),
    pl.Float64(),
    pl.Float64(),
    pl.Float64(),
    pl.Int64(),
]

DATE_WIDTH = 8
TIME_WIDTH = 6

_DATE_FORMAT = "%Y%m%d"
_TIME_FORMAT = "%H%M%S"

#: Longest first line :func:`detect` will read before giving up. A file whose
#: first line is longer than this is not a line-oriented CSV, whatever else it
#: is, and the bound keeps detection from paging in a single-line gigabyte.
_MAX_HEADER_BYTES = 4096

#: Default ceiling for :func:`parse`. The largest export shipped with this
#: project is 422 MiB (GBPCHF), so every real file clears it while anything
#: larger has to be admitted deliberately — the frame is collected into memory
#: and a caller who has not thought about that should find out here. Measured
#: on EURGBP, 411.6 MiB and 7,994,547 rows: 3.2 s, a 457 MiB frame and a peak
#: RSS of 1066 MiB, so budget roughly 2.5x the file size in memory.
DEFAULT_MAX_BYTES = 512 * 1024 * 1024


@dataclass(frozen=True)
class DuplicateStamp:
    """One ``(date, time)`` that the file carries more than once.

    Attributes:
        date_str: Padded ``YYYYMMDD`` date, as it appears in the frame.
        time_str: Padded ``HHMMSS`` time, as it appears in the frame.
        count: How many rows share this stamp. Always at least two.
    """

    date_str: str
    time_str: str
    count: int


@dataclass(frozen=True)
class ForexiteFile:
    """One parsed vendor file, together with what was wrong with it.

    Attributes:
        frame: The rows, with :data:`COLUMNS` in order. Duplicated stamps are
            still present; nothing has been removed.
        duplicates: Stamps appearing more than once, ascending. Empty when the
            file is clean.
    """

    frame: pl.DataFrame
    duplicates: tuple[DuplicateStamp, ...]


def detect(path: Path) -> bool:
    """Whether ``path`` looks like a Forexite ASCII export.

    The test is deliberately literal and cheap: the first line must contain
    :data:`HEADER_MARKER`. No attempt is made to validate the remaining column
    names, because a file carrying ``<TICKER>`` and a different column order is
    a Forexite file this reader does not yet handle — which :func:`parse` should
    say, rather than detection quietly answering "not my format".

    Args:
        path: File to inspect.

    Returns:
        ``True`` when the first line contains ``<TICKER>``.

    Raises:
        OSError: If the file cannot be opened. A missing file is a caller error,
            not a negative detection.
    """
    with path.open("rb") as handle:
        first_line = handle.readline(_MAX_HEADER_BYTES)
    return HEADER_MARKER.encode("ascii") in first_line


def parse(
    path: Path,
    expected_symbol: str,
    *,
    max_bytes: int = DEFAULT_MAX_BYTES,
) -> ForexiteFile:
    """Read one vendor file into a frame, verifying what can be verified.

    The scan is lazy and collected through the streaming engine, so the file is
    read in chunks rather than materialised as text; the resulting frame is
    still held in memory, which is what ``max_bytes`` guards.

    Args:
        path: File to read.
        expected_symbol: Symbol the caller believes this file holds. Compared
            against the ``<TICKER>`` field of every row, exactly and
            case-sensitively — a file named ``EURGBP.txt`` whose rows say
            something else is the case this argument exists for.
        max_bytes: Refuse files larger than this. Raise it deliberately when a
            genuinely larger export has to be read.

    Returns:
        The rows and any duplicated stamps.

    Raises:
        DataError: If the file is larger than ``max_bytes``, does not carry the
            Forexite header, holds a ticker other than ``expected_symbol``, or
            contains a date or time that is not a real instant.
    """
    size = path.stat().st_size
    if size > max_bytes:
        raise DataError(
            f"{path}: {size} bytes exceeds max_bytes={max_bytes}; the frame is "
            "collected into memory, so raise max_bytes deliberately if this "
            "file really has to be read whole"
        )

    if not detect(path):
        raise DataError(
            f"{path}: first line does not contain {HEADER_MARKER!r}, so this is "
            "not a Forexite ASCII export"
        )

    frame = (
        pl.scan_csv(
            path,
            has_header=True,
            new_columns=list(COLUMNS),
            schema_overrides=_DTYPES,
        )
        .with_columns(
            pl.col("date_str").str.zfill(DATE_WIDTH),
            pl.col("time_str").str.zfill(TIME_WIDTH),
        )
        .collect(engine="streaming")
    )

    _check_ticker(frame, path=path, expected_symbol=expected_symbol)
    _check_stamps(frame, path=path)
    return ForexiteFile(frame=frame, duplicates=_duplicate_stamps(frame))


def _check_ticker(frame: pl.DataFrame, *, path: Path, expected_symbol: str) -> None:
    """Reject a file whose rows name a symbol other than the one asked for."""
    found = sorted(value for value in frame["ticker"].unique().to_list() if value is not None)
    if found == [expected_symbol]:
        return
    raise DataError(
        f"{path}: expected ticker {expected_symbol!r} but the file holds {found!r}; "
        "the comparison is case-sensitive and exact"
    )


def _check_stamps(frame: pl.DataFrame, *, path: Path) -> None:
    """Reject dates or times that do not denote a real instant.

    Validity is delegated to polars' own strptime rather than to a hand-written
    range check, so that "is this a clock time" has one answer in one place.
    ``246000`` fails here; ``030000`` does not, which is the pair of cases the
    padding above exists to keep distinguishable.
    """
    for column, fmt, width, kind in (
        ("date_str", _DATE_FORMAT, DATE_WIDTH, "date"),
        ("time_str", _TIME_FORMAT, TIME_WIDTH, "time"),
    ):
        parsed = (
            pl.col(column).str.to_date(fmt, strict=False)
            if kind == "date"
            else pl.col(column).str.to_time(fmt, strict=False)
        )
        bad = frame.filter(parsed.is_null()).select(column).unique().sort(column)
        if bad.height == 0:
            continue
        samples = bad[column].head(5).to_list()
        raise DataError(
            f"{path}: {bad.height} distinct invalid {kind} value(s) in {column!r}, "
            f"expected {width} digits parsable as {fmt!r}; first: {samples!r}"
        )


def _duplicate_stamps(frame: pl.DataFrame) -> tuple[DuplicateStamp, ...]:
    """Collect ``(date, time)`` pairs carried by more than one row."""
    repeated = (
        frame.group_by(["date_str", "time_str"])
        .len()
        .filter(pl.col("len") > 1)
        .sort(["date_str", "time_str"])
    )
    return tuple(
        DuplicateStamp(date_str=row[0], time_str=row[1], count=int(row[2]))
        for row in repeated.iter_rows()
    )
