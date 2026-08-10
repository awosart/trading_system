"""Turning a vendor's own rows into the project's canonical bars.

This is the stage the format readers deliberately stop short of. It answers the
three questions they refuse to: what instant a stamp denotes, what the volume
column means, and what to do about a period nobody has calibrated.

**There is no ``source_tz`` here, and its absence is the point.** A constant
timezone is what this pipeline started with, and measurement destroyed it: the
Forexite export's offset drifts between 0, +1 and +2 hours over the history,
with transitions that only partly coincide with European DST dates. A constant —
any constant — puts whole years of bars on the wrong hour and nothing downstream
notices. So the offset arrives as an :class:`~trading_system.data_pipeline.
utc_calibration.OffsetTable`: measured where it could be measured, silent where
it could not, and unable to answer for a date it never saw.

**Partial failure is the normal case, not an error path.** The calibration
covers 2019-01-02 to 2023-09-29 while the export starts in 2001, and inside that
window sixteen days fall between runs. A file therefore routinely contains rows
the table cannot place. Refusing the whole file for them would throw away four
usable years because of a Christmas week; placing them under a neighbouring
offset would be the silent guess the calibration exists to prevent. So they are
excluded, by day, with the reason attached, and the caller gets both halves.

**The symbol is checked once, before any row is touched.** A symbol whose clock
was inherited rather than measured is not a per-row condition, and finding out
about it eight million times through raised exceptions would be slower and no
more informative than finding out once.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from typing import Literal

import polars as pl

from trading_system.core.exceptions import DataError
from trading_system.data_pipeline.utc_calibration import OffsetTable

#: Why a day's rows could not be normalised.
#:
#: ``no_coverage`` — the date lies outside the calibrated window entirely. For
#: the Forexite export that is everything before 2019 and after Sep 2023.
#:
#: ``uncovered_gap`` — the date is inside the window but between two runs, so
#: the offset either side of it is known and the offset *on* it is not. In the
#: shipped table all sixteen such days are Christmas, New Year or a weekend
#: straddling a change of offset.
#:
#: ``offset_step`` — the hours around a change of offset between two adjacent
#: trading days. Found on real data, not by reasoning: normalising EURJPY
#: produced 60 non-increasing timestamps, all at ``2022-12-05 (+2)`` meeting
#: ``2022-12-06 (+1)``, the one boundary in the shipped table that both drops an
#: hour and has no weekend to absorb it. The last hour of the earlier day and
#: the first hour of the later one land on the same UTC hour, and the
#: calibration — which resolves a day at a time — cannot say which label owns
#: it, because it does not know at what hour of the boundary the vendor's clock
#: actually moved. Both sides are dropped. Letting them through would leave
#: duplicate stamps for ``OHLCVFrame.from_raw`` to de-duplicate under a rule
#: written for re-sent vendor corrections, which this is not.
#:
#: There is deliberately no ``ambiguous`` member. Every one of the 61 ambiguous
#: days in the shipped table falls *inside* a run and is therefore covered by
#: its neighbours; a reason that cannot occur would be a branch nobody could
#: ever exercise. That those rows lean on a neighbour rather than their own
#: measurement is still worth knowing, so it is counted instead — see
#: :attr:`NormalizedBatch.rows_on_ambiguous_days`.
ExclusionReason = Literal["no_coverage", "uncovered_gap", "offset_step"]

#: Written into the ``volume`` column, which :class:`~trading_system.data.models
#: .OHLCVFrame` requires and this vendor cannot fill. Forexite's own column is
#: the constant 4 across every file measured — 29.2 million rows, one distinct
#: value — so it carries nothing. Zero is chosen over that constant because
#: ``quality.py`` has a ``zero_volume`` detector and no constant-value detector:
#: writing 0.0 makes "this feed has no volume" a finding the existing checks
#: report on every bar, where writing 4.0 would sail through them and reach
#: VWAP, MFI, OBV and RelativeVolume as a plausible-looking weight.
ABSENT_VOLUME = 0.0

_VENDOR_DATE_FORMAT = "%Y%m%d"
_VENDOR_TIME_FORMAT = "%H%M%S"

#: Canonical column order, matching ``OHLCVFrame``'s requirement exactly.
NORMALISED_COLUMNS: tuple[str, ...] = (
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "volume",
)


@dataclass(frozen=True)
class ExcludedDay:
    """One day whose rows could not be placed on the UTC clock.

    Aggregated by day rather than listed per row: the reason is a property of
    the date, so a million identical entries would say nothing a count does not.

    Attributes:
        day: The vendor-labelled date, as written in the export.
        reason: Why the table could not place it.
        rows: How many rows were dropped for it.
    """

    day: date
    reason: ExclusionReason
    rows: int


@dataclass(frozen=True)
class NormalizedBatch:
    """One symbol's rows after normalisation, and what did not survive it.

    Attributes:
        symbol: Instrument these bars belong to.
        frame: Bars with :data:`NORMALISED_COLUMNS`, tz-aware UTC timestamps,
            ascending. Ready for ``OHLCVFrame.from_raw``.
        excluded: Days dropped for want of a calibrated offset, ascending.
        rows_on_ambiguous_days: Rows placed under an offset their own day did
            not establish — the day was ambiguous and inherited the run around
            it. Not an error and not excluded; a number worth seeing before
            trusting a result that leans on those bars.
    """

    symbol: str
    frame: pl.DataFrame
    excluded: tuple[ExcludedDay, ...]
    rows_on_ambiguous_days: int

    @property
    def excluded_rows(self) -> int:
        """Total rows dropped across every excluded day."""
        return sum(day.rows for day in self.excluded)


def normalize_timestamp(
    date_str: str,
    time_str: str,
    symbol: str,
    offset_table: OffsetTable,
) -> datetime:
    """Place one vendor stamp on the UTC clock.

    Args:
        date_str: Vendor date, ``YYYYMMDD``, zero-padded by the reader.
        time_str: Vendor time, ``HHMMSS``, zero-padded by the reader.
        symbol: Instrument the row belongs to. Used to name the failure; the
            offset itself does not depend on it, which is a measured property of
            this export rather than an assumption — see ``OffsetTable``.
        offset_table: Measured offsets.

    Returns:
        The tz-aware UTC instant the row denotes.

    Raises:
        DataError: If the stamp is unparsable, or the table has no offset for
            its date. Never falls back to zero or to a neighbouring run: an
            uncalibrated date has no offset, and inventing one here is the exact
            failure the calibration was built to prevent.
    """
    try:
        moment = datetime.strptime(
            f"{date_str}{time_str}", _VENDOR_DATE_FORMAT + _VENDOR_TIME_FORMAT
        )
    except ValueError as error:
        raise DataError(
            f"{symbol}: {date_str!r}/{time_str!r} is not a valid vendor stamp"
        ) from error

    # Caught only to name the symbol, then re-raised with the original attached.
    # `offset_for` knows the date and the vendor but not which instrument was
    # being imported, and that is the first thing a reader of the message wants.
    # This is not swallowing the failure: nothing is returned on this path.
    try:
        offset_hours = offset_table.offset_for(moment)
    except DataError as error:
        raise DataError(f"{symbol}: {error}") from error

    return (moment + timedelta(hours=offset_hours)).replace(tzinfo=UTC)


def normalize_batch(
    frame: pl.DataFrame,
    symbol: str,
    offset_table: OffsetTable,
) -> NormalizedBatch:
    """Normalise one symbol's rows, dropping only the days that cannot be placed.

    Args:
        frame: Rows from a format reader, carrying at least ``date_str``,
            ``time_str``, ``open``, ``high``, ``low`` and ``close``. Any volume
            column is ignored — see :data:`ABSENT_VOLUME`.
        symbol: Instrument the rows belong to.
        offset_table: Measured offsets.

    Returns:
        The placed bars and the days that were dropped.

    Raises:
        DataError: If required columns are missing, or the table does not vouch
            for ``symbol``. Both are conditions of the whole request, checked
            before any row is read.
    """
    offset_table.check_symbol(symbol)

    required = {"date_str", "time_str", "open", "high", "low", "close"}
    missing = required - set(frame.columns)
    if missing:
        raise DataError(f"{symbol}: rows are missing columns {sorted(missing)}")

    if frame.height == 0:
        return NormalizedBatch(
            symbol=symbol,
            frame=_empty_normalised(),
            excluded=(),
            rows_on_ambiguous_days=0,
        )

    dated = frame.with_columns(pl.col("date_str").str.to_date(_VENDOR_DATE_FORMAT).alias("_day"))
    unparsable = dated.filter(pl.col("_day").is_null())
    if unparsable.height:
        sample = unparsable["date_str"].head(3).to_list()
        raise DataError(
            f"{symbol}: {unparsable.height} row(s) carry an unparsable date, e.g. {sample}"
        )

    lookup = _offset_lookup(offset_table)
    joined = dated.join(lookup, left_on="_day", right_on="day", how="left").with_columns(
        pl.col("time_str").str.slice(0, 2).cast(pl.Int32).alias("_hour")
    )

    uncovered = pl.col("offset_hours").is_null()
    stepped = ~uncovered & (
        (pl.col("_hour") < pl.col("drop_first_hours"))
        | (pl.col("_hour") >= 24 - pl.col("drop_last_hours"))
    )

    placed = joined.filter(~uncovered & ~stepped)
    excluded = _summarise_exclusions(
        joined.filter(uncovered), offset_table
    ) + _summarise_exclusions(joined.filter(stepped), offset_table, reason="offset_step")

    return NormalizedBatch(
        symbol=symbol,
        frame=_to_utc_frame(placed),
        excluded=tuple(sorted(excluded, key=lambda day: (day.day, day.reason))),
        rows_on_ambiguous_days=int(
            placed.filter(pl.col("_day").is_in(list(offset_table.ambiguous_days))).height
        ),
    )


def _offset_lookup(offset_table: OffsetTable) -> pl.DataFrame:
    """Expand the runs into one row per covered day, marking the step boundaries.

    A join against a few thousand dates is the whole of the offset arithmetic,
    and it keeps the per-row work inside polars. Walking the runs in Python for
    each of eight million rows would be the same answer, arrived at slowly.

    ``drop_first_hours`` and ``drop_last_hours`` mark the hours made ambiguous
    by a step down in offset between two *adjacent* days. A step with a weekend
    inside it needs no marking: the market was shut, so the hours that would
    have collided carry no bars.
    """
    by_day: dict[date, int] = {}
    for run in offset_table.runs:
        day = run.start
        while day <= run.end:
            by_day[day] = run.offset_hours
            day += timedelta(days=1)

    days = sorted(by_day)
    first: list[int] = []
    last: list[int] = []
    for day in days:
        previous = by_day.get(day - timedelta(days=1))
        following = by_day.get(day + timedelta(days=1))
        offset = by_day[day]
        first.append(max(0, previous - offset) if previous is not None else 0)
        last.append(max(0, offset - following) if following is not None else 0)

    return pl.DataFrame(
        {
            "day": days,
            "offset_hours": [by_day[day] for day in days],
            "drop_first_hours": first,
            "drop_last_hours": last,
        },
        schema={
            "day": pl.Date,
            "offset_hours": pl.Int32,
            "drop_first_hours": pl.Int32,
            "drop_last_hours": pl.Int32,
        },
    )


def _to_utc_frame(placed: pl.DataFrame) -> pl.DataFrame:
    """Build the canonical frame from rows that have an offset."""
    if placed.height == 0:
        return _empty_normalised()
    stamp = (
        pl.concat_str([pl.col("date_str"), pl.col("time_str")])
        .str.to_datetime(_VENDOR_DATE_FORMAT + _VENDOR_TIME_FORMAT)
        .dt.offset_by(pl.format("{}h", pl.col("offset_hours")))
        .dt.replace_time_zone("UTC")
        .alias("timestamp")
    )
    return (
        placed.with_columns(stamp, pl.lit(ABSENT_VOLUME, dtype=pl.Float64).alias("volume"))
        .select(NORMALISED_COLUMNS)
        .sort("timestamp")
    )


def _summarise_exclusions(
    dropped: pl.DataFrame,
    offset_table: OffsetTable,
    *,
    reason: ExclusionReason | None = None,
) -> tuple[ExcludedDay, ...]:
    """Group dropped rows by day and say why each day was dropped.

    Args:
        dropped: Rows that will not be normalised.
        offset_table: Used to tell a date outside the calibration from a hole
            inside it, when ``reason`` does not already say.
        reason: Fixed reason for every row, when the caller already knows it.
            A step boundary is the case: those days are covered, so deriving the
            reason from coverage would mislabel them.
    """
    if dropped.height == 0:
        return ()
    counted = dropped.group_by("_day").len().sort("_day")
    return tuple(
        ExcludedDay(
            day=day,
            reason=reason if reason is not None else _reason_for(day, offset_table),
            rows=int(rows),
        )
        for day, rows in counted.iter_rows()
    )


def _reason_for(day: date, offset_table: OffsetTable) -> ExclusionReason:
    """Distinguish a date outside the calibration from a hole inside it."""
    if offset_table.covered_from <= day <= offset_table.covered_to:
        return "uncovered_gap"
    return "no_coverage"


def _empty_normalised() -> pl.DataFrame:
    """A zero-row frame carrying the canonical schema."""
    return pl.DataFrame(
        schema={
            "timestamp": pl.Datetime(time_unit="us", time_zone="UTC"),
            **dict.fromkeys(("open", "high", "low", "close", "volume"), pl.Float64),
        }
    )


def normalise_symbols(
    batches: Sequence[tuple[str, pl.DataFrame]],
    offset_table: OffsetTable,
) -> tuple[NormalizedBatch, ...]:
    """Normalise several symbols, failing on the first the table does not cover.

    Args:
        batches: ``(symbol, rows)`` pairs.
        offset_table: Measured offsets.

    Returns:
        One result per input, in order.

    Raises:
        DataError: From the first symbol the table will not vouch for. Not
            collected and reported at the end: an uncalibrated symbol is a
            mistake in the request, and importing the other fifteen while
            quietly setting it aside is how it would go unnoticed.
    """
    return tuple(normalize_batch(rows, symbol, offset_table) for symbol, rows in batches)
