"""Recovering the UTC offset of a vendor export by measuring it against a reference.

A vendor whose timestamps are not UTC, and does not say so, cannot be imported
safely: an hour of error puts every bar on the wrong side of the London open and
nothing downstream will notice. The Forexite export is such a vendor — measured
against the Dukascopy series already in the store, its offset is not one zone but
a sequence of them, drifting between 0, +1 and +2 hours with transitions that
only partly coincide with European DST dates.

This module turns that from a guess into a measurement. Given the vendor's hourly
closes and a reference series known to be UTC, it decides the offset one day at a
time and collapses the agreeing days into runs. What it cannot decide it says so
about, and the resulting :class:`OffsetTable` refuses to answer for a date it did
not cover, so a period nobody measured cannot be imported under an offset nobody
established.

**Why closing prices and not a session profile.** The obvious detector — align
the intraday volatility profile, whose London and New York peaks are visible by
eye — was built first and rejected on measurement: against known-good answers it
got 37 of 54 months right, and *every* one of the seventeen misses was off by
exactly one hour. The session profile is too smooth to resolve the very quantity
being recovered. Comparing closing prices bar by bar against a real reference is
crude by comparison and discriminates cleanly: on GBPUSD the median absolute
difference is 2.0-2.3 pips at the true offset and 5-16 pips at any other.

**Why the margin is a ratio and not a threshold.** The residual at the correct
offset is the difference in price basis between two vendors, and it grows with
volatility. A fixed cap therefore rejects exactly the days a cap is supposed to
protect: 2019-01-02 scored 5.5 against a runner-up of 18.4 — decided by any
reasonable reading — and was thrown out by an absolute limit of 4.0. The winner
has to beat the runner-up by a factor; how large the winner is in absolute terms
says more about the day's range than about the alignment.

**One table serves every symbol in an export, and that is measured too.** For
Forexite it was checked by triangular identity — ``GBPJPY = GBPUSD x USDJPY`` and
six others covering all fifteen FX symbols — which held to a median of 0.7-4.9
pips over roughly 7.9 million minutes each, where a one-hour disagreement between
symbols would show as tens of pips. The clock belongs to the export, not to the
symbol. A metal in the same export has no such identity available and inherits
the finding rather than confirming it; :attr:`OffsetTable.symbol_scope` is where
that distinction is written down.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, datetime
from typing import Literal

import polars as pl
from pydantic import BaseModel, ConfigDict, Field, model_validator

from trading_system.core.exceptions import DataError

#: Offsets, in whole hours, that a day is tested against. Whole hours because
#: every offset ever observed in this export is one; a vendor on a half-hour
#: zone would need this widened and would be found by every day coming back
#: ambiguous rather than by a wrong answer.
DEFAULT_CANDIDATES: tuple[int, ...] = (0, 1, 2, 3)

#: Matched hours below which a day is treated as closed rather than undecided.
#: A Saturday has none; a holiday half-session has a handful, and deciding an
#: offset off four quiet hours is how a wrong answer gets made confidently.
DEFAULT_MIN_HOURS = 12

#: How much better the winning offset must be than the runner-up.
DEFAULT_MARGIN_RATIO = 1.5

#: Shortest run allowed to stand on its own in :func:`smooth_isolated_runs`.
DEFAULT_MIN_RUN_DAYS = 2

DayStatus = Literal["decided", "ambiguous", "closed"]


@dataclass(frozen=True)
class DayVerdict:
    """What the comparison concluded about one calendar day.

    Attributes:
        day: The UTC day, as labelled by the reference.
        status: ``decided`` when one offset won by the required margin,
            ``ambiguous`` when none did, ``closed`` when the day carried too
            few matched hours to ask.
        offset_hours: Hours to add to a vendor stamp to get UTC. ``None``
            unless ``status`` is ``decided``.
        best_median: Median absolute price difference at the winning offset, in
            the price units of the input. ``None`` for a closed day.
        runner_up_median: The same at the second-best offset, which is what the
            margin is measured against.
        matched_hours: Hours the two series had in common at the best offset.
    """

    day: date
    status: DayStatus
    offset_hours: int | None
    best_median: float | None
    runner_up_median: float | None
    matched_hours: int


class OffsetRun(BaseModel):
    """A stretch of days sharing one offset.

    Attributes:
        start: First day of the run.
        end: Last day, inclusive. Inclusive because the boundaries are trading
            days and a half-open end would name a day the market was shut.
        offset_hours: Hours to add to a vendor stamp inside this run to get UTC.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    start: date
    end: date
    offset_hours: int

    @model_validator(mode="after")
    def _check_order(self) -> "OffsetRun":
        """Reject a run that ends before it starts."""
        if self.end < self.start:
            raise ValueError(f"run ends {self.end} before it starts {self.start}")
        return self

    def covers(self, day: date) -> bool:
        """Whether ``day`` falls inside this run."""
        return self.start <= day <= self.end


class OffsetTable(BaseModel):
    """Measured offsets for one vendor export, and what they rest on.

    The provenance fields are not decoration. A table of numbers with no record
    of what they were measured against, over what period, and by what rule is a
    table nobody can re-derive or challenge — and this one exists precisely
    because the vendor's own claim about its timestamps could not be trusted.

    Attributes:
        vendor: Export the table describes.
        reference: Series the offsets were measured against, named well enough
            to repeat the measurement.
        method: One sentence on how a day was decided.
        generated_on: When the measurement was run.
        symbol_scope: Prose account of how the scope was established, verbatim
            enough to repeat. It explains ``verified_symbols`` and
            ``inherited_symbols``; it is not what code reads to decide.
        verified_symbols: Symbols whose clock was confirmed by measurement.
        inherited_symbols: Symbols in the same export whose clock was *not*
            measured and is assumed to match. Structured rather than left to the
            prose because an importer has to act on the difference, and a
            consumer that had to grep ``symbol_scope`` for a symbol name would
            be a second authority free to disagree with the first.
        runs: Offsets by period, ascending and non-overlapping.
        ambiguous_days: Market days no offset won. They are usually inside a run
            decided by their neighbours; listed so that "covered" is never
            confused with "measured".
        smoothed_days: Days whose measured offset was overridden by
            :func:`smooth_isolated_runs`.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    vendor: str = Field(min_length=1)
    reference: str = Field(min_length=1)
    method: str = Field(min_length=1)
    generated_on: date
    symbol_scope: str = Field(min_length=1)
    verified_symbols: tuple[str, ...] = ()
    inherited_symbols: tuple[str, ...] = ()
    runs: tuple[OffsetRun, ...]
    ambiguous_days: tuple[date, ...] = ()
    smoothed_days: tuple[date, ...] = ()

    @model_validator(mode="after")
    def _check_symbol_sets(self) -> "OffsetTable":
        """Reject a symbol claimed as both measured and assumed."""
        both = sorted(set(self.verified_symbols) & set(self.inherited_symbols))
        if both:
            raise ValueError(f"symbols listed as both verified and inherited: {both}")
        return self

    def check_symbol(self, symbol: str) -> None:
        """Assert that this table may be applied to ``symbol``.

        Called once per file rather than per row: a symbol the table does not
        vouch for is a property of the request, not of any particular bar, and
        discovering it eight million times is not discovering it better.

        Args:
            symbol: Instrument the caller wants to normalise.

        Raises:
            DataError: If the symbol's clock was assumed rather than measured,
                or if the table says nothing about it at all. The two get
                different messages because they need different work: the first
                needs a reference series for that symbol, the second needs
                somebody to establish whether it belongs to this export.
        """
        if symbol in self.verified_symbols:
            return
        if symbol in self.inherited_symbols:
            raise DataError(
                f"no calibration for {symbol}: its clock is inherited from the rest of "
                f"the {self.vendor} export, not measured. Every other symbol was "
                "confirmed by triangular identity, which a metal does not admit. "
                "A separate reference series for this symbol is needed before it "
                "can be imported."
            )
        raise DataError(
            f"no calibration for {symbol}: it is not among the symbols this "
            f"{self.vendor} table covers ({', '.join(sorted(self.verified_symbols))})"
        )

    @model_validator(mode="after")
    def _check_runs(self) -> "OffsetTable":
        """Reject an empty, unsorted or overlapping run list.

        Overlap is the failure that matters: two runs covering one day would
        make :meth:`offset_for` depend on iteration order, which is exactly the
        kind of silent ambiguity this table exists to remove.
        """
        if not self.runs:
            raise ValueError("an offset table with no runs cannot answer anything")
        for earlier, later in zip(self.runs, self.runs[1:], strict=False):
            if later.start <= earlier.end:
                raise ValueError(
                    f"runs overlap or are unsorted: {earlier.start}..{earlier.end} "
                    f"then {later.start}..{later.end}"
                )
        return self

    @property
    def covered_from(self) -> date:
        """First day the table speaks for."""
        return self.runs[0].start

    @property
    def covered_to(self) -> date:
        """Last day the table speaks for."""
        return self.runs[-1].end

    def offset_for(self, moment: datetime) -> int:
        """Hours to add to a vendor stamp to reach UTC.

        Args:
            moment: A vendor timestamp, as written in the export.

        Returns:
            The offset in whole hours.

        Raises:
            DataError: If no run covers the date. Deliberately not a fallback to
                the nearest run: outside the measured period the offset is
                unknown, and answering anyway is the failure this module was
                written to prevent.
        """
        day = moment.date()
        for run in self.runs:
            if run.covers(day):
                return run.offset_hours
        raise DataError(
            f"{self.vendor}: no measured UTC offset for {day}; the table covers "
            f"{self.covered_from}..{self.covered_to}. Extend the calibration "
            "before importing this period rather than assuming an offset."
        )


@dataclass(frozen=True)
class Calibration:
    """Everything one calibration pass concluded.

    Attributes:
        days: Per-day verdicts, ascending, including closed and ambiguous ones.
        runs: The decided days collapsed into runs.
    """

    days: tuple[DayVerdict, ...]
    runs: tuple[OffsetRun, ...]

    @property
    def decided(self) -> tuple[DayVerdict, ...]:
        """Days that produced an offset."""
        return tuple(day for day in self.days if day.status == "decided")

    @property
    def ambiguous(self) -> tuple[DayVerdict, ...]:
        """Market days that did not."""
        return tuple(day for day in self.days if day.status == "ambiguous")

    @property
    def closed(self) -> tuple[DayVerdict, ...]:
        """Days with too little overlap to ask about."""
        return tuple(day for day in self.days if day.status == "closed")


def calibrate(
    vendor: pl.DataFrame,
    reference: pl.DataFrame,
    *,
    candidates: Sequence[int] = DEFAULT_CANDIDATES,
    min_hours: int = DEFAULT_MIN_HOURS,
    margin_ratio: float = DEFAULT_MARGIN_RATIO,
) -> Calibration:
    """Measure the vendor's UTC offset day by day against a known-UTC reference.

    Both inputs are hourly closes keyed by the hour's opening instant. The
    vendor's stamps are taken as written — no timezone is assumed for them,
    which is the entire point — and the reference's are UTC.

    Args:
        vendor: Columns ``ts`` (tz-naive datetime, the vendor's own label) and
            ``close``. One row per hour.
        reference: Columns ``ts`` (tz-naive datetime, UTC) and ``close``.
        candidates: Offsets in whole hours to test.
        min_hours: Below this many matched hours a day is reported ``closed``.
        margin_ratio: The runner-up's median must exceed the winner's by at
            least this factor for the day to be ``decided``.

    Returns:
        Per-day verdicts and the runs they collapse into.

    Raises:
        DataError: If either frame lacks the required columns, or ``candidates``
            holds fewer than two offsets — with one candidate there is no
            runner-up and every day would be "decided" by default.
    """
    for name, frame in (("vendor", vendor), ("reference", reference)):
        missing = {"ts", "close"} - set(frame.columns)
        if missing:
            raise DataError(f"{name} frame is missing columns {sorted(missing)}")
    if len(set(candidates)) < 2:
        raise DataError(
            f"calibration needs at least two distinct candidates, got {list(candidates)}; "
            "with one there is nothing to beat and every day decides itself"
        )

    scores: dict[date, dict[int, tuple[float, int]]] = {}
    for offset in sorted(set(candidates)):
        shifted = vendor.with_columns(
            (pl.col("ts") + pl.duration(hours=offset)).alias("ts")
        ).select("ts", pl.col("close").alias("vendor_close"))
        joined = shifted.join(reference.select("ts", "close"), on="ts", how="inner")
        if joined.height == 0:
            continue
        per_day = (
            joined.with_columns(
                (pl.col("vendor_close") - pl.col("close")).abs().alias("gap"),
                pl.col("ts").dt.date().alias("day"),
            )
            .group_by("day")
            .agg(pl.col("gap").median().alias("median_gap"), pl.len().alias("hours"))
        )
        for day, median_gap, hours in per_day.iter_rows():
            scores.setdefault(day, {})[offset] = (float(median_gap), int(hours))

    verdicts = tuple(
        _judge(day, scores[day], min_hours=min_hours, margin_ratio=margin_ratio)
        for day in sorted(scores)
    )
    return Calibration(days=verdicts, runs=_collapse(verdicts))


def _judge(
    day: date,
    offsets: dict[int, tuple[float, int]],
    *,
    min_hours: int,
    margin_ratio: float,
) -> DayVerdict:
    """Decide one day from its per-offset medians."""
    matched = max(hours for _, hours in offsets.values())
    if matched < min_hours:
        return DayVerdict(
            day=day,
            status="closed",
            offset_hours=None,
            best_median=None,
            runner_up_median=None,
            matched_hours=matched,
        )

    ranked = sorted(offsets.items(), key=lambda item: item[1][0])
    (best_offset, (best_median, best_hours)) = ranked[0]
    runner_up_median = ranked[1][1][0] if len(ranked) > 1 else None

    # Strictly greater, not `>=`, because of the degenerate case its own test
    # found: a frozen or flat stretch scores 0.0 at every offset, and `0 >= 0`
    # would report a confident offset for a period in which nothing whatsoever
    # distinguishes one alignment from another.
    decided = runner_up_median is not None and runner_up_median > best_median * margin_ratio
    return DayVerdict(
        day=day,
        status="decided" if decided else "ambiguous",
        offset_hours=best_offset if decided else None,
        best_median=best_median,
        runner_up_median=runner_up_median,
        matched_hours=best_hours,
    )


def _collapse(verdicts: Sequence[DayVerdict]) -> tuple[OffsetRun, ...]:
    """Merge consecutive decided days sharing an offset into runs.

    Ambiguous and closed days neither break a run nor extend it: a weekend
    between two Fridays at the same offset is one run, and a day nobody could
    decide does not become a boundary just because it was unreadable.
    """
    runs: list[OffsetRun] = []
    for verdict in verdicts:
        if verdict.status != "decided" or verdict.offset_hours is None:
            continue
        if runs and runs[-1].offset_hours == verdict.offset_hours:
            runs[-1] = runs[-1].model_copy(update={"end": verdict.day})
        else:
            runs.append(
                OffsetRun(start=verdict.day, end=verdict.day, offset_hours=verdict.offset_hours)
            )
    return tuple(runs)


def smooth_isolated_runs(
    runs: Sequence[OffsetRun],
    *,
    min_run_days: int = DEFAULT_MIN_RUN_DAYS,
) -> tuple[tuple[OffsetRun, ...], tuple[date, ...]]:
    """Absorb runs too short to be a real change of clock into their neighbours.

    A single day disagreeing with the months on either side of it is a misread,
    not a timezone that lasted one session: the vendor would have had to change
    zone on a Monday and change back on the Tuesday. Absorbing it is therefore
    the more truthful reading — but it is still an override of a measurement, so
    it happens in its own function, off by default in :func:`calibrate`, and
    hands back exactly which days it touched.

    Only runs strictly between two neighbours that agree with each other are
    absorbed. A short run at either end has nothing to be absorbed into and is
    left alone, because there it might be the start of a genuine change.

    Args:
        runs: Runs to smooth, ascending and non-overlapping.
        min_run_days: Runs shorter than this many days are candidates.

    Returns:
        The smoothed runs and the days whose offset was overridden.
    """
    if len(runs) < 3:
        return tuple(runs), ()

    kept: list[OffsetRun] = [runs[0]]
    overridden: list[date] = []
    for index in range(1, len(runs) - 1):
        current, following = runs[index], runs[index + 1]
        span_days = (current.end - current.start).days + 1
        absorbable = span_days < min_run_days and kept[-1].offset_hours == following.offset_hours
        if not absorbable:
            kept.append(current)
            continue
        day = current.start
        while day <= current.end:
            overridden.append(day)
            day = date.fromordinal(day.toordinal() + 1)
        kept[-1] = kept[-1].model_copy(update={"end": current.end})
    kept.append(runs[-1])

    merged: list[OffsetRun] = [kept[0]]
    for run in kept[1:]:
        if merged[-1].offset_hours == run.offset_hours:
            merged[-1] = merged[-1].model_copy(update={"end": run.end})
        else:
            merged.append(run)
    return tuple(merged), tuple(overridden)
