"""Tests for UTC offset calibration.

The synthetic series is a random walk sampled hourly: the detector works by
comparing prices bar to bar, so the fixture has to move enough that a wrong
alignment is visibly worse than the right one — which is exactly the property
real market data has and a flat or monotonic fixture does not.
"""

import random
from datetime import date, datetime, timedelta
from pathlib import Path

import polars as pl
import pytest

from trading_system.core.exceptions import DataError
from trading_system.data_pipeline.utc_calibration import (
    Calibration,
    OffsetRun,
    OffsetTable,
    calibrate,
    smooth_isolated_runs,
)

START = datetime(2021, 1, 4)  # a Monday


def walk(hours: int, *, seed: int = 7, step: float = 0.0006) -> pl.DataFrame:
    """An hourly random walk, standing in for a real price series."""
    rng = random.Random(seed)
    price, rows = 1.3000, []
    for index in range(hours):
        price += rng.uniform(-step, step)
        rows.append((START + timedelta(hours=index), price))
    return pl.DataFrame(rows, schema={"ts": pl.Datetime, "close": pl.Float64}, orient="row")


def shifted_by(reference: pl.DataFrame, offset_hours: int) -> pl.DataFrame:
    """The same series labelled as a vendor whose clock is ``offset_hours`` behind UTC."""
    return reference.with_columns(pl.col("ts") - pl.duration(hours=offset_hours))


class TestARecoverableOffset:
    @pytest.mark.parametrize("offset", [0, 1, 2, 3])
    def test_a_constant_offset_is_recovered(self, offset: int) -> None:
        reference = walk(24 * 10)
        result = calibrate(shifted_by(reference, offset), reference)
        assert {day.offset_hours for day in result.decided} == {offset}

    def test_the_recovered_days_collapse_into_one_run(self) -> None:
        reference = walk(24 * 10)
        result = calibrate(shifted_by(reference, 2), reference)
        assert len(result.runs) == 1
        assert result.runs[0].offset_hours == 2

    def test_a_change_of_offset_mid_series_is_found_at_the_right_day(self) -> None:
        reference = walk(24 * 12)
        boundary = START + timedelta(days=6)
        early = shifted_by(reference.filter(pl.col("ts") < boundary), 1)
        late = shifted_by(reference.filter(pl.col("ts") >= boundary), 2)
        result = calibrate(pl.concat([early, late]), reference)
        offsets = [(run.offset_hours, run.start, run.end) for run in result.runs]
        assert [item[0] for item in offsets] == [1, 2]
        assert offsets[1][1] == boundary.date()


class TestWhatItRefusesToDecide:
    def test_a_day_with_too_little_overlap_is_closed_not_ambiguous(self) -> None:
        # "The market was shut" and "the data disagreed" are different findings
        # and must not be counted together.
        reference = walk(24 * 3)
        vendor = shifted_by(reference, 1).head(4)
        result = calibrate(vendor, reference)
        assert result.decided == ()
        assert {day.status for day in result.days} == {"closed"}

    def test_a_series_that_does_not_discriminate_comes_back_ambiguous(self) -> None:
        # A flat series is identical under every shift, so no offset can win.
        flat = pl.DataFrame(
            [(START + timedelta(hours=i), 1.3) for i in range(24 * 5)],
            schema={"ts": pl.Datetime, "close": pl.Float64},
            orient="row",
        )
        result = calibrate(flat, flat)
        assert result.decided == ()
        assert len(result.ambiguous) > 0

    def test_ambiguous_days_are_reported_rather_than_filled_in(self) -> None:
        flat = pl.DataFrame(
            [(START + timedelta(hours=i), 1.3) for i in range(24 * 5)],
            schema={"ts": pl.Datetime, "close": pl.Float64},
            orient="row",
        )
        result = calibrate(flat, flat)
        assert all(day.offset_hours is None for day in result.ambiguous)

    def test_a_single_candidate_is_refused(self) -> None:
        reference = walk(24 * 3)
        with pytest.raises(DataError, match="at least two distinct candidates"):
            calibrate(reference, reference, candidates=(0,))

    def test_a_frame_without_the_required_columns_is_refused(self) -> None:
        reference = walk(24 * 3)
        with pytest.raises(DataError, match="missing columns"):
            calibrate(reference.rename({"close": "px"}), reference)


class TestGapsDoNotBreakRuns:
    def test_a_weekend_does_not_split_one_run_in_two(self) -> None:
        # Two working weeks with the market shut between them is one run, not
        # two: a day nobody could read is not a boundary.
        reference = walk(24 * 14)
        weekend = (START + timedelta(days=5), START + timedelta(days=7))
        trading = reference.filter((pl.col("ts") < weekend[0]) | (pl.col("ts") >= weekend[1]))
        result = calibrate(shifted_by(trading, 1), reference)
        assert len(result.runs) == 1
        assert result.runs[0].offset_hours == 1


class TestSmoothingIsolatedRuns:
    def test_a_one_day_island_between_agreeing_neighbours_is_absorbed(self) -> None:
        runs = (
            OffsetRun(start=date(2021, 4, 1), end=date(2021, 7, 22), offset_hours=2),
            OffsetRun(start=date(2021, 7, 26), end=date(2021, 7, 26), offset_hours=1),
            OffsetRun(start=date(2021, 7, 27), end=date(2021, 10, 29), offset_hours=2),
        )
        smoothed, overridden = smooth_isolated_runs(runs)
        assert len(smoothed) == 1
        assert smoothed[0].offset_hours == 2
        assert overridden == (date(2021, 7, 26),)

    def test_the_overridden_days_are_handed_back(self) -> None:
        # Smoothing overrides a measurement. Doing that without saying which
        # days were touched would make the table unauditable.
        runs = (
            OffsetRun(start=date(2021, 1, 1), end=date(2021, 2, 1), offset_hours=0),
            OffsetRun(start=date(2021, 2, 2), end=date(2021, 2, 2), offset_hours=3),
            OffsetRun(start=date(2021, 2, 3), end=date(2021, 3, 1), offset_hours=0),
        )
        _, overridden = smooth_isolated_runs(runs)
        assert overridden == (date(2021, 2, 2),)

    def test_a_short_run_between_disagreeing_neighbours_survives(self) -> None:
        # Here the short run is the transition itself, not noise.
        runs = (
            OffsetRun(start=date(2021, 1, 1), end=date(2021, 2, 1), offset_hours=0),
            OffsetRun(start=date(2021, 2, 2), end=date(2021, 2, 2), offset_hours=1),
            OffsetRun(start=date(2021, 2, 3), end=date(2021, 3, 1), offset_hours=2),
        )
        smoothed, overridden = smooth_isolated_runs(runs)
        assert len(smoothed) == 3
        assert overridden == ()

    def test_a_short_run_at_the_edge_is_left_alone(self) -> None:
        # At the start there is nothing to absorb it into, and it may be the
        # beginning of a real change rather than a misread.
        runs = (
            OffsetRun(start=date(2021, 1, 1), end=date(2021, 1, 1), offset_hours=1),
            OffsetRun(start=date(2021, 1, 2), end=date(2021, 3, 1), offset_hours=2),
            OffsetRun(start=date(2021, 3, 2), end=date(2021, 5, 1), offset_hours=0),
        )
        smoothed, overridden = smooth_isolated_runs(runs)
        assert smoothed[0].offset_hours == 1
        assert overridden == ()

    def test_long_runs_are_untouched(self) -> None:
        runs = (
            OffsetRun(start=date(2021, 1, 1), end=date(2021, 2, 1), offset_hours=0),
            OffsetRun(start=date(2021, 2, 2), end=date(2021, 3, 1), offset_hours=1),
            OffsetRun(start=date(2021, 3, 2), end=date(2021, 4, 1), offset_hours=0),
        )
        smoothed, overridden = smooth_isolated_runs(runs)
        assert len(smoothed) == 3
        assert overridden == ()


class TestTheTableRefusesWhatItDoesNotKnow:
    @pytest.fixture
    def table(self) -> OffsetTable:
        return OffsetTable(
            vendor="forexite",
            reference="Dukascopy GBPUSD H1 in the local store",
            method="daily median absolute close difference, winner beats runner-up by 1.5x",
            generated_on=date(2026, 8, 10),
            symbol_scope="all sixteen exports",
            runs=(
                OffsetRun(start=date(2021, 1, 4), end=date(2021, 3, 26), offset_hours=1),
                OffsetRun(start=date(2021, 3, 29), end=date(2021, 10, 29), offset_hours=2),
            ),
        )

    def test_a_covered_date_answers(self, table: OffsetTable) -> None:
        assert table.offset_for(datetime(2021, 5, 3, 14, 30)) == 2
        assert table.offset_for(datetime(2021, 2, 1, 9, 0)) == 1

    def test_a_date_before_coverage_raises(self, table: OffsetTable) -> None:
        with pytest.raises(DataError, match="no measured UTC offset"):
            table.offset_for(datetime(2019, 6, 1, 12, 0))

    def test_a_date_after_coverage_raises(self, table: OffsetTable) -> None:
        with pytest.raises(DataError, match="no measured UTC offset"):
            table.offset_for(datetime(2023, 6, 1, 12, 0))

    def test_a_gap_between_runs_raises_rather_than_picking_a_neighbour(
        self, table: OffsetTable
    ) -> None:
        # 2021-03-27 and -28 are a weekend nobody measured. Answering from the
        # nearest run would be the assumption this module exists to refuse.
        with pytest.raises(DataError, match="no measured UTC offset"):
            table.offset_for(datetime(2021, 3, 27, 12, 0))

    def test_the_error_names_the_covered_period(self, table: OffsetTable) -> None:
        with pytest.raises(DataError, match=r"2021-01-04\.\.2021-10-29"):
            table.offset_for(datetime(2019, 6, 1, 12, 0))

    def test_overlapping_runs_are_rejected_at_construction(self) -> None:
        with pytest.raises(ValueError, match="overlap or are unsorted"):
            OffsetTable(
                vendor="v",
                reference="r",
                method="m",
                generated_on=date(2026, 8, 10),
                symbol_scope="s",
                runs=(
                    OffsetRun(start=date(2021, 1, 1), end=date(2021, 6, 1), offset_hours=0),
                    OffsetRun(start=date(2021, 5, 1), end=date(2021, 9, 1), offset_hours=1),
                ),
            )

    def test_an_empty_table_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="no runs"):
            OffsetTable(
                vendor="v",
                reference="r",
                method="m",
                generated_on=date(2026, 8, 10),
                symbol_scope="s",
                runs=(),
            )

    def test_a_backwards_run_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="ends .* before it starts"):
            OffsetRun(start=date(2021, 6, 1), end=date(2021, 1, 1), offset_hours=0)


class TestTheShippedTable:
    """The generated artifact is data the import will trust, so it is asserted."""

    @pytest.fixture
    def shipped(self) -> OffsetTable:
        path = Path(__file__).resolve().parents[2] / "configs" / "forexite_utc_offsets.json"
        return OffsetTable.model_validate_json(path.read_text())

    def test_it_parses_and_carries_its_provenance(self, shipped: OffsetTable) -> None:
        # A table of numbers nobody can re-derive is worse than no table.
        assert "Dukascopy" in shipped.reference
        assert "triangular" in shipped.symbol_scope
        assert shipped.method

    def test_it_covers_the_overlap_with_the_reference_and_no_more(
        self, shipped: OffsetTable
    ) -> None:
        assert shipped.covered_from == date(2019, 1, 2)
        assert shipped.covered_to == date(2023, 9, 29)

    def test_the_offset_is_never_assumed_outside_that_window(self, shipped: OffsetTable) -> None:
        # The Forexite export starts in 2001; everything before the reference
        # exists is unmeasured and must stay unimportable.
        with pytest.raises(DataError):
            shipped.offset_for(datetime(2015, 6, 1, 12, 0))

    def test_the_one_smoothed_day_is_recorded(self, shipped: OffsetTable) -> None:
        # An overridden measurement that leaves no trace is an unauditable one.
        assert shipped.smoothed_days == (date(2021, 7, 26),)

    def test_the_measured_drift_is_present_rather_than_flattened(
        self, shipped: OffsetTable
    ) -> None:
        # The whole finding is that this export has no single offset. A table
        # that had collapsed to one run would mean the drift had been lost.
        assert len({run.offset_hours for run in shipped.runs}) == 3
        assert len(shipped.runs) > 5


class TestTheCalibrationReportsEveryDay:
    def test_closed_and_ambiguous_and_decided_partition_the_days(self) -> None:
        reference = walk(24 * 10)
        result: Calibration = calibrate(shifted_by(reference, 1), reference)
        total = len(result.decided) + len(result.ambiguous) + len(result.closed)
        assert total == len(result.days)
