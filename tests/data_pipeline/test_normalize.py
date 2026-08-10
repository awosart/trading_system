"""Tests for normalisation onto the UTC clock.

The offset table under test is the shipped one, not a fixture. The dates below
are real entries from ``configs/forexite_utc_offsets.json``, chosen because the
behaviour that matters — which day gets which offset, and which day gets none —
is a property of that measured table and would be asserted away by a synthetic
one built to agree with the code.
"""

from datetime import UTC, date, datetime
from pathlib import Path

import polars as pl
import pytest

from trading_system.core.exceptions import DataError
from trading_system.core.types import Timeframe
from trading_system.data.models import OHLCVFrame
from trading_system.data_pipeline.normalize import (
    ABSENT_VOLUME,
    NORMALISED_COLUMNS,
    normalize_batch,
    normalize_timestamp,
)
from trading_system.data_pipeline.utc_calibration import OffsetTable

TABLE_PATH = Path(__file__).resolve().parents[2] / "configs" / "forexite_utc_offsets.json"

# Real rows of the shipped table:
#   2019-01-02 .. 2020-09-30  ->  0
#   2021-03-29 .. 2021-10-29  -> +2
#   2022-12-06 .. 2023-03-24  -> +1
IN_RUN_ZERO = ("20190603", 0)
IN_RUN_PLUS_TWO = ("20210503", 2)
IN_RUN_PLUS_ONE = ("20230103", 1)

BEFORE_COVERAGE = "20150601"  # the export reaches back to 2001; the reference does not
AFTER_COVERAGE = "20231201"  # the calibration stops at 2023-09-29
INSIDE_GAP = "20201225"  # Christmas, between the runs ending 12-24 and starting 12-28

SYMBOL = "EURJPY"


@pytest.fixture(scope="module")
def table() -> OffsetTable:
    return OffsetTable.model_validate_json(TABLE_PATH.read_text())


#: Schema stated explicitly so that `rows()` with no stamps is an empty frame
#: with the reader's columns, not a frame with no columns at all — the latter
#: would fail the column check and never reach the behaviour under test.
READER_SCHEMA = {
    "ticker": pl.String,
    "date_str": pl.String,
    "time_str": pl.String,
    "open": pl.Float64,
    "high": pl.Float64,
    "low": pl.Float64,
    "close": pl.Float64,
    "source_vol": pl.Int64,
}


def rows(*stamps: tuple[str, str]) -> pl.DataFrame:
    """Reader-shaped rows: padded stamp strings and prices."""
    return pl.DataFrame(
        [
            {
                "ticker": SYMBOL,
                "date_str": day,
                "time_str": clock,
                "open": 130.0,
                "high": 130.5,
                "low": 129.5,
                "close": 130.2,
                "source_vol": 4,
            }
            for day, clock in stamps
        ],
        schema=READER_SCHEMA,
    )


class TestOneStamp:
    @pytest.mark.parametrize(("day", "offset"), [IN_RUN_ZERO, IN_RUN_PLUS_TWO, IN_RUN_PLUS_ONE])
    def test_a_covered_day_is_shifted_by_its_measured_offset(
        self, table: OffsetTable, day: str, offset: int
    ) -> None:
        result = normalize_timestamp(day, "093000", SYMBOL, table)
        assert result == datetime(
            int(day[:4]), int(day[4:6]), int(day[6:]), 9 + offset, 30, tzinfo=UTC
        )

    def test_the_result_is_tz_aware_utc(self, table: OffsetTable) -> None:
        assert normalize_timestamp(IN_RUN_ZERO[0], "000000", SYMBOL, table).tzinfo is UTC

    def test_a_padded_early_hour_survives_the_shift(self, table: OffsetTable) -> None:
        # 03:01 under +2 is 05:01, not 31:00 or 00:31.
        result = normalize_timestamp(IN_RUN_PLUS_TWO[0], "030100", SYMBOL, table)
        assert (result.hour, result.minute) == (5, 1)

    def test_a_shift_can_cross_midnight_into_the_next_day(self, table: OffsetTable) -> None:
        result = normalize_timestamp(IN_RUN_PLUS_TWO[0], "230000", SYMBOL, table)
        assert result == datetime(2021, 5, 4, 1, 0, tzinfo=UTC)


class TestAnUncoveredDateRaises:
    @pytest.mark.parametrize("day", [BEFORE_COVERAGE, AFTER_COVERAGE, INSIDE_GAP])
    def test_it_raises_rather_than_returning_anything(self, table: OffsetTable, day: str) -> None:
        with pytest.raises(DataError):
            normalize_timestamp(day, "093000", SYMBOL, table)

    @pytest.mark.parametrize("day", [BEFORE_COVERAGE, AFTER_COVERAGE, INSIDE_GAP])
    def test_the_message_names_the_date(self, table: OffsetTable, day: str) -> None:
        # A failure that does not say which bar it was about sends the reader
        # back to the file to find out.
        expected = f"{day[:4]}-{day[4:6]}-{day[6:]}"
        with pytest.raises(DataError, match=expected):
            normalize_timestamp(day, "093000", SYMBOL, table)

    def test_the_message_names_the_symbol(self, table: OffsetTable) -> None:
        with pytest.raises(DataError, match=SYMBOL):
            normalize_timestamp(BEFORE_COVERAGE, "093000", SYMBOL, table)

    def test_a_gap_inside_the_window_is_refused_like_one_outside_it(
        self, table: OffsetTable
    ) -> None:
        # Christmas 2020 sits between a run at +1 and a run at 0. Both
        # neighbours are known, which is exactly the temptation to interpolate.
        assert table.covered_from < date(2020, 12, 25) < table.covered_to
        with pytest.raises(DataError):
            normalize_timestamp(INSIDE_GAP, "120000", SYMBOL, table)

    def test_an_unparsable_stamp_is_refused(self, table: OffsetTable) -> None:
        with pytest.raises(DataError, match="not a valid vendor stamp"):
            normalize_timestamp(IN_RUN_ZERO[0], "246000", SYMBOL, table)


class TestABatchFailsPartially:
    def test_covered_rows_survive_uncovered_ones(self, table: OffsetTable) -> None:
        batch = normalize_batch(
            rows(
                (IN_RUN_ZERO[0], "093000"),
                (BEFORE_COVERAGE, "093000"),
                (IN_RUN_PLUS_TWO[0], "093000"),
                (INSIDE_GAP, "093000"),
            ),
            SYMBOL,
            table,
        )
        assert batch.frame.height == 2
        assert batch.excluded_rows == 2

    def test_the_excluded_days_carry_their_date_reason_and_count(self, table: OffsetTable) -> None:
        batch = normalize_batch(
            rows(
                (BEFORE_COVERAGE, "093000"),
                (BEFORE_COVERAGE, "094000"),
                (INSIDE_GAP, "093000"),
                (AFTER_COVERAGE, "093000"),
            ),
            SYMBOL,
            table,
        )
        assert [(day.day, day.reason, day.rows) for day in batch.excluded] == [
            (date(2015, 6, 1), "no_coverage", 2),
            (date(2020, 12, 25), "uncovered_gap", 1),
            (date(2023, 12, 1), "no_coverage", 1),
        ]

    def test_a_hole_in_the_middle_does_not_abort_the_file(self, table: OffsetTable) -> None:
        # The whole reason for partial failure: sixteen uncovered days inside
        # the window would otherwise throw away four usable years.
        stamps = [(IN_RUN_ZERO[0], f"{hour:02d}0000") for hour in range(10)]
        stamps.insert(5, (INSIDE_GAP, "120000"))
        batch = normalize_batch(rows(*stamps), SYMBOL, table)
        assert batch.frame.height == 10
        assert batch.excluded_rows == 1

    def test_nothing_covered_yields_an_empty_frame_rather_than_an_error(
        self, table: OffsetTable
    ) -> None:
        batch = normalize_batch(rows((BEFORE_COVERAGE, "093000")), SYMBOL, table)
        assert batch.frame.height == 0
        assert batch.excluded_rows == 1

    def test_an_empty_input_is_not_an_error(self, table: OffsetTable) -> None:
        empty = rows().with_columns()
        batch = normalize_batch(empty, SYMBOL, table)
        assert batch.frame.height == 0
        assert batch.excluded == ()


class TestTheOutputShape:
    def test_the_columns_are_the_canonical_ones_in_order(self, table: OffsetTable) -> None:
        batch = normalize_batch(rows((IN_RUN_ZERO[0], "093000")), SYMBOL, table)
        assert tuple(batch.frame.columns) == NORMALISED_COLUMNS

    def test_volume_is_zero_not_the_vendors_constant(self, table: OffsetTable) -> None:
        # Forexite's own column is the constant 4 across every file measured.
        # Carrying it forward would sail past quality.py's detectors and reach
        # VWAP and MFI as a plausible weight.
        batch = normalize_batch(rows((IN_RUN_ZERO[0], "093000")), SYMBOL, table)
        assert batch.frame["volume"].to_list() == [ABSENT_VOLUME]

    def test_the_rows_come_back_sorted(self, table: OffsetTable) -> None:
        batch = normalize_batch(
            rows(
                (IN_RUN_PLUS_TWO[0], "120000"),
                (IN_RUN_ZERO[0], "120000"),
                (IN_RUN_PLUS_ONE[0], "120000"),
            ),
            SYMBOL,
            table,
        )
        assert batch.frame["timestamp"].is_sorted()

    def test_the_result_is_accepted_by_ohlcvframe(self, table: OffsetTable) -> None:
        # The point of the canonical shape: the next stage must not have to
        # repair it.
        batch = normalize_batch(
            rows((IN_RUN_ZERO[0], "093000"), (IN_RUN_ZERO[0], "094000")), SYMBOL, table
        )
        frame = OHLCVFrame(batch.frame, SYMBOL, Timeframe.M1)
        assert len(frame) == 2

    def test_an_empty_result_is_still_a_valid_shape(self, table: OffsetTable) -> None:
        batch = normalize_batch(rows((BEFORE_COVERAGE, "093000")), SYMBOL, table)
        assert tuple(batch.frame.columns) == NORMALISED_COLUMNS
        assert OHLCVFrame(batch.frame, SYMBOL, Timeframe.M1).is_empty


class TestTheSymbolIsCheckedFirst:
    def test_an_inherited_symbol_is_refused_before_any_row_is_read(
        self, table: OffsetTable
    ) -> None:
        # XAGUSD's clock was never measured — no triangular identity exists for
        # a metal — so it must not ride along on the FX finding.
        with pytest.raises(DataError, match="no calibration for XAGUSD"):
            normalize_batch(rows((IN_RUN_ZERO[0], "093000")), "XAGUSD", table)

    def test_the_refusal_says_what_is_missing(self, table: OffsetTable) -> None:
        with pytest.raises(DataError, match="separate reference series"):
            normalize_batch(rows((IN_RUN_ZERO[0], "093000")), "XAGUSD", table)

    def test_it_is_refused_even_when_there_are_no_rows_at_all(self, table: OffsetTable) -> None:
        # Proof the check precedes the data rather than falling out of it.
        with pytest.raises(DataError, match="no calibration for XAGUSD"):
            normalize_batch(rows(), "XAGUSD", table)

    def test_a_symbol_outside_the_export_gets_a_different_message(self, table: OffsetTable) -> None:
        # "Not measured" and "not one of ours" need different work.
        with pytest.raises(DataError, match="not among the symbols"):
            normalize_batch(rows((IN_RUN_ZERO[0], "093000")), "EURUSD", table)

    def test_missing_columns_are_refused(self, table: OffsetTable) -> None:
        with pytest.raises(DataError, match="missing columns"):
            normalize_batch(rows((IN_RUN_ZERO[0], "093000")).drop("close"), SYMBOL, table)


class TestTheStepBetweenAdjacentRuns:
    """Regression: real data, not reasoning, found this one.

    Normalising the real EURJPY export produced 60 non-increasing timestamps.
    All of them sat at ``2022-12-05 (+2)`` meeting ``2022-12-06 (+1)`` — the one
    boundary in the shipped table that drops an hour with no weekend to absorb
    it, so the earlier day's last hour and the later day's first hour land on
    the same UTC hour.
    """

    def test_both_sides_of_the_step_are_dropped(self, table: OffsetTable) -> None:
        batch = normalize_batch(rows(("20221205", "230000"), ("20221206", "000000")), SYMBOL, table)
        assert batch.frame.height == 0
        assert [(day.day, day.reason, day.rows) for day in batch.excluded] == [
            (date(2022, 12, 5), "offset_step", 1),
            (date(2022, 12, 6), "offset_step", 1),
        ]

    def test_hours_away_from_the_step_are_untouched(self, table: OffsetTable) -> None:
        batch = normalize_batch(rows(("20221205", "220000"), ("20221206", "010000")), SYMBOL, table)
        assert batch.frame.height == 2
        assert batch.excluded == ()

    def test_the_result_has_strictly_increasing_stamps_across_the_step(
        self, table: OffsetTable
    ) -> None:
        # The property the defect broke. Without the exclusion, 23:00 on the 5th
        # (+2 -> 01:00 on the 6th) collides with 00:00 on the 6th (+1 -> 01:00).
        stamps = [("20221205", f"{hour:02d}0000") for hour in range(20, 24)]
        stamps += [("20221206", f"{hour:02d}0000") for hour in range(4)]
        batch = normalize_batch(rows(*stamps), SYMBOL, table)
        series = batch.frame["timestamp"]
        assert series.n_unique() == series.len()
        assert series.is_sorted()

    def test_a_step_with_a_weekend_inside_it_drops_nothing(self, table: OffsetTable) -> None:
        # 2021-10-29 (+2) -> 2021-11-01 (0) is a two-hour drop, but the market
        # was shut in between, so no bar can collide and nothing needs dropping.
        batch = normalize_batch(rows(("20211029", "230000"), ("20211101", "000000")), SYMBOL, table)
        assert batch.frame.height == 2
        assert batch.excluded == ()

    def test_a_step_up_loses_no_rows(self, table: OffsetTable) -> None:
        # 2020-09-30 (0) -> 2020-10-01 (+1) is adjacent too, but a step up
        # opens an hour-wide hole rather than overlapping: nothing is ambiguous.
        batch = normalize_batch(rows(("20200930", "230000"), ("20201001", "000000")), SYMBOL, table)
        assert batch.frame.height == 2
        assert batch.excluded == ()


class TestAmbiguousDaysAreCountedNotExcluded:
    def test_a_row_on_an_ambiguous_day_is_normalised_and_counted(self, table: OffsetTable) -> None:
        # All 61 ambiguous days in the shipped table fall inside a run, so they
        # are covered by their neighbours. They are still leaning on a
        # measurement that is not their own, which the count makes visible.
        assert table.ambiguous_days, "the shipped table should carry ambiguous days"
        day = table.ambiguous_days[0]
        batch = normalize_batch(rows((day.strftime("%Y%m%d"), "093000")), SYMBOL, table)
        assert batch.frame.height == 1
        assert batch.excluded == ()
        assert batch.rows_on_ambiguous_days == 1

    def test_an_ordinary_day_does_not_count_towards_it(self, table: OffsetTable) -> None:
        batch = normalize_batch(rows((IN_RUN_ZERO[0], "093000")), SYMBOL, table)
        assert batch.rows_on_ambiguous_days == 0
