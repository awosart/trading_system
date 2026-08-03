"""Data-quality detectors: broken data must produce the expected severities."""

from datetime import UTC, datetime, timedelta
from typing import cast

import polars as pl
import pytest

from trading_system.core.exceptions import DataError
from trading_system.core.types import Timeframe
from trading_system.data.quality import (
    DataQualityReport,
    QualityConfig,
    Severity,
    check_frame,
    check_quality,
)
from trading_system.data.sessions import AssetClass, TradingCalendar

from .conftest import make_frame

START = datetime(2024, 1, 3, 0, 0, tzinfo=UTC)  # a Wednesday


def clean(count: int = 30) -> pl.DataFrame:
    """Build a plausible, defect-free hourly dataset."""
    return pl.DataFrame(
        {
            "timestamp": [START + timedelta(hours=i) for i in range(count)],
            "open": [100.0 + i * 0.1 for i in range(count)],
            "high": [100.5 + i * 0.1 for i in range(count)],
            "low": [99.5 + i * 0.1 for i in range(count)],
            "close": [100.2 + i * 0.1 for i in range(count)],
            "volume": [1000.0 + i for i in range(count)],
        }
    ).with_columns(pl.col("timestamp").dt.replace_time_zone("UTC"))


def report_for(df: pl.DataFrame, config: QualityConfig | None = None) -> DataQualityReport:
    """Run the detectors over ``df``."""
    return check_quality(df, symbol="EURUSD", timeframe=Timeframe.H1, config=config)


def test_clean_data_raises_nothing() -> None:
    report = report_for(clean())
    assert report.issues == ()
    assert report.max_severity is None
    assert not report.has_errors
    assert report.bars_checked == 30


class TestErrorSeverityDefects:
    """Logically impossible values must be graded ERROR, not merely warned about."""

    def test_high_below_low(self) -> None:
        df = clean()
        df[5, "high"] = 90.0
        df[5, "low"] = 110.0
        report = report_for(df)
        issue = report.issue("high_below_low")
        assert issue is not None
        assert issue.severity is Severity.ERROR
        assert issue.count == 1
        assert issue.samples == (START + timedelta(hours=5),)
        assert report.has_errors

    def test_close_outside_range(self) -> None:
        df = clean()
        df[7, "close"] = 500.0
        issue = report_for(df).issue("close_outside_range")
        assert issue is not None
        assert issue.severity is Severity.ERROR
        assert issue.count == 1

    def test_open_outside_range(self) -> None:
        df = clean()
        df[9, "open"] = 0.01
        issue = report_for(df).issue("open_outside_range")
        assert issue is not None
        assert issue.severity is Severity.ERROR

    def test_negative_volume(self) -> None:
        df = clean()
        df[3, "volume"] = -5.0
        issue = report_for(df).issue("negative_volume")
        assert issue is not None
        assert issue.severity is Severity.ERROR
        assert issue.count == 1

    def test_non_positive_price(self) -> None:
        df = clean()
        for column in ("open", "high", "low", "close"):
            df[4, column] = 0.0
        issue = report_for(df).issue("non_positive_price")
        assert issue is not None
        assert issue.severity is Severity.ERROR

    def test_duplicate_timestamps(self) -> None:
        df = pl.concat([clean(), clean().head(2)], how="vertical")
        issue = report_for(df).issue("duplicate_timestamp")
        assert issue is not None
        assert issue.severity is Severity.ERROR
        assert issue.count == 2


class TestWarnSeverityDefects:
    """Suspicious but survivable data is graded WARN."""

    def test_zero_volume(self) -> None:
        df = clean()
        df[6, "volume"] = 0.0
        issue = report_for(df).issue("zero_volume")
        assert issue is not None
        assert issue.severity is Severity.WARN

    def test_frozen_bars(self) -> None:
        df = clean()
        for row in (10, 11, 12, 13):
            for column in ("open", "high", "low", "close"):
                df[row, column] = 100.0
        issue = report_for(df).issue("frozen_bars")
        assert issue is not None
        assert issue.severity is Severity.WARN
        assert issue.count == 4

    def test_short_flat_run_is_not_frozen(self) -> None:
        """A single flat bar is normal in thin markets; it must not be flagged."""
        df = clean()
        for column in ("open", "high", "low", "close"):
            df[10, column] = 100.0
        assert report_for(df).issue("frozen_bars") is None

    def test_price_gap_against_atr(self) -> None:
        df = clean(60)
        df[40, "open"] = 200.0
        df[40, "high"] = 200.5
        issue = report_for(df).issue("price_gap")
        assert issue is not None
        assert issue.severity is Severity.WARN

    def test_return_outlier(self) -> None:
        df = clean(60)
        df[31, "close"] = 400.0
        df[31, "high"] = 400.0
        issue = report_for(df).issue("return_outlier")
        assert issue is not None
        assert issue.severity is Severity.WARN

    def test_lone_outlier_is_not_masked_by_its_own_dispersion(self) -> None:
        """A single bad tick must be caught in a small sample.

        Scored classically, one outlier inflates the standard deviation it is
        compared against: with n returns the largest attainable z-score is about
        ``(n-1)/sqrt(n)``, roughly 7.6 here, so a threshold of 8 would never
        fire. The MAD-based score is unaffected by the outlier's own magnitude.
        """
        df = clean(60)
        df[31, "close"] = 400.0
        df[31, "high"] = 400.0

        returns = (df["close"] / df["close"].shift(1) - 1).drop_nulls()
        classical_peak = float(
            cast(float, ((returns - returns.mean()) / returns.std()).abs().max())
        )
        assert classical_peak < 8.0  # the masking effect, demonstrated

        assert report_for(df).issue("return_outlier") is not None


class TestMissingBars:
    def test_hole_is_reported(self) -> None:
        df = clean(30)
        with_hole = df.filter(df["timestamp"] != START + timedelta(hours=10))
        issue = report_for(with_hole).issue("missing_bars")
        assert issue is not None
        assert issue.severity is Severity.WARN
        assert issue.count == 1
        assert issue.samples == (START + timedelta(hours=10),)

    def test_calendar_excludes_the_weekend(self) -> None:
        """The FX weekend break is not missing data.

        In January the FX week closes 17:00 EST (22:00 UTC) Friday and reopens
        22:00 UTC Sunday, so a series running to 21:00 Friday and resuming at
        22:00 Sunday has no genuine hole.
        """
        friday_last_bar = datetime(2024, 1, 5, 21, 0, tzinfo=UTC)
        sunday_reopen = datetime(2024, 1, 7, 22, 0, tzinfo=UTC)
        df = pl.DataFrame(
            {
                "timestamp": [friday_last_bar, sunday_reopen],
                "open": [1.0, 1.0],
                "high": [1.1, 1.1],
                "low": [0.9, 0.9],
                "close": [1.0, 1.0],
                "volume": [5.0, 5.0],
            }
        ).with_columns(pl.col("timestamp").dt.replace_time_zone("UTC"))

        # Without a calendar every closed hour looks like a hole.
        assert report_for(df).issue("missing_bars") is not None

        config = QualityConfig(calendar=TradingCalendar(AssetClass.FX))
        assert report_for(df, config).issue("missing_bars") is None

    def test_calendar_still_reports_holes_during_open_hours(self) -> None:
        """Suppressing the weekend must not suppress a genuine mid-week gap."""
        df = clean(30)
        with_hole = df.filter(df["timestamp"] != START + timedelta(hours=10))
        config = QualityConfig(calendar=TradingCalendar(AssetClass.FX))
        issue = report_for(with_hole, config).issue("missing_bars")
        assert issue is not None
        assert issue.count == 1


def test_issues_are_ordered_most_severe_first() -> None:
    df = clean()
    df[5, "high"] = 90.0
    df[5, "low"] = 110.0
    df[6, "volume"] = 0.0
    severities = [issue.severity for issue in report_for(df).issues]
    assert severities[0] is Severity.ERROR
    assert Severity.WARN in severities
    assert severities == sorted(severities, key=lambda s: {"ERROR": 0, "WARN": 1, "INFO": 2}[s])


def test_counts_by_severity_totals_affected_bars() -> None:
    df = clean()
    df[5, "volume"] = -1.0
    df[6, "volume"] = 0.0
    totals = report_for(df).counts_by_severity()
    assert totals[Severity.ERROR] == 1
    assert totals[Severity.WARN] == 1


def test_samples_are_capped() -> None:
    df = clean(30)
    for row in range(20):
        df[row, "volume"] = 0.0
    issue = report_for(df, QualityConfig(max_samples=3)).issue("zero_volume")
    assert issue is not None
    assert issue.count == 20
    assert len(issue.samples) == 3


def test_check_frame_accepts_a_validated_frame() -> None:
    report = check_frame(make_frame(START, 20, timeframe=Timeframe.H1))
    assert report.symbol == "EURUSD"
    assert report.bars_checked == 20


def test_missing_columns_are_rejected() -> None:
    with pytest.raises(DataError, match="missing required columns"):
        report_for(clean().drop("volume"))


def test_codes_lists_raised_issues() -> None:
    df = clean()
    df[3, "volume"] = -1.0
    assert "negative_volume" in report_for(df).codes()
