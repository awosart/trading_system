"""Data-quality detectors producing a structured, severity-graded report.

These checks grade rather than reject. Real vendor data always contains some
defects, so a loader that refuses to load anything imperfect is useless; the
caller decides what is tolerable. Structural guarantees — schema, ordering,
nullability — are enforced by :class:`~trading_system.data.models.OHLCVFrame`
instead, because those genuinely make the data unusable.

Checks run on a raw DataFrame so that duplicate timestamps are still detectable;
a constructed frame can no longer contain any, and reporting zero for a defect
the type system already excluded would be misleading.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum

import polars as pl

from trading_system.core.exceptions import DataError
from trading_system.core.types import Timeframe
from trading_system.data.models import REQUIRED_COLUMNS, TIMESTAMP_COLUMN, OHLCVFrame
from trading_system.data.sessions import TradingCalendar


class Severity(StrEnum):
    """How badly a detected issue compromises the data.

    Attributes:
        INFO: Worth knowing, harmless.
        WARN: Suspicious; may distort statistics but the data remains usable.
        ERROR: Logically impossible values. Backtests on this data are invalid.
    """

    INFO = "INFO"
    WARN = "WARN"
    ERROR = "ERROR"


_SEVERITY_ORDER: dict[str, int] = {"INFO": 0, "WARN": 1, "ERROR": 2}


@dataclass(frozen=True)
class QualityConfig:
    """Thresholds governing the detectors.

    Attributes:
        atr_period: Lookback for the ATR used to scale gap detection.
        gap_atr_multiple: A bar-to-bar gap wider than this many ATRs is flagged.
        return_zscore_threshold: Robust z-score of a close-to-close return above
            which the bar is reported as an outlier. Scored against the median
            absolute deviation, so the default is deliberately loose: financial
            returns are fat-tailed and a tighter bound flags ordinary volatility.
        frozen_run_length: Number of consecutive flat bars that constitutes a
            frozen feed.
        max_samples: Example timestamps retained per issue.
        calendar: Market calendar used to tell a genuine missing bar from a
            legitimate weekend or holiday. Without one, every closed period is
            reported as missing data.
        max_missing_scan: Upper bound on missing-bar candidates classified
            against the calendar, guarding against pathological gaps.
    """

    atr_period: int = 14
    gap_atr_multiple: float = 5.0
    return_zscore_threshold: float = 8.0
    frozen_run_length: int = 3
    max_samples: int = 5
    calendar: TradingCalendar | None = None
    max_missing_scan: int = 200_000


@dataclass(frozen=True)
class QualityIssue:
    """One detected defect, aggregated across the dataset.

    Attributes:
        code: Stable machine-readable identifier.
        severity: How serious the defect is.
        count: Number of affected bars.
        message: Human-readable explanation.
        samples: Up to ``max_samples`` example timestamps.
    """

    code: str
    severity: Severity
    count: int
    message: str
    samples: tuple[datetime, ...] = ()


@dataclass(frozen=True)
class DataQualityReport:
    """Outcome of running every detector over one dataset.

    Attributes:
        symbol: Instrument the data belongs to.
        timeframe: Bar size checked.
        bars_checked: Number of rows examined.
        issues: Detected defects, most severe first.
    """

    symbol: str
    timeframe: Timeframe
    bars_checked: int
    issues: tuple[QualityIssue, ...] = field(default_factory=tuple)

    @property
    def has_errors(self) -> bool:
        """Whether any ``ERROR``-severity issue was found."""
        return any(issue.severity is Severity.ERROR for issue in self.issues)

    @property
    def max_severity(self) -> Severity | None:
        """The worst severity present, or ``None`` when the data is clean."""
        if not self.issues:
            return None
        return max(self.issues, key=lambda issue: _SEVERITY_ORDER[issue.severity]).severity

    def counts_by_severity(self) -> dict[Severity, int]:
        """Total affected bars grouped by severity."""
        totals: dict[Severity, int] = {}
        for issue in self.issues:
            totals[issue.severity] = totals.get(issue.severity, 0) + issue.count
        return totals

    def issue(self, code: str) -> QualityIssue | None:
        """Return the issue with ``code``, or ``None`` if it was not raised."""
        return next((issue for issue in self.issues if issue.code == code), None)

    def codes(self) -> set[str]:
        """Set of raised issue codes."""
        return {issue.code for issue in self.issues}


def check_frame(frame: OHLCVFrame, config: QualityConfig | None = None) -> DataQualityReport:
    """Run every detector over a validated frame.

    Args:
        frame: Data to inspect.
        config: Detector thresholds; defaults are used when omitted.

    Returns:
        The resulting report.
    """
    return check_quality(
        frame.df, symbol=frame.symbol, timeframe=frame.timeframe, config=config
    )


def check_quality(
    df: pl.DataFrame,
    *,
    symbol: str,
    timeframe: Timeframe,
    config: QualityConfig | None = None,
) -> DataQualityReport:
    """Run every detector over a raw OHLCV DataFrame.

    Args:
        df: Bars to inspect. Must carry the canonical OHLCV column names but need
            not be sorted, deduplicated, or correctly typed.
        symbol: Instrument identifier, for reporting.
        timeframe: Bar size, used for missing-bar detection.
        config: Detector thresholds; defaults are used when omitted.

    Returns:
        The resulting report, with issues ordered most severe first.

    Raises:
        DataError: If required columns are absent.
    """
    settings = config or QualityConfig()
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing_columns:
        raise DataError(f"{symbol}: missing required columns {missing_columns}")

    working = df.select(REQUIRED_COLUMNS).with_columns(
        *[pl.col(column).cast(pl.Float64) for column in ("open", "high", "low", "close", "volume")]
    )

    issues: list[QualityIssue] = []
    issues.extend(_check_duplicates(working, settings))
    ordered = working.sort(TIMESTAMP_COLUMN).unique(
        subset=[TIMESTAMP_COLUMN], keep="last", maintain_order=True
    )
    issues.extend(_check_ohlc_relations(ordered, settings))
    issues.extend(_check_volume(ordered, settings))
    issues.extend(_check_frozen(ordered, settings))
    issues.extend(_check_gaps(ordered, settings))
    issues.extend(_check_return_outliers(ordered, settings))
    issues.extend(_check_missing_bars(ordered, timeframe, settings))

    issues.sort(key=lambda issue: (-_SEVERITY_ORDER[issue.severity], issue.code))
    return DataQualityReport(
        symbol=symbol,
        timeframe=timeframe,
        bars_checked=working.height,
        issues=tuple(issues),
    )


def _samples(df: pl.DataFrame, mask: pl.Expr, limit: int) -> tuple[datetime, ...]:
    """Collect up to ``limit`` timestamps where ``mask`` holds."""
    matched = df.filter(mask).head(limit)[TIMESTAMP_COLUMN].to_list()
    return tuple(matched)


def _issue_if_any(
    df: pl.DataFrame,
    mask: pl.Expr,
    *,
    code: str,
    severity: Severity,
    message: str,
    limit: int,
) -> list[QualityIssue]:
    """Build a single-issue list when ``mask`` matches at least one row."""
    count = int(df.select(mask.sum()).item() or 0)
    if not count:
        return []
    return [
        QualityIssue(
            code=code,
            severity=severity,
            count=count,
            message=message,
            samples=_samples(df, mask, limit),
        )
    ]


def _check_duplicates(df: pl.DataFrame, config: QualityConfig) -> list[QualityIssue]:
    """Detect repeated timestamps."""
    duplicated = df.filter(pl.col(TIMESTAMP_COLUMN).is_duplicated())
    if duplicated.is_empty():
        return []
    distinct = duplicated[TIMESTAMP_COLUMN].unique().sort()
    extra = duplicated.height - distinct.len()
    return [
        QualityIssue(
            code="duplicate_timestamp",
            severity=Severity.ERROR,
            count=extra,
            message=(
                f"{extra} duplicate bar(s) across {distinct.len()} timestamp(s); "
                "the same instant cannot have two different bars"
            ),
            samples=tuple(distinct.head(config.max_samples).to_list()),
        )
    ]


def _check_ohlc_relations(df: pl.DataFrame, config: QualityConfig) -> list[QualityIssue]:
    """Detect logically impossible OHLC arrangements."""
    issues: list[QualityIssue] = []
    issues.extend(
        _issue_if_any(
            df,
            pl.col("high") < pl.col("low"),
            code="high_below_low",
            severity=Severity.ERROR,
            message="high is below low",
            limit=config.max_samples,
        )
    )
    for column in ("open", "close"):
        issues.extend(
            _issue_if_any(
                df,
                (pl.col(column) > pl.col("high")) | (pl.col(column) < pl.col("low")),
                code=f"{column}_outside_range",
                severity=Severity.ERROR,
                message=f"{column} falls outside the bar's [low, high] range",
                limit=config.max_samples,
            )
        )
    issues.extend(
        _issue_if_any(
            df,
            pl.min_horizontal("open", "high", "low", "close") <= 0,
            code="non_positive_price",
            severity=Severity.ERROR,
            message="price is zero or negative",
            limit=config.max_samples,
        )
    )
    return issues


def _check_volume(df: pl.DataFrame, config: QualityConfig) -> list[QualityIssue]:
    """Detect negative and zero volume."""
    return [
        *_issue_if_any(
            df,
            pl.col("volume") < 0,
            code="negative_volume",
            severity=Severity.ERROR,
            message="volume is negative",
            limit=config.max_samples,
        ),
        *_issue_if_any(
            df,
            pl.col("volume") == 0,
            code="zero_volume",
            severity=Severity.WARN,
            message="volume is zero, suggesting an untraded or synthetic bar",
            limit=config.max_samples,
        ),
    ]


def _check_frozen(df: pl.DataFrame, config: QualityConfig) -> list[QualityIssue]:
    """Detect runs of completely flat bars, the signature of a stalled feed."""
    if df.height < config.frozen_run_length:
        return []
    flat = (
        (pl.col("open") == pl.col("high"))
        & (pl.col("high") == pl.col("low"))
        & (pl.col("low") == pl.col("close"))
    )
    marked = df.with_columns(flat.alias("_flat"))
    # Number consecutive groups so runs of flat bars can be measured.
    marked = marked.with_columns(
        (pl.col("_flat") != pl.col("_flat").shift(1).fill_null(False)).cum_sum().alias("_run")
    )
    runs = (
        marked.filter(pl.col("_flat"))
        .group_by("_run")
        .agg(
            pl.len().alias("length"),
            pl.col(TIMESTAMP_COLUMN).first().alias("started_at"),
        )
        .filter(pl.col("length") >= config.frozen_run_length)
        .sort("started_at")
    )
    if runs.is_empty():
        return []
    affected = int(runs["length"].sum())
    return [
        QualityIssue(
            code="frozen_bars",
            severity=Severity.WARN,
            count=affected,
            message=(
                f"{runs.height} run(s) of >= {config.frozen_run_length} bars with "
                f"identical OHLC, covering {affected} bars"
            ),
            samples=tuple(runs["started_at"].head(config.max_samples).to_list()),
        )
    ]


def _check_gaps(df: pl.DataFrame, config: QualityConfig) -> list[QualityIssue]:
    """Detect price jumps between bars that are large relative to recent range."""
    if df.height < config.atr_period + 2:
        return []
    previous_close = pl.col("close").shift(1)
    true_range = pl.max_horizontal(
        pl.col("high") - pl.col("low"),
        (pl.col("high") - previous_close).abs(),
        (pl.col("low") - previous_close).abs(),
    )
    enriched = df.with_columns(true_range.alias("_tr")).with_columns(
        pl.col("_tr").rolling_mean(window_size=config.atr_period).shift(1).alias("_atr"),
        (pl.col("open") - pl.col("close").shift(1)).abs().alias("_gap"),
    )
    mask = (
        pl.col("_atr").is_not_null()
        & (pl.col("_atr") > 0)
        & (pl.col("_gap") > config.gap_atr_multiple * pl.col("_atr"))
    )
    return _issue_if_any(
        enriched,
        mask,
        code="price_gap",
        severity=Severity.WARN,
        message=f"open gaps more than {config.gap_atr_multiple}x ATR from the previous close",
        limit=config.max_samples,
    )


def _check_return_outliers(df: pl.DataFrame, config: QualityConfig) -> list[QualityIssue]:
    """Detect close-to-close returns far outside the sample distribution.

    Scoring uses the median and median absolute deviation rather than the mean
    and standard deviation. A single bad tick inflates the standard deviation it
    would be measured against — with n samples the largest attainable classical
    z-score is only about ``(n-1)/sqrt(n)``, so in a few hundred bars one severe
    outlier masks itself and is never flagged. The MAD is unmoved by it.
    """
    if df.height < 3:
        return []
    enriched = df.with_columns((pl.col("close") / pl.col("close").shift(1) - 1).alias("_ret"))
    median = enriched.select(pl.col("_ret").median()).item()
    if median is None:
        return []

    deviation = (pl.col("_ret") - median).abs()
    mad = enriched.select(deviation.median()).item()
    if mad is not None and mad > 0:
        # 0.6745 rescales the MAD so the score matches a standard deviation
        # under a normal distribution (Iglewicz-Hoaglin).
        score = 0.6745 * deviation / mad
        basis = "median absolute deviations"
    else:
        sigma = enriched.select(pl.col("_ret").std()).item()
        if sigma is None or sigma == 0:
            return []
        score = deviation / sigma
        basis = "standard deviations"

    return _issue_if_any(
        enriched,
        pl.col("_ret").is_not_null() & (score > config.return_zscore_threshold),
        code="return_outlier",
        severity=Severity.WARN,
        message=(
            f"close-to-close return exceeds {config.return_zscore_threshold} {basis} "
            "from the median, suggesting a bad tick"
        ),
        limit=config.max_samples,
    )


def _check_missing_bars(
    df: pl.DataFrame, timeframe: Timeframe, config: QualityConfig
) -> list[QualityIssue]:
    """Detect bars absent from an otherwise continuous series.

    With a calendar, weekends and holidays are excluded so only genuine holes are
    reported. Without one, every closed period counts as missing — which is why
    supplying a calendar matters for anything but crypto.
    """
    if df.height < 2 or timeframe is Timeframe.D1:
        return []

    first = df[TIMESTAMP_COLUMN].item(0)
    last = df[TIMESTAMP_COLUMN].item(-1)
    expected = pl.datetime_range(
        first, last, interval=timeframe.duration, time_zone="UTC", eager=True
    ).alias(TIMESTAMP_COLUMN)

    present = set(df[TIMESTAMP_COLUMN].to_list())
    candidates = [ts for ts in expected.to_list() if ts not in present]
    if not candidates:
        return []

    truncated = len(candidates) > config.max_missing_scan
    scanned = candidates[: config.max_missing_scan]
    calendar = config.calendar
    if calendar is not None:
        scanned = [ts for ts in scanned if calendar.is_open(ts)]
        if not scanned:
            return []

    note = " (scan truncated)" if truncated else ""
    return [
        QualityIssue(
            code="missing_bars",
            severity=Severity.WARN,
            count=len(scanned),
            message=(
                f"{len(scanned)} expected {timeframe} bar(s) absent between "
                f"{first:%Y-%m-%d %H:%M} and {last:%Y-%m-%d %H:%M}{note}"
            ),
            samples=tuple(scanned[: config.max_samples]),
        )
    ]
