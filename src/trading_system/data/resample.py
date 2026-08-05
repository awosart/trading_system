"""Aggregation of fine-grained bars into coarser timeframes.

Two properties matter more than anything else here.

**No lookahead.** A resampled bar is emitted only once the source data covers its
entire window. Resampling 1m bars that stop at 12:30 must not produce a 12:00 H1
bar, because that bar's close is not yet knowable at 12:30 — emitting it would
leak future information into every backtest built on top.

**Session-anchored days.** A daily FX bar closes at the market's close, not at UTC
midnight. The anchor is a local wall-clock time in a real timezone, so it tracks
DST: a day anchored to 17:00 New York ends at 22:00 UTC in winter and 21:00 UTC in
summer, and the transition days are 23 or 25 hours long.
"""

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo

import polars as pl

from trading_system.core.exceptions import DataError
from trading_system.core.types import Timeframe
from trading_system.data.models import TIMESTAMP_COLUMN, OHLCVFrame

_POLARS_EVERY: dict[Timeframe, str] = {
    Timeframe.M1: "1m",
    Timeframe.M5: "5m",
    Timeframe.M15: "15m",
    Timeframe.H1: "1h",
    Timeframe.H4: "4h",
    Timeframe.D1: "1d",
}


@dataclass(frozen=True)
class DayOrigin:
    """Where the trading day starts, for session-anchored aggregation.

    Attributes:
        tz: IANA timezone the anchor is expressed in, e.g. ``"America/New_York"``.
        at: Local wall-clock time at which the trading day begins.
    """

    tz: str
    at: time = time(0, 0)

    def offset_expression(self) -> str:
        """Render the anchor as a polars duration offset from local midnight."""
        parts = []
        if self.at.hour:
            parts.append(f"{self.at.hour}h")
        if self.at.minute:
            parts.append(f"{self.at.minute}m")
        if self.at.second:
            parts.append(f"{self.at.second}s")
        return "".join(parts) or "0s"


#: FX convention: the trading day rolls at 17:00 New York, tracking US DST.
FX_DAY_ORIGIN = DayOrigin(tz="America/New_York", at=time(17, 0))


def anchor_offset(origin: DayOrigin) -> timedelta:
    """Distance from local midnight to the trading day's anchor.

    Args:
        origin: Trading-day anchor.

    Returns:
        The offset as a duration.
    """
    return timedelta(hours=origin.at.hour, minutes=origin.at.minute, seconds=origin.at.second)


def trading_day(timestamp: datetime, origin: DayOrigin) -> date:
    """Which trading day an instant belongs to under ``origin``.

    The one definition of "what day is it" in this system. A session VWAP resets
    on this boundary and so does a daily loss limit; if those were two functions
    they could disagree, and the disagreement would be invisible — a trade booked
    into one day by the indicator and the next day by the risk limit.

    **The anchor instant is never constructed**, only the day label is computed.
    Building "02:30 local on the reset date" and comparing against it breaks on
    the spring-forward Sunday, when that local time does not exist, and is
    ambiguous on the autumn one, when it happens twice. Converting the instant
    into the zone and subtracting the offset in wall-clock terms is defined
    everywhere and monotone. The consequence is correct rather than merely
    tolerable: on a transition day the trading day really is 23 or 25 hours long.

    Args:
        timestamp: A tz-aware instant.
        origin: Trading-day anchor: an IANA zone plus a local wall-clock time.
            Never a fixed UTC offset — 17:00 New York is 22:00 UTC in winter and
            21:00 UTC in summer, and a stored offset silently drifts twice a year.

    Returns:
        The date labelling the trading day the instant falls in.

    Raises:
        ValueError: If ``timestamp`` is naive.
    """
    if timestamp.tzinfo is None:
        raise ValueError("timestamp must be tz-aware to be placed in a trading day")
    local = timestamp.astimezone(ZoneInfo(origin.tz)).replace(tzinfo=None)
    return (local - anchor_offset(origin)).date()


def resample(
    frame: OHLCVFrame,
    target: Timeframe,
    *,
    origin: DayOrigin | None = None,
    drop_incomplete: bool = True,
) -> OHLCVFrame:
    """Aggregate ``frame`` up to a coarser timeframe.

    Bars are labelled with the OPEN time of their window and windows are
    half-open ``[start, end)``, matching the project-wide bar convention.

    Args:
        frame: Source bars. Must be finer than ``target``.
        target: Timeframe to aggregate into.
        origin: Trading-day anchor. ``None`` aligns windows to UTC midnight,
            which is correct for crypto and for intraday timeframes that divide
            the day evenly; FX daily bars should pass :data:`FX_DAY_ORIGIN`.
        drop_incomplete: Discard a trailing window the source does not fully
            cover. Leave enabled outside of live streaming — a partial bar is
            the classic lookahead bug.

    Returns:
        A new frame at ``target`` resolution.

    Raises:
        DataError: If ``target`` is not a whole multiple of the source timeframe,
            or is not strictly coarser than it.
    """
    source = frame.timeframe
    _validate_step(source, target)

    if frame.is_empty:
        return OHLCVFrame.empty(frame.symbol, target)

    every = _POLARS_EVERY[target]
    grouping_tz = origin.tz if origin is not None else "UTC"
    offset = origin.offset_expression() if origin is not None else "0s"

    localised = frame.df.with_columns(pl.col(TIMESTAMP_COLUMN).dt.convert_time_zone(grouping_tz))
    aggregated = localised.group_by_dynamic(
        TIMESTAMP_COLUMN,
        every=every,
        offset=offset,
        closed="left",
        label="left",
    ).agg(
        pl.col("open").first(),
        pl.col("high").max(),
        pl.col("low").min(),
        pl.col("close").last(),
        pl.col("volume").sum(),
    )

    if drop_incomplete:
        aggregated = _drop_incomplete_windows(aggregated, frame, every)

    resampled = aggregated.with_columns(
        pl.col(TIMESTAMP_COLUMN).dt.convert_time_zone("UTC").dt.cast_time_unit("us")
    ).sort(TIMESTAMP_COLUMN)
    return OHLCVFrame(resampled, frame.symbol, target)


def _drop_incomplete_windows(
    aggregated: pl.DataFrame, frame: OHLCVFrame, every: str
) -> pl.DataFrame:
    """Remove windows the source data does not fully cover, at either end.

    A trailing partial window is the lookahead hazard: its close is not yet
    knowable. A leading partial window is a subtler defect — it is stamped with a
    window open time it has no data for, so its open price, extremes and volume
    all describe a shorter period than the label claims. Neither is a bar that
    represents its window, so neither is published.

    The source's last bar opens at ``last`` and covers ``[last, last + source)``,
    so data is known through ``last + source``.
    """
    first_open = frame.timestamps.item(0)
    known_through = frame.timestamps.item(-1) + frame.timeframe.duration
    # offset_by respects DST, so a 1d window over a transition is 23h or 25h.
    window_start = pl.col(TIMESTAMP_COLUMN).dt.convert_time_zone("UTC")
    window_end = pl.col(TIMESTAMP_COLUMN).dt.offset_by(every).dt.convert_time_zone("UTC")
    return aggregated.filter((window_start >= first_open) & (window_end <= known_through))


def _validate_step(source: Timeframe, target: Timeframe) -> None:
    """Reject aggregations that are not a clean whole-number step upwards."""
    if target is source:
        raise DataError(f"target timeframe {target} equals the source; nothing to resample")
    source_seconds = int(source.duration.total_seconds())
    target_seconds = int(target.duration.total_seconds())
    if target_seconds < source_seconds:
        raise DataError(
            f"cannot resample {source} up to finer timeframe {target}; "
            "downsampling would have to invent data"
        )
    if target_seconds % source_seconds:
        raise DataError(
            f"{target} is not a whole multiple of {source}; "
            f"{target_seconds}s / {source_seconds}s leaves a remainder"
        )
