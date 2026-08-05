"""When a bar closes, which is the only instant at which it may be published.

``Bar.timestamp`` is a bar's OPEN time. Merging several streams on that field is
the single most destructive defect available in this layer: a D1 bar opening at
21:00 today would enter the stream at 21:00 today, twenty-four hours before it
closes, and a strategy filtering on it would be reading tomorrow's high, low and
close. Every stream is therefore keyed on ``close_ts``, computed here.

**``Timeframe.duration`` is not that answer for D1.** Its own docstring says 24
hours is nominal and that a calendar day spanning a DST transition is 23 or 25
hours long. Adding 24 hours to a daily bar's open puts its close an hour off
twice a year, in a way that shows up as a daily filter that updates at the wrong
time for a fortnight and then corrects itself.

**The boundary instant is found, not constructed.** P10 stage 2 established that
"17:00 local on date D" must never be built — it does not exist on the
spring-forward Sunday and happens twice on the autumn one — and that only day
*labels* may be compared. This module keeps that rule: it takes the candidate
``open + 24h``, and accepts the neighbour an hour either side if that is where
:func:`~trading_system.data.resample.trading_day` says the label actually turns
over. The authority is the label function throughout; the search is bounded
because a DST shift moves the boundary by at most an hour.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta

from trading_system.core.types import Timeframe
from trading_system.data.resample import DayOrigin, trading_day

#: Sort rank per timeframe, finest first. Fixes the order of several bars closing
#: at one instant so a run is reproducible; correctness does not depend on it,
#: because publication and evaluation are separate phases.
TIMEFRAME_RANK: dict[Timeframe, int] = {
    timeframe: rank
    for rank, timeframe in enumerate(
        (
            Timeframe.M1,
            Timeframe.M5,
            Timeframe.M15,
            Timeframe.H1,
            Timeframe.H4,
            Timeframe.D1,
        )
    )
}

#: How far either side of ``open + 24h`` the daily boundary may sit. One hour is
#: every DST shift in the IANA database that a trading-day anchor can meet.
_DAY_BOUNDARY_SEARCH_HOURS = (-1, 0, 1)

#: Granularity at which "the instant before the boundary" is probed.
_EPSILON = timedelta(microseconds=1)


@dataclass(frozen=True, order=True)
class StreamKey:
    """One instrument at one timeframe: the unit a bar stream is keyed by.

    Ordered, so several streams closing at the same instant sort deterministically.

    Attributes:
        symbol: Instrument.
        timeframe: Bar size.
    """

    symbol: str
    timeframe: Timeframe

    def __str__(self) -> str:
        """``EURUSD@H1``."""
        return f"{self.symbol}@{self.timeframe.value}"

    @property
    def rank(self) -> int:
        """Sort rank of this stream's timeframe, finest first."""
        return TIMEFRAME_RANK[self.timeframe]


def bar_close_ts(timeframe: Timeframe, bar_open: datetime, day_origin: DayOrigin) -> datetime:
    """The instant a bar is complete, and therefore the instant it may be seen.

    Args:
        timeframe: Size of the bar.
        bar_open: Its open time, tz-aware UTC.
        day_origin: Where the trading day starts, used for ``D1`` only.

    Returns:
        The close instant, tz-aware UTC.
    """
    if timeframe is not Timeframe.D1:
        return bar_open + timeframe.duration
    return day_close_ts(bar_open, day_origin)


def day_close_ts(bar_open: datetime, day_origin: DayOrigin) -> datetime:
    """The instant the trading day containing ``bar_open`` rolls over.

    Args:
        bar_open: Open time of a daily bar, tz-aware.
        day_origin: Anchor the day is cut on.

    Returns:
        The first instant belonging to the following trading day.

    Raises:
        ValueError: If the boundary is not within an hour of ``open + 24h``,
            which means the bar's open is not on the day anchor at all — a
            resampling or loading defect, not a market condition.
    """
    label = trading_day(bar_open, day_origin)
    base = bar_open + timedelta(days=1)
    for shift in _DAY_BOUNDARY_SEARCH_HOURS:
        candidate = base + timedelta(hours=shift)
        if (
            trading_day(candidate, day_origin) > label
            and trading_day(candidate - _EPSILON, day_origin) <= label
        ):
            return candidate
    raise ValueError(
        f"daily bar opening {bar_open:%F %T%z} does not sit on the {day_origin.tz} "
        f"{day_origin.at} anchor: the day labelled {label} does not turn over within an "
        "hour of 24h later. A daily series must be cut on the same origin the run uses."
    )
