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

**A bar's close can land inside a window the market never opened in at all.**
This module gives every intraday timeframe a plain ``open + duration`` close —
correct as an instant, since that really is when the bar is complete and may
be published. But for a timeframe whose duration does not evenly divide the
trading week (H4 chief among them), the very last bar before a weekly close can
finish a few hours *into* the following ``trading_day`` window, and FX's week
is shut for the whole of that window. ``trading_day`` labels it anyway — it
only knows about 24-hour slices, not about when a market is open — and the
result is a trading day with exactly one bar in it, a fact nothing downstream
expects. The instant computed here is not what changes: it really is when the
bar closes, and moving it would be lookahead by another name. What changes is
which *label* consumers of that instant attach to it —
:func:`~trading_system.data.resample.trading_day_of_close` is where that
correction lives, next to :func:`~trading_system.data.resample.trading_day`
itself rather than here, because both :class:`~trading_system.backtest.portfolio.Portfolio`
and :class:`~trading_system.risk.circuit_breakers.CircuitBreakers` need it and
``risk`` sits below ``backtest`` in this project's import direction — a
function only this module could reach would have put the fix out of one of
its two consumers' reach.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta

from trading_system.core.types import Timeframe
from trading_system.data.resample import DayOrigin, trading_day

#: Sort rank per timeframe, finest first. Fixes the order of several bars closing
#: at one instant so a run is reproducible; correctness does not depend on it,
#: because publication and evaluation are separate phases.
#:
#: Derived from ``Timeframe.duration`` rather than listing the members, because a
#: hand-written list is a second authority on which timeframes exist and it went
#: out of date the moment one was added: ``M30`` arrived with the third vendor
#: import and this table did not learn about it, so every M30 stream died on a
#: ``KeyError`` deep inside the heap merge — a timeframe the store held, the
#: schema accepted and the engine could not run. Ordering by duration cannot
#: drift that way, and it reproduces the previous order of the other six exactly.
#: The rank is a tie-break only: it never leaves this module and never enters a
#: digest, so the values themselves are free to shift when a member is added.
TIMEFRAME_RANK: dict[Timeframe, int] = {
    timeframe: rank for rank, timeframe in enumerate(sorted(Timeframe, key=lambda tf: tf.duration))
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
