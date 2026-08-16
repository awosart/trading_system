"""Cutting a holdout off every series before anything is allowed to look at it.

A screen over thousands of strategies is a selection, and after it has run every
bar it touched is contaminated: the best row is best partly because it was
chosen, and re-measuring it on the same bars cannot tell the two apart. The only
uncontaminated evidence is a stretch of history the screen never saw, and there
is exactly one moment at which such a stretch can be created — before the screen
runs. Afterwards it does not exist and cannot be manufactured.

So the cut is made here, once, and what the screen is handed is a frame that
**does not contain** the holdout bars rather than one it is asked not to read.
That is the same construction P15 stage 2 uses to keep out-of-sample bars away
from a parameter selector: ``ISWindowView`` refuses to hold a bar at or after
its window's end, so the selector cannot read one whatever it does. A rule that
lives in a docstring is a rule somebody eventually breaks; a bar that is not in
the object cannot be read at all.

The boundary is snapped to a trading-day boundary with the same
:class:`~trading_system.data.resample.DayOrigin` the fold splitter uses, so a
later walk-forward on the holdout starts where a fold would have started and the
join between screen and holdout is not a partial day.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime

from trading_system.data.models import OHLCVFrame
from trading_system.data.resample import FX_DAY_ORIGIN, DayOrigin
from trading_system.validation.splitting import _snap_forward

#: Fraction of each series held back from the screen. A fifth: enough bars for a
#: confirming walk-forward to have several folds on the coarser timeframes, and
#: little enough that the screen still sees the bulk of the history it is
#: ranking on.
DEFAULT_HOLDOUT_FRACTION = 0.2


@dataclass(frozen=True)
class HoldoutBoundary:
    """Where one series was cut, and what each side got.

    Attributes:
        symbol: Instrument.
        timeframe: Bar size, as its value.
        boundary: First instant belonging to the holdout. The screen's frame
            ends strictly before it.
        screen_bars: Bars the screen may see.
        holdout_bars: Bars withheld.
        screen_end: Open time of the last bar the screen may see.
    """

    symbol: str
    timeframe: str
    boundary: datetime
    screen_bars: int
    holdout_bars: int
    screen_end: datetime | None

    def as_record(self) -> dict[str, object]:
        """The boundary as a manifest entry."""
        return {
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "boundary": self.boundary.isoformat(),
            "screen_bars": self.screen_bars,
            "holdout_bars": self.holdout_bars,
            "screen_end": self.screen_end.isoformat() if self.screen_end else None,
        }


def holdout_boundary(
    frame: OHLCVFrame,
    *,
    fraction: float = DEFAULT_HOLDOUT_FRACTION,
    day_origin: DayOrigin = FX_DAY_ORIGIN,
) -> datetime:
    """The first instant that belongs to the holdout.

    Args:
        frame: The whole series.
        fraction: Share of bars to withhold, from the end.
        day_origin: Where the trading day starts, for snapping the cut.

    Returns:
        The boundary, snapped forward to a trading-day boundary. Snapping
        forward rather than to the nearest means the screen never gains a bar
        it would not have had — the error, where there is one, is always in the
        direction of showing the screen less.

    Raises:
        ValueError: If ``fraction`` is not strictly between 0 and 1, or the
            frame is empty.
    """
    if not 0.0 < fraction < 1.0:
        raise ValueError(f"holdout fraction must be in (0, 1), got {fraction}")
    if frame.is_empty:
        raise ValueError("cannot cut a holdout from an empty frame")
    timestamps = frame.timestamps
    index = int(len(timestamps) * (1.0 - fraction))
    index = min(max(index, 1), len(timestamps) - 1)
    return _snap_forward(timestamps[index], day_origin)


def screen_frame(
    frame: OHLCVFrame,
    *,
    fraction: float = DEFAULT_HOLDOUT_FRACTION,
    day_origin: DayOrigin = FX_DAY_ORIGIN,
) -> tuple[OHLCVFrame, HoldoutBoundary]:
    """The part of a series a screen may see, and where it was cut.

    Args:
        frame: The whole series.
        fraction: Share of bars to withhold, from the end.
        day_origin: Where the trading day starts.

    Returns:
        ``(frame, boundary)``. The frame holds no bar at or after the boundary —
        not "should not be read past it": the rows are gone.
    """
    boundary = holdout_boundary(frame, fraction=fraction, day_origin=day_origin)
    visible = frame.slice(None, boundary)
    return visible, HoldoutBoundary(
        symbol=frame.symbol,
        timeframe=frame.timeframe.value,
        boundary=boundary,
        screen_bars=len(visible),
        holdout_bars=len(frame) - len(visible),
        screen_end=visible.end,
    )


def boundaries_for(
    frames: Mapping[tuple[str, str], OHLCVFrame],
    *,
    fraction: float = DEFAULT_HOLDOUT_FRACTION,
    day_origin: DayOrigin = FX_DAY_ORIGIN,
) -> tuple[HoldoutBoundary, ...]:
    """Where every series in a universe would be cut.

    Computed up front so the manifest can record the cut for the whole run
    rather than one boundary per worker, and so a reader can see that every
    series was cut before any of them was screened.

    Args:
        frames: Series by ``(symbol, timeframe value)``.
        fraction: Share of bars to withhold.
        day_origin: Where the trading day starts.

    Returns:
        One boundary per series, in key order.
    """
    return tuple(
        screen_frame(frame, fraction=fraction, day_origin=day_origin)[1]
        for _key, frame in sorted(frames.items())
    )
