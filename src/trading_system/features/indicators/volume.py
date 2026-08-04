"""Volume indicators: OBV, session and anchored VWAP, volume MA, relative volume.

Both VWAPs accumulate rather than roll, which makes their session boundary the
interesting part. A session is defined by a wall-clock anchor in a real
timezone, reusing :class:`~trading_system.data.resample.DayOrigin`, so an FX day
anchored to 17:00 New York keeps rolling at 17:00 local across DST instead of
drifting an hour twice a year. Bucketing is done on local wall-clock time for
exactly that reason.

Zero-volume bars are real — FX feeds quoting tick volume produce them in thin
sessions — so a session whose volume is entirely zero falls back to the
unweighted mean of the typical price rather than dividing by zero.
"""

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import ClassVar
from zoneinfo import ZoneInfo

import polars as pl

from trading_system.data.models import TIMESTAMP_COLUMN
from trading_system.data.resample import FX_DAY_ORIGIN, DayOrigin
from trading_system.features.base import BaseStreaming, ScalarIndicator
from trading_system.features.expressions import (
    safe_divide,
    source_expression,
    typical_price,
)
from trading_system.features.rolling import RollingWindow


def _anchor_offset(origin: DayOrigin) -> timedelta:
    """Distance from local midnight to the session anchor."""
    return timedelta(hours=origin.at.hour, minutes=origin.at.minute, seconds=origin.at.second)


def _session_key(origin: DayOrigin) -> pl.Expr:
    """Expression bucketing bars into trading days under ``origin``.

    Converting to the origin's zone and *then* dropping the offset makes the
    arithmetic wall-clock, so a 23- or 25-hour DST day still maps to one bucket.
    """
    local = pl.col(TIMESTAMP_COLUMN).dt.convert_time_zone(origin.tz).dt.replace_time_zone(None)
    return (local - _anchor_offset(origin)).dt.date()


def _session_of(timestamp: datetime, origin: DayOrigin, offset: timedelta) -> object:
    """Trading day a timestamp falls in, matching :func:`_session_key` exactly."""
    local = timestamp.astimezone(ZoneInfo(origin.tz)).replace(tzinfo=None)
    return (local - offset).date()


@dataclass(frozen=True)
class OBV(ScalarIndicator):
    """On-balance volume.

    Volume is added on an up close and subtracted on a down close; an unchanged
    close contributes nothing. The running total starts at zero on the second
    bar — the first has no previous close to compare against, so it carries no
    value at all rather than an arbitrary zero.
    """

    @property
    def name(self) -> str:
        """Identifier."""
        return "obv"

    @property
    def warmup(self) -> int:
        """The first bar has no previous close to take a direction from."""
        return 1

    def _expression(self) -> pl.Expr:
        """Running sum of signed volume."""
        change = pl.col("close").diff()
        direction = pl.when(change > 0).then(1.0).when(change < 0).then(-1.0).otherwise(0.0)
        signed = pl.when(change.is_null()).then(0.0).otherwise(direction * pl.col("volume"))
        return signed.cum_sum()

    def streaming(self) -> "OBVStream":
        """Build the incremental counterpart."""
        return OBVStream(self)


class OBVStream(BaseStreaming[OBV, float]):
    """Incremental :class:`OBV`."""

    def reset(self) -> None:
        """Zero the running total and forget the previous close."""
        self._total = 0.0
        self._previous_close: float | None = None

    def step(
        self,
        _timestamp: datetime,
        _open_price: float,
        _high: float,
        _low: float,
        close: float,
        volume: float,
        /,
    ) -> float | None:
        """Add or subtract the bar's volume according to its close direction."""
        previous_close = self._previous_close
        self._previous_close = close
        if previous_close is None:
            return None
        if close > previous_close:
            self._total += volume
        elif close < previous_close:
            self._total -= volume
        return self._total


@dataclass(frozen=True)
class SessionVwap(ScalarIndicator):
    """Volume-weighted average price, reset at each session boundary.

    Attributes:
        origin: Where the trading day starts. Defaults to the FX convention of
            17:00 New York.
    """

    requires_timestamps: ClassVar[bool] = True

    origin: DayOrigin = FX_DAY_ORIGIN

    @property
    def name(self) -> str:
        """Identifier, e.g. ``vwap_session_America/New_York_17:00:00``."""
        return f"vwap_session_{self.origin.tz}_{self.origin.at}"

    @property
    def warmup(self) -> int:
        """None: the first bar of a session is already a valid one-bar average."""
        return 0

    def _expression(self) -> pl.Expr:
        """Cumulative volume-weighted typical price within each session."""
        key = _session_key(self.origin)
        typical = source_expression("hlc3")
        flow = (typical * pl.col("volume")).cum_sum().over(key)
        weight = pl.col("volume").cum_sum().over(key)
        bars = typical.cum_count().over(key).cast(pl.Float64)
        return (
            pl.when(weight == 0).then(typical.cum_sum().over(key) / bars).otherwise(flow / weight)
        )

    def streaming(self) -> "SessionVwapStream":
        """Build the incremental counterpart."""
        return SessionVwapStream(self)


class SessionVwapStream(BaseStreaming[SessionVwap, float]):
    """Incremental :class:`SessionVwap`."""

    def reset(self) -> None:
        """Drop the accumulators and the current session key."""
        self._origin = self._indicator.origin
        self._offset = _anchor_offset(self._origin)
        self._session: object | None = None
        self._flow = 0.0
        self._weight = 0.0
        self._typical_total = 0.0
        self._bars = 0

    def step(
        self,
        timestamp: datetime,
        _open_price: float,
        high: float,
        low: float,
        close: float,
        volume: float,
        /,
    ) -> float | None:
        """Accumulate within the session, restarting when the trading day rolls."""
        session = _session_of(timestamp, self._origin, self._offset)
        if session != self._session:
            self._session = session
            self._flow = 0.0
            self._weight = 0.0
            self._typical_total = 0.0
            self._bars = 0

        typical = typical_price(high, low, close)
        self._flow += typical * volume
        self._weight += volume
        self._typical_total += typical
        self._bars += 1
        if self._weight == 0:
            return self._typical_total / self._bars
        return self._flow / self._weight


@dataclass(frozen=True)
class AnchoredVwap(ScalarIndicator):
    """Volume-weighted average price accumulated from a fixed instant.

    Unlike every other indicator here, validity is a function of the data rather
    than of a bar count: bars before the anchor carry no value, and ``warmup``
    reports 0 because the anchor may fall anywhere in the frame — or outside it.
    Callers that need "is this row usable" must test for null, which is the
    check the pipeline applies anyway.

    Attributes:
        anchor: Instant to accumulate from, tz-aware. ``None`` anchors to the
            first bar of the frame, which is the form the registry builds.
    """

    requires_timestamps: ClassVar[bool] = True

    anchor: datetime | None = None

    def __post_init__(self) -> None:
        """Reject a naive anchor, which would have no defined instant."""
        if self.anchor is not None and self.anchor.tzinfo is None:
            raise ValueError("anchor must be tz-aware")

    @property
    def name(self) -> str:
        """Identifier, e.g. ``avwap_20240102T000000Z``."""
        if self.anchor is None:
            return "avwap_start"
        return f"avwap_{self.anchor.astimezone(UTC):%Y%m%dT%H%M%SZ}"

    @property
    def warmup(self) -> int:
        """None; pre-anchor bars are null rather than part of a fixed warmup."""
        return 0

    def _expression(self) -> pl.Expr:
        """Cumulative volume-weighted typical price from the anchor onwards."""
        typical = source_expression("hlc3")
        if self.anchor is None:
            # pl.lit(True) would be a *scalar* that broadcasts, so the running
            # bar count below would stick at 1 instead of counting rows.
            active = pl.repeat(value=True, n=pl.len(), dtype=pl.Boolean)
        else:
            active = pl.col(TIMESTAMP_COLUMN) >= self.anchor
        flow = pl.when(active).then(typical * pl.col("volume")).otherwise(0.0).cum_sum()
        weight = pl.when(active).then(pl.col("volume")).otherwise(0.0).cum_sum()
        typical_total = pl.when(active).then(typical).otherwise(0.0).cum_sum()
        bars = active.cast(pl.Float64).cum_sum()
        unweighted = safe_divide(typical_total, bars, 0.0)
        return (
            pl.when(~active).then(None).when(weight == 0).then(unweighted).otherwise(flow / weight)
        )

    def streaming(self) -> "AnchoredVwapStream":
        """Build the incremental counterpart."""
        return AnchoredVwapStream(self)


class AnchoredVwapStream(BaseStreaming[AnchoredVwap, float]):
    """Incremental :class:`AnchoredVwap`."""

    def reset(self) -> None:
        """Drop the accumulators."""
        self._anchor = self._indicator.anchor
        self._flow = 0.0
        self._weight = 0.0
        self._typical_total = 0.0
        self._bars = 0

    def step(
        self,
        timestamp: datetime,
        _open_price: float,
        high: float,
        low: float,
        close: float,
        volume: float,
        /,
    ) -> float | None:
        """Accumulate once the anchor has been reached."""
        if self._anchor is not None and timestamp < self._anchor:
            return None
        typical = typical_price(high, low, close)
        self._flow += typical * volume
        self._weight += volume
        self._typical_total += typical
        self._bars += 1
        if self._weight == 0:
            return self._typical_total / self._bars
        return self._flow / self._weight


@dataclass(frozen=True)
class VolumeMA(ScalarIndicator):
    """Simple moving average of volume.

    Attributes:
        period: Number of bars averaged.
    """

    period: int = 20

    def __post_init__(self) -> None:
        """Reject non-positive periods."""
        if self.period < 1:
            raise ValueError(f"period must be positive, got {self.period}")

    @property
    def name(self) -> str:
        """Identifier, e.g. ``volume_ma_20``."""
        return f"volume_ma_{self.period}"

    @property
    def warmup(self) -> int:
        """Bars consumed before the window is full."""
        return self.period - 1

    def _expression(self) -> pl.Expr:
        """Rolling mean of volume."""
        return pl.col("volume").rolling_mean(self.period)

    def streaming(self) -> "VolumeMAStream":
        """Build the incremental counterpart."""
        return VolumeMAStream(self)


class VolumeMAStream(BaseStreaming[VolumeMA, float]):
    """Incremental :class:`VolumeMA`."""

    def reset(self) -> None:
        """Empty the window."""
        self._window = RollingWindow(self._indicator.period)

    def step(
        self,
        _timestamp: datetime,
        _open_price: float,
        _high: float,
        _low: float,
        _close: float,
        volume: float,
        /,
    ) -> float | None:
        """Push one bar's volume and return the mean once full."""
        self._window.push(volume)
        return self._window.mean() if self._window.full else None


@dataclass(frozen=True)
class RelativeVolume(ScalarIndicator):
    """Volume as a multiple of the *preceding* ``period`` bars' average.

    The current bar is excluded from its own baseline; including it would drag
    the reference towards the value being measured and mute exactly the spikes
    the indicator exists to find. A baseline of zero carries no information
    about what is normal, so it reports 1.0 — "average" — rather than an
    infinity.

    Attributes:
        period: Baseline length.
    """

    period: int = 20

    def __post_init__(self) -> None:
        """Reject non-positive periods."""
        if self.period < 1:
            raise ValueError(f"period must be positive, got {self.period}")

    @property
    def name(self) -> str:
        """Identifier, e.g. ``rvol_20``."""
        return f"rvol_{self.period}"

    @property
    def warmup(self) -> int:
        """A full baseline must sit strictly before the measured bar."""
        return self.period

    def _expression(self) -> pl.Expr:
        """Volume over the mean of the previous ``period`` volumes."""
        baseline = pl.col("volume").shift(1).rolling_mean(self.period)
        return safe_divide(pl.col("volume"), baseline, 1.0)

    def streaming(self) -> "RelativeVolumeStream":
        """Build the incremental counterpart."""
        return RelativeVolumeStream(self)


class RelativeVolumeStream(BaseStreaming[RelativeVolume, float]):
    """Incremental :class:`RelativeVolume`."""

    def reset(self) -> None:
        """Empty the baseline window."""
        self._window = RollingWindow(self._indicator.period)

    def step(
        self,
        _timestamp: datetime,
        _open_price: float,
        _high: float,
        _low: float,
        _close: float,
        volume: float,
        /,
    ) -> float | None:
        """Compare this bar against the baseline, then fold it in."""
        ready = self._window.full
        baseline = self._window.mean() if ready else 0.0
        self._window.push(volume)
        if not ready:
            return None
        if baseline == 0:
            return 1.0
        return volume / baseline
