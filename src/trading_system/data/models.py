"""The OHLCV frame: market data that has already been proven well-formed.

An :class:`OHLCVFrame` guarantees *structural* correctness only — schema, dtypes,
tz-aware UTC ordering, no duplicates, no nulls. It deliberately says nothing
about whether the data is *plausible*: a bar whose close sits outside its own
high/low range is structurally valid and will be constructed without complaint.
Semantic defects are the job of :mod:`trading_system.data.quality`, which grades
them by severity instead of hard-failing, because real vendor data always has
some and a loader that refuses to load is useless.

The timestamp column holds the bar's OPEN time, per the project-wide convention.
"""

from datetime import datetime, timedelta

import polars as pl

from trading_system.core.exceptions import DataError
from trading_system.core.types import Timeframe

TIMESTAMP_COLUMN = "timestamp"
OHLCV_COLUMNS = ("open", "high", "low", "close", "volume")
REQUIRED_COLUMNS = (TIMESTAMP_COLUMN, *OHLCV_COLUMNS)

TIMESTAMP_DTYPE = pl.Datetime(time_unit="us", time_zone="UTC")


def empty_ohlcv_dataframe() -> pl.DataFrame:
    """Build a zero-row DataFrame carrying the canonical OHLCV schema.

    Returns:
        An empty DataFrame that satisfies every :class:`OHLCVFrame` invariant.
    """
    return pl.DataFrame(
        schema={
            TIMESTAMP_COLUMN: TIMESTAMP_DTYPE,
            **dict.fromkeys(OHLCV_COLUMNS, pl.Float64),
        }
    )


class OHLCVFrame:
    """A validated table of OHLCV bars for one symbol and timeframe.

    Instances are treated as immutable. Every method that narrows or transforms
    the data returns a new frame; the wrapped DataFrame must never be mutated in
    place by callers.
    """

    __slots__ = ("_df", "_symbol", "_timeframe")

    def __init__(self, df: pl.DataFrame, symbol: str, timeframe: Timeframe) -> None:
        """Validate ``df`` and wrap it.

        Args:
            df: Bars to wrap. Must already satisfy every invariant; use
                :meth:`from_raw` to normalise untrusted input first.
            symbol: Instrument identifier the bars belong to.
            timeframe: Bar size the data is sampled at.

        Raises:
            DataError: If the schema, dtypes, ordering, or nullability of ``df``
                violates the frame's guarantees.
            ValueError: If ``symbol`` is empty.
        """
        if not symbol:
            raise ValueError("symbol must be a non-empty string")
        _validate(df, symbol=symbol)
        self._df = df
        self._symbol = symbol
        self._timeframe = timeframe

    @classmethod
    def from_raw(
        cls,
        df: pl.DataFrame,
        symbol: str,
        timeframe: Timeframe,
        *,
        assume_tz: str | None = None,
    ) -> "OHLCVFrame":
        """Normalise untrusted bars into a valid frame.

        Casts the OHLCV columns to ``Float64``, moves timestamps to UTC, sorts by
        time and drops duplicate timestamps keeping the last occurrence — vendors
        commonly re-send a bar with corrected values, and the later copy wins.

        Args:
            df: Bars to normalise. Must contain the required column names.
            symbol: Instrument identifier.
            timeframe: Bar size.
            assume_tz: IANA zone used to localise naive timestamps, e.g. a broker
                export recorded in ``"Europe/Riga"``. Naive input without this
                argument is rejected rather than silently assumed to be UTC.

        Returns:
            A validated frame.

        Raises:
            DataError: If required columns are missing, or timestamps are naive
                and ``assume_tz`` was not supplied.
        """
        missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
        if missing:
            raise DataError(f"{symbol}: missing required columns {missing}")

        timestamp = pl.col(TIMESTAMP_COLUMN)
        dtype = df.schema[TIMESTAMP_COLUMN]
        if not isinstance(dtype, pl.Datetime):
            raise DataError(
                f"{symbol}: '{TIMESTAMP_COLUMN}' must be a Datetime column, got {dtype}"
            )
        if dtype.time_zone is None:
            if assume_tz is None:
                raise DataError(
                    f"{symbol}: timestamps are tz-naive; pass assume_tz to state which "
                    "zone they were recorded in"
                )
            timestamp = timestamp.dt.replace_time_zone(assume_tz)
        timestamp = timestamp.dt.convert_time_zone("UTC").dt.cast_time_unit("us")

        normalised = (
            df.with_columns(
                timestamp.alias(TIMESTAMP_COLUMN),
                *[pl.col(column).cast(pl.Float64) for column in OHLCV_COLUMNS],
            )
            .select(REQUIRED_COLUMNS)
            .sort(TIMESTAMP_COLUMN)
            .unique(subset=[TIMESTAMP_COLUMN], keep="last", maintain_order=True)
        )
        return cls(normalised, symbol, timeframe)

    @classmethod
    def empty(cls, symbol: str, timeframe: Timeframe) -> "OHLCVFrame":
        """Build an empty frame for ``symbol`` and ``timeframe``."""
        return cls(empty_ohlcv_dataframe(), symbol, timeframe)

    @property
    def df(self) -> pl.DataFrame:
        """The wrapped DataFrame. Treat as read-only."""
        return self._df

    @property
    def symbol(self) -> str:
        """Instrument identifier."""
        return self._symbol

    @property
    def timeframe(self) -> Timeframe:
        """Bar size of the contained data."""
        return self._timeframe

    @property
    def timestamps(self) -> pl.Series:
        """The timestamp column, ascending."""
        return self._df[TIMESTAMP_COLUMN]

    @property
    def start(self) -> datetime | None:
        """Open time of the first bar, or ``None`` when empty."""
        return None if self.is_empty else self.timestamps.item(0)

    @property
    def end(self) -> datetime | None:
        """Open time of the last bar, or ``None`` when empty."""
        return None if self.is_empty else self.timestamps.item(-1)

    @property
    def is_empty(self) -> bool:
        """Whether the frame holds no bars."""
        return self._df.height == 0

    def __len__(self) -> int:
        """Number of bars."""
        return self._df.height

    def __repr__(self) -> str:
        """Compact description including the covered range."""
        if self.is_empty:
            span = "empty"
        else:
            span = f"{self.start:%Y-%m-%d %H:%M}..{self.end:%Y-%m-%d %H:%M}"
        return f"OHLCVFrame({self._symbol} {self._timeframe} n={len(self)} {span})"

    def with_df(self, df: pl.DataFrame) -> "OHLCVFrame":
        """Return a frame carrying the same identity but different rows.

        Args:
            df: Replacement bars, subject to the usual validation.

        Returns:
            A new frame with this frame's symbol and timeframe.
        """
        return OHLCVFrame(df, self._symbol, self._timeframe)

    def slice(self, start: datetime | None = None, end: datetime | None = None) -> "OHLCVFrame":
        """Restrict the frame to a half-open time range ``[start, end)``.

        The range is half-open so that adjacent slices tile without repeating a
        bar at the boundary.

        Args:
            start: Inclusive lower bound on bar open time. ``None`` means no
                lower bound.
            end: Exclusive upper bound on bar open time. ``None`` means no upper
                bound.

        Returns:
            A new frame holding the bars inside the range.

        Raises:
            ValueError: If either bound is tz-naive, or ``start`` is after ``end``.
        """
        if start is not None and start.tzinfo is None:
            raise ValueError("slice start must be tz-aware")
        if end is not None and end.tzinfo is None:
            raise ValueError("slice end must be tz-aware")
        if start is not None and end is not None and start > end:
            raise ValueError(f"slice start {start} is after end {end}")

        predicates = []
        if start is not None:
            predicates.append(pl.col(TIMESTAMP_COLUMN) >= start)
        if end is not None:
            predicates.append(pl.col(TIMESTAMP_COLUMN) < end)
        if not predicates:
            return self
        filtered = self._df
        for predicate in predicates:
            filtered = filtered.filter(predicate)
        return self.with_df(filtered)

    def last(self, n: int) -> "OHLCVFrame":
        """Return the most recent ``n`` bars.

        Args:
            n: Number of bars to keep. Larger than the frame yields everything.

        Returns:
            A new frame with at most ``n`` bars.

        Raises:
            ValueError: If ``n`` is negative.
        """
        if n < 0:
            raise ValueError(f"n must be non-negative, got {n}")
        return self.with_df(self._df.tail(n))

    def equals(self, other: "OHLCVFrame") -> bool:
        """Whether ``other`` has the same identity and rows."""
        return (
            self._symbol == other._symbol
            and self._timeframe is other._timeframe
            and self._df.equals(other._df)
        )


def _validate(df: pl.DataFrame, *, symbol: str) -> None:
    """Assert every structural invariant, raising :class:`DataError` on the first breach."""
    if tuple(df.columns) != REQUIRED_COLUMNS:
        raise DataError(f"{symbol}: expected columns {list(REQUIRED_COLUMNS)}, got {df.columns}")

    timestamp_dtype = df.schema[TIMESTAMP_COLUMN]
    if timestamp_dtype != TIMESTAMP_DTYPE:
        raise DataError(
            f"{symbol}: '{TIMESTAMP_COLUMN}' must be {TIMESTAMP_DTYPE}, got {timestamp_dtype}"
        )

    for column in OHLCV_COLUMNS:
        dtype = df.schema[column]
        if dtype != pl.Float64:
            raise DataError(f"{symbol}: column '{column}' must be Float64, got {dtype}")

    null_counts = df.null_count().row(0)
    nulls = {column: count for column, count in zip(df.columns, null_counts, strict=True) if count}
    if nulls:
        raise DataError(f"{symbol}: null values present in {nulls}")

    if df.height > 1:
        deltas = df[TIMESTAMP_COLUMN].diff().drop_nulls()
        non_increasing = int((deltas <= timedelta(0)).sum())
        if non_increasing:
            raise DataError(
                f"{symbol}: timestamps must be strictly increasing; found "
                f"{non_increasing} non-increasing step(s). Use OHLCVFrame.from_raw to "
                "sort and de-duplicate untrusted input."
            )
