"""Bar-local view of history: what a strategy is allowed to see on bar ``t``.

:class:`BarContext` is the only thing a compiled entry condition is handed, and
it is shaped so that reaching bar ``t+1`` is not a rule but an absence. There is
no ``next()``, no ``bar_at(index)``, no ``frame``/``df`` property, no iterator,
no slice. Every accessor takes a single ``lookback`` argument counted *backwards*
from ``t``; forwards is not a value the API accepts, it is a direction the API
does not have.

That claim is only worth what tests make of it, so the suite proves it two ways:
by enumerating the public surface and asserting the accessor set and their
signatures, and by an equivalence property — a context built at index ``t`` over
the whole series must return, for every accessor and every argument, exactly what
a context built at ``t`` over a series *truncated* at ``t`` returns. A method
that could see the future would have to disagree.

:class:`BarSeries` is the immutable column store contexts view into. Columns are
materialised into Python lists once: a compiled condition does tens of scalar
lookups per bar, and ``Series.item()`` through polars costs about a microsecond
each, which alone would eat the per-bar budget.
"""

from collections.abc import Iterator, Mapping, Sequence
from datetime import datetime

from trading_system.core.exceptions import ValidationError
from trading_system.core.types import Timeframe
from trading_system.data.models import OHLCV_COLUMNS, TIMESTAMP_COLUMN, OHLCVFrame
from trading_system.entry.labels import LabelCategory, label_columns
from trading_system.features.pipeline import FeatureSet

#: Price fields a condition may reference, matching P04's ``price:<field>``.
PRICE_FIELDS: tuple[str, ...] = OHLCV_COLUMNS


class BarSeries:
    """Aligned price and feature columns for one symbol, indexed by bar.

    Treated as immutable: the columns are copied in at construction and no method
    hands the underlying lists back out.
    """

    __slots__ = (
        "_features",
        "_labels",
        "_length",
        "_prices",
        "_symbol",
        "_timeframe",
        "_timestamps",
    )

    def __init__(
        self,
        *,
        symbol: str,
        timeframe: Timeframe,
        timestamps: Sequence[datetime],
        prices: Mapping[str, Sequence[float]],
        features: Mapping[str, Sequence[float | None]],
        labels: Mapping[str, Sequence[frozenset[str] | None]] | None = None,
    ) -> None:
        """Bundle aligned columns into a series.

        Args:
            symbol: Instrument the bars belong to.
            timeframe: Bar size, used to derive each bar's close time.
            timestamps: Bar OPEN times, ascending, tz-aware UTC.
            prices: One column per entry of :data:`PRICE_FIELDS`.
            features: One column per feature key, nulls allowed for warmup.
            labels: One column per :class:`~trading_system.entry.labels.LabelCategory`,
                each holding the bar's label set or ``None`` where the bar cannot
                be classified. Omitted entirely for a strategy that asks for none.

        Raises:
            ValidationError: If a price column is missing or any column's length
                differs from ``timestamps``.
        """
        length = len(timestamps)
        missing = [field for field in PRICE_FIELDS if field not in prices]
        if missing:
            raise ValidationError(f"{symbol}: missing price columns {missing}")
        labels = labels or {}
        for name, column in (*prices.items(), *features.items(), *labels.items()):
            if len(column) != length:
                raise ValidationError(
                    f"{symbol}: column {name!r} has {len(column)} rows, expected {length}"
                )

        self._symbol = symbol
        self._timeframe = timeframe
        self._timestamps = list(timestamps)
        self._prices = {field: list(prices[field]) for field in PRICE_FIELDS}
        self._features = {key: list(column) for key, column in features.items()}
        self._labels = {key: list(column) for key, column in labels.items()}
        self._length = length

    @classmethod
    def from_frame(
        cls,
        frame: OHLCVFrame,
        features: FeatureSet | None = None,
        categories: Sequence["LabelCategory"] = (),
    ) -> "BarSeries":
        """Build a series from a validated frame and, optionally, its features.

        Args:
            frame: Bars to expose.
            features: Features computed over exactly these bars. ``None`` means
                the series carries prices only.
            categories: Label categories to classify the bars into. Empty means
                none, which is what a strategy that never asks about patterns or
                sessions should pay for.

        Returns:
            The series.

        Raises:
            ValidationError: If ``features`` was computed over different bars —
                a different symbol, timeframe, length, or timestamps. Silently
                accepting a misalignment here would offset every feature lookup
                by an unknown number of bars, which is lookahead in one
                direction and stale data in the other.
        """
        if features is not None:
            if features.symbol != frame.symbol:
                raise ValidationError(
                    f"features are for {features.symbol!r}, frame is {frame.symbol!r}"
                )
            if features.timeframe is not frame.timeframe:
                raise ValidationError(
                    f"features are {features.timeframe}, frame is {frame.timeframe}"
                )
            if len(features) != len(frame):
                raise ValidationError(
                    f"features cover {len(features)} bars, frame has {len(frame)}"
                )
            if not features.timestamps.equals(frame.timestamps):
                raise ValidationError("feature timestamps do not match the frame's")

        feature_columns: dict[str, Sequence[float | None]] = (
            {name: features.df[name].to_list() for name in features.feature_columns}
            if features is not None
            else {}
        )
        return cls(
            symbol=frame.symbol,
            timeframe=frame.timeframe,
            timestamps=frame.df[TIMESTAMP_COLUMN].to_list(),
            prices={field: frame.df[field].to_list() for field in PRICE_FIELDS},
            features=feature_columns,
            labels=label_columns(frame, categories),
        )

    def __len__(self) -> int:
        """Number of bars."""
        return self._length

    @property
    def symbol(self) -> str:
        """Instrument identifier."""
        return self._symbol

    @property
    def timeframe(self) -> Timeframe:
        """Bar size."""
        return self._timeframe

    @property
    def feature_keys(self) -> tuple[str, ...]:
        """Feature columns this series carries."""
        return tuple(self._features)

    @property
    def label_categories(self) -> tuple[str, ...]:
        """Label categories this series carries."""
        return tuple(self._labels)

    def truncated(self, length: int) -> "BarSeries":
        """A copy holding only the first ``length`` bars.

        Exists for the no-lookahead tests, which compare a context built over the
        full history against the same context built over history that stops at
        the bar under test.

        Args:
            length: Number of leading bars to keep.

        Returns:
            The shortened series.

        Raises:
            ValueError: If ``length`` is negative or exceeds the series.
        """
        if not 0 <= length <= self._length:
            raise ValueError(f"length must be in [0, {self._length}], got {length}")
        return BarSeries(
            symbol=self._symbol,
            timeframe=self._timeframe,
            timestamps=self._timestamps[:length],
            prices={field: column[:length] for field, column in self._prices.items()},
            features={key: column[:length] for key, column in self._features.items()},
            labels={key: column[:length] for key, column in self._labels.items()},
        )

    def context(self, index: int) -> "BarContext":
        """The view a strategy gets on bar ``index``.

        Args:
            index: Bar to position at, counted from the start of the series.

        Returns:
            A context restricted to bars ``<= index``.

        Raises:
            IndexError: If ``index`` is outside the series.
        """
        if not 0 <= index < self._length:
            raise IndexError(f"bar index {index} outside [0, {self._length})")
        return BarContext(self, index)

    def contexts(self) -> Iterator["BarContext"]:
        """Yield one context per bar, oldest first.

        Yields:
            A context per bar, in the order a live loop would see them.
        """
        for index in range(self._length):
            yield BarContext(self, index)

    def _open_time(self, index: int) -> datetime:
        """Open time of the bar at ``index``."""
        return self._timestamps[index]

    def _price_column(self, field: str) -> list[float]:
        """The named price column.

        Raises:
            ValidationError: If ``field`` is not a price field.
        """
        column = self._prices.get(field)
        if column is None:
            raise ValidationError(f"unknown price field {field!r}; known: {list(PRICE_FIELDS)}")
        return column

    def _feature_column(self, key: str) -> list[float | None]:
        """The named feature column.

        Raises:
            ValidationError: If ``key`` is not carried by this series.
        """
        column = self._features.get(key)
        if column is None:
            raise ValidationError(
                f"feature {key!r} is not in this series; carried: {sorted(self._features)}"
            )
        return column

    def _label_column(self, category: str) -> list[frozenset[str] | None]:
        """The named label column.

        Raises:
            ValidationError: If ``category`` is not carried by this series.
        """
        column = self._labels.get(category)
        if column is None:
            raise ValidationError(
                f"label category {category!r} is not in this series; "
                f"carried: {sorted(self._labels)}"
            )
        return column


class BarContext:
    """Everything visible on bar ``t``, and nothing else.

    Access is always backwards: ``lookback=0`` is bar ``t``, ``lookback=1`` is
    ``t-1``. A lookback that runs off the start of the series returns ``None``
    ("no such bar"), which the operator layer propagates as *unknown* rather than
    as ``False`` — see :mod:`trading_system.entry.operators`.

    A negative lookback raises: it is the one way to phrase a forward reference
    with the vocabulary this class does have, so it is rejected loudly rather
    than allowed to wrap around into the future the way a Python index would.
    """

    __slots__ = ("_index", "_series")

    def __init__(self, series: BarSeries, index: int) -> None:
        """Position a view at one bar of a series.

        Args:
            series: Columns to read from.
            index: Bar to position at.
        """
        self._series = series
        self._index = index

    def __repr__(self) -> str:
        """Compact description naming the symbol and bar."""
        closed = f"{self.bar_close_ts:%F %T}"
        return f"BarContext({self._series.symbol} bar={self._index} close={closed})"

    @property
    def index(self) -> int:
        """Position of bar ``t`` in the series."""
        return self._index

    @property
    def symbol(self) -> str:
        """Instrument the bars belong to."""
        return self._series.symbol

    @property
    def timeframe(self) -> Timeframe:
        """Bar size."""
        return self._series.timeframe

    @property
    def bar_close_ts(self) -> datetime:
        """Close time of bar ``t``, UTC.

        Derived as open time plus one timeframe, per the project convention that a
        bar's stored timestamp is its OPEN. This is the instant from which an
        order derived from this bar may be filled.
        """
        return self._series._open_time(self._index) + self._series.timeframe.duration

    def price(self, field: str, lookback: int = 0) -> float | None:
        """A price of the bar ``lookback`` bars before ``t``.

        Args:
            field: One of :data:`PRICE_FIELDS`.
            lookback: Bars back from ``t``. Zero is the current bar.

        Returns:
            The price, or ``None`` if that bar precedes the series.

        Raises:
            ValueError: If ``lookback`` is negative.
            ValidationError: If ``field`` is not a price field.
        """
        position = self._position(lookback)
        if position is None:
            return None
        return self._series._price_column(field)[position]

    def feature(self, key: str, lookback: int = 0) -> float | None:
        """A feature value of the bar ``lookback`` bars before ``t``.

        Args:
            key: Feature column name, as produced by
                :func:`~trading_system.entry.features.feature_key`.
            lookback: Bars back from ``t``. Zero is the current bar.

        Returns:
            The value, or ``None`` if that bar precedes the series or the
            indicator had not warmed up on it. The two are deliberately not
            distinguished: both mean "this condition cannot be decided here".

        Raises:
            ValueError: If ``lookback`` is negative.
            ValidationError: If ``key`` is not carried by this series.
        """
        column = self._series._feature_column(key)
        position = self._position(lookback)
        if position is None:
            return None
        return column[position]

    def labels(self, category: str, lookback: int = 0) -> frozenset[str] | None:
        """The categorical labels of the bar ``lookback`` bars before ``t``.

        Args:
            category: A :class:`~trading_system.entry.labels.LabelCategory`
                value, e.g. ``"pattern"``.
            lookback: Bars back from ``t``. Zero is the current bar.

        Returns:
            The label set, or ``None`` if that bar precedes the series or cannot
            be classified. An *empty* set is not ``None``: it means the bar was
            classified and carries none of that category's labels, which is a
            decidable answer, whereas ``None`` is not.

        Raises:
            ValueError: If ``lookback`` is negative.
            ValidationError: If ``category`` is not carried by this series.
        """
        column = self._series._label_column(category)
        position = self._position(lookback)
        if position is None:
            return None
        return column[position]

    def feature_snapshot(self) -> dict[str, float | None]:
        """Every feature's value on bar ``t``, for the signal's audit trail.

        Returns:
            Feature key to value, reading bar ``t`` only.
        """
        return {key: self.feature(key) for key in self._series.feature_keys}

    def _position(self, lookback: int) -> int | None:
        """Translate a lookback into a series position.

        Raises:
            ValueError: If ``lookback`` is negative.
        """
        if lookback < 0:
            raise ValueError(
                f"lookback must be non-negative, got {lookback}; bars after t are not "
                "reachable from a BarContext"
            )
        position = self._index - lookback
        return position if position >= 0 else None
