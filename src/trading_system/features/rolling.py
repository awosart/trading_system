"""Incremental primitives the streaming indicators are assembled from.

Two deliberate choices govern this module.

**Window aggregates are recomputed, not accumulated.** A running sum updated by
``total += new - evicted`` is O(1) but its rounding error random-walks with the
number of bars processed; on a million-bar volume series that error passes the
``1e-9`` parity tolerance and the incremental path starts disagreeing with the
vectorised one. polars evaluates each window independently, so this module does
too. The cost is O(period) per bar with ``period`` a small constant — still O(1)
in the length of the history, which is what a live process cares about.

**Extremes are the exception.** A monotonic deque gives rolling max and min in
O(1) amortised time *exactly*, with no rounding involved at all, so Donchian
channels and Stochastic ranges use one.
"""

import math
from collections import deque
from collections.abc import Sequence


class RollingWindow:
    """The most recent ``size`` values, with the aggregates indicators need."""

    __slots__ = ("_size", "_values")

    def __init__(self, size: int) -> None:
        """Create an empty window.

        Args:
            size: Number of values retained.

        Raises:
            ValueError: If ``size`` is not positive.
        """
        if size < 1:
            raise ValueError(f"window size must be positive, got {size}")
        self._size = size
        self._values: deque[float] = deque(maxlen=size)

    @property
    def size(self) -> int:
        """Capacity of the window."""
        return self._size

    @property
    def full(self) -> bool:
        """Whether the window holds ``size`` values."""
        return len(self._values) == self._size

    @property
    def values(self) -> tuple[float, ...]:
        """The retained values, oldest first."""
        return tuple(self._values)

    @property
    def oldest(self) -> float:
        """The value that will be evicted next.

        Raises:
            IndexError: If the window is empty.
        """
        return self._values[0]

    @property
    def newest(self) -> float:
        """The most recently pushed value.

        Raises:
            IndexError: If the window is empty.
        """
        return self._values[-1]

    def push(self, value: float) -> float | None:
        """Append a value, evicting the oldest once the window is full.

        Args:
            value: Value to append.

        Returns:
            The evicted value, or ``None`` if nothing was evicted.
        """
        evicted = self._values[0] if self.full else None
        self._values.append(value)
        return evicted

    def clear(self) -> None:
        """Drop every retained value."""
        self._values.clear()

    def sum(self) -> float:
        """Sum of the retained values."""
        return sum(self._values)

    def mean(self) -> float:
        """Arithmetic mean of the retained values.

        Raises:
            ZeroDivisionError: If the window is empty.
        """
        return sum(self._values) / len(self._values)

    def std(self, ddof: int = 0) -> float:
        """Standard deviation of the retained values, two-pass.

        Args:
            ddof: Delta degrees of freedom. ``0`` matches the population
                convention used by Bollinger Bands and by polars' default here.

        Returns:
            The standard deviation.
        """
        count = len(self._values)
        mean = sum(self._values) / count
        variance = sum((value - mean) ** 2 for value in self._values) / (count - ddof)
        return math.sqrt(variance)

    def mean_absolute_deviation(self) -> float:
        """Mean absolute deviation about the window mean, as CCI defines it."""
        count = len(self._values)
        mean = sum(self._values) / count
        return sum(abs(value - mean) for value in self._values) / count

    def weighted_mean(self, weights: Sequence[float]) -> float:
        """Weighted mean, ``weights[-1]`` applying to the newest value.

        Args:
            weights: One weight per retained value, oldest first.

        Returns:
            ``sum(w * x) / sum(w)``.

        Raises:
            ValueError: If the number of weights does not match the contents.
        """
        if len(weights) != len(self._values):
            raise ValueError(f"expected {len(self._values)} weights, got {len(weights)}")
        total = 0.0
        for weight, value in zip(weights, self._values, strict=True):
            total += weight * value
        return total / sum(weights)

    def __len__(self) -> int:
        """Number of values currently retained."""
        return len(self._values)


class RollingExtreme:
    """Rolling maximum or minimum in O(1) amortised time.

    Holds a deque of ``(index, value)`` pairs kept monotonic, so the extreme of
    the current window is always at the front. Each value is pushed and popped
    at most once.
    """

    __slots__ = ("_index", "_largest", "_pairs", "_size")

    def __init__(self, size: int, *, largest: bool) -> None:
        """Create an empty rolling extreme.

        Args:
            size: Window length.
            largest: ``True`` tracks the maximum, ``False`` the minimum.

        Raises:
            ValueError: If ``size`` is not positive.
        """
        if size < 1:
            raise ValueError(f"window size must be positive, got {size}")
        self._size = size
        self._largest = largest
        self._pairs: deque[tuple[int, float]] = deque()
        self._index = 0

    @property
    def full(self) -> bool:
        """Whether ``size`` values have been pushed."""
        return self._index >= self._size

    def push(self, value: float) -> float | None:
        """Append a value and report the window's extreme.

        Args:
            value: Value to append.

        Returns:
            The extreme over the last ``size`` values, or ``None`` until that
            many have been seen.
        """
        while self._pairs and (
            self._pairs[-1][1] <= value if self._largest else self._pairs[-1][1] >= value
        ):
            self._pairs.pop()
        self._pairs.append((self._index, value))
        oldest_allowed = self._index - self._size + 1
        while self._pairs[0][0] < oldest_allowed:
            self._pairs.popleft()
        self._index += 1
        return self._pairs[0][1] if self.full else None

    def clear(self) -> None:
        """Forget every value seen so far."""
        self._pairs.clear()
        self._index = 0


class SeededEma:
    """Exponential moving average seeded with the SMA of its first inputs.

    Seeding matters for parity. polars' ``ewm_mean(alpha, adjust=False)`` seeds
    on the first non-null input and then applies ``y += alpha * (x - y)``; the
    vectorised helpers feed it a series whose seed slot carries the SMA, and
    this class reproduces both halves exactly — the recurrence below is the same
    floating-point expression polars evaluates, not an algebraic rearrangement
    of it, which would differ in the last bit.

    Wilder's smoothing (RSI, ATR, ADX) is this with ``alpha = 1 / period``.
    """

    __slots__ = ("_alpha", "_period", "_seed", "_value")

    def __init__(self, period: int, alpha: float | None = None) -> None:
        """Create an EMA that warms up over ``period`` values.

        Args:
            period: Number of values averaged into the seed.
            alpha: Smoothing factor. Defaults to ``2 / (period + 1)``; pass
                ``1 / period`` for Wilder's smoothing.

        Raises:
            ValueError: If ``period`` is not positive or ``alpha`` is outside
                ``(0, 1]``.
        """
        if period < 1:
            raise ValueError(f"period must be positive, got {period}")
        resolved = 2.0 / (period + 1) if alpha is None else alpha
        if not 0.0 < resolved <= 1.0:
            raise ValueError(f"alpha must be in (0, 1], got {resolved}")
        self._period = period
        self._alpha = resolved
        self._seed: list[float] = []
        self._value: float | None = None

    @property
    def period(self) -> int:
        """Number of values averaged into the seed."""
        return self._period

    @property
    def alpha(self) -> float:
        """Smoothing factor."""
        return self._alpha

    @property
    def value(self) -> float | None:
        """Current average, or ``None`` while seeding."""
        return self._value

    def push(self, value: float) -> float | None:
        """Fold one value into the average.

        Args:
            value: Next input.

        Returns:
            The updated average, or ``None`` until ``period`` values have been
            seen.
        """
        if self._value is None:
            self._seed.append(value)
            if len(self._seed) < self._period:
                return None
            self._value = sum(self._seed) / self._period
            self._seed.clear()
            return self._value
        self._value += self._alpha * (value - self._value)
        return self._value

    def clear(self) -> None:
        """Return to the pre-seed state."""
        self._seed.clear()
        self._value = None


def wilder_alpha(period: int) -> float:
    """Smoothing factor for Wilder's moving average.

    Args:
        period: Averaging period.

    Returns:
        ``1 / period``.
    """
    return 1.0 / period


def linear_weights(period: int) -> tuple[float, ...]:
    """Weights for a linearly weighted moving average, oldest first.

    Args:
        period: Averaging period.

    Returns:
        ``(1.0, 2.0, ..., period)``.
    """
    return tuple(float(i + 1) for i in range(period))
