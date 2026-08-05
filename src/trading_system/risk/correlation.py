"""Measured co-movement between instruments, as of a stated instant.

A correlation matrix looks like a property of the market. It is not: it is a
function of *when you asked*, and treating it as a standing fact is the easiest
lookahead bug in this system to write and the hardest to notice, because nothing
about ``matrix[i][j]`` suggests a timestamp. Four things make it hard to write
here, and the fourth is the one that actually proves it:

1. **There is no matrix object obtainable without an instant.** The only entry
   point is :meth:`CorrelationProvider.matrix`, whose ``as_of`` has no default.
   A matrix cannot be built once and kept in an attribute "because the market is
   the same market".
2. **The cut-off is bar close**, the same expression
   :class:`~trading_system.risk.conversion.BarFxConverter` uses: a daily bar
   counts when ``timestamp + timeframe <= as_of``. The bar containing ``as_of``
   is still open and its close is not knowable.
3. **The cache is keyed by the trading day** ``as_of`` falls in, never held as a
   mutable "current matrix". A single refreshed attribute is precisely the leak:
   a decision on bar ``t`` would read a matrix recomputed at ``t + k``. Keyed,
   a later recomputation is simply a different entry.
4. **The equivalence test**, the same instrument P06 used to pin ``BarContext``:
   ``matrix(as_of=t)`` over the full history must equal ``matrix(as_of=t)`` over
   history truncated at ``t``. Any future bar reaching the window separates them.

**Returns are daily, and every instrument is bucketed by one shared day origin.**
On the signal timeframe, the correlation between NAS100 and GBPJPY mostly
measures non-overlapping trading hours: half of one instrument's bars fall while
the other is closed, and a sample correlation over asynchronous bars is biased
toward zero. Zero reads as "independent", which is the most *permissive* value
available — the defect would hand out permission to concentrate precisely where
there is no evidence. Bucketing every instrument with the same
:class:`~trading_system.data.resample.DayOrigin` matters more than giving each
one its native session: what is being measured is whether two things move
together, and that needs a common ruler.

**Clustering can only ever merge.** The configured manual groups are the floor —
``EURUSD``, ``GBPUSD`` and ``AUDUSD`` are one bet against the dollar whether or
not any history exists to prove it — and measured correlation may join clusters
further, never split them. That is what makes a short history degrade gracefully
instead of catastrophically: with no data the manual prior stands, and a missing
measurement is never more permissive than a present one.
"""

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from itertools import combinations

import polars as pl
from pydantic import BaseModel, ConfigDict, Field, model_validator

from trading_system.core.types import Timeframe, ensure_utc
from trading_system.data.models import TIMESTAMP_COLUMN, OHLCVFrame
from trading_system.data.resample import DayOrigin, trading_day

#: Correlation of an instrument with itself.
PERFECT = 1.0


class CorrelationConfig(BaseModel):
    """How co-movement is measured.

    Attributes:
        timeframe: Bar size returns are computed on. Daily by default — see the
            module docstring on why the signal timeframe is the wrong ruler.
        window: How many of the most recent returns to measure over.
        min_periods: Fewest *overlapping* observations a pair needs before its
            correlation is used at all. Counted per pair rather than against
            ``window``, because two instruments trading different calendars —
            crypto against FX — share fewer days than either has.
        threshold: Absolute correlation at or above which two instruments are
            merged into one cluster. Absolute, because a pair at −0.9 is also one
            bet, just with a leg inverted.
        day_origin: The single boundary every instrument's returns are bucketed
            by. One origin for all of them, deliberately.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    timeframe: Timeframe = Timeframe.D1
    window: int = Field(default=60, gt=1)
    min_periods: int = Field(default=40, gt=1)
    threshold: float = Field(default=0.7, gt=0.0, le=1.0)
    day_origin: DayOrigin = DayOrigin(tz="UTC")

    @model_validator(mode="after")
    def _minimum_is_reachable(self) -> "CorrelationConfig":
        """Reject a minimum that the window can never satisfy.

        Returns:
            The validated config.

        Raises:
            ValueError: If ``min_periods`` exceeds ``window``, which would make
                every correlation permanently unavailable — a configuration that
                silently disables the whole module.
        """
        if self.min_periods > self.window:
            raise ValueError(
                f"min_periods {self.min_periods} exceeds window {self.window}, so no pair "
                "could ever have enough history and correlation would never be used"
            )
        return self


@dataclass(frozen=True)
class CorrelationMatrix:
    """Pairwise correlations measured as of one instant.

    Attributes:
        as_of: The instant this was measured at. Carried on the value itself, so
            a matrix that is passed around still says when it is true of.
        values: ``(symbol_a, symbol_b)`` to correlation, both orderings present.
            A pair with too little overlapping history is **absent**, never zero:
            zero is a measurement meaning "independent", and using it for "not
            measured" would turn missing data into permission to concentrate.
        observations: ``(symbol_a, symbol_b)`` to how many overlapping returns
            the figure rests on, for pairs that have any.
    """

    as_of: datetime
    values: Mapping[tuple[str, str], float]
    observations: Mapping[tuple[str, str], int]

    def get(self, symbol_a: str, symbol_b: str) -> float | None:
        """Correlation between two instruments, or ``None`` if unmeasured.

        Args:
            symbol_a: First instrument.
            symbol_b: Second instrument.

        Returns:
            The correlation, ``1.0`` for an instrument against itself, or
            ``None`` when the pair lacked enough overlapping history.
        """
        if symbol_a == symbol_b:
            return PERFECT
        return self.values.get((symbol_a, symbol_b))

    @property
    def symbols(self) -> tuple[str, ...]:
        """Every symbol appearing in at least one measured pair, sorted."""
        return tuple(sorted({symbol for pair in self.values for symbol in pair}))


def daily_returns(
    frame: OHLCVFrame, *, as_of: datetime, config: CorrelationConfig
) -> Mapping[date, float]:
    """Log returns of ``frame``, bucketed by trading day, up to ``as_of``.

    Log rather than simple returns: they are additive across time and symmetric
    between a rise and the fall that undoes it, so a window straddling a large
    round trip is not biased by which came first.

    Args:
        frame: Bars for one instrument, at any timeframe finer than or equal to
            the configured one.
        as_of: Instant the measurement is made at. Bars that have not closed by
            then are excluded, which is where no-lookahead is enforced.
        config: Measurement settings.

    Returns:
        Trading day to that day's log return, oldest first. A day is present only
        when the day before it also has a close to measure against.

    Raises:
        ValueError: If ``as_of`` is naive.
    """
    moment = ensure_utc(as_of)
    if frame.is_empty:
        return {}

    # A bar's timestamp is its OPEN, so a bar has closed by `moment` exactly when
    # it opened by `moment - duration`. Identical rule to BarFxConverter.
    cutoff = moment - frame.timeframe.duration
    closed = frame.df.filter(pl.col(TIMESTAMP_COLUMN) <= cutoff)
    if closed.height < 2:
        return {}

    # Last close within each trading day, under the one shared origin.
    days = [trading_day(timestamp, config.day_origin) for timestamp in closed[TIMESTAMP_COLUMN]]
    closes = closed["close"].to_list()
    last_close_of_day: dict[date, float] = {}
    for day, close in zip(days, closes, strict=True):
        last_close_of_day[day] = close

    ordered = sorted(last_close_of_day)
    returns: dict[date, float] = {}
    for previous, current in zip(ordered, ordered[1:], strict=False):
        before, after = last_close_of_day[previous], last_close_of_day[current]
        if before > 0 and after > 0:
            returns[current] = math.log(after / before)
    return returns


def _pearson(paired: Sequence[tuple[float, float]]) -> float | None:
    """Pearson correlation of paired observations, or ``None`` if undefined.

    Args:
        paired: Observations as ``(x, y)`` pairs.

    Returns:
        The correlation, or ``None`` when either series has zero variance — a
        flat window carries no information about co-movement, and reporting zero
        would misrepresent that as evidence of independence.
    """
    count = len(paired)
    mean_x = sum(x for x, _ in paired) / count
    mean_y = sum(y for _, y in paired) / count
    covariance = sum((x - mean_x) * (y - mean_y) for x, y in paired)
    variance_x = sum((x - mean_x) ** 2 for x, _ in paired)
    variance_y = sum((y - mean_y) ** 2 for _, y in paired)
    if variance_x <= 0 or variance_y <= 0:
        return None
    return covariance / math.sqrt(variance_x * variance_y)


class CorrelationProvider:
    """Computes correlation matrices from the same bars the backtest runs on.

    Holding the frames rather than a store is deliberate: a provider that loaded
    its own data could load a different date range than the run, and the two
    would disagree in a way no test of either alone would catch.
    """

    __slots__ = ("_cache", "_config", "_series")

    def __init__(
        self, series: Mapping[str, OHLCVFrame], *, config: CorrelationConfig | None = None
    ) -> None:
        """Build a provider over loaded instrument series.

        Args:
            series: Symbol to its bars.
            config: Measurement settings; defaults are daily returns over 60 days.
        """
        self._series = dict(series)
        self._config = config if config is not None else CorrelationConfig()
        self._cache: dict[date, CorrelationMatrix] = {}

    def __repr__(self) -> str:
        """Compact description naming the universe and the window."""
        return f"CorrelationProvider({sorted(self._series)}, window={self._config.window})"

    @property
    def config(self) -> CorrelationConfig:
        """Measurement settings."""
        return self._config

    def matrix(self, *, as_of: datetime) -> CorrelationMatrix:
        """Correlations as they could have been measured at ``as_of``.

        Args:
            as_of: The instant of the decision this will inform. Required and
                without a default: a matrix with no instant attached is the
                lookahead bug this module exists to prevent.

        Returns:
            The matrix. Pairs with fewer than ``min_periods`` overlapping returns
            are absent from it rather than present as zero.

        Raises:
            ValueError: If ``as_of`` is naive.
        """
        moment = ensure_utc(as_of)
        # Keyed by trading day, never a mutable "current matrix" attribute: a
        # single refreshed slot would let a later bar's measurement be read by an
        # earlier decision, which is exactly the leak being guarded against.
        key = trading_day(moment, self._config.day_origin)
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        computed = self._compute(moment)
        self._cache[key] = computed
        return computed

    def _compute(self, as_of: datetime) -> CorrelationMatrix:
        """Measure every pair over the trailing window.

        Args:
            as_of: Instant of measurement, already normalised to UTC.

        Returns:
            The matrix.
        """
        returns = {
            symbol: daily_returns(frame, as_of=as_of, config=self._config)
            for symbol, frame in self._series.items()
        }
        values: dict[tuple[str, str], float] = {}
        observations: dict[tuple[str, str], int] = {}

        for symbol_a, symbol_b in combinations(sorted(returns), 2):
            days_a, days_b = returns[symbol_a], returns[symbol_b]
            # Pairwise intersection, then the trailing window of it: crypto and
            # FX do not share a calendar, so a window counted on either series
            # alone would silently compare different spans.
            shared = sorted(set(days_a) & set(days_b))[-self._config.window :]
            if len(shared) < self._config.min_periods:
                continue
            correlation = _pearson([(days_a[day], days_b[day]) for day in shared])
            if correlation is None:
                continue
            values[(symbol_a, symbol_b)] = correlation
            values[(symbol_b, symbol_a)] = correlation
            observations[(symbol_a, symbol_b)] = len(shared)
            observations[(symbol_b, symbol_a)] = len(shared)

        return CorrelationMatrix(as_of=as_of, values=values, observations=observations)
