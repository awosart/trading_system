"""Polars building blocks shared by the vectorised indicator implementations.

Each helper here has a counterpart in :mod:`trading_system.features.rolling` or
in an indicator's state machine, and the pair is what the parity tests hold to
``1e-9``. Where a formula could be written several algebraically equivalent
ways, the form that matches the incremental path bit-for-bit is the one used.
"""

from collections.abc import Sequence

import polars as pl

#: Price series an indicator may be computed over. The synthetic ones are the
#: usual averages: ``hl2`` the bar midpoint, ``hlc3`` the typical price, and
#: ``ohlc4`` the full average.
PRICE_SOURCES: tuple[str, ...] = (
    "open",
    "high",
    "low",
    "close",
    "volume",
    "hl2",
    "hlc3",
    "ohlc4",
)


#: polars lowers a division by a literal into a multiplication by its reciprocal.
#: For a power of two that is exact, but ``x / 3`` and ``x * (1/3)`` disagree in
#: the last bit for roughly a third of all inputs. One ULP is normally beneath
#: notice — except where an indicator makes a *strict comparison* between
#: consecutive derived values, which is exactly how MFI decides whether money
#: flowed in or out. Mirroring the reciprocal here keeps the two evaluation paths
#: bit-identical, and the parity tests fail loudly if polars ever stops doing it.
_ONE_THIRD = 1.0 / 3.0


def typical_price(high: float, low: float, close: float) -> float:
    """Mean of a bar's high, low and close, rounded as polars rounds it.

    Args:
        high: Bar high.
        low: Bar low.
        close: Bar close.

    Returns:
        The typical price.
    """
    return (high + low + close) * _ONE_THIRD


def validate_source(source: str) -> str:
    """Check that ``source`` names a known price series.

    Args:
        source: Candidate source name.

    Returns:
        The source unchanged.

    Raises:
        ValueError: If the name is not one of :data:`PRICE_SOURCES`.
    """
    if source not in PRICE_SOURCES:
        raise ValueError(f"unknown price source {source!r}; expected one of {list(PRICE_SOURCES)}")
    return source


def source_suffix(source: str) -> str:
    """Name fragment distinguishing a non-default price source.

    Args:
        source: One of :data:`PRICE_SOURCES`.

    Returns:
        An empty string for ``"close"``, otherwise ``"_<source>"``.
    """
    return "" if source == "close" else f"_{source}"


def source_expression(source: str) -> pl.Expr:
    """Return the expression selecting a price series.

    Args:
        source: One of :data:`PRICE_SOURCES`.

    Returns:
        The corresponding expression.

    Raises:
        ValueError: If the name is unknown.
    """
    validate_source(source)
    if source == "hl2":
        return (pl.col("high") + pl.col("low")) / 2
    if source == "hlc3":
        return (pl.col("high") + pl.col("low") + pl.col("close")) / 3
    if source == "ohlc4":
        return (pl.col("open") + pl.col("high") + pl.col("low") + pl.col("close")) / 4
    return pl.col(source)


def source_value(
    source: str,
    open_price: float,
    high: float,
    low: float,
    close: float,
    volume: float,
) -> float:
    """Compute a price source from one bar's raw fields.

    The arithmetic mirrors :func:`source_expression` term for term so the
    incremental path cannot drift from the vectorised one.

    Args:
        source: One of :data:`PRICE_SOURCES`.
        open_price: Bar open.
        high: Bar high.
        low: Bar low.
        close: Bar close.
        volume: Bar volume.

    Returns:
        The selected value.

    Raises:
        ValueError: If the name is unknown.
    """
    match source:
        case "open":
            return open_price
        case "high":
            return high
        case "low":
            return low
        case "close":
            return close
        case "volume":
            return volume
        case "hl2":
            return (high + low) / 2
        case "hlc3":
            return typical_price(high, low, close)
        case "ohlc4":
            return (open_price + high + low + close) / 4
        case _:
            raise ValueError(
                f"unknown price source {source!r}; expected one of {list(PRICE_SOURCES)}"
            )


def true_range() -> pl.Expr:
    """True range, undefined on the first bar.

    Wilder's true range needs the previous close, which the first bar does not
    have. Many libraries substitute ``high - low`` there; this one returns null
    instead, so a warmup count stays an honest statement about how much history
    an indicator consumed rather than an off-by-one convention.

    Returns:
        The true range expression.
    """
    previous_close = pl.col("close").shift(1)
    span = pl.max_horizontal(
        pl.col("high") - pl.col("low"),
        (pl.col("high") - previous_close).abs(),
        (pl.col("low") - previous_close).abs(),
    )
    return pl.when(previous_close.is_null()).then(None).otherwise(span)


def seeded_ema(values: pl.Expr, period: int, alpha: float) -> pl.Expr:
    """Exponential moving average seeded with the SMA of the first ``period`` values.

    ``ewm_mean(adjust=False)`` seeds on the first non-null element it sees, so
    replacing that element with the rolling mean produces an SMA-seeded EMA
    without leaving polars. Leading nulls in ``values`` — the first bar of a
    true-range series, for instance — are carried through, so the seed lands on
    the first index where a full window of *real* values exists.

    Args:
        values: Series to smooth. Must contain no interior nulls.
        period: Number of values averaged into the seed.
        alpha: Smoothing factor.

    Returns:
        The smoothed expression, null until the seed index.
    """
    seed = values.rolling_mean(period)
    at_seed = seed.is_not_null() & (seed.is_not_null().cum_sum() == 1)
    return (
        pl.when(at_seed)
        .then(seed)
        .when(seed.is_null())
        .then(None)
        .otherwise(values)
        .ewm_mean(alpha=alpha, adjust=False, ignore_nulls=False)
    )


def wilder_rma(values: pl.Expr, period: int) -> pl.Expr:
    """Wilder's smoothed moving average, ``alpha = 1 / period``.

    Args:
        values: Series to smooth.
        period: Averaging period.

    Returns:
        The smoothed expression.
    """
    return seeded_ema(values, period, 1.0 / period)


def rolling_mad(values: pl.Expr, period: int) -> pl.Expr:
    """Mean absolute deviation about the rolling mean.

    Expanded into ``period`` shifted terms rather than evaluated with
    ``rolling_map``: the deviation is not separable into a running statistic,
    but a horizontal sum of lagged columns stays fully vectorised and runs about
    twenty times faster than a Python-per-window callback.

    Args:
        values: Series to measure.
        period: Window length.

    Returns:
        The mean absolute deviation expression, null through the warmup.
    """
    mean = values.rolling_mean(period)
    deviations = [(values.shift(lag) - mean).abs() for lag in range(period)]
    return pl.sum_horizontal(deviations, ignore_nulls=False) / period


def weighted_rolling_mean(values: pl.Expr, weights: Sequence[float]) -> pl.Expr:
    """Rolling weighted mean, ``weights[-1]`` applying to the newest bar.

    polars' own ``rolling_mean(weights=...)`` panics on a series containing
    nulls, which rules it out for chained averages such as the Hull MA, whose
    outer window consumes the warmup nulls of its inner ones. Expanding the
    window into shifted terms sidesteps that and keeps the summation order —
    oldest to newest — identical to
    :meth:`~trading_system.features.rolling.RollingWindow.weighted_mean`.

    Args:
        values: Series to average.
        weights: One weight per bar of the window, oldest first.

    Returns:
        The weighted mean expression, null through the warmup.
    """
    period = len(weights)
    terms = [
        values.shift(period - 1 - position) * weight for position, weight in enumerate(weights)
    ]
    return pl.sum_horizontal(terms, ignore_nulls=False) / sum(weights)


def safe_divide(numerator: pl.Expr, denominator: pl.Expr, fallback: float) -> pl.Expr:
    """Divide, substituting ``fallback`` where the denominator vanishes.

    Degenerate windows are real: a flat range makes Stochastic's denominator
    zero, and a session of zero-volume bars does the same to VWAP. Each caller
    states the value the ratio collapses to rather than letting an infinity or a
    NaN travel downstream.

    Args:
        numerator: Dividend expression.
        denominator: Divisor expression.
        fallback: Value to use where the divisor is exactly zero.

    Returns:
        The guarded ratio.
    """
    return (
        pl.when(denominator == 0)
        .then(pl.lit(fallback, dtype=pl.Float64))
        .otherwise(numerator / denominator)
    )
