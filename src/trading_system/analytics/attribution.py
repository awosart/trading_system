"""Where a run's result came from: PnL by slice, quality against realised R, excursions.

Three things live here, and they answer different questions.

**Slices** cut the closed trades by strategy, symbol, session, weekday, hour,
direction and quality level. Every slice carries its own trade count, because
a mean R over four trades and a mean R over four hundred are not the same kind
of statement and nothing else on the row says which one you are reading.

**Quality against realised R** asks whether the score the Entry Engine attaches
to a setup predicts anything. It branches on **how many distinct quality values
were actually observed**, not on a trade-count threshold — a decile needs ten
distinguishable values to exist, and no number of trades creates them. Every
strategy in the library today scores with one modifier, so quality takes
exactly two values and the honest presentation is two groups and the gap
between them, not ten buckets of which eight are empty.

**Excursions (MAE/MFE)** need the bars, which a closed trade does not carry.
They are passed in explicitly rather than recorded on the trade — a decision
carried over from stage 1 and unchanged: a position that stores its own price
history turns every trade into a copy of the market data, and the frames are
already in the caller's hands.

**Everything per-trade is measured in R, never in account currency.** Sizing is
a fraction of equity, so a trade's money result depends on when it happened and
its R does not; a slice mean over money would rank the hours of the day partly
by how large the account was when they came up. ``net`` is still reported per
slice as a total, where it means what it says.
"""

import math
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import StrEnum

from scipy import stats

from trading_system.analytics.metrics import field_meta
from trading_system.backtest.clock import StreamKey
from trading_system.backtest.portfolio import TradeRecord
from trading_system.core.types import Side
from trading_system.data.models import OHLCVFrame
from trading_system.data.sessions import Session, session_of

#: Distinct quality values at or above which deciles are built instead of
#: per-level groups. Ten, because that is what a decile is: fewer distinct
#: values than buckets guarantees empty buckets, and a chart of empty buckets
#: reads as a distribution while being a handful of points.
DECILE_LEVEL_THRESHOLD = 10

#: Trades below which a slice row is flagged thin. Not a filter — the row is
#: still shown, on the same principle by which a fold below
#: ``min_trades_per_fold`` stays in the walk-forward aggregate.
THIN_SLICE_TRADES = 10

#: Label for trades whose entry fell outside every named session.
OUTSIDE_SESSIONS = "OUTSIDE_SESSIONS"

_WEEKDAYS = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")


@dataclass(frozen=True)
class SliceRow:
    """One bucket of a slice, with the sample it was computed over.

    Attributes:
        label: The bucket's name.
        count: Trades in the bucket.
        net: Total ``net`` of those trades, account currency. A sum, so it is
            meaningful in money where a mean would not be.
        total_r: Sum of realised R.
        expectancy_r: Mean realised R.
        median_r: Median realised R.
        win_rate: Share of trades with ``net > 0``.
        thin: Whether ``count`` is below :data:`THIN_SLICE_TRADES`.
    """

    label: str = field(metadata=field_meta("fact"))
    count: int = field(metadata=field_meta("n"))
    net: Decimal = field(metadata=field_meta("value", n_field="count"))
    total_r: float = field(metadata=field_meta("value", n_field="count"))
    expectancy_r: float = field(metadata=field_meta("value", n_field="count"))
    median_r: float = field(metadata=field_meta("value", n_field="count"))
    win_rate: float = field(metadata=field_meta("value", n_field="count"))
    thin: bool = field(metadata=field_meta("fact"))


@dataclass(frozen=True)
class Attribution:
    """One dimension cut into buckets.

    Attributes:
        dimension: What was cut by.
        rows: The buckets, ordered by the dimension's own natural order where
            it has one (hour, weekday, quality) and by descending ``net``
            where it does not.
        n_trades: Trades the cut was made over.
        partitions: Whether the buckets partition the trades — one trade in
            exactly one bucket. False for sessions, which overlap by
            construction (``LONDON_NY_OVERLAP`` is active precisely when
            London and New York both are), so the rows sum to more than
            ``n_trades`` and their ``net`` sums to more than the run's. Stated
            as a field rather than left to a reader who knows the session
            table, because a total that does not add up is otherwise read as
            an arithmetic bug.
    """

    dimension: str = field(metadata=field_meta("fact"))
    rows: tuple[SliceRow, ...] = field(metadata=field_meta("raw"))
    n_trades: int = field(metadata=field_meta("n"))
    partitions: bool = field(metadata=field_meta("fact"))


def _row(label: str, trades: Sequence[TradeRecord]) -> SliceRow:
    """Summarise one bucket's trades."""
    rs = [trade.realized_r for trade in trades]
    wins = sum(1 for trade in trades if trade.net > 0)
    return SliceRow(
        label=label,
        count=len(trades),
        net=sum((trade.net for trade in trades), Decimal(0)),
        total_r=math.fsum(rs),
        expectancy_r=math.fsum(rs) / len(rs),
        median_r=statistics.median(rs),
        win_rate=wins / len(trades),
        thin=len(trades) < THIN_SLICE_TRADES,
    )


def _attribute(
    trades: Sequence[TradeRecord],
    dimension: str,
    labels: "Mapping[str, Sequence[TradeRecord]]",
    *,
    order: Sequence[str] | None = None,
    partitions: bool = True,
) -> Attribution:
    """Assemble an :class:`Attribution` from already-bucketed trades."""
    if order is not None:
        names = [name for name in order if name in labels]
        names += sorted(set(labels) - set(order))
    else:
        names = sorted(labels, key=lambda name: -sum(t.net for t in labels[name]))
    return Attribution(
        dimension=dimension,
        rows=tuple(_row(name, labels[name]) for name in names),
        n_trades=len(trades),
        partitions=partitions,
    )


def by_strategy(trades: Sequence[TradeRecord]) -> Attribution:
    """Cut by the strategy that opened each trade."""
    buckets: dict[str, list[TradeRecord]] = {}
    for trade in trades:
        buckets.setdefault(trade.strategy_id, []).append(trade)
    return _attribute(trades, "strategy", buckets)


def by_symbol(trades: Sequence[TradeRecord]) -> Attribution:
    """Cut by instrument."""
    buckets: dict[str, list[TradeRecord]] = {}
    for trade in trades:
        buckets.setdefault(trade.symbol, []).append(trade)
    return _attribute(trades, "symbol", buckets)


def by_direction(trades: Sequence[TradeRecord]) -> Attribution:
    """Cut by side held."""
    buckets: dict[str, list[TradeRecord]] = {}
    for trade in trades:
        buckets.setdefault(trade.side.value, []).append(trade)
    return _attribute(trades, "direction", buckets, order=[Side.BUY.value, Side.SELL.value])


def by_hour(trades: Sequence[TradeRecord]) -> Attribution:
    """Cut by UTC hour of entry.

    UTC and not local time, because the run has no single locale: the same hour
    label has to mean the same instant across instruments.
    """
    buckets: dict[str, list[TradeRecord]] = {}
    for trade in trades:
        buckets.setdefault(f"{trade.opened_at.hour:02d}", []).append(trade)
    return _attribute(trades, "hour", buckets, order=[f"{h:02d}" for h in range(24)])


def by_weekday(trades: Sequence[TradeRecord]) -> Attribution:
    """Cut by weekday of entry."""
    buckets: dict[str, list[TradeRecord]] = {}
    for trade in trades:
        buckets.setdefault(_WEEKDAYS[trade.opened_at.weekday()], []).append(trade)
    return _attribute(trades, "weekday", buckets, order=list(_WEEKDAYS))


def by_session(trades: Sequence[TradeRecord]) -> Attribution:
    """Cut by the sessions active at entry.

    A trade lands in **every** session active at its entry, so the rows overlap
    and do not sum to the total — see :attr:`Attribution.partitions`. Attributing
    each trade to one "primary" session would need a precedence rule that does
    not exist anywhere in this system, and inventing one here would make the
    London/New York overlap disappear into whichever of the two won.
    """
    buckets: dict[str, list[TradeRecord]] = {}
    for trade in trades:
        active = session_of(trade.opened_at)
        names = sorted(session.value for session in active) or [OUTSIDE_SESSIONS]
        for name in names:
            buckets.setdefault(name, []).append(trade)
    order = [session.value for session in Session] + [OUTSIDE_SESSIONS]
    return _attribute(trades, "session", buckets, order=order, partitions=False)


def by_quality(trades: Sequence[TradeRecord]) -> Attribution:
    """Cut by the observed quality values, ascending."""
    buckets: dict[str, list[TradeRecord]] = {}
    for trade in trades:
        buckets.setdefault(f"{trade.quality:.4g}", []).append(trade)
    order = sorted(buckets, key=float)
    return _attribute(trades, "quality", buckets, order=order)


#: Every slice, by name. A mapping rather than a list of calls at the call
#: site, so a dimension added here reaches the report without the report
#: needing to know it was added.
SLICES = {
    "strategy": by_strategy,
    "symbol": by_symbol,
    "direction": by_direction,
    "session": by_session,
    "weekday": by_weekday,
    "hour": by_hour,
    "quality": by_quality,
}


def attribute_all(trades: Sequence[TradeRecord]) -> dict[str, Attribution]:
    """Every slice in :data:`SLICES`, computed over the same trades."""
    return {name: cut(trades) for name, cut in SLICES.items()}


# ---------------------------------------------------------------------------
# Quality against realised R
# ---------------------------------------------------------------------------


class QualityMode(StrEnum):
    """How the quality analysis was presented. See :class:`QualityAnalysis`."""

    #: No trades at all — nothing was scored against an outcome.
    NO_TRADES = "no_trades"

    #: One distinct quality value: a correlation does not exist.
    CONSTANT = "constant"

    #: Two to nine distinct values: grouped by the values themselves.
    LEVELS = "levels"

    #: Ten or more distinct values: grouped into deciles.
    DECILES = "deciles"


@dataclass(frozen=True)
class QualityGroup:
    """One quality value (or decile) and how its trades turned out.

    Attributes:
        label: The quality value, or the decile's range.
        quality_mean: Mean quality inside the group — equal to the value
            itself for a single-level group, and a range midpoint in spirit
            for a decile.
        count: Trades in the group.
        expectancy_r: Mean realised R.
        median_r: Median realised R.
        win_rate: Share with ``net > 0``.
    """

    label: str = field(metadata=field_meta("fact"))
    quality_mean: float = field(metadata=field_meta("fact"))
    count: int = field(metadata=field_meta("n"))
    expectancy_r: float = field(metadata=field_meta("value", n_field="count"))
    median_r: float = field(metadata=field_meta("value", n_field="count"))
    win_rate: float = field(metadata=field_meta("value", n_field="count"))


@dataclass(frozen=True)
class QualityGap:
    """The difference between the top and bottom quality group, with its uncertainty.

    Welch's two-sample interval, which does not assume the two groups share a
    variance — they routinely do not, since the high-quality group is usually
    the smaller one.

    Attributes:
        low_label: The lower group.
        high_label: The higher group.
        n_low: Trades in the lower group.
        n_high: Trades in the higher group.
        difference: ``mean R(high) - mean R(low)``. Positive means the score
            ranked the trades in the direction it claims to.
        ci_low: Lower bound of the 95% interval, or ``None`` when either group
            has fewer than two trades and no variance can be estimated.
        ci_high: Upper bound.
        t_statistic: Welch's t, or ``None`` when undefined.
        p_value: Two-sided p, or ``None`` when undefined. Two-sided because
            the interesting failure is not only "quality predicts nothing" but
            "quality predicts backwards", and a one-sided test cannot report
            the second.
    """

    low_label: str = field(metadata=field_meta("fact"))
    high_label: str = field(metadata=field_meta("fact"))
    n_low: int = field(metadata=field_meta("n"))
    n_high: int = field(metadata=field_meta("n"))
    difference: float = field(metadata=field_meta("value", n_field="n_high"))
    ci_low: float | None = field(metadata=field_meta("value", n_field="n_high"))
    ci_high: float | None = field(metadata=field_meta("value", n_field="n_high"))
    t_statistic: float | None = field(metadata=field_meta("value", n_field="n_high"))
    p_value: float | None = field(metadata=field_meta("value", n_field="n_high"))


@dataclass(frozen=True)
class QualityAnalysis:
    """Whether the Entry Engine's quality score predicted realised R.

    **The branch is on the number of distinct quality values observed, not on
    the number of trades.** With one value the correlation is undefined and is
    reported as ``None`` with a reason — never as a dash, which reads as "we
    computed it and it was nothing", and never by omitting the section, which
    reads as "we forgot". A constant quality is itself worth seeing: it means
    either that the spec declares no modifiers, or that a modifier's condition
    is never true, and the second is a defect that looks from outside exactly
    like a score that does not work.

    Attributes:
        mode: Which presentation was chosen — see :class:`QualityMode`.
        n_trades: Trades analysed.
        n_levels: Distinct quality values observed.
        groups: The per-level (or per-decile) breakdown, ascending.
        correlation: Pearson correlation of quality against realised R, or
            ``None`` when undefined. With exactly two levels this is the
            point-biserial correlation and is a rescaling of ``gap.difference``
            — published beside the gap and never instead of it, because a
            correlation alone hides both group sizes.
        note: Why the correlation is what it is, in words. Always present,
            including when the correlation is a number.
        gap: The top-minus-bottom difference with its interval, or ``None``
            when there are fewer than two groups.
    """

    mode: str = field(metadata=field_meta("fact"))
    n_trades: int = field(metadata=field_meta("n"))
    n_levels: int = field(metadata=field_meta("fact"))
    groups: tuple[QualityGroup, ...] = field(metadata=field_meta("raw"))
    correlation: float | None = field(metadata=field_meta("value", n_field="n_trades"))
    note: str = field(metadata=field_meta("fact"))
    gap: QualityGap | None = field(metadata=field_meta("raw"))


def _group(label: str, trades: Sequence[TradeRecord]) -> QualityGroup:
    """Summarise one quality group."""
    rs = [trade.realized_r for trade in trades]
    return QualityGroup(
        label=label,
        quality_mean=math.fsum(trade.quality for trade in trades) / len(trades),
        count=len(trades),
        expectancy_r=math.fsum(rs) / len(rs),
        median_r=statistics.median(rs),
        win_rate=sum(1 for trade in trades if trade.net > 0) / len(trades),
    )


def _welch(low: Sequence[float], high: Sequence[float]) -> tuple[float | None, ...]:
    """Welch's t, two-sided p and the 95% interval for ``mean(high) - mean(low)``."""
    if len(low) < 2 or len(high) < 2:
        return (None, None, None, None)
    var_low, var_high = statistics.variance(low), statistics.variance(high)
    se_squared = var_low / len(low) + var_high / len(high)
    if se_squared <= 0.0:
        return (None, None, None, None)
    se = math.sqrt(se_squared)
    difference = math.fsum(high) / len(high) - math.fsum(low) / len(low)
    df_numerator = se_squared**2
    df_denominator = (var_low / len(low)) ** 2 / (len(low) - 1) + (var_high / len(high)) ** 2 / (
        len(high) - 1
    )
    if df_denominator <= 0.0:
        return (None, None, None, None)
    df = df_numerator / df_denominator
    t_stat = difference / se
    critical = float(stats.t.ppf(0.975, df))
    p_value = float(2.0 * stats.t.sf(abs(t_stat), df))
    return (t_stat, p_value, difference - critical * se, difference + critical * se)


def quality_vs_r(trades: Sequence[TradeRecord]) -> QualityAnalysis:
    """Ask whether quality predicted realised R, and say what could be asked.

    Args:
        trades: Closed trades, in any order.

    Returns:
        The analysis. Never raises on a degenerate sample: no trades, one
        quality level and a level with a single trade are all states this
        returns rather than states it refuses.
    """
    if not trades:
        return QualityAnalysis(
            mode=QualityMode.NO_TRADES,
            n_trades=0,
            n_levels=0,
            groups=(),
            correlation=None,
            note="no trades: quality was never scored against an outcome",
            gap=None,
        )

    levels: dict[float, list[TradeRecord]] = {}
    for trade in trades:
        levels.setdefault(trade.quality, []).append(trade)
    ordered = sorted(levels)

    if len(ordered) == 1:
        only = ordered[0]
        return QualityAnalysis(
            mode=QualityMode.CONSTANT,
            n_trades=len(trades),
            n_levels=1,
            groups=(_group(f"{only:.4g}", levels[only]),),
            correlation=None,
            note=(
                f"quality is constant at {only:.4g} across all {len(trades)} trades, so a "
                "correlation is undefined. Either the spec declares no quality modifiers, or "
                "a modifier's condition never held — the second is a defect that looks from "
                "outside like a score that does not work."
            ),
            gap=None,
        )

    qualities = [trade.quality for trade in trades]
    realised = [trade.realized_r for trade in trades]
    try:
        correlation = statistics.correlation(qualities, realised)
    except statistics.StatisticsError:  # realised R is itself constant
        correlation = None

    if len(ordered) >= DECILE_LEVEL_THRESHOLD:
        groups = _deciles(trades)
        mode = QualityMode.DECILES
        note = (
            f"{len(ordered)} distinct quality values over {len(trades)} trades: "
            "presented as deciles."
        )
    else:
        groups = tuple(_group(f"{value:.4g}", levels[value]) for value in ordered)
        mode = QualityMode.LEVELS
        note = (
            f"{len(ordered)} distinct quality values over {len(trades)} trades — fewer than "
            f"{DECILE_LEVEL_THRESHOLD}, so the trades are grouped by the values themselves "
            "rather than into deciles, which would leave most buckets empty. With two levels "
            "the correlation below is the point-biserial one and carries the same information "
            "as the gap; read them together."
        )

    low_trades = [trade for trade in trades if trade.quality == ordered[0]]
    high_trades = [trade for trade in trades if trade.quality == ordered[-1]]
    t_stat, p_value, ci_low, ci_high = _welch(
        [trade.realized_r for trade in low_trades],
        [trade.realized_r for trade in high_trades],
    )
    gap = QualityGap(
        low_label=f"{ordered[0]:.4g}",
        high_label=f"{ordered[-1]:.4g}",
        n_low=len(low_trades),
        n_high=len(high_trades),
        difference=(
            math.fsum(trade.realized_r for trade in high_trades) / len(high_trades)
            - math.fsum(trade.realized_r for trade in low_trades) / len(low_trades)
        ),
        ci_low=ci_low,
        ci_high=ci_high,
        t_statistic=t_stat,
        p_value=p_value,
    )
    return QualityAnalysis(
        mode=mode,
        n_trades=len(trades),
        n_levels=len(ordered),
        groups=groups,
        correlation=correlation,
        note=note,
        gap=gap,
    )


def _deciles(trades: Sequence[TradeRecord]) -> tuple[QualityGroup, ...]:
    """Ten equal-count buckets by ascending quality, thinnest possible ties respected."""
    ranked = sorted(trades, key=lambda trade: trade.quality)
    size = len(ranked) / 10.0
    groups: list[QualityGroup] = []
    for index in range(10):
        chunk = ranked[int(round(index * size)) : int(round((index + 1) * size))]
        if not chunk:
            continue
        label = f"D{index + 1} [{chunk[0].quality:.3g}..{chunk[-1].quality:.3g}]"
        groups.append(_group(label, chunk))
    return tuple(groups)


# ---------------------------------------------------------------------------
# Excursions
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Excursion:
    """How far one trade ran against and for its holder before it closed.

    Both are measured in R against the trade's own frozen
    ``initial_risk_distance``, and both are non-negative: MAE is how far the
    price went the wrong way, MFE how far it went the right way, each reported
    as a magnitude with its direction implied by which field it is.

    Attributes:
        position_id: The trade.
        mae_r: Maximum adverse excursion, in R, ``>= 0``.
        mfe_r: Maximum favourable excursion, in R, ``>= 0``.
        realized_r: What the trade actually made, for comparison.
        bars: Bars the measurement covered.
    """

    position_id: str = field(metadata=field_meta("fact"))
    mae_r: float = field(metadata=field_meta("fact"))
    mfe_r: float = field(metadata=field_meta("fact"))
    realized_r: float = field(metadata=field_meta("fact"))
    bars: int = field(metadata=field_meta("n"))


@dataclass(frozen=True)
class ExcursionStats:
    """Excursions across a run, and the two readings they support.

    Attributes:
        excursions: Per-trade detail, in trade order.
        count: Trades measured.
        skipped: Trades whose stream was not supplied or which covered no bars
            — reported rather than silently absent, so a scatter with fewer
            points than the run has trades is explained on the page.
        mean_mae_r: Mean MAE in R. Well below 1.0 says stops were rarely
            approached: either they are wider than the trades need, or the
            exits close first.
        median_mae_r: Median MAE in R.
        mean_mfe_r: Mean MFE in R.
        median_mfe_r: Median MFE in R.
        capture: ``sum(realized_r) / sum(mfe_r)`` across every measured trade —
            the share of the best available movement that was actually taken.

            **An aggregate ratio, deliberately not the mean of per-trade
            ratios.** The per-trade mean is unusable here and was measured
            being so: on ``channel-breakout-h4`` it came out at −1.33, driven
            by a single trade that lost 1.01R after running 0.03R in favour —
            a ratio of −32 that no amount of sample size dilutes, because the
            denominator, not the numerator, is what went to zero. A trade that
            barely moved in favour before losing is an ordinary trade, not a
            32× catastrophe, and the aggregate says so: it weights each trade
            by how much movement was actually on offer.
        median_capture: Median of the per-trade ratios among trades that ran
            favourably at all. Published beside ``capture`` because they
            answer different questions — one is "of all the movement offered,
            how much was taken", the other "what did a typical trade take".
        losers_with_favourable_run: Trades that finished at or below zero
            after having been at least 1R in favour. The concrete form of
            "targets too far away".
    """

    excursions: tuple[Excursion, ...] = field(metadata=field_meta("raw"))
    count: int = field(metadata=field_meta("n"))
    skipped: int = field(metadata=field_meta("fact"))
    mean_mae_r: float | None = field(metadata=field_meta("value", n_field="count"))
    median_mae_r: float | None = field(metadata=field_meta("value", n_field="count"))
    mean_mfe_r: float | None = field(metadata=field_meta("value", n_field="count"))
    median_mfe_r: float | None = field(metadata=field_meta("value", n_field="count"))
    capture: float | None = field(metadata=field_meta("value", n_field="count"))
    median_capture: float | None = field(metadata=field_meta("value", n_field="count"))
    losers_with_favourable_run: int = field(metadata=field_meta("value", n_field="count"))


def _stream_for(trade: TradeRecord, streams: Mapping[StreamKey, OHLCVFrame]) -> OHLCVFrame | None:
    """The frame carrying this trade's symbol, finest timeframe first.

    A run may hold several timeframes of one symbol. The finest one is chosen
    because an excursion is a maximum over the path, and a coarser bar can only
    understate it — a daily bar's high is the same number as the highest hourly
    high inside it, but a daily bar spanning the entry hides where inside the
    day the entry happened.
    """
    candidates = [frame for key, frame in streams.items() if key.symbol == trade.symbol]
    if not candidates:
        return None
    return min(candidates, key=lambda frame: frame.timeframe.duration)


def excursions(
    trades: Sequence[TradeRecord], streams: Mapping[StreamKey, OHLCVFrame]
) -> ExcursionStats:
    """Measure maximum adverse and favourable excursion for each trade.

    **The bars measured are those that opened strictly after the entry fill and
    at or before the close.** The entry bar itself is excluded: its high and low
    include movement that happened before the fill, so counting it would credit
    the trade with an excursion it was not in the market for — the same reason
    P06 does not apply invalidation to the trigger bar.

    Args:
        trades: Closed trades.
        streams: The run's bars, by stream. Passed explicitly: a closed trade
            does not carry price history and should not start to.

    Returns:
        Per-trade excursions and their summary. A trade whose symbol is absent
        from ``streams``, or which spans no bars after its entry, is counted in
        ``skipped`` rather than dropped silently.
    """
    measured: list[Excursion] = []
    skipped = 0
    for trade in trades:
        frame = _stream_for(trade, streams)
        if frame is None or trade.initial_risk_distance <= 0.0:
            skipped += 1
            continue
        window = _window(frame, trade.opened_at, trade.closed_at)
        if not window:
            skipped += 1
            continue
        highs, lows = window
        if trade.side is Side.BUY:
            favourable = max(highs) - trade.entry_price
            adverse = trade.entry_price - min(lows)
        else:
            favourable = trade.entry_price - min(lows)
            adverse = max(highs) - trade.entry_price
        measured.append(
            Excursion(
                position_id=trade.position_id,
                mae_r=max(0.0, adverse / trade.initial_risk_distance),
                mfe_r=max(0.0, favourable / trade.initial_risk_distance),
                realized_r=trade.realized_r,
                bars=len(highs),
            )
        )
    return _excursion_stats(tuple(measured), skipped)


def _window(
    frame: OHLCVFrame, opened_at: datetime, closed_at: datetime
) -> tuple[list[float], list[float]] | None:
    """Highs and lows of the bars strictly after ``opened_at`` and at or before ``closed_at``."""
    df = frame.df.filter((frame.df["timestamp"] > opened_at) & (frame.df["timestamp"] <= closed_at))
    if df.is_empty():
        return None
    return list(df["high"]), list(df["low"])


def _excursion_stats(measured: tuple[Excursion, ...], skipped: int) -> ExcursionStats:
    """Summarise per-trade excursions."""
    if not measured:
        return ExcursionStats(
            excursions=(),
            count=0,
            skipped=skipped,
            mean_mae_r=None,
            median_mae_r=None,
            mean_mfe_r=None,
            median_mfe_r=None,
            capture=None,
            median_capture=None,
            losers_with_favourable_run=0,
        )
    maes = [item.mae_r for item in measured]
    mfes = [item.mfe_r for item in measured]
    ratios = [item.realized_r / item.mfe_r for item in measured if item.mfe_r > 0.0]
    total_mfe = math.fsum(mfes)
    return ExcursionStats(
        excursions=measured,
        count=len(measured),
        skipped=skipped,
        mean_mae_r=math.fsum(maes) / len(maes),
        median_mae_r=statistics.median(maes),
        mean_mfe_r=math.fsum(mfes) / len(mfes),
        median_mfe_r=statistics.median(mfes),
        capture=(math.fsum(item.realized_r for item in measured) / total_mfe)
        if total_mfe > 0.0
        else None,
        median_capture=statistics.median(ratios) if ratios else None,
        losers_with_favourable_run=sum(
            1 for item in measured if item.realized_r <= 0.0 and item.mfe_r >= 1.0
        ),
    )
