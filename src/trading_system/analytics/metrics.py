"""Descriptive performance metrics: what happened, with its sample size attached.

This module answers *what happened*, never *whether it's real* — that question,
and the "n=19 isn't enough" judgment, belongs to :mod:`trading_system.analytics.statistical`,
which is why nothing here returns a ``reliable: bool``. What this module owns
instead is a structural guarantee: **every value derived by aggregating a
variable-size sample carries that sample's size on the same object, as a named
field, always.** A caller can misread a lone float with no context glued to it;
a caller cannot misread a field it has to name to read the value next to it.

Two disciplines run through every function here:

**Annualisation is measured, never asserted.** ``periods_per_year`` is not
252 by convention — a fixed constant is right for a US equity calendar, quietly
wrong by about 3% for FX (which prints five ``trading_day`` labels a week,
~257-261/year depending on holidays) and badly wrong for a crypto stream that
trades every calendar day (~365). All three come out of the same formula,
``len(daily.days) / years_elapsed(daily)``, with no branch on asset class —
the same principle P10 applies to point value and P13 to correlation: one
formula, not one per instrument.

**Every risk and risk-adjusted metric takes a** :class:`DailyCurve`, **never
a raw equity curve or a bare sequence of floats.**
:class:`~trading_system.backtest.portfolio.EquityPoint` carries one row per
instant of the merged stream, and that row count is a function of portfolio
composition — adding a second stream adds rows without any strategy
changing. Accepting anything looser than ``DailyCurve`` here would
let a caller annualise from ``len(curve)`` by accident, exactly the mistake
this module exists to make structurally impossible. ``daily_curve()`` is the
single door through which any equity curve enters this module, taking the last
row of each ``EquityPoint.day`` label — the same label VWAP resets on and the
daily loss limit measures against, so this module does not invent a second
definition of "day" either.

**Sharpe and Sortino are named for their frequency.** ``sharpe_daily`` and
``sortino_daily``, not ``sharpe_ratio`` — "Sharpe ratio" has competing everyday
readings (per bar, per day, per trade) that the type system alone does not
head off, because a reader can imagine a per-trade Sharpe even where the
signature only accepts a :class:`DailyCurve`. The other risk-adjusted metrics
(Ulcer Index, VaR, CVaR, Omega) do not carry the suffix: nobody in this
codebase's context expects a "per-trade Ulcer Index," so the type alone
already disambiguates them.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date, timedelta
from decimal import Decimal
from statistics import correlation, quantiles

from trading_system.backtest.portfolio import EquityPoint, TradeRecord

#: Threshold above which a single trade's share of total net PnL is flagged as
#: the result "holding together on one trade" rather than a distributed edge.
BEST_TRADE_CONCENTRATION_THRESHOLD = 0.30


def field_meta(kind: str, *, n_field: str | None = None) -> dict[str, str | None]:
    """Build field metadata tagging a dataclass field's role, for the sample-size invariant.

    Used as ``field(metadata=field_meta(...))`` rather than wrapped inside a
    ``field()``-returning helper: mypy's dataclass support recognises a field
    as required only when it sees a literal call to ``dataclasses.field``
    carrying no ``default``/``default_factory`` — a helper that returns
    ``field(...)`` from inside its own body defeats that recognition and would
    let a required argument silently type-check as optional.

    Args:
        kind: One of ``"raw"`` (the field IS the sample, e.g. a tuple of
            observations), ``"n"`` (the field is a sample size), ``"value"``
            (a statistic derived from a sample — must name the sibling ``"n"``
            field it was computed over), or ``"fact"`` (a deterministic
            quantity with no sampling-noise concept: an input echo, a date, a
            ratio of two already-known numbers).
        n_field: For ``kind="value"``, the name of the sibling field on the
            same dataclass that carries the sample size.

    Returns:
        Metadata a reflective test walks — see
        ``tests/analytics/test_field_invariants.py``. Every field of every
        dataclass in this module must carry this metadata; the test
        enumerates the module's dataclasses itself; a field added without it
        fails immediately rather than trusting anyone to remember.
    """
    return {"kind": kind, "n_field": n_field}


def _years_elapsed(days: Sequence[date]) -> float:
    """Calendar years spanned by a sorted sequence of day labels.

    Args:
        days: Trading-day labels, oldest first.

    Returns:
        Elapsed time in years, using the same 365.25 convention throughout
        this module so nothing here can disagree with itself about a year's
        length.

    Raises:
        ValueError: If ``days`` spans zero time — CAGR and a measured
            annualisation factor are both undefined over an instant.
    """
    span = (days[-1] - days[0]).days
    if span <= 0:
        raise ValueError("cannot annualise over a curve spanning zero elapsed days")
    return span / 365.25


def measured_periods_per_year(daily: "DailyCurve") -> float:
    """The run's own trading-day frequency, measured rather than assumed.

    Args:
        daily: The daily-bucketed curve to measure.

    Returns:
        ``len(daily.days) / years_elapsed`` — ~252 for a US equity calendar,
        ~257-261 for FX (five ``trading_day`` labels a week), ~365 for an
        instrument that trades every calendar day. One formula, no branch on
        asset class: the day-label count already encodes how often the
        instrument trades, so nothing else needs to.
    """
    return len(daily.days) / _years_elapsed(daily.days)


# ---------------------------------------------------------------------------
# The daily curve
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DailyCurve:
    """One equity curve, bucketed to its last row per ``trading_day`` label.

    The single door every risk and risk-adjusted metric in this module enters
    through. Built by :func:`daily_curve`, never by hand — hand-assembling one
    would be exactly the kind of ad hoc day-cutting P14 is required not to
    invent.

    Attributes:
        days: Trading-day labels, strictly increasing.
        equity: Equity at the last instant of each day, aligned to ``days``.
        balance: Balance at the last instant of each day, aligned to ``days``.
    """

    days: tuple[date, ...] = field(metadata=field_meta("raw"))
    equity: tuple[Decimal, ...] = field(metadata=field_meta("raw"))
    balance: tuple[Decimal, ...] = field(metadata=field_meta("raw"))


def daily_curve(curve: Sequence[EquityPoint]) -> DailyCurve:
    """Bucket an equity curve to one row per trading day.

    Args:
        curve: The run's equity curve, chronologically ordered — the order
            :class:`~trading_system.backtest.portfolio.Portfolio` already
            appends it in.

    Returns:
        The daily curve, one row per distinct ``EquityPoint.day``, holding
        that day's last mark.

    Raises:
        ValueError: If ``curve`` is empty, or its timestamps are not
            non-decreasing — the precondition this function relies on rather
            than re-sorting, since re-sorting a curve that is not actually
            chronological would hide the defect instead of surfacing it.
    """
    if not curve:
        raise ValueError("cannot build a daily curve from an empty equity curve")
    days: list[date] = []
    equity: list[Decimal] = []
    balance: list[Decimal] = []
    last_ts = curve[0].ts
    last_day: date | None = None
    for point in curve:
        if point.ts < last_ts:
            raise ValueError(
                f"equity curve is not chronologically ordered: {point.ts} follows {last_ts}"
            )
        last_ts = point.ts
        if point.day == last_day:
            equity[-1] = point.equity
            balance[-1] = point.balance
        else:
            days.append(point.day)
            equity.append(point.equity)
            balance.append(point.balance)
            last_day = point.day
    return DailyCurve(days=tuple(days), equity=tuple(equity), balance=tuple(balance))


def simple_returns(daily: DailyCurve) -> tuple[float, ...]:
    """Day-over-day simple returns of a daily curve.

    Args:
        daily: The daily curve.

    Returns:
        ``equity[i] / equity[i-1] - 1`` for each consecutive pair, length
        ``len(daily.days) - 1``. Computed in :class:`~decimal.Decimal` and
        converted to ``float`` only at the return, since a return is a ratio,
        not money.

    Raises:
        ValueError: If any day's equity is not strictly positive — a return
            is undefined against a wiped-out or negative account.
    """
    returns: list[float] = []
    for previous, current in zip(daily.equity, daily.equity[1:], strict=False):
        if previous <= 0:
            raise ValueError(f"cannot compute a return over non-positive equity {previous}")
        returns.append(float(current / previous - 1))
    return tuple(returns)


# ---------------------------------------------------------------------------
# Returns
# ---------------------------------------------------------------------------


def total_return(daily: DailyCurve) -> float:
    """Whole-run return from first day's equity to last.

    Args:
        daily: The daily curve.

    Returns:
        ``equity[-1] / equity[0] - 1``. A ratio of two fixed endpoints, not an
        aggregate over a sample — no sample size to attach.
    """
    first = daily.equity[0]
    if first <= 0:
        raise ValueError(f"cannot compute a return over non-positive starting equity {first}")
    return float(daily.equity[-1] / first - 1)


@dataclass(frozen=True)
class CagrResult:
    """Compound annual growth rate, with the window it was extrapolated from.

    Attributes:
        value: CAGR — ``(equity[-1] / equity[0]) ** (1 / years_elapsed) - 1``.
        years_elapsed: Calendar years the curve spans. Not a sample size (CAGR
            is a closed-form function of two endpoints, not an aggregate), but
            the field a reader needs to judge extrapolation risk: three months
            annualised to a year is the same kind of overreach as a small ``n``
            elsewhere, so it travels with the value for the same reason.
    """

    value: float = field(metadata=field_meta("fact"))
    years_elapsed: float = field(metadata=field_meta("fact"))


def cagr(daily: DailyCurve) -> CagrResult:
    """Compound annual growth rate of a daily curve.

    Args:
        daily: The daily curve.

    Returns:
        The CAGR and the elapsed years it was computed over.
    """
    first = daily.equity[0]
    if first <= 0:
        raise ValueError(f"cannot compute CAGR over non-positive starting equity {first}")
    years = _years_elapsed(daily.days)
    growth = float(daily.equity[-1] / first)
    if growth <= 0:
        raise ValueError("cannot compute CAGR: equity went to zero or negative")
    return CagrResult(value=growth ** (1 / years) - 1, years_elapsed=years)


@dataclass(frozen=True)
class MonthlyReturn:
    """One calendar month's return.

    Attributes:
        year: Calendar year.
        month: Calendar month, 1-12.
        return_pct: Return from the prior month's last equity to this month's
            last equity. A ratio of two fixed month-end points, not a sample
            aggregate.
    """

    year: int = field(metadata=field_meta("fact"))
    month: int = field(metadata=field_meta("fact"))
    return_pct: float = field(metadata=field_meta("fact"))


def _month_end_equity(daily: DailyCurve) -> list[tuple[int, int, Decimal]]:
    """Last equity value of each calendar month present in ``daily``."""
    months: list[tuple[int, int, Decimal]] = []
    last_key: tuple[int, int] | None = None
    for day, equity in zip(daily.days, daily.equity, strict=True):
        key = (day.year, day.month)
        if key == last_key:
            months[-1] = (*key, equity)
        else:
            months.append((*key, equity))
            last_key = key
    return months


def monthly_returns(daily: DailyCurve) -> tuple[MonthlyReturn, ...]:
    """Month-over-month returns table.

    Args:
        daily: The daily curve.

    Returns:
        One :class:`MonthlyReturn` per calendar month after the first, oldest
        first. The first month has no prior month-end to return from and is
        excluded, the same way :func:`simple_returns` excludes day zero.
    """
    months = _month_end_equity(daily)
    results: list[MonthlyReturn] = []
    for (_, _, previous), (year, month, current) in zip(months, months[1:], strict=False):
        if previous <= 0:
            raise ValueError(f"cannot compute a monthly return over non-positive equity {previous}")
        results.append(
            MonthlyReturn(year=year, month=month, return_pct=float(current / previous - 1))
        )
    return tuple(results)


@dataclass(frozen=True)
class AnnualReturn:
    """One calendar year's return.

    Attributes:
        year: Calendar year.
        return_pct: Return from the prior year's last equity to this year's
            last equity.
    """

    year: int = field(metadata=field_meta("fact"))
    return_pct: float = field(metadata=field_meta("fact"))


def annual_returns(daily: DailyCurve) -> tuple[AnnualReturn, ...]:
    """Year-over-year returns table.

    Args:
        daily: The daily curve.

    Returns:
        One :class:`AnnualReturn` per calendar year after the first, oldest
        first.
    """
    years: list[tuple[int, Decimal]] = []
    last_year: int | None = None
    for day, equity in zip(daily.days, daily.equity, strict=True):
        if day.year == last_year:
            years[-1] = (day.year, equity)
        else:
            years.append((day.year, equity))
            last_year = day.year
    results: list[AnnualReturn] = []
    for (_, previous), (year, current) in zip(years, years[1:], strict=False):
        if previous <= 0:
            raise ValueError(f"cannot compute an annual return over non-positive equity {previous}")
        results.append(AnnualReturn(year=year, return_pct=float(current / previous - 1)))
    return tuple(results)


# ---------------------------------------------------------------------------
# Risk
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DrawdownPeriod:
    """One episode of equity below its running peak.

    Attributes:
        peak_day: Day the running peak this episode falls from was set.
        trough_day: Day of the episode's worst equity.
        recovery_day: Day equity first closed back at or above ``peak_equity``,
            or ``None`` if the curve ends still under water.
        peak_equity: Equity at ``peak_day``.
        trough_equity: Equity at ``trough_day``.
        depth_pct: ``trough_equity / peak_equity - 1``, negative.
        depth_money: ``trough_equity - peak_equity``, negative.
        duration_days: Trading days from peak to trough.
        recovery_days: Trading days from trough to recovery, or ``None`` if
            unrecovered.
    """

    peak_day: date = field(metadata=field_meta("fact"))
    trough_day: date = field(metadata=field_meta("fact"))
    recovery_day: date | None = field(metadata=field_meta("fact"))
    peak_equity: Decimal = field(metadata=field_meta("fact"))
    trough_equity: Decimal = field(metadata=field_meta("fact"))
    depth_pct: float = field(metadata=field_meta("fact"))
    depth_money: Decimal = field(metadata=field_meta("fact"))
    duration_days: int = field(metadata=field_meta("fact"))
    recovery_days: int | None = field(metadata=field_meta("fact"))


@dataclass(frozen=True)
class DrawdownStats:
    """Every drawdown episode in a curve, and the worst of them.

    Attributes:
        max_drawdown_pct: Depth, in percent, of the worst episode by depth.
        max_drawdown_money: Depth, in money, of that same episode.
        max_drawdown: The worst episode in full.
        all_drawdowns: Every episode, oldest first, including ones still
            running at the end of the curve.
    """

    max_drawdown_pct: float = field(metadata=field_meta("fact"))
    max_drawdown_money: Decimal = field(metadata=field_meta("fact"))
    max_drawdown: DrawdownPeriod = field(metadata=field_meta("fact"))
    all_drawdowns: tuple[DrawdownPeriod, ...] = field(metadata=field_meta("raw"))


def drawdown_stats(daily: DailyCurve) -> DrawdownStats:
    """Walk a daily curve and segment it into peak-to-recovery episodes.

    Args:
        daily: The daily curve.

    Returns:
        Every drawdown episode and the worst one by depth.

    Raises:
        ValueError: If ``daily`` has fewer than two days — a drawdown needs a
            peak and something after it to compare.
    """
    if len(daily.days) < 2:
        raise ValueError("drawdown statistics need at least two days")

    episodes: list[DrawdownPeriod] = []
    peak_day, peak_equity = daily.days[0], daily.equity[0]
    in_drawdown = False
    trough_day, trough_equity = peak_day, peak_equity

    def _close_episode(recovery_day: date | None) -> None:
        depth_pct = float(trough_equity / peak_equity - 1)
        depth_money = trough_equity - peak_equity
        duration = (trough_day - peak_day).days
        recovery = (recovery_day - trough_day).days if recovery_day is not None else None
        episodes.append(
            DrawdownPeriod(
                peak_day=peak_day,
                trough_day=trough_day,
                recovery_day=recovery_day,
                peak_equity=peak_equity,
                trough_equity=trough_equity,
                depth_pct=depth_pct,
                depth_money=depth_money,
                duration_days=duration,
                recovery_days=recovery,
            )
        )

    for day, equity in zip(daily.days[1:], daily.equity[1:], strict=True):
        if equity >= peak_equity:
            if in_drawdown:
                _close_episode(recovery_day=day)
                in_drawdown = False
            peak_day, peak_equity = day, equity
            trough_day, trough_equity = day, equity
            continue
        if not in_drawdown or equity < trough_equity:
            trough_day, trough_equity = day, equity
        in_drawdown = True

    if in_drawdown:
        _close_episode(recovery_day=None)

    if not episodes:
        raise ValueError("curve never dipped below its running peak; no drawdown to report")

    worst = min(episodes, key=lambda episode: episode.depth_pct)
    return DrawdownStats(
        max_drawdown_pct=worst.depth_pct,
        max_drawdown_money=worst.depth_money,
        max_drawdown=worst,
        all_drawdowns=tuple(episodes),
    )


@dataclass(frozen=True)
class SampleStat:
    """A single float derived from a daily-return sample, with that sample's size.

    Attributes:
        value: The statistic.
        n: Number of daily returns it was computed over.
    """

    value: float = field(metadata=field_meta("value", n_field="n"))
    n: int = field(metadata=field_meta("n"))


def ulcer_index(daily: DailyCurve) -> SampleStat:
    """Root-mean-square of the drawdown-from-peak percentage, one term per day.

    Args:
        daily: The daily curve.

    Returns:
        ``sqrt(mean(drawdown_pct_i ** 2))`` over every day (0 at a new peak),
        and the number of days it was measured over.
    """
    peak = daily.equity[0]
    squared_drawdowns: list[float] = []
    for equity in daily.equity:
        peak = max(peak, equity)
        drawdown_pct = float(equity / peak - 1)
        squared_drawdowns.append(drawdown_pct**2)
    value = math.sqrt(sum(squared_drawdowns) / len(squared_drawdowns))
    return SampleStat(value=value, n=len(squared_drawdowns))


def value_at_risk(daily: DailyCurve, *, confidence: float = 0.95) -> SampleStat:
    """Historical (empirical) Value at Risk of daily returns.

    Args:
        daily: The daily curve.
        confidence: Confidence level, e.g. ``0.95`` for the worst 5% tail.

    Returns:
        The ``(1 - confidence)``-quantile daily return by nearest-rank —
        negative, the loss threshold — and the number of daily returns it was
        measured over.

    Raises:
        ValueError: If ``daily`` has fewer than two days, so there is at
            least one return to rank.
    """
    returns = sorted(simple_returns(daily))
    if not returns:
        raise ValueError("value at risk needs at least one daily return")
    index = max(math.ceil((1 - confidence) * len(returns)) - 1, 0)
    return SampleStat(value=returns[index], n=len(returns))


def conditional_value_at_risk(daily: DailyCurve, *, confidence: float = 0.95) -> SampleStat:
    """Historical (empirical) Conditional VaR — the mean of the tail beyond VaR.

    Args:
        daily: The daily curve.
        confidence: Confidence level, matching :func:`value_at_risk`.

    Returns:
        Mean of every daily return at or below the VaR threshold, and the
        number of daily returns it was measured over.
    """
    returns = sorted(simple_returns(daily))
    if not returns:
        raise ValueError("conditional value at risk needs at least one daily return")
    index = max(math.ceil((1 - confidence) * len(returns)) - 1, 0)
    tail = returns[: index + 1]
    return SampleStat(value=sum(tail) / len(tail), n=len(returns))


def downside_deviation(
    daily: DailyCurve, *, mar: float = 0.0, periods_per_year: float | None = None
) -> SampleStat:
    """Annualised standard deviation of returns below a minimum acceptable return.

    Args:
        daily: The daily curve.
        mar: Minimum acceptable per-period return. Zero by default.
        periods_per_year: Annualisation factor. Measured from ``daily`` via
            :func:`measured_periods_per_year` when ``None``.

    Returns:
        ``sqrt(mean(min(r - mar, 0) ** 2) * periods_per_year)``, and the
        number of daily returns it was computed over.
    """
    returns = simple_returns(daily)
    if not returns:
        raise ValueError("downside deviation needs at least one daily return")
    factor = periods_per_year if periods_per_year is not None else measured_periods_per_year(daily)
    downside_sq = [min(r - mar, 0.0) ** 2 for r in returns]
    value = math.sqrt(sum(downside_sq) / len(downside_sq) * factor)
    return SampleStat(value=value, n=len(returns))


def omega_ratio(daily: DailyCurve, *, threshold: float = 0.0) -> SampleStat:
    """Ratio of gains above a threshold to losses below it, over daily returns.

    Args:
        daily: The daily curve.
        threshold: Per-period return threshold. Zero by default.

    Returns:
        ``sum(gains above threshold) / sum(losses below threshold)``, and the
        number of daily returns it was computed over.

    Raises:
        ValueError: If there are no returns below the threshold to divide by —
            Omega is undefined without a loss side, not infinite by
            convention.
    """
    returns = simple_returns(daily)
    if not returns:
        raise ValueError("omega ratio needs at least one daily return")
    excess = [r - threshold for r in returns]
    gains = sum(e for e in excess if e > 0)
    losses = sum(-e for e in excess if e < 0)
    if losses == 0:
        raise ValueError("omega ratio is undefined: no returns fell below the threshold")
    return SampleStat(value=gains / losses, n=len(returns))


# ---------------------------------------------------------------------------
# Risk-adjusted
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SharpeResult:
    """Annualised Sharpe ratio of daily returns, with what it was computed against.

    Attributes:
        value: The annualised Sharpe ratio.
        risk_free_rate: Annual risk-free rate used, as given.
        periods_per_year: Annualisation factor used, as given or measured.
        n_periods: Number of daily returns the ratio was computed over.
    """

    value: float = field(metadata=field_meta("value", n_field="n_periods"))
    risk_free_rate: float = field(metadata=field_meta("fact"))
    periods_per_year: float = field(metadata=field_meta("fact"))
    n_periods: int = field(metadata=field_meta("n"))


def sharpe_daily(
    daily: DailyCurve, *, risk_free_rate: float = 0.0, periods_per_year: float | None = None
) -> SharpeResult:
    """Annualised Sharpe ratio computed from the daily-bucketed equity curve.

    Args:
        daily: The daily curve. Never a raw curve or a bare float sequence —
            the type is what stops this from being annualised against the
            wrong period, see the module docstring.
        risk_free_rate: Annual risk-free rate.
        periods_per_year: Annualisation factor. Measured from ``daily`` via
            :func:`measured_periods_per_year` when ``None``.

    Returns:
        The Sharpe ratio and everything it was computed against.

    Raises:
        ValueError: If there are fewer than two daily returns, or their
            standard deviation is zero — both leave the ratio undefined, and
            this module raises rather than returning a silent ``nan``.
    """
    returns = simple_returns(daily)
    if len(returns) < 2:
        raise ValueError("sharpe_daily needs at least two daily returns")
    factor = periods_per_year if periods_per_year is not None else measured_periods_per_year(daily)
    rf_period = (1 + risk_free_rate) ** (1 / factor) - 1
    excess = [r - rf_period for r in returns]
    mean_excess = sum(excess) / len(excess)
    variance = sum((e - mean_excess) ** 2 for e in excess) / (len(excess) - 1)
    stdev = math.sqrt(variance)
    if stdev == 0:
        raise ValueError("sharpe_daily is undefined: returns have zero variance")
    value = (mean_excess / stdev) * math.sqrt(factor)
    return SharpeResult(
        value=value, risk_free_rate=risk_free_rate, periods_per_year=factor, n_periods=len(returns)
    )


@dataclass(frozen=True)
class SortinoResult:
    """Annualised Sortino ratio of daily returns, with what it was computed against.

    Attributes:
        value: The annualised Sortino ratio.
        risk_free_rate: Annual risk-free rate used, as given.
        periods_per_year: Annualisation factor used, as given or measured.
        mar: Minimum acceptable per-period return used for the downside term.
        n_periods: Number of daily returns the ratio was computed over.
        downside_deviation: The annualised downside deviation used as the
            denominator.
    """

    value: float = field(metadata=field_meta("value", n_field="n_periods"))
    risk_free_rate: float = field(metadata=field_meta("fact"))
    periods_per_year: float = field(metadata=field_meta("fact"))
    mar: float = field(metadata=field_meta("fact"))
    n_periods: int = field(metadata=field_meta("n"))
    downside_deviation: float = field(metadata=field_meta("value", n_field="n_periods"))


def sortino_daily(
    daily: DailyCurve,
    *,
    risk_free_rate: float = 0.0,
    periods_per_year: float | None = None,
    mar: float = 0.0,
) -> SortinoResult:
    """Annualised Sortino ratio computed from the daily-bucketed equity curve.

    Args:
        daily: The daily curve.
        risk_free_rate: Annual risk-free rate.
        periods_per_year: Annualisation factor. Measured from ``daily`` via
            :func:`measured_periods_per_year` when ``None``.
        mar: Minimum acceptable per-period return for the downside term.

    Returns:
        The Sortino ratio and everything it was computed against.

    Raises:
        ValueError: If there are no daily returns, or the downside deviation
            is zero.
    """
    returns = simple_returns(daily)
    if not returns:
        raise ValueError("sortino_daily needs at least one daily return")
    factor = periods_per_year if periods_per_year is not None else measured_periods_per_year(daily)
    rf_period = (1 + risk_free_rate) ** (1 / factor) - 1
    mean_excess = sum(r - rf_period for r in returns) / len(returns)
    downside = downside_deviation(daily, mar=mar, periods_per_year=factor)
    if downside.value == 0:
        raise ValueError("sortino_daily is undefined: downside deviation is zero")
    value = (mean_excess * factor) / downside.value
    return SortinoResult(
        value=value,
        risk_free_rate=risk_free_rate,
        periods_per_year=factor,
        mar=mar,
        n_periods=len(returns),
        downside_deviation=downside.value,
    )


def calmar_ratio(daily: DailyCurve, *, window_years: float = 3.0) -> float:
    """CAGR over max drawdown, on the trailing window (industry convention: 3 years).

    Args:
        daily: The daily curve.
        window_years: Trailing window. Falls back to the full curve when it is
            shorter than this.

    Returns:
        ``CAGR / abs(max_drawdown_pct)`` over the trailing window. A ratio of
        two already-summarised figures, not a fresh sample aggregate — no
        sample size to attach beyond what each of those two already carries.
    """
    cutoff_days = window_years * 365.25
    start_index = 0
    for index in range(len(daily.days) - 1, -1, -1):
        if (daily.days[-1] - daily.days[index]).days > cutoff_days:
            start_index = index + 1
            break
    window = DailyCurve(
        days=daily.days[start_index:],
        equity=daily.equity[start_index:],
        balance=daily.balance[start_index:],
    )
    window_cagr = cagr(window)
    window_drawdown = drawdown_stats(window)
    if window_drawdown.max_drawdown_pct == 0:
        raise ValueError("calmar_ratio is undefined: no drawdown in the window")
    return window_cagr.value / abs(window_drawdown.max_drawdown_pct)


def mar_ratio(daily: DailyCurve) -> float:
    """CAGR over max drawdown, full history — unlike Calmar, no trailing window.

    Args:
        daily: The daily curve.

    Returns:
        ``CAGR / abs(max_drawdown_pct)`` over the whole curve.
    """
    full_cagr = cagr(daily)
    full_drawdown = drawdown_stats(daily)
    if full_drawdown.max_drawdown_pct == 0:
        raise ValueError("mar_ratio is undefined: no drawdown in the curve")
    return full_cagr.value / abs(full_drawdown.max_drawdown_pct)


def recovery_factor(daily: DailyCurve) -> float:
    """Total money made over the worst money drawdown.

    Args:
        daily: The daily curve.

    Returns:
        ``(equity[-1] - equity[0]) / abs(max_drawdown_money)``.
    """
    drawdown = drawdown_stats(daily)
    if drawdown.max_drawdown_money == 0:
        raise ValueError("recovery_factor is undefined: no drawdown in the curve")
    net = daily.equity[-1] - daily.equity[0]
    return float(net / abs(drawdown.max_drawdown_money))


# ---------------------------------------------------------------------------
# Trades
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RDistribution:
    """Distribution of realised R-multiples across trades.

    Attributes:
        values: Realised R per trade, chronological.
        count: Number of trades — ``len(values)``, promoted to a named field
            so a reader never has to reconstruct it.
        mean: Mean realised R.
        median: Median realised R.
        stdev: Sample standard deviation of realised R.
        minimum: Worst realised R.
        maximum: Best realised R.
        p10: 10th percentile.
        p25: 25th percentile.
        p75: 75th percentile.
        p90: 90th percentile.
    """

    values: tuple[float, ...] = field(metadata=field_meta("raw"))
    count: int = field(metadata=field_meta("n"))
    mean: float = field(metadata=field_meta("value", n_field="count"))
    median: float = field(metadata=field_meta("value", n_field="count"))
    stdev: float = field(metadata=field_meta("value", n_field="count"))
    minimum: float = field(metadata=field_meta("value", n_field="count"))
    maximum: float = field(metadata=field_meta("value", n_field="count"))
    p10: float = field(metadata=field_meta("value", n_field="count"))
    p25: float = field(metadata=field_meta("value", n_field="count"))
    p75: float = field(metadata=field_meta("value", n_field="count"))
    p90: float = field(metadata=field_meta("value", n_field="count"))


def r_distribution(trades: Sequence[TradeRecord]) -> RDistribution:
    """Summarise the distribution of realised R-multiples across trades.

    Percentiles below the 1st or above the 99th trade of a small sample are
    interpolations near a single observation, not a resolved distribution —
    this function still computes them (see the module docstring: that
    judgment is :mod:`~trading_system.analytics.statistical`'s, not this
    module's), but it always reports ``count`` on the same object so nothing
    reads the percentile without also seeing how thin it is.

    Args:
        trades: Completed trades.

    Returns:
        The distribution.

    Raises:
        ValueError: If ``trades`` is empty, or has exactly one trade (standard
            deviation is undefined for ``n=1``).
    """
    if len(trades) < 2:
        raise ValueError("r_distribution needs at least two trades")
    values = tuple(trade.realized_r for trade in trades)
    n = len(values)
    mean = sum(values) / n
    variance = sum((v - mean) ** 2 for v in values) / (n - 1)
    sorted_values = sorted(values)
    median = (
        sorted_values[n // 2] if n % 2 else (sorted_values[n // 2 - 1] + sorted_values[n // 2]) / 2
    )
    percentiles = quantiles(values, n=100, method="inclusive")
    return RDistribution(
        values=values,
        count=n,
        mean=mean,
        median=median,
        stdev=math.sqrt(variance),
        minimum=sorted_values[0],
        maximum=sorted_values[-1],
        p10=percentiles[9],
        p25=percentiles[24],
        p75=percentiles[74],
        p90=percentiles[89],
    )


@dataclass(frozen=True)
class TradeStats:
    """Trade-level performance summary.

    A "win" is ``net > 0``, a "loss" is ``net < 0``; a breakeven trade
    (``net == 0``) counts toward ``count`` but not toward either bucket, the
    common industry reading of winrate.

    Attributes:
        count: Number of trades.
        winrate: Fraction of trades with ``net > 0``.
        profit_factor: Gross profit over gross loss. ``math.inf`` when there
            are wins and no losses (a real, named edge case, not a division
            silently avoided).
        expectancy_r: Mean realised R across trades.
        avg_win_r: Mean realised R of winning trades, or ``None`` if there are
            none.
        avg_loss_r: Mean realised R of losing trades, or ``None`` if there are
            none.
        payoff_ratio: ``avg_win_r / abs(avg_loss_r)``, or ``None`` if either
            side is missing.
        avg_win_money: Mean ``net`` of winning trades, or ``None``.
        avg_loss_money: Mean ``net`` of losing trades, or ``None``.
        max_consecutive_wins: Longest streak of winning trades, by
            ``closed_at`` order.
        max_consecutive_losses: Longest streak of losing trades, by
            ``closed_at`` order.
        avg_time_in_trade: Mean ``closed_at - opened_at``.
    """

    count: int = field(metadata=field_meta("n"))
    winrate: float = field(metadata=field_meta("value", n_field="count"))
    profit_factor: float = field(metadata=field_meta("value", n_field="count"))
    expectancy_r: float = field(metadata=field_meta("value", n_field="count"))
    avg_win_r: float | None = field(metadata=field_meta("value", n_field="count"))
    avg_loss_r: float | None = field(metadata=field_meta("value", n_field="count"))
    payoff_ratio: float | None = field(metadata=field_meta("value", n_field="count"))
    avg_win_money: Decimal | None = field(metadata=field_meta("value", n_field="count"))
    avg_loss_money: Decimal | None = field(metadata=field_meta("value", n_field="count"))
    max_consecutive_wins: int = field(metadata=field_meta("value", n_field="count"))
    max_consecutive_losses: int = field(metadata=field_meta("value", n_field="count"))
    avg_time_in_trade: timedelta = field(metadata=field_meta("value", n_field="count"))


def _max_streak(flags: Sequence[bool]) -> int:
    """Longest run of consecutive ``True`` in ``flags``."""
    longest = current = 0
    for flag in flags:
        current = current + 1 if flag else 0
        longest = max(longest, current)
    return longest


def trade_stats(trades: Sequence[TradeRecord]) -> TradeStats:
    """Summarise win rate, expectancy, streaks and payoff across trades.

    Args:
        trades: Completed trades, in any order — sorted internally by
            ``closed_at`` for the streak calculation, since streaks are a
            property of trade sequence, not of list order.

    Returns:
        The summary.

    Raises:
        ValueError: If ``trades`` is empty.
    """
    if not trades:
        raise ValueError("trade_stats needs at least one trade")
    ordered = sorted(trades, key=lambda trade: trade.closed_at)
    count = len(ordered)
    wins = [trade for trade in ordered if trade.net > 0]
    losses = [trade for trade in ordered if trade.net < 0]
    winrate = len(wins) / count

    gross_profit = sum((trade.net for trade in wins), Decimal(0))
    gross_loss = sum((-trade.net for trade in losses), Decimal(0))
    if gross_loss == 0:
        profit_factor = math.inf if gross_profit > 0 else math.nan
    else:
        profit_factor = float(gross_profit / gross_loss)

    expectancy_r = sum(trade.realized_r for trade in ordered) / count
    avg_win_r = sum(trade.realized_r for trade in wins) / len(wins) if wins else None
    avg_loss_r = sum(trade.realized_r for trade in losses) / len(losses) if losses else None
    payoff_ratio = (
        avg_win_r / abs(avg_loss_r)
        if avg_win_r is not None and avg_loss_r is not None and avg_loss_r != 0
        else None
    )
    avg_win_money = sum((trade.net for trade in wins), Decimal(0)) / len(wins) if wins else None
    avg_loss_money = (
        sum((trade.net for trade in losses), Decimal(0)) / len(losses) if losses else None
    )

    max_consecutive_wins = _max_streak([trade.net > 0 for trade in ordered])
    max_consecutive_losses = _max_streak([trade.net < 0 for trade in ordered])
    total_duration = sum((trade.closed_at - trade.opened_at for trade in ordered), timedelta())
    avg_time_in_trade = total_duration / count

    return TradeStats(
        count=count,
        winrate=winrate,
        profit_factor=profit_factor,
        expectancy_r=expectancy_r,
        avg_win_r=avg_win_r,
        avg_loss_r=avg_loss_r,
        payoff_ratio=payoff_ratio,
        avg_win_money=avg_win_money,
        avg_loss_money=avg_loss_money,
        max_consecutive_wins=max_consecutive_wins,
        max_consecutive_losses=max_consecutive_losses,
        avg_time_in_trade=avg_time_in_trade,
    )


# ---------------------------------------------------------------------------
# Stability
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StabilityStats:
    """How much the result depends on smoothness, calendar coverage, and one trade.

    Attributes:
        r_squared: R² of a linear fit of daily equity against day index — how
            close the curve is to a straight line.
        n_days: Number of days ``r_squared`` was fit over.
        profitable_months_fraction: Fraction of calendar months with a
            positive month-over-month return.
        n_months: Number of calendar months ``profitable_months_fraction`` was
            computed over.
        best_trade_contribution: Best single trade's ``net`` as a fraction of
            gross turnover — ``sum(abs(trade.net) for trade in trades)`` —
            not of net PnL. Net PnL is unusable as a denominator on a losing
            or breakeven run: it can be zero (division by zero), negative
            (flipping the sign of an otherwise-informative fraction), or
            merely small (inflating the ratio past 100% for a reason that has
            nothing to do with concentration). A gross-turnover denominator
            answers "what share of everything this system did was its single
            best trade," which stays meaningful — and is exactly the question
            that matters — whether the run finished up or down: 20 trades
            that net negative overall, one of them +2.6R, is precisely the
            "held together by (and in this case, in spite of) one trade" case
            this field exists to catch, and a net-PnL denominator would have
            hidden it entirely as ``None``. ``None`` only when every trade's
            ``net`` is exactly zero (turnover itself is zero, 0/0).
        best_trade_position_id: ``position_id`` of the best trade.
        concentrated: Whether ``best_trade_contribution`` exceeds
            :data:`BEST_TRADE_CONCENTRATION_THRESHOLD` — the result holding
            together on one trade.
    """

    r_squared: float = field(metadata=field_meta("value", n_field="n_days"))
    n_days: int = field(metadata=field_meta("n"))
    profitable_months_fraction: float = field(metadata=field_meta("value", n_field="n_months"))
    n_months: int = field(metadata=field_meta("n"))
    best_trade_contribution: float | None = field(metadata=field_meta("value", n_field="n_trades"))
    n_trades: int = field(metadata=field_meta("n"))
    best_trade_position_id: str | None = field(metadata=field_meta("fact"))
    concentrated: bool = field(metadata=field_meta("fact"))


def stability_stats(daily: DailyCurve, trades: Sequence[TradeRecord]) -> StabilityStats:
    """How linear the equity curve is, how many months were profitable, and PnL concentration.

    Args:
        daily: The daily curve.
        trades: Completed trades.

    Returns:
        The summary.

    Raises:
        ValueError: If ``daily`` has fewer than two days, or ``trades`` is
            empty.
    """
    if len(daily.days) < 2:
        raise ValueError("stability_stats needs at least two days")
    if not trades:
        raise ValueError("stability_stats needs at least one trade")

    x = [float(index) for index in range(len(daily.days))]
    y = [float(equity) for equity in daily.equity]
    r_squared = correlation(x, y) ** 2

    months = monthly_returns(daily)
    if months:
        profitable = sum(1 for month in months if month.return_pct > 0)
        profitable_fraction = profitable / len(months)
    else:
        profitable_fraction = math.nan
    n_months = len(months)

    gross_turnover = sum((abs(trade.net) for trade in trades), Decimal(0))
    best_trade = max(trades, key=lambda trade: trade.net)
    contribution = float(best_trade.net / gross_turnover) if gross_turnover > 0 else None
    concentrated = contribution is not None and contribution > BEST_TRADE_CONCENTRATION_THRESHOLD

    return StabilityStats(
        r_squared=r_squared,
        n_days=len(daily.days),
        profitable_months_fraction=profitable_fraction,
        n_months=n_months,
        best_trade_contribution=contribution,
        n_trades=len(trades),
        best_trade_position_id=best_trade.position_id if contribution is not None else None,
        concentrated=concentrated,
    )
