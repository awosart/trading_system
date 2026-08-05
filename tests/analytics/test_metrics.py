"""Metric-by-metric verification against hand-computed reference values.

Every non-trivial expected number in this file is derived independently of
``metrics.py`` in the test itself — either as closed-form arithmetic written
out in a comment, or (for the R-distribution percentiles) via the documented
"inclusive" interpolation formula computed by hand. ``pytest.approx`` is used
wherever the reference involves an irrational (a square root, a fractional
power), the same discipline the rest of this codebase applies to FX-rate
arithmetic.
"""

from datetime import UTC, date, datetime, timedelta
from decimal import Decimal

import pytest

from tests.analytics.conftest import curve_from_returns, point, trade
from trading_system.analytics.metrics import (
    BEST_TRADE_CONCENTRATION_THRESHOLD,
    DailyCurve,
    annual_returns,
    cagr,
    calmar_ratio,
    conditional_value_at_risk,
    daily_curve,
    downside_deviation,
    drawdown_stats,
    mar_ratio,
    measured_periods_per_year,
    monthly_returns,
    omega_ratio,
    r_distribution,
    recovery_factor,
    sharpe_daily,
    simple_returns,
    sortino_daily,
    stability_stats,
    total_return,
    trade_stats,
    ulcer_index,
    value_at_risk,
)
from trading_system.backtest.portfolio import TradeRecord
from trading_system.data.resample import FX_DAY_ORIGIN, DayOrigin, trading_day
from trading_system.data.sessions import AssetClass, TradingCalendar

# ---------------------------------------------------------------------------
# DailyCurve
# ---------------------------------------------------------------------------


class TestDailyCurve:
    def test_it_keeps_the_last_row_of_each_day(self) -> None:
        d1, d2 = date(2024, 1, 1), date(2024, 1, 2)
        curve = [
            point(d1, 100, ts=datetime(2024, 1, 1, 10, tzinfo=UTC)),
            point(d1, 105, ts=datetime(2024, 1, 1, 20, tzinfo=UTC)),
            point(d2, 110, ts=datetime(2024, 1, 2, 10, tzinfo=UTC)),
        ]
        daily = daily_curve(curve)
        assert daily.days == (d1, d2)
        assert daily.equity == (Decimal(105), Decimal(110))

    def test_an_empty_curve_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            daily_curve([])

    def test_a_curve_out_of_chronological_order_is_rejected(self) -> None:
        d1 = date(2024, 1, 1)
        curve = [
            point(d1, 100, ts=datetime(2024, 1, 1, 20, tzinfo=UTC)),
            point(d1, 105, ts=datetime(2024, 1, 1, 10, tzinfo=UTC)),
        ]
        with pytest.raises(ValueError, match="chronolog"):
            daily_curve(curve)


class TestSimpleReturns:
    def test_returns_are_day_over_day_ratios(self) -> None:
        # 110/100 - 1 = 0.10 exactly; 99/110 - 1 = -0.10 exactly.
        daily = DailyCurve(
            days=(date(2024, 1, 1), date(2024, 1, 2), date(2024, 1, 3)),
            equity=(Decimal(100), Decimal(110), Decimal(99)),
            balance=(Decimal(100), Decimal(110), Decimal(99)),
        )
        assert simple_returns(daily) == pytest.approx((0.10, -0.10))


# ---------------------------------------------------------------------------
# Returns
# ---------------------------------------------------------------------------


class TestTotalReturn:
    def test_it_is_the_ratio_of_first_to_last_equity(self) -> None:
        daily = DailyCurve(
            days=(date(2024, 1, 1), date(2024, 1, 2)),
            equity=(Decimal(100), Decimal(150)),
            balance=(Decimal(100), Decimal(150)),
        )
        assert total_return(daily) == pytest.approx(0.5)


class TestCagr:
    def test_it_matches_the_closed_form_growth_rate(self) -> None:
        # 2020-01-01 -> 2024-01-01 spans exactly 1461 days (one leap day,
        # 2020), and 1461 / 365.25 = 4.0 exactly: a clean elapsed-years
        # figure with an ordinary calendar span. Growth 100 -> 200 doubles,
        # so CAGR = 2 ** (1/4) - 1 = sqrt(sqrt(2)) - 1.
        # sqrt(2) = 1.4142135624; sqrt(1.4142135624) = 1.1892071150.
        daily = DailyCurve(
            days=(date(2020, 1, 1), date(2024, 1, 1)),
            equity=(Decimal(100), Decimal(200)),
            balance=(Decimal(100), Decimal(200)),
        )
        result = cagr(daily)
        assert result.years_elapsed == pytest.approx(4.0)
        assert result.value == pytest.approx(0.1892071150, rel=1e-6)


class TestMonthlyReturns:
    def test_month_end_to_month_end_ratios(self) -> None:
        daily = DailyCurve(
            days=(date(2024, 1, 31), date(2024, 2, 29), date(2024, 3, 29)),
            equity=(Decimal(100), Decimal(110), Decimal(121)),
            balance=(Decimal(100), Decimal(110), Decimal(121)),
        )
        result = monthly_returns(daily)
        assert [(r.year, r.month) for r in result] == [(2024, 2), (2024, 3)]
        assert [r.return_pct for r in result] == pytest.approx([0.10, 0.10])


class TestAnnualReturns:
    def test_year_end_to_year_end_ratios(self) -> None:
        daily = DailyCurve(
            days=(date(2023, 12, 31), date(2024, 12, 31), date(2025, 12, 31)),
            equity=(Decimal(100), Decimal(150), Decimal(180)),
            balance=(Decimal(100), Decimal(150), Decimal(180)),
        )
        result = annual_returns(daily)
        assert [r.year for r in result] == [2024, 2025]
        assert [r.return_pct for r in result] == pytest.approx([0.5, 0.2])


# ---------------------------------------------------------------------------
# Risk
# ---------------------------------------------------------------------------


class TestDrawdownStats:
    """DoD: max drawdown on a synthetic curve with two drawdowns of different depth."""

    def _two_drawdown_curve(self) -> DailyCurve:
        # day1: 100 (peak) -> day2: 90 (-10%, trough) -> day3: 100 (recovers)
        # -> day4: 130 (new peak) -> day5: 91 (-30%, trough; 91/130 = 0.7
        # exactly) -> day6: 130 (recovers) -> day7: 140 (new final peak).
        days = tuple(date(2024, 1, d) for d in range(1, 8))
        equity = (
            Decimal(100),
            Decimal(90),
            Decimal(100),
            Decimal(130),
            Decimal(91),
            Decimal(130),
            Decimal(140),
        )
        return DailyCurve(days=days, equity=equity, balance=equity)

    def test_it_finds_both_episodes_and_the_worse_one_as_max(self) -> None:
        daily = self._two_drawdown_curve()
        stats = drawdown_stats(daily)

        assert len(stats.all_drawdowns) == 2
        first, second = stats.all_drawdowns
        assert first.depth_pct == pytest.approx(-0.10)
        assert first.depth_money == Decimal(-10)
        assert first.peak_day == date(2024, 1, 1)
        assert first.trough_day == date(2024, 1, 2)
        assert first.recovery_day == date(2024, 1, 3)
        assert first.duration_days == 1
        assert first.recovery_days == 1

        assert second.depth_pct == pytest.approx(-0.30)
        assert second.depth_money == Decimal(-39)
        assert second.peak_day == date(2024, 1, 4)
        assert second.trough_day == date(2024, 1, 5)
        assert second.recovery_day == date(2024, 1, 6)

        assert stats.max_drawdown is second
        assert stats.max_drawdown_pct == pytest.approx(-0.30)
        assert stats.max_drawdown_money == Decimal(-39)

    def test_an_unrecovered_drawdown_reports_no_recovery(self) -> None:
        days = tuple(date(2024, 1, d) for d in range(1, 4))
        equity = (Decimal(100), Decimal(90), Decimal(80))
        daily = DailyCurve(days=days, equity=equity, balance=equity)
        stats = drawdown_stats(daily)
        assert len(stats.all_drawdowns) == 1
        episode = stats.all_drawdowns[0]
        assert episode.recovery_day is None
        assert episode.recovery_days is None
        assert episode.trough_day == date(2024, 1, 3)

    def test_a_curve_that_never_dips_below_peak_has_no_drawdown_to_report(self) -> None:
        days = tuple(date(2024, 1, d) for d in range(1, 4))
        equity = (Decimal(100), Decimal(110), Decimal(120))
        daily = DailyCurve(days=days, equity=equity, balance=equity)
        with pytest.raises(ValueError, match="never dipped"):
            drawdown_stats(daily)

    def test_fewer_than_two_days_is_rejected(self) -> None:
        daily = DailyCurve(
            days=(date(2024, 1, 1),), equity=(Decimal(100),), balance=(Decimal(100),)
        )
        with pytest.raises(ValueError, match="at least two days"):
            drawdown_stats(daily)


class TestUlcerIndex:
    def test_it_matches_the_root_mean_square_drawdown(self) -> None:
        # peak stays 100 throughout; drawdown_pct per day: 0, -0.10, 0.
        # mean squared = 0.01 / 3; ulcer = sqrt(0.01 / 3) = 0.0577350269.
        days = tuple(date(2024, 1, d) for d in range(1, 4))
        equity = (Decimal(100), Decimal(90), Decimal(100))
        daily = DailyCurve(days=days, equity=equity, balance=equity)
        result = ulcer_index(daily)
        assert result.value == pytest.approx(0.0577350269, rel=1e-6)
        assert result.n == 3


class TestValueAtRiskAndCVaR:
    def test_they_match_the_hand_ranked_tail(self) -> None:
        # Hand-picked returns, sorted ascending:
        # [-0.30, -0.20, -0.10, -0.05, 0.01, 0.02, 0.03, 0.05, 0.08, 0.15]
        # n=10. At 95%: index = ceil(0.05*10) - 1 = 0 -> VaR = -0.30,
        # CVaR = mean(sorted[:1]) = -0.30.
        # At 80%: index = ceil(0.20*10) - 1 = 1 -> VaR = -0.20,
        # CVaR = mean(sorted[:2]) = mean(-0.30, -0.20) = -0.25.
        returns = [
            Decimal("0.05"),
            Decimal("-0.10"),
            Decimal("0.02"),
            Decimal("-0.20"),
            Decimal("0.15"),
            Decimal("-0.05"),
            Decimal("0.08"),
            Decimal("-0.30"),
            Decimal("0.01"),
            Decimal("0.03"),
        ]
        daily = curve_from_returns(returns)

        var95 = value_at_risk(daily, confidence=0.95)
        assert var95.value == pytest.approx(-0.30)
        assert var95.n == 10

        var80 = value_at_risk(daily, confidence=0.80)
        assert var80.value == pytest.approx(-0.20)

        cvar80 = conditional_value_at_risk(daily, confidence=0.80)
        assert cvar80.value == pytest.approx(-0.25)
        assert cvar80.n == 10


class TestDownsideDeviation:
    def test_it_only_counts_shortfalls_below_the_mar(self) -> None:
        # returns = [0.10, -0.10, 0.10, -0.10], mar=0.
        # downside terms: 0, 0.01, 0, 0.01 -> mean = 0.005.
        # periods_per_year=4 -> sqrt(0.005 * 4) = sqrt(0.02) = 0.1414213562.
        returns = [Decimal("0.10"), Decimal("-0.10"), Decimal("0.10"), Decimal("-0.10")]
        daily = curve_from_returns(returns)
        result = downside_deviation(daily, mar=0.0, periods_per_year=4.0)
        assert result.value == pytest.approx(0.1414213562, rel=1e-6)
        assert result.n == 4


class TestOmegaRatio:
    def test_it_is_gains_over_losses_relative_to_the_threshold(self) -> None:
        # returns = [0.10, -0.05, 0.20, -0.10]; gains = 0.30, losses = 0.15.
        returns = [Decimal("0.10"), Decimal("-0.05"), Decimal("0.20"), Decimal("-0.10")]
        daily = curve_from_returns(returns)
        result = omega_ratio(daily, threshold=0.0)
        assert result.value == pytest.approx(2.0)
        assert result.n == 4

    def test_it_is_undefined_with_no_losing_returns(self) -> None:
        returns = [Decimal("0.10"), Decimal("0.05")]
        daily = curve_from_returns(returns)
        with pytest.raises(ValueError, match="undefined"):
            omega_ratio(daily)


# ---------------------------------------------------------------------------
# Risk-adjusted
# ---------------------------------------------------------------------------


class TestSharpeDaily:
    def test_it_matches_the_hand_computed_ratio(self) -> None:
        # returns = [0.04, 0.00, -0.02, 0.02], mean = 0.01.
        # deviations: 0.03, -0.01, -0.03, 0.01; squares sum = 0.0020.
        # sample variance = 0.0020 / 3; stdev = sqrt(2/3000).
        # mean / stdev = 0.01 * sqrt(1500) = sqrt(0.15) = 0.3872983346.
        # periods_per_year=4 -> annualised = sqrt(0.15) * 2 = 0.7745966692.
        returns = [Decimal("0.04"), Decimal("0.00"), Decimal("-0.02"), Decimal("0.02")]
        daily = curve_from_returns(returns)
        result = sharpe_daily(daily, risk_free_rate=0.0, periods_per_year=4.0)
        assert result.value == pytest.approx(0.7745966692, rel=1e-6)
        assert result.risk_free_rate == 0.0
        assert result.periods_per_year == 4.0
        assert result.n_periods == 4

    def test_zero_variance_is_rejected(self) -> None:
        returns = [Decimal("0.01"), Decimal("0.01"), Decimal("0.01")]
        daily = curve_from_returns(returns)
        with pytest.raises(ValueError, match="zero variance"):
            sharpe_daily(daily, periods_per_year=4.0)

    def test_fewer_than_two_returns_is_rejected(self) -> None:
        daily = curve_from_returns([Decimal("0.01")])
        with pytest.raises(ValueError, match="at least two"):
            sharpe_daily(daily, periods_per_year=4.0)


class TestSortinoDaily:
    def test_it_matches_the_hand_computed_ratio(self) -> None:
        # Same returns as the Sharpe test: mean = 0.01.
        # downside terms: 0, 0, 0.0004, 0 -> mean = 0.0001.
        # downside deviation = sqrt(0.0001 * 4) = 0.02.
        # sortino = (0.01 * 4) / 0.02 = 2.0 exactly.
        returns = [Decimal("0.04"), Decimal("0.00"), Decimal("-0.02"), Decimal("0.02")]
        daily = curve_from_returns(returns)
        result = sortino_daily(daily, risk_free_rate=0.0, periods_per_year=4.0, mar=0.0)
        assert result.value == pytest.approx(2.0, rel=1e-6)
        assert result.downside_deviation == pytest.approx(0.02, rel=1e-6)
        assert result.n_periods == 4


class TestCalmarAndMarRatio:
    def test_calmar_reduces_to_mar_when_the_window_covers_the_whole_curve(self) -> None:
        daily = DailyCurve(
            days=(date(2020, 1, 1), date(2021, 1, 1), date(2022, 1, 1), date(2024, 1, 1)),
            equity=(Decimal(100), Decimal(80), Decimal(100), Decimal(200)),
            balance=(Decimal(100), Decimal(80), Decimal(100), Decimal(200)),
        )
        assert calmar_ratio(daily, window_years=10.0) == pytest.approx(mar_ratio(daily))

    def test_mar_ratio_matches_cagr_over_drawdown(self) -> None:
        # 2020-01-01 -> 2024-01-01: CAGR = sqrt(sqrt(2)) - 1 = 0.1892071150
        # (same span as TestCagr). The only drawdown is 100 -> 80 = -20%.
        # mar = 0.1892071150 / 0.20 = 0.9460355750.
        daily = DailyCurve(
            days=(date(2020, 1, 1), date(2021, 1, 1), date(2022, 1, 1), date(2024, 1, 1)),
            equity=(Decimal(100), Decimal(80), Decimal(100), Decimal(200)),
            balance=(Decimal(100), Decimal(80), Decimal(100), Decimal(200)),
        )
        assert mar_ratio(daily) == pytest.approx(0.9460355750, rel=1e-6)

    def test_a_short_window_excludes_an_old_drawdown(self) -> None:
        # An ancient, deep drawdown (2020, -50%) followed by flat recovery,
        # then a recent, shallow one (2023, -10%) close to the curve's end.
        days = (
            date(2020, 1, 1),
            date(2020, 6, 1),
            date(2021, 1, 1),
            date(2023, 6, 1),
            date(2023, 9, 1),
            date(2024, 1, 1),
        )
        equity = (
            Decimal(100),
            Decimal(50),
            Decimal(100),
            Decimal(200),
            Decimal(180),
            Decimal(220),
        )
        daily = DailyCurve(days=days, equity=equity, balance=equity)

        # Hand-traced against calmar_ratio's own window-selection loop: walking
        # backward from the last day (2024-01-01) with a 1-year cutoff of
        # 365.25 days, the 2021-01-01 row is the first whose distance from the
        # last day (1095 days) exceeds the cutoff, so the window starts at the
        # row after it: index 3 (2023-06-01) onward.
        window = DailyCurve(days=days[3:], equity=equity[3:], balance=equity[3:])
        expected = cagr(window).value / abs(drawdown_stats(window).max_drawdown_pct)

        assert calmar_ratio(daily, window_years=1.0) == pytest.approx(expected)
        # And that window's -10% drawdown is materially different from the
        # whole curve's -50% one: the short window must not have seen 2020.
        assert calmar_ratio(daily, window_years=1.0) != pytest.approx(mar_ratio(daily), rel=1e-2)


class TestRecoveryFactor:
    def test_it_is_net_profit_over_worst_drawdown_money(self) -> None:
        # Same two-drawdown curve as TestDrawdownStats: net = 140 - 100 = 40,
        # worst drawdown money = -39. 40 / 39 = 1.0256410256.
        days = tuple(date(2024, 1, d) for d in range(1, 8))
        equity = (
            Decimal(100),
            Decimal(90),
            Decimal(100),
            Decimal(130),
            Decimal(91),
            Decimal(130),
            Decimal(140),
        )
        daily = DailyCurve(days=days, equity=equity, balance=equity)
        assert recovery_factor(daily) == pytest.approx(1.0256410256, rel=1e-6)


# ---------------------------------------------------------------------------
# Trades
# ---------------------------------------------------------------------------


def _r_distribution_trades() -> list[TradeRecord]:
    values = [1.0, -0.5, 2.0, -1.0, 0.5, 1.5, -0.5, 3.0, 0.0, -2.0]
    base = datetime(2024, 1, 1, tzinfo=UTC)
    return [
        trade(
            position_id=f"t{i}",
            opened_at=base + timedelta(days=i),
            closed_at=base + timedelta(days=i, hours=4),
            net=100.0 * r,
            realized_r=r,
        )
        for i, r in enumerate(values)
    ]


class TestRDistribution:
    def test_central_tendency_and_spread_match_hand_computation(self) -> None:
        # sum = 4.0, n=10 -> mean = 0.4.
        # sorted: [-2,-1,-0.5,-0.5,0,0.5,1,1.5,2,3]; median = (0+0.5)/2 = 0.25.
        # deviations^2 sum = 20.40 (see module comment history); sample
        # variance = 20.40/9 = 2.2666667; stdev = sqrt(2.2666667) ~ 1.505546.
        dist = r_distribution(_r_distribution_trades())
        assert dist.count == 10
        assert dist.mean == pytest.approx(0.4)
        assert dist.median == pytest.approx(0.25)
        assert dist.minimum == pytest.approx(-2.0)
        assert dist.maximum == pytest.approx(3.0)
        assert dist.stdev == pytest.approx(1.505546, rel=1e-4)

    def test_percentiles_match_the_inclusive_interpolation_formula(self) -> None:
        # "inclusive" (Type 7) percentile: rank = q * (n - 1) on the sorted
        # sample, linearly interpolated between the bracketing order
        # statistics. sorted = [-2,-1,-0.5,-0.5,0,0.5,1,1.5,2,3], n=10.
        # p10: rank=0.9  -> between [-2,-1],   frac=0.9 -> -2 + 0.9*1    = -1.1
        # p25: rank=2.25 -> between [-0.5,-0.5],frac=.25 -> -0.5         = -0.5
        # p75: rank=6.75 -> between [1, 1.5],   frac=0.75 -> 1 + 0.75*.5 = 1.375
        # p90: rank=8.1  -> between [2, 3],     frac=0.1 -> 2 + 0.1*1   = 2.1
        dist = r_distribution(_r_distribution_trades())
        assert dist.p10 == pytest.approx(-1.1)
        assert dist.p25 == pytest.approx(-0.5)
        assert dist.p75 == pytest.approx(1.375)
        assert dist.p90 == pytest.approx(2.1)

    def test_fewer_than_two_trades_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="at least two"):
            r_distribution(_r_distribution_trades()[:1])


def _streak_trades() -> list[TradeRecord]:
    # In closed_at order: win, win, loss, loss, loss, win.
    base = datetime(2024, 1, 1, tzinfo=UTC)
    nets_and_r = [
        (100.0, 1.0),
        (200.0, 2.0),
        (-50.0, -1.0),
        (-100.0, -2.0),
        (-200.0, -3.0),
        (300.0, 4.0),
    ]
    trades: list[TradeRecord] = []
    for i, (net, r) in enumerate(nets_and_r):
        opened = base + timedelta(days=i)
        trades.append(
            trade(
                position_id=f"s{i}",
                opened_at=opened,
                closed_at=opened + timedelta(days=1),
                net=net,
                realized_r=r,
            )
        )
    return trades


class TestTradeStats:
    def test_every_figure_matches_hand_computation(self) -> None:
        # wins: 100, 200, 300 (3); losses: -50, -100, -200 (3); count=6.
        # winrate = 3/6 = 0.5.
        # gross_profit=600, gross_loss=350 -> profit_factor=600/350=1.7142857.
        # expectancy_r = (1+2-1-2-3+4)/6 = 1/6 = 0.1666667.
        # avg_win_r=(1+2+4)/3=2.3333333; avg_loss_r=(-1-2-3)/3=-2.0.
        # payoff_ratio = 2.3333333/2.0 = 1.1666667.
        # avg_win_money=(100+200+300)/3=200; avg_loss_money=(-50-100-200)/3=-116.6666667.
        # order is win,win,loss,loss,loss,win -> max_consecutive_wins=2,
        # max_consecutive_losses=3.
        # every trade lasts exactly 1 day -> avg_time_in_trade = 1 day.
        stats = trade_stats(_streak_trades())
        assert stats.count == 6
        assert stats.winrate == pytest.approx(0.5)
        assert stats.profit_factor == pytest.approx(600 / 350)
        assert stats.expectancy_r == pytest.approx(1 / 6)
        assert stats.avg_win_r == pytest.approx(7 / 3)
        assert stats.avg_loss_r == pytest.approx(-2.0)
        assert stats.payoff_ratio == pytest.approx((7 / 3) / 2.0)
        assert stats.avg_win_money == Decimal(200)
        assert stats.avg_loss_money == pytest.approx(Decimal("-116.6666666666666666666666667"))
        assert stats.max_consecutive_wins == 2
        assert stats.max_consecutive_losses == 3
        assert stats.avg_time_in_trade == timedelta(days=1)

    def test_profit_factor_is_infinite_with_no_losses(self) -> None:
        base = datetime(2024, 1, 1, tzinfo=UTC)
        trades = [
            trade(opened_at=base, closed_at=base + timedelta(days=1), net=100.0, realized_r=1.0),
            trade(
                opened_at=base + timedelta(days=1),
                closed_at=base + timedelta(days=2),
                net=50.0,
                realized_r=0.5,
            ),
        ]
        stats = trade_stats(trades)
        assert stats.profit_factor == float("inf")
        assert stats.avg_loss_r is None
        assert stats.payoff_ratio is None

    def test_empty_trades_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="at least one trade"):
            trade_stats([])


# ---------------------------------------------------------------------------
# Stability
# ---------------------------------------------------------------------------


class TestStabilityStats:
    def test_a_perfectly_linear_curve_has_r_squared_one(self) -> None:
        days = (
            date(2024, 1, 29),
            date(2024, 1, 30),
            date(2024, 1, 31),
            date(2024, 2, 1),
            date(2024, 2, 2),
        )
        equity = (Decimal(100), Decimal(110), Decimal(120), Decimal(130), Decimal(140))
        daily = DailyCurve(days=days, equity=equity, balance=equity)
        trades = _streak_trades()

        result = stability_stats(daily, trades)
        assert result.r_squared == pytest.approx(1.0)
        assert result.n_days == 5
        # Only one month-over-month transition exists (Jan -> Feb: 120 -> 140,
        # a gain), so profitable_months_fraction is 1/1.
        assert result.n_months == 1
        assert result.profitable_months_fraction == pytest.approx(1.0)

    def test_best_trade_contribution_and_concentration_flag(self) -> None:
        # total_net = 100+200-50-100-200+300 = 250; best trade nets 300,
        # i.e. 300/250 = 1.2 of the total (>100% is correct and expected:
        # the other trades were net losers, so the best trade contributes
        # more than the whole final result).
        days = (date(2024, 1, 1), date(2024, 1, 2))
        equity = (Decimal(1000), Decimal(1010))
        daily = DailyCurve(days=days, equity=equity, balance=equity)
        trades = _streak_trades()

        result = stability_stats(daily, trades)
        assert result.n_trades == 6
        assert result.best_trade_contribution == pytest.approx(1.2)
        assert result.best_trade_position_id == "s5"
        assert BEST_TRADE_CONCENTRATION_THRESHOLD == 0.30
        assert result.concentrated is True

    def test_a_non_positive_total_net_reports_no_contribution(self) -> None:
        days = (date(2024, 1, 1), date(2024, 1, 2))
        equity = (Decimal(1000), Decimal(900))
        daily = DailyCurve(days=days, equity=equity, balance=equity)
        base = datetime(2024, 1, 1, tzinfo=UTC)
        trades = [
            trade(opened_at=base, closed_at=base + timedelta(days=1), net=-100.0, realized_r=-1.0),
            trade(
                opened_at=base + timedelta(days=1),
                closed_at=base + timedelta(days=2),
                net=-50.0,
                realized_r=-0.5,
            ),
        ]
        result = stability_stats(daily, trades)
        assert result.best_trade_contribution is None
        assert result.best_trade_position_id is None
        assert result.concentrated is False

    def test_fewer_than_two_days_is_rejected(self) -> None:
        daily = DailyCurve(
            days=(date(2024, 1, 1),), equity=(Decimal(100),), balance=(Decimal(100),)
        )
        with pytest.raises(ValueError, match="at least two days"):
            stability_stats(daily, _streak_trades())

    def test_empty_trades_is_rejected(self) -> None:
        days = (date(2024, 1, 1), date(2024, 1, 2))
        equity = (Decimal(100), Decimal(110))
        daily = DailyCurve(days=days, equity=equity, balance=equity)
        with pytest.raises(ValueError, match="at least one trade"):
            stability_stats(daily, [])


# ---------------------------------------------------------------------------
# Annualisation factor: measured, never asserted, no branch on asset class.
# ---------------------------------------------------------------------------


class TestMeasuredPeriodsPerYear:
    """DoD: measured correctly on an FX series (~257) and a crypto one (~365)."""

    def test_fx_measures_to_about_257_trading_days_a_year(self) -> None:
        # trading_day() under FX_DAY_ORIGIN prints five labels a week — FX
        # never trades Fri 17:00 NY -> Sun 17:00 NY — so two years of open
        # hours should measure to noticeably less than 365/year and close to
        # the ~252-261 range real calendars produce, never asserted as a
        # constant inside metrics.py itself.
        calendar = TradingCalendar(AssetClass.FX)
        start = datetime(2022, 1, 1, tzinfo=UTC)
        hours = [start + timedelta(hours=h) for h in range(24 * 365 * 2)]
        open_days = sorted({trading_day(ts, FX_DAY_ORIGIN) for ts in hours if calendar.is_open(ts)})
        equity = tuple(Decimal(100) for _ in open_days)
        daily = DailyCurve(days=tuple(open_days), equity=equity, balance=equity)

        factor = measured_periods_per_year(daily)
        assert 250 <= factor <= 262

    def test_crypto_measures_to_about_365_trading_days_a_year(self) -> None:
        # Same function, same code path — crypto trades every calendar day,
        # so the day-label count alone (not a branch on asset class) yields
        # ~365/year.
        origin = DayOrigin(tz="UTC")
        start = datetime(2022, 1, 1, tzinfo=UTC)
        hours = [start + timedelta(hours=h) for h in range(24 * 365 * 2)]
        days = sorted({trading_day(ts, origin) for ts in hours})
        equity = tuple(Decimal(100) for _ in days)
        daily = DailyCurve(days=tuple(days), equity=equity, balance=equity)

        factor = measured_periods_per_year(daily)
        assert 363 <= factor <= 366
