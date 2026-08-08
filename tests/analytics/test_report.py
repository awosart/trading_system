"""The HTML report: what it renders, and what it refuses to render silently.

These tests are about the page's claims, not its styling. A number that the
sample cannot support has to arrive marked; a stitched walk-forward curve has to
arrive labelled as a procedure rather than as an account; a quality score with
no variation has to arrive as a stated absence rather than as a dash.
"""

from dataclasses import replace
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from trading_system.analytics.metrics import daily_curve, sharpe_daily, trade_stats
from trading_system.analytics.report import (
    RELIABLE_TRADES,
    CostSensitivityPoint,
    CostSensitivityReport,
    FoldSegment,
    FoldSelection,
    ReportSource,
    SearchSummary,
    SourceKind,
    all_metric_rows,
    build,
    build_comparison,
    export_metrics,
    metric_rows,
    money_summary,
    sharpe_family,
    write,
)
from trading_system.backtest.portfolio import EquityPoint, TradeRecord
from trading_system.core.types import Price, Side
from trading_system.validation.report import FoldReport
from trading_system.validation.splitting import FoldWindow

START = datetime(2024, 1, 1, tzinfo=UTC)


def curve(n: int, *, start: float = 100_000.0, step: float = 25.0) -> list[EquityPoint]:
    """A rising curve of ``n`` daily points."""
    points = []
    for index in range(n):
        ts = START + timedelta(days=index)
        equity = Decimal(str(start + step * index))
        points.append(
            EquityPoint(
                ts=ts,
                day=ts.date(),
                balance=equity,
                equity=equity,
                realized=Decimal(0),
                unrealized=Decimal(0),
                commission_paid=Decimal(0),
                swap_paid=Decimal(0),
                open_positions=0,
            )
        )
    return points


def trades(n: int, *, quality: float | None = None) -> list[TradeRecord]:
    """``n`` trades alternating win and loss; ``quality`` fixes the score."""
    made = []
    for index in range(n):
        ts = START + timedelta(days=index)
        r = 1.0 if index % 2 else -0.8
        made.append(
            TradeRecord(
                position_id=f"p{index}",
                symbol="EURUSD",
                strategy_id="demo",
                side=Side.BUY if index % 2 else Side.SELL,
                size=Decimal("1"),
                opened_at=ts,
                closed_at=ts + timedelta(hours=6),
                entry_price=Price(1.1),
                quality=quality if quality is not None else (0.55 if index % 3 else 0.70),
                initial_risk_distance=0.001,
                gross=Decimal("0"),
                commission=Decimal("0"),
                swap=Decimal("0"),
                net=Decimal(str(round(r * 100, 2))),
                realized_r=r,
                legs=1,
            )
        )
    return made


def run_source(n_trades: int, **kwargs: object) -> ReportSource:
    """A flat-run source with ``n_trades`` trades."""
    quality = kwargs.pop("quality", None)
    assert not kwargs
    return ReportSource(
        kind=SourceKind.RUN,
        label="demo-run",
        curve=curve(120),
        trades=trades(n_trades, quality=quality),  # type: ignore[arg-type]
        streams={},
        money_is_real=True,
    )


class TestSampleSizeMarkingComesFromTheMetadataNotFromAList:
    """A metric added upstream must arrive flagged without this module being edited."""

    def test_a_thin_sample_marks_every_derived_statistic(self) -> None:
        rows = metric_rows("trades", trade_stats(trades(10)))
        derived = [row for row in rows if row.sample is not None and row.name != "count"]
        assert derived
        assert all(row.unreliable for row in derived)

    def test_a_large_sample_marks_none_of_them(self) -> None:
        rows = metric_rows("trades", trade_stats(trades(RELIABLE_TRADES + 20)))
        assert not any(row.unreliable for row in rows)

    def test_every_row_either_names_its_sample_or_declares_it_has_none(self) -> None:
        rows = all_metric_rows(curve(200), trades(150))
        assert rows
        for row in rows:
            assert row.sample is None or row.sample >= 0

    def test_raw_sample_fields_are_not_rendered_as_table_cells(self) -> None:
        # r_distribution.values is the whole sample; a tuple of 150 floats is
        # not a cell, and rendering it would push the real numbers off screen.
        rows = all_metric_rows(curve(200), trades(150))
        assert not any(row.name == "values" for row in rows)

    def test_a_non_dataclass_is_refused_rather_than_rendered_empty(self) -> None:
        with pytest.raises(TypeError):
            metric_rows("nonsense", {"count": 3})


class TestASingleRunPage:
    """The flat-run report."""

    def test_every_section_renders(self) -> None:
        html = build(run_source(120)).html
        for heading in (
            "Equity",
            "Metrics",
            "Distribution of R",
            "Monthly",
            "Excursions",
            "Quality against realised R",
            "Attribution",
        ):
            assert f"<h2>{heading}</h2>" in html

    def test_the_charts_are_actually_drawn_not_escaped_into_text(self) -> None:
        # The failure this guards is real and silent: Jinja's autoescape turns
        # a figure's HTML into visible angle brackets, and the page still
        # renders — just with no charts on it.
        html = build(run_source(120)).html
        assert html.count('class="plotly-graph-div') >= 3
        assert "&lt;div" not in html

    def test_a_thin_run_says_so_at_the_top_of_the_metrics(self) -> None:
        html = build(run_source(20)).html
        assert f"below the {RELIABLE_TRADES}" in html

    def test_a_run_without_bars_says_excursions_were_not_measured(self) -> None:
        html = build(run_source(40)).html
        assert "Not measured" in html

    def test_the_page_is_self_contained(self) -> None:
        # The file has to open from disk with no network months later, so
        # nothing may be fetched: no external script, stylesheet or font.
        # Checked as tags rather than by searching for a CDN hostname — the
        # bundled plotly source contains "cdn.plot.ly" as the default topojson
        # location, which is only ever requested by geo charts, and this report
        # draws none.
        html = build(run_source(40)).html
        assert "<script src=" not in html
        assert "<link " not in html
        assert "@import" not in html

    def test_metrics_export_to_json(self, tmp_path: object) -> None:
        rendered = build(run_source(40))
        path = export_metrics(rendered, tmp_path / "m.json")  # type: ignore[operator]
        assert path.exists()
        assert '"metrics"' in path.read_text()

    def test_the_page_writes_to_disk(self, tmp_path: object) -> None:
        path = write(build(run_source(40)), tmp_path / "r.html")  # type: ignore[operator]
        assert path.exists() and path.stat().st_size > 100_000


class TestConstantQualityIsStatedNotDashed:
    """The behaviour settled before implementation: say it, do not hide it."""

    def test_the_page_explains_why_there_is_no_correlation(self) -> None:
        html = build(run_source(60, quality=0.55)).html
        assert "constant at 0.55" in html
        assert "never held" in html

    def test_the_section_is_present_rather_than_skipped(self) -> None:
        html = build(run_source(60, quality=0.55)).html
        assert "<h2>Quality against realised R</h2>" in html

    def test_two_levels_render_a_group_table_and_a_gap(self) -> None:
        html = build(run_source(120)).html
        assert "Top minus bottom" in html
        assert "95% interval" in html


class TestAWalkForwardPageCannotBeMistakenForOneStrategy:
    """The four structural defences, checked on the rendered page."""

    def source(self) -> ReportSource:
        segments = tuple(
            FoldSegment(
                index=index,
                start=START + timedelta(days=30 * index),
                end=START + timedelta(days=30 * (index + 1)),
                parameters={"ema_period_50": 35 + index, "multiple": 2},
                trade_count=12 + index,
                expectancy_r=0.1 - 0.05 * index,
                insufficient=index == 3,
            )
            for index in range(4)
        )
        return ReportSource(
            kind=SourceKind.WALKFORWARD,
            label="wf-demo",
            curve=curve(120),
            trades=trades(120),
            streams={},
            money_is_real=False,
            folds=(),
            segments=segments,
            is_oos_pairs=((0.2, 0.1), (0.3, -0.05), (0.1, 0.02), (None, None)),
        )

    def test_the_axis_is_multiples_not_money(self) -> None:
        html = build(self.source()).html
        assert "Multiple of starting point" in html
        assert "Equity (account currency)" not in html

    def test_the_page_says_it_is_a_procedure(self) -> None:
        html = build(self.source()).html
        assert "procedure, not one strategy" in html

    def test_the_seams_carry_the_parameters_that_segment_traded_on(self) -> None:
        html = build(self.source()).html
        assert "ema_period_50=35" in html
        assert "ema_period_50=38" in html

    def test_no_ratio_between_is_and_oos_appears_anywhere(self) -> None:
        html = build(self.source()).html
        assert "no ratio is taken" in html

    def test_a_baseline_that_traded_nothing_does_not_read_as_a_search_on_nothing(self) -> None:
        """The two in-sample samples are different, and the page has to say which is which.

        A fold whose in-sample run closed no trades was still optimised over a
        real sample: the run uses the strategy file's own parameters, the search
        evaluates its own trials afterwards. Measured on the real
        ``channel-breakout-h4`` fold 5 — baseline zero trades, 75 of 125 trials
        scored — so the page must show both numbers rather than one.
        """
        source = replace(
            self.source(),
            folds=(
                FoldReport(
                    index=0,
                    is_window=FoldWindow(
                        START, START + timedelta(days=1), START + timedelta(days=2)
                    ),
                    oos_window=FoldWindow(
                        START + timedelta(days=3),
                        START + timedelta(days=4),
                        START + timedelta(days=5),
                    ),
                    is_trade_count=0,
                    oos_trade_count=11,
                    is_expectancy_r=None,
                    oos_expectancy_r=-0.36,
                    is_sortino=None,
                    oos_sortino=None,
                    is_boundary_residual=None,
                    oos_boundary_residual=None,
                    drain_truncated=0,
                    rejections={},
                    degradations={},
                    exit_drops={},
                    entry_drops={},
                    signal_drops={},
                    atr_unavailable_fraction=0.0,
                    insufficient=False,
                ),
            ),
            segments=(
                FoldSegment(
                    index=0,
                    start=START,
                    end=START + timedelta(days=5),
                    parameters={"ema_period": 100},
                    trade_count=11,
                    expectancy_r=-0.36,
                    insufficient=False,
                    selection=FoldSelection(
                        parameters={"ema_period": 100},
                        n_trials=125,
                        n_scored=75,
                        selected_score=6.36,
                    ),
                    baseline_is_traded=False,
                ),
            ),
        )
        html = build(source).html
        assert "IS baseline trades" in html
        assert "Search scored / trials" in html
        assert "75 / 125" in html
        assert "template found none" in html
        assert "not the sample the search worked on" in html

    def test_the_fold_charts_are_present(self) -> None:
        html = build(self.source()).html
        assert html.count('class="plotly-graph-div') >= 5

    def test_a_flat_run_gets_neither_fold_chart(self) -> None:
        rendered = build(run_source(40))
        assert "expectancy by fold" not in rendered.html


class TestComparison:
    """Several runs on one page."""

    def sources(self, count: int) -> list[ReportSource]:
        return [
            ReportSource(
                kind=SourceKind.RUN,
                label=f"strategy-{index}",
                curve=curve(80 + 10 * index, start=100_000.0, step=10.0 * (index + 1)),
                trades=trades(30 + 20 * index),
                streams={},
                money_is_real=True,
            )
            for index in range(count)
        ]

    def test_all_curves_land_on_one_chart(self) -> None:
        html = build_comparison(self.sources(4)).html
        for index in range(4):
            assert f"strategy-{index}" in html
        assert html.count('class="plotly-graph-div') == 1

    def test_curves_are_normalised_because_the_accounts_are_not_comparable(self) -> None:
        html = build_comparison(self.sources(2)).html
        assert "normalised to their own starting points" in html

    def test_metrics_line_up_one_column_per_run(self) -> None:
        rendered = build_comparison(self.sources(3))
        assert set(rendered.metrics) == {"strategy-0", "strategy-1", "strategy-2"}
        assert "<th>strategy-2</th>" in rendered.html

    def test_a_metric_missing_from_one_run_leaves_a_gap_not_a_shifted_row(self) -> None:
        # One source with no trades at all: its trade metrics do not exist, and
        # the row must still align with the other source's.
        rich = self.sources(1)[0]
        empty = ReportSource(
            kind=SourceKind.RUN,
            label="no-trades",
            curve=curve(40),
            trades=[],
            streams={},
            money_is_real=True,
        )
        rendered = build_comparison([rich, empty])
        assert "—" in rendered.html
        assert "no-trades" in rendered.metrics


class TestMoneySummary:
    """The pure function, independent of the page it feeds."""

    def test_a_flat_run_reports_real_money_with_no_hedge(self) -> None:
        summary = money_summary(curve(60), trades(20), hypothetical=False)
        assert summary is not None
        assert summary.hypothetical is False
        assert summary.starting_equity == Decimal("100000.0")
        assert summary.trade_count == 20

    def test_pnl_and_drawdown_are_computed_the_same_way_regardless_of_hypothetical(self) -> None:
        real = money_summary(curve(60), [], hypothetical=False)
        stitched = money_summary(curve(60), [], hypothetical=True)
        assert real is not None and stitched is not None
        assert real.pnl_money == stitched.pnl_money
        assert real.pnl_pct == stitched.pnl_pct
        assert real.hypothetical is False
        assert stitched.hypothetical is True

    def test_an_empty_curve_returns_none_rather_than_a_zeroed_summary(self) -> None:
        assert money_summary([], [], hypothetical=False) is None

    def test_a_monotonic_curve_reports_zero_drawdown_rather_than_raising(self) -> None:
        # drawdown_stats raises on a curve that never dipped below its peak —
        # curve() is exactly that shape. The summary must absorb this, not
        # propagate it: this is a page that has to render regardless.
        summary = money_summary(curve(40), [], hypothetical=False)
        assert summary is not None
        assert summary.max_drawdown_money == Decimal(0)
        assert summary.max_drawdown_pct == 0.0

    def test_pnl_percent_matches_the_money_figure(self) -> None:
        summary = money_summary(curve(60, start=100_000.0, step=100.0), [], hypothetical=False)
        assert summary is not None
        expected_pct = float(summary.pnl_money / summary.starting_equity)
        assert summary.pnl_pct == pytest.approx(expected_pct)


class TestTheMoneyBlockCaptionsItselfCorrectly:
    """The top-of-page block must not let a stitched curve read as a real account."""

    def test_a_flat_run_gets_no_hypothetical_warning(self) -> None:
        html = build(run_source(60)).html
        assert "Hypothetical account" not in html
        assert "Starting capital" in html

    def test_a_walk_forward_curve_is_captioned_hypothetical(self) -> None:
        source = replace(
            ReportSource(
                kind=SourceKind.WALKFORWARD,
                label="wf-money",
                curve=curve(60),
                trades=trades(30),
                streams={},
                money_is_real=False,
            ),
        )
        html = build(source).html
        assert "Hypothetical account, not a simulated one" in html
        assert "Nominal starting" in html
        assert (
            "no run ever actually held this balance continuously" in html.lower()
            or "no run ever actually held this balance" in html
        )

    def test_the_block_is_present_for_both_kinds_not_only_the_flat_run(self) -> None:
        # Point 1's resolution: omitting money for WF would lose the "simple
        # money at a glance" value that was explicitly asked for.
        source = ReportSource(
            kind=SourceKind.WALKFORWARD,
            label="wf-money-2",
            curve=curve(60),
            trades=trades(30),
            streams={},
            money_is_real=False,
        )
        html = build(source).html
        assert "money-grid" in html
        assert "Max drawdown" in html


class TestSharpeFamily:
    """PSR and DSR, each independently definable or not."""

    def test_dsr_is_absent_with_a_reason_when_no_search_was_recorded(self) -> None:
        family = sharpe_family(curve(60), None)
        assert family is not None
        assert family.dsr is None
        assert family.dsr_reason is not None
        assert "no parameter search" in family.dsr_reason

    def test_dsr_is_computed_when_n_trials_is_given(self) -> None:
        family = sharpe_family(curve(60), 1000)
        assert family is not None
        assert family.dsr is not None
        assert family.dsr.n_trials == 1000
        assert family.dsr_reason is None

    def test_dsr_with_more_trials_is_never_higher_than_with_fewer(self) -> None:
        few = sharpe_family(curve(60), 2)
        many = sharpe_family(curve(60), 1000)
        assert few is not None and many is not None
        assert few.dsr is not None and many.dsr is not None
        assert many.dsr.value <= few.dsr.value

    def test_psr_and_dsr_agree_when_n_trials_is_one(self) -> None:
        # DSR's own docstring: n_trials <= 1 makes expected_max_sr zero, and
        # DSR "reduces exactly to PSR(0)".
        family = sharpe_family(curve(60), 1)
        assert family is not None
        assert family.psr is not None and family.dsr is not None
        assert family.psr.value == pytest.approx(family.dsr.value)

    def test_an_empty_curve_returns_none(self) -> None:
        assert sharpe_family([], 100) is None

    def test_too_few_daily_points_leaves_sharpe_and_dsr_with_reasons_not_crashes(self) -> None:
        family = sharpe_family(curve(1), 100)
        assert family is not None
        assert family.sharpe is None and family.sharpe_reason is not None
        assert family.dsr is None and family.dsr_reason is not None


class TestSharpeFamilyOnThePage:
    def test_the_section_renders_with_all_three_figures(self) -> None:
        html = build(run_source(120)).html
        assert "The Sharpe family" in html
        assert "DSR" in html
        assert "no parameter search was recorded" in html

    def test_dsr_appears_with_n_trials_when_a_search_is_attached(self) -> None:
        source = replace(
            run_source(120),
            search=SearchSummary(
                method="grid",
                objective="sortino_sqrt_n",
                trial_budget=125,
                feasible_size=125,
                truncated=False,
                total_trials=1000,
                total_scored=850,
                n_folds_searched=8,
            ),
        )
        html = build(source).html
        assert "1000" in html
        assert "no parameter search was recorded" not in html


class TestUnderwaterAndDurationSections:
    def test_underwater_curve_renders(self) -> None:
        html = build(run_source(60)).html
        assert "Underwater curve" in html

    def test_time_in_trade_section_renders_both_charts(self) -> None:
        html = build(run_source(60)).html
        assert "<h2>Time in trade</h2>" in html
        assert (
            html.count('class="plotly-graph-div') >= 5
        )  # equity, underwater, r, monthly, excursions... + 2 duration


class TestSearchSummaryOnThePage:
    def source_with_search(self, *, truncated: bool) -> ReportSource:
        base = ReportSource(
            kind=SourceKind.WALKFORWARD,
            label="wf-search",
            curve=curve(60),
            trades=trades(30),
            streams={},
            money_is_real=False,
        )
        return replace(
            base,
            search=SearchSummary(
                method="grid",
                objective="sortino_sqrt_n",
                trial_budget=125,
                feasible_size=125 if not truncated else 400,
                truncated=truncated,
                total_trials=1000,
                total_scored=850,
                n_folds_searched=8,
            ),
        )

    def test_a_full_coverage_search_says_so(self) -> None:
        html = build(self.source_with_search(truncated=False)).html
        assert "covers the whole space" in html

    def test_a_truncated_search_is_flagged(self) -> None:
        html = build(self.source_with_search(truncated=True)).html
        assert "truncated" in html

    def test_a_flat_run_has_no_search_section(self) -> None:
        html = build(run_source(60)).html
        assert "Search, across the whole walk-forward" not in html


class TestTooltips:
    def test_a_known_metric_carries_a_tooltip(self) -> None:
        rows = metric_rows("sharpe", sharpe_daily(daily_curve(curve(60))))
        (row,) = [r for r in rows if r.name == "value"]
        assert row.tooltip is not None
        assert "annualised" in row.tooltip

    def test_an_unlisted_metric_carries_no_tooltip(self) -> None:
        rows = metric_rows("trades", trade_stats(trades(30)))
        (row,) = [r for r in rows if r.name == "max_consecutive_wins"]
        assert row.tooltip is None

    def test_the_bubble_markup_appears_only_for_rows_with_a_tooltip(self) -> None:
        html = build(run_source(120)).html
        assert 'class="tip"' in html
        assert 'class="bubble"' in html


class TestCostSensitivityOnThePage:
    def report(self) -> CostSensitivityReport:
        return CostSensitivityReport(
            points=(
                CostSensitivityPoint(1.0, 100, 0.05, 1.2, 0.8),
                CostSensitivityPoint(1.5, 96, 0.01, 1.05, 1.2),
                CostSensitivityPoint(2.0, 90, -0.03, 0.92, 1.6),
                CostSensitivityPoint(3.0, 80, -0.12, 0.71, 2.4),
            ),
            note="Spread scaled on the instrument directly; slippage and gap via perturb_costs.",
        )

    def test_the_table_renders_every_multiplier(self) -> None:
        source = replace(run_source(60), cost_sensitivity=self.report())
        html = build(source).html
        assert "×1" in html
        assert "×1.5" in html
        assert "×2" in html
        assert "×3" in html

    def test_a_source_without_it_has_no_section(self) -> None:
        html = build(run_source(60)).html
        assert "<h2>Cost sensitivity</h2>" not in html

    def test_no_trades_at_a_multiplier_is_a_dash_not_a_crash(self) -> None:
        report = CostSensitivityReport(
            points=(CostSensitivityPoint(3.0, 0, None, None, 2.4),), note="n/a"
        )
        source = replace(run_source(60), cost_sensitivity=report)
        html = build(source).html
        assert "—" in html
