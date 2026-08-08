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

from trading_system.analytics.metrics import trade_stats
from trading_system.analytics.report import (
    RELIABLE_TRADES,
    FoldSegment,
    FoldSelection,
    ReportSource,
    SourceKind,
    all_metric_rows,
    build,
    build_comparison,
    export_metrics,
    metric_rows,
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
