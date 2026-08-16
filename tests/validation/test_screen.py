"""The screen: a holdout that cannot be reached, a log that survives being killed.

Two of these tests are the ones the whole stage exists for. The first walks the
object graph a screen task actually runs on and requires that no path leads to a
holdout bar — a promise in a docstring would be worth nothing here, because the
holdout only has value while nothing has looked at it and there is no way to
tell afterwards whether something did. The second kills a screen half-way and
requires that the finished rows are still on disk, because the first attempt at
this sweep lost 2 400 results to one interruption.
"""

import json
from dataclasses import fields, is_dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import polars as pl
import pytest

from trading_system.core.instruments import InstrumentClass
from trading_system.core.types import Timeframe
from trading_system.data.models import OHLCVFrame
from trading_system.data.resample import FX_DAY_ORIGIN, trading_day
from trading_system.strategies.normalize.coverage import MarketCoverage, SeriesCoverage
from trading_system.validation.holdout import (
    holdout_boundary,
    screen_frame,
)
from trading_system.validation.screen import (
    ScreenRow,
    ScreenStore,
    ScreenTask,
    build_screen_inputs,
    cross_sectional_z,
    distinct_signatures,
    load_manifest,
    plan_tasks,
    run_screen,
    screen_id,
)
from trading_system.validation.trials import (
    correlation_matrix,
    estimate,
    extrapolate,
    participation_ratio,
)

EXIT_LIBRARY = Path("src/trading_system/exit/library.json")
INSTRUMENTS = Path("configs/instruments.yaml")


def _frame(bars: int, *, symbol: str = "EURUSD", start: datetime | None = None) -> OHLCVFrame:
    """A synthetic H1 series of ``bars`` bars."""
    origin = start or datetime(2020, 1, 1, tzinfo=UTC)
    stamps = [origin + timedelta(hours=index) for index in range(bars)]
    prices = [1.10 + 0.0001 * (index % 50) for index in range(bars)]
    return OHLCVFrame.from_raw(
        pl.DataFrame(
            {
                "timestamp": stamps,
                "open": prices,
                "high": [price + 0.0005 for price in prices],
                "low": [price - 0.0005 for price in prices],
                "close": prices,
                "volume": [100.0] * bars,
            }
        ),
        symbol,
        Timeframe.H1,
    )


def _reachable_frames(root: Any, seen: set[int] | None = None) -> list[OHLCVFrame]:
    """Every :class:`OHLCVFrame` reachable from ``root`` through the object graph.

    Reflective rather than a list of places to look, for the reason the same
    walker exists in the optimiser's tests: the claim is that *no* path leads to
    a withheld bar, and a hand-written list of attributes can only check the
    paths its author thought of.
    """
    seen = set() if seen is None else seen
    if id(root) in seen:
        return []
    seen.add(id(root))
    if isinstance(root, OHLCVFrame):
        return [root]
    if isinstance(root, (str, bytes, int, float, bool)) or root is None:
        return []
    found: list[OHLCVFrame] = []
    if isinstance(root, dict):
        for key, value in root.items():
            found.extend(_reachable_frames(key, seen))
            found.extend(_reachable_frames(value, seen))
        return found
    if isinstance(root, (list, tuple, set, frozenset)):
        for item in root:
            found.extend(_reachable_frames(item, seen))
        return found
    if is_dataclass(root) and not isinstance(root, type):
        for spec in fields(root):
            found.extend(_reachable_frames(getattr(root, spec.name, None), seen))
        return found
    for value in getattr(root, "__dict__", {}).values():
        found.extend(_reachable_frames(value, seen))
    for name in getattr(type(root), "__slots__", ()):
        found.extend(_reachable_frames(getattr(root, name, None), seen))
    return found


class TestTheHoldoutIsCutBeforeAnythingRuns:
    """The bars are not in the object, which is stronger than not being read."""

    def test_the_screen_frame_holds_no_bar_at_or_after_the_boundary(self) -> None:
        whole = _frame(2000)
        visible, boundary = screen_frame(whole, fraction=0.2)
        assert visible.end is not None
        assert visible.end < boundary.boundary
        assert boundary.holdout_bars > 0
        assert boundary.screen_bars + boundary.holdout_bars == len(whole)

    def test_the_boundary_sits_on_a_trading_day_boundary(self) -> None:
        whole = _frame(2000)
        boundary = holdout_boundary(whole, fraction=0.2)
        # The label changes exactly at the boundary: the instant before belongs
        # to the previous trading day, the boundary itself to the next.
        before = trading_day(boundary - timedelta(microseconds=1), FX_DAY_ORIGIN)
        assert trading_day(boundary, FX_DAY_ORIGIN) > before

    def test_snapping_only_ever_shows_the_screen_less(self) -> None:
        whole = _frame(2000)
        naive = whole.timestamps[int(2000 * 0.8)]
        assert holdout_boundary(whole, fraction=0.2) >= naive

    @pytest.mark.parametrize("fraction", [0.0, 1.0, -0.1, 1.5])
    def test_a_nonsense_fraction_is_refused(self, fraction: float) -> None:
        with pytest.raises(ValueError, match="holdout fraction"):
            holdout_boundary(_frame(100), fraction=fraction)

    def test_no_path_from_a_real_task_reaches_a_holdout_bar(self, tmp_path: Path) -> None:
        # The one that matters: not a mock, not a promise. Build what a worker
        # actually runs and walk every object it can reach.
        spec = next(Path("strategies/test_strategies/specs").rglob("*.json"))
        task = ScreenTask(spec_path=spec, symbol="EURUSD", bar_budget=3_000, holdout_fraction=0.2)
        inputs, boundary = build_screen_inputs(task, Path("data"), INSTRUMENTS, EXIT_LIBRARY)
        frames = _reachable_frames(inputs)
        assert frames, "the walk found no frames at all, so it proves nothing"
        for frame in frames:
            assert frame.end is not None
            assert frame.end < boundary.boundary, (
                f"{frame.symbol}@{frame.timeframe.value} reaches {frame.end}, "
                f"at or past the holdout boundary {boundary.boundary}"
            )
        del tmp_path


class TestTheLogSurvivesInterruption:
    """An interruption costs the task in flight and nothing else."""

    def _task(self, name: str) -> ScreenTask:
        return ScreenTask(spec_path=Path(f"/nonexistent/{name}.json"), symbol=name)

    def test_rows_are_on_disk_before_the_run_returns(self, tmp_path: Path) -> None:
        store = ScreenStore(tmp_path)
        seen: list[str] = []

        def observe(row: ScreenRow) -> None:
            # Read the log from disk, not from memory: the claim is about what
            # survives the process, so the check must not consult the process.
            seen.append(row.key)
            assert row.key in ScreenStore(tmp_path).completed()

        run_screen(
            [self._task("a"), self._task("b")],
            store,
            data_dir=tmp_path,
            instruments_path=tmp_path / "none.yaml",
            exit_library=tmp_path / "none.json",
            on_row=observe,
        )
        assert seen == ["a@a", "b@b"]

    def test_a_second_call_skips_what_the_first_finished(self, tmp_path: Path) -> None:
        store = ScreenStore(tmp_path)
        run_screen(
            [self._task("a")],
            store,
            data_dir=tmp_path,
            instruments_path=tmp_path / "none.yaml",
            exit_library=tmp_path / "none.json",
        )
        ran: list[str] = []
        rows = run_screen(
            [self._task("a"), self._task("b")],
            store,
            data_dir=tmp_path,
            instruments_path=tmp_path / "none.yaml",
            exit_library=tmp_path / "none.json",
            on_row=lambda row: ran.append(row.key),
        )
        assert ran == ["b@b"], "the finished task was run a second time"
        assert [row.key for row in rows] == ["a@a", "b@b"]

    def test_a_failing_task_becomes_a_row_rather_than_ending_the_screen(
        self, tmp_path: Path
    ) -> None:
        store = ScreenStore(tmp_path)
        rows = run_screen(
            [self._task("a")],
            store,
            data_dir=tmp_path,
            instruments_path=tmp_path / "none.yaml",
            exit_library=tmp_path / "none.json",
        )
        assert len(rows) == 1
        assert rows[0].error is not None
        assert rows[0].trades == 0

    def test_the_parquet_is_materialised_from_the_log(self, tmp_path: Path) -> None:
        store = ScreenStore(tmp_path)
        run_screen(
            [self._task("a")],
            store,
            data_dir=tmp_path,
            instruments_path=tmp_path / "none.yaml",
            exit_library=tmp_path / "none.json",
        )
        rows_path, returns_path = store.materialise()
        assert rows_path.is_file()
        assert returns_path is None
        table = pl.read_parquet(rows_path)
        assert table.height == 1
        assert "daily_returns" not in table.columns


class TestIdentity:
    """The same inputs are the same screen; a different anything is not."""

    def test_the_same_inputs_give_the_same_id(self, tmp_path: Path) -> None:
        paths = [tmp_path / "a.json", tmp_path / "b.json"]
        first = screen_id(
            paths,
            universe=["EURUSD"],
            bar_budget=100,
            holdout_fraction=0.2,
            risk_pct=0.01,
            symbols_per_spec=1,
        )
        second = screen_id(
            paths,
            universe=["EURUSD"],
            bar_budget=100,
            holdout_fraction=0.2,
            risk_pct=0.01,
            symbols_per_spec=1,
        )
        assert first == second

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("universe", ["EURUSD", "GBPUSD"]),
            ("bar_budget", 200),
            ("holdout_fraction", 0.3),
            ("risk_pct", 0.02),
            ("symbols_per_spec", 3),
        ],
    )
    def test_changing_any_setting_changes_the_id(
        self, tmp_path: Path, field: str, value: object
    ) -> None:
        base: dict[str, Any] = {
            "universe": ["EURUSD"],
            "bar_budget": 100,
            "holdout_fraction": 0.2,
            "risk_pct": 0.01,
            "symbols_per_spec": 1,
        }
        paths = [tmp_path / "a.json"]
        changed = {**base, field: value}
        assert screen_id(paths, **base) != screen_id(paths, **changed)

    def test_a_finished_screen_is_recognised_by_its_manifest(self, tmp_path: Path) -> None:
        assert load_manifest(tmp_path) is None
        (tmp_path / "manifest.json").write_text('{"screen_id": "x"}', encoding="utf-8")
        found = load_manifest(tmp_path)
        assert found is not None and found["screen_id"] == "x"


class TestCrossSectionalZ:
    """The screening unit: standardised inside its own market, or absent."""

    def _row(self, key: str, symbol: str, sortino: float | None) -> ScreenRow:
        return ScreenRow(
            key=key,
            spec_id=key,
            symbol=symbol,
            timeframe="H1",
            bars=1000,
            start=None,
            end=None,
            cost_ratio=0.1,
            trades=100,
            fills=200,
            expired_orders=0,
            open_at_end=0,
            expectancy_r=0.01,
            winrate=0.4,
            profit_factor=1.1,
            total_return=0.1,
            sharpe=0.2,
            sortino=sortino,
            max_drawdown_pct=0.1,
            dominant_reason="",
            dominant_count=0,
            kept_run=False,
            result_digest="d",
            seconds=1.0,
        )

    def test_a_row_is_scored_against_its_own_market_only(self) -> None:
        rows = [
            self._row("a", "EURUSD", 1.0),
            self._row("b", "EURUSD", 2.0),
            self._row("c", "EURUSD", 3.0),
            self._row("d", "XAUUSD", 100.0),
            self._row("e", "XAUUSD", 200.0),
            self._row("f", "XAUUSD", 300.0),
        ]
        scores = cross_sectional_z(rows)
        # The middle row of each market is at its market's mean, whatever the
        # market's scale — which is the whole point of the unit.
        assert scores["b"] == pytest.approx(0.0)
        assert scores["e"] == pytest.approx(0.0)
        assert scores["a"] == pytest.approx(scores["d"])

    def test_a_market_with_too_few_peers_has_no_score(self) -> None:
        scores = cross_sectional_z([self._row("a", "EURUSD", 1.0), self._row("b", "EURUSD", 2.0)])
        assert scores == {}

    def test_a_market_where_every_row_is_identical_has_no_score(self) -> None:
        rows = [self._row(name, "EURUSD", 1.0) for name in ("a", "b", "c")]
        # Zero spread is no reference at all; a zero score would say "average"
        # about a market that has no distribution to be average in.
        assert cross_sectional_z(rows) == {}

    def test_a_failed_row_is_not_scored(self) -> None:
        good = [self._row(name, "EURUSD", float(index)) for index, name in enumerate("abc", 1)]
        broken = self._row("d", "EURUSD", None)
        assert "d" not in cross_sectional_z([*good, broken])


class TestEffectiveTrials:
    """The discount, and the two ends of its range."""

    def test_uncorrelated_variables_count_as_themselves(self) -> None:
        identity = [[1.0 if i == j else 0.0 for j in range(5)] for i in range(5)]
        assert participation_ratio(identity).effective == pytest.approx(5.0, abs=1e-6)

    def test_one_variable_repeated_counts_as_one(self) -> None:
        ones = [[1.0] * 5 for _ in range(5)]
        assert participation_ratio(ones).effective == pytest.approx(1.0, abs=1e-6)

    def test_a_block_of_clones_counts_the_blocks(self) -> None:
        # Two perfectly-correlated pairs: two ideas wearing four names.
        matrix = [
            [1.0, 1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 1.0],
            [0.0, 0.0, 1.0, 1.0],
        ]
        assert participation_ratio(matrix).effective == pytest.approx(2.0, abs=1e-6)

    def test_correlation_of_identical_series_is_one(self) -> None:
        series = [0.1, -0.2, 0.3, 0.05, -0.1]
        matrix = correlation_matrix([series, series])
        assert matrix[0][1] == pytest.approx(1.0)

    def test_a_flat_series_correlates_with_nothing(self) -> None:
        matrix = correlation_matrix([[0.0, 0.0, 0.0, 0.0], [0.1, -0.2, 0.3, 0.05]])
        assert matrix[0][1] == 0.0

    def test_extrapolation_is_bounded_by_the_population(self) -> None:
        assert extrapolate(10.0, 10, 100) == pytest.approx(100.0)
        assert extrapolate(50.0, 10, 100) == 100.0
        assert extrapolate(0.0, 10, 100) == 1.0

    def test_the_signature_floor_caps_the_strategy_factor(self) -> None:
        returns = {f"s{index}": [0.1, -0.1, 0.2, 0.05, -0.02] for index in range(6)}
        markets = {"EURUSD": [0.1, -0.1, 0.2], "XAUUSD": [-0.1, 0.1, -0.2]}
        capped = estimate(
            strategy_returns=returns,
            market_returns=markets,
            n_strategies=600,
            n_markets=2,
            n_trials_raw=1200,
            signature_floor=7,
        )
        assert capped.strategies_effective <= 7.0
        assert capped.n_trials_effective <= 1200

    def test_the_estimate_never_exceeds_the_raw_count(self) -> None:
        found = estimate(
            strategy_returns={},
            market_returns={},
            n_strategies=10,
            n_markets=10,
            n_trials_raw=12,
            signature_floor=None,
        )
        assert found.n_trials_effective <= 12


class TestSignatures:
    """Distinct ideas, not distinct files."""

    def _write(self, path: Path, spec_id: str, indicators: list[str]) -> Path:
        payload = {
            "id": spec_id,
            "entries": [{"trigger": {"left": {"indicator": name}}} for name in indicators],
        }
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def test_two_specs_with_the_same_indicators_and_family_are_one_idea(
        self, tmp_path: Path
    ) -> None:
        first = self._write(tmp_path / "a.json", "a", ["ema", "rsi"])
        second = self._write(tmp_path / "b.json", "b", ["rsi", "ema"])
        families = {"a": {"family": "TREND"}, "b": {"family": "TREND"}}
        assert distinct_signatures([first, second], families) == 1

    def test_a_different_family_makes_it_a_different_idea(self, tmp_path: Path) -> None:
        first = self._write(tmp_path / "a.json", "a", ["ema"])
        second = self._write(tmp_path / "b.json", "b", ["ema"])
        families = {"a": {"family": "TREND"}, "b": {"family": "BREAKOUT"}}
        assert distinct_signatures([first, second], families) == 2

    def test_an_unreadable_spec_is_skipped_rather_than_counted(self, tmp_path: Path) -> None:
        broken = tmp_path / "broken.json"
        broken.write_text("{not json", encoding="utf-8")
        assert distinct_signatures([broken], {}) == 0


class TestASeriesThatCannotBeClosedIsExcludedByName:
    """A task that cannot start is not a failure to be counted — it is a plan entry.

    Every daily series in the delivered store is cut by the vendor on UTC
    midnight while runs cut their trading day at 17:00 New York, so no daily bar
    has a close instant under that origin. Letting those tasks run and fail would
    file a data convention under `ValueError` and bury it among real errors.
    """

    def _spec(self, tmp_path: Path, timeframe: str) -> Path:
        import json

        path = tmp_path / "s.json"
        path.write_text(
            json.dumps(
                {
                    "id": "s",
                    "version": "0.1.0",
                    "type": "POSITION",
                    "timeframes": {"signal_tf": timeframe, "entry_tf": timeframe},
                    "instruments": {
                        "allowed_classes": ["FX"],
                        "allowed_symbols": ["EURUSD"],
                        "denied_symbols": [],
                    },
                    "market_regimes": [],
                    "entries": [
                        {
                            "direction": "LONG",
                            "trigger": {
                                "type": "leaf",
                                "op": "gt",
                                "left": "price:close",
                                "right": {"indicator": "ema", "params": {"period": 50}},
                            },
                            "confirmation": [],
                            "confirmation_window_bars": 0,
                            "invalidation": {
                                "price_level": {"indicator": "ema", "params": {"period": 50}}
                            },
                            "entry_order": {
                                "order": {"type": "MARKET"},
                                "expire_after_bars": 1,
                            },
                        }
                    ],
                    "exit_ref": "conservative_2r",
                    "filters": [],
                    "risk_profile": {
                        "base_quality": 0.5,
                        "quality_modifiers": [],
                        "stop_reference": {"kind": "ATR", "period": 14, "multiple": 1.5},
                        "max_concurrent_positions": 1,
                        "cooldown_bars_after_loss": 0,
                    },
                }
            ),
            encoding="utf-8",
        )
        return path

    def _coverage(self, timeframe: Timeframe, *, anchor_ok: bool) -> MarketCoverage:
        return MarketCoverage(
            series={
                ("EURUSD", timeframe): SeriesCoverage(
                    symbol="EURUSD",
                    asset_class=InstrumentClass.FX,
                    timeframe=timeframe,
                    bars=5000,
                    start=datetime(2010, 1, 1, tzinfo=UTC),
                    end=datetime(2026, 1, 1, tzinfo=UTC),
                    median_range_points=10.0,
                    has_volume=True,
                    day_anchor_ok=anchor_ok,
                    cost_points=1.0,
                )
            }
        )

    def test_it_produces_no_task_and_says_why(self, tmp_path: Path) -> None:
        spec = self._spec(tmp_path, "D1")
        tasks, skipped = plan_tasks(
            [spec], self._coverage(Timeframe.D1, anchor_ok=False), symbols_per_spec=0
        )
        assert tasks == ()
        assert "s@EURUSD" in skipped
        assert "day anchor" in skipped["s@EURUSD"]

    def test_the_same_series_with_a_usable_anchor_is_planned(self, tmp_path: Path) -> None:
        spec = self._spec(tmp_path, "D1")
        tasks, skipped = plan_tasks(
            [spec], self._coverage(Timeframe.D1, anchor_ok=True), symbols_per_spec=0
        )
        assert [task.symbol for task in tasks] == ["EURUSD"]
        assert skipped == {}
