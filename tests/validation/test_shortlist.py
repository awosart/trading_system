"""Choosing candidates: the freeze, the struck-out axes, and the same holdout.

Three of these guard decisions that are only worth anything if they cannot be
walked back. A shortlist that can be re-frozen is a threshold chosen with the
answers visible. A search space that can tune the position limit is a search
optimising the confound this stage measured. And a stage-two run that can reach
a holdout bar destroys the only uncontaminated evidence there will ever be.
"""

import json
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any

import pytest

from trading_system.data.models import OHLCVFrame
from trading_system.strategies.schema import StrategySpec
from trading_system.validation.holdout import screen_frame
from trading_system.validation.screen import ScreenRow
from trading_system.validation.shortlist import (
    EXCLUDED_TIMEFRAMES,
    MIN_TRADES,
    THROTTLE_AXES,
    Candidate,
    ShortlistManifest,
    build_walkforward_inputs,
    drop_share,
    now_iso,
    residual_z,
    select,
    without_throttles,
)
from trading_system.validation.space_builder import build_space_document

EXIT_LIBRARY = Path("src/trading_system/exit/library.json")
INSTRUMENTS = Path("configs/instruments.yaml")


def _row(
    key: str,
    *,
    symbol: str = "EURUSD",
    timeframe: str = "H4",
    trades: int = 200,
    expectancy_r: float = 0.05,
    dominant: str = "",
    dominant_count: int = 0,
) -> ScreenRow:
    """A completed screen row."""
    return ScreenRow(
        key=key,
        spec_id=key,
        symbol=symbol,
        timeframe=timeframe,
        bars=20_000,
        start=None,
        end=None,
        cost_ratio=0.1,
        trades=trades,
        fills=trades * 2,
        expired_orders=0,
        open_at_end=0,
        expectancy_r=expectancy_r,
        winrate=0.4,
        profit_factor=1.1,
        total_return=0.1,
        sharpe=0.2,
        sortino=0.3,
        max_drawdown_pct=0.1,
        dominant_reason=dominant,
        dominant_count=dominant_count,
        kept_run=False,
        result_digest="d",
        seconds=1.0,
    )


def _reachable_frames(root: Any, seen: set[int] | None = None) -> list[OHLCVFrame]:
    """Every frame reachable from ``root`` — the same walk the screen's test uses."""
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


class TestTheConfoundIsMeasuredAndRemoved:
    """The exposure confound, and what regressing it out does and does not do."""

    def test_the_drop_share_is_zero_when_something_else_dominated(self) -> None:
        assert drop_share(_row("a", dominant="reject:margin", dominant_count=500)) == 0.0

    def test_the_drop_share_counts_dropped_against_taken(self) -> None:
        row = _row("a", trades=100, dominant="drop:position_limit", dominant_count=300)
        assert drop_share(row) == pytest.approx(0.75)

    def test_residualising_leaves_no_linear_correlation(self) -> None:
        import statistics

        shares = [0.1, 0.3, 0.5, 0.7, 0.9, 0.2, 0.4]
        scores = [0.5 + 2.0 * share + (index % 3) * 0.1 for index, share in enumerate(shares)]
        residual = residual_z(scores, shares)
        assert statistics.correlation(shares, residual) == pytest.approx(0.0, abs=1e-9)

    def test_a_confound_with_no_variance_leaves_the_scores_alone(self) -> None:
        scores = [1.0, 2.0, 3.0]
        assert residual_z(scores, [0.5, 0.5, 0.5]) == scores


class TestSelection:
    """Who gets promoted, and who cannot be."""

    def _paths(self, *names: str) -> dict[str, Path]:
        return {name: Path(f"/specs/{name}.json") for name in names}

    def test_an_excluded_timeframe_is_never_promoted(self) -> None:
        rows = [_row(f"s{i}", timeframe="M1", expectancy_r=1.0) for i in range(5)]
        z = {row.key: 5.0 for row in rows}
        assert select(rows, z, self._paths(*(r.key for r in rows)), limit=5) == ()

    def test_a_row_below_the_trade_floor_is_never_promoted(self) -> None:
        rows = [_row(f"s{i}", trades=MIN_TRADES - 1, expectancy_r=1.0) for i in range(5)]
        z = {row.key: 5.0 for row in rows}
        assert select(rows, z, self._paths(*(r.key for r in rows)), limit=5) == ()

    def test_the_ranking_is_by_residual_not_raw_z(self) -> None:
        # Five rows whose raw z is exactly explained by how much they were
        # thinned, plus one that was not thinned at all and still scored above
        # the line. The thinned rows have the higher raw z; the unthinned one
        # is the only one carrying anything the confound does not account for.
        rows = []
        z: dict[str, float] = {}
        for index, dropped in enumerate((100, 300, 500, 700, 900)):
            key = f"thinned{index}"
            rows.append(
                _row(key, dominant="drop:position_limit", dominant_count=dropped, trades=100)
            )
            z[key] = 1.0 + 2.0 * (dropped / (dropped + 100))
        rows.append(_row("clean", trades=100))
        z["clean"] = 1.5

        raw_best = max(z, key=lambda name: z[name])
        assert raw_best != "clean", "the fixture must give a thinned row the better raw z"

        chosen = select(rows, z, self._paths(*z), limit=6)
        assert chosen[0].key == "clean"
        # The residual is centred, so it is not comparable in level with the raw
        # z; what matters is the order, and the order changed.
        assert chosen[0].residual_z > max(
            item.residual_z for item in chosen if item.key != "clean"
        )

    def test_every_excluded_timeframe_is_named_rather_than_implied(self) -> None:
        assert set(EXCLUDED_TIMEFRAMES) == {"M1", "M5"}


class TestTheFreezeCannotBeWalkedBack:
    """A second selection over the same screen is a threshold chosen after the fact."""

    def _manifest(self) -> ShortlistManifest:
        return ShortlistManifest(
            screen_id="screen",
            generated=now_iso(),
            holdout_fraction=0.2,
            thresholds={"limit": 1},
            confound={"corr": 0.17},
            candidates=(
                Candidate(
                    key="a@EURUSD",
                    spec_id="a",
                    spec_path="/specs/a.json",
                    symbol="EURUSD",
                    timeframe="H4",
                    trades=200,
                    expectancy_r=0.05,
                    z=1.0,
                    drop_share=0.0,
                    residual_z=1.0,
                ),
            ),
        )

    def test_writing_over_a_frozen_shortlist_is_refused(self, tmp_path: Path) -> None:
        path = tmp_path / "shortlist.json"
        self._manifest().write(path)
        with pytest.raises(FileExistsError, match="already frozen"):
            self._manifest().write(path)

    def test_it_reads_back_as_it_was_written(self, tmp_path: Path) -> None:
        path = tmp_path / "shortlist.json"
        self._manifest().write(path)
        read = ShortlistManifest.read(path)
        assert read is not None
        assert read.candidates[0].key == "a@EURUSD"
        assert read.thresholds["limit"] == 1

    def test_a_missing_shortlist_reads_as_none(self, tmp_path: Path) -> None:
        assert ShortlistManifest.read(tmp_path / "absent.json") is None

    def test_the_frozen_file_says_what_it_is_not(self, tmp_path: Path) -> None:
        path = tmp_path / "shortlist.json"
        self._manifest().write(path)
        payload = json.loads(path.read_text())
        assert "holdout is untouched" in payload["what_this_is"]


class TestTheThrottlesAreStruckOut:
    """A search that can tune the position limit optimises the confound itself."""

    def _spec(self) -> StrategySpec:
        path = next(Path("strategies/test_strategies/specs").rglob("*.json"))
        return StrategySpec.model_validate_json(path.read_text())

    def test_the_generator_still_offers_them(self) -> None:
        # If this stops being true the striking-out is dead code, and a test
        # that passes for the wrong reason is worse than no test.
        document = build_space_document(self._spec())
        offered = {axis["name"] for axis in document["axes"]}
        assert offered & set(THROTTLE_AXES), (
            "the space generator no longer offers a throttle axis; "
            "without_throttles now guards nothing"
        )

    def test_both_axes_are_gone_afterwards(self) -> None:
        document, removed = without_throttles(build_space_document(self._spec()))
        remaining = {axis["name"] for axis in document["axes"]}
        assert not remaining & set(THROTTLE_AXES)
        assert set(removed) <= set(THROTTLE_AXES)
        assert removed, "nothing was struck out, so nothing was guarded"

    def test_a_constraint_naming_a_struck_axis_goes_with_it(self) -> None:
        document = {
            "axes": [
                {"name": "max_concurrent_positions", "pointers": ["/a"], "values": [1, 2]},
                {"name": "ema_fast", "pointers": ["/b"], "values": [10, 20]},
                {"name": "ema_slow", "pointers": ["/c"], "values": [50, 60]},
            ],
            "constraints": [
                {"less": "ema_fast", "greater": "ema_slow"},
                {"less": "max_concurrent_positions", "greater": "ema_slow"},
            ],
        }
        trimmed, removed = without_throttles(document)
        # A constraint referring to an axis that no longer exists would make the
        # space refuse to build at all.
        assert removed == ("max_concurrent_positions",)
        assert trimmed["constraints"] == [{"less": "ema_fast", "greater": "ema_slow"}]

    def test_the_original_document_is_not_mutated(self) -> None:
        document = build_space_document(self._spec())
        before = len(document["axes"])
        without_throttles(document)
        assert len(document["axes"]) == before


class TestStageTwoCannotReachTheHoldout:
    """The same structural claim as stage one, on the object stage two runs."""

    def test_no_path_from_a_candidate_reaches_a_holdout_bar(self) -> None:
        spec_path = next(Path("strategies/test_strategies/specs").rglob("*.json"))
        candidate = Candidate(
            key="probe",
            spec_id=spec_path.stem,
            spec_path=str(spec_path),
            symbol="EURUSD",
            timeframe="H1",
            trades=200,
            expectancy_r=0.05,
            z=1.0,
            drop_share=0.0,
            residual_z=1.0,
        )
        inputs, spec = build_walkforward_inputs(
            candidate, Path("data"), INSTRUMENTS, EXIT_LIBRARY, holdout_fraction=0.2
        )
        from trading_system.core.config import Settings
        from trading_system.data.store import ParquetStore

        # The timeframe comes from the spec, not from the candidate's label:
        # the boundary to compare against is the one for the series actually run.
        whole = ParquetStore(Settings().data_dir).get("EURUSD", spec.timeframes.signal_tf)
        _visible, boundary = screen_frame(whole, fraction=0.2)

        frames = _reachable_frames(inputs)
        assert frames, "the walk found no frames at all, so it proves nothing"
        for frame in frames:
            assert frame.end is not None
            assert frame.end < boundary.boundary, (
                f"{frame.symbol}@{frame.timeframe.value} reaches {frame.end}, "
                f"at or past the holdout boundary {boundary.boundary}"
            )
