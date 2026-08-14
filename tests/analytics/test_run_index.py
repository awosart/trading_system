"""The index over stored runs: what it lists, what it refuses to hide, what it links.

The page itself is a table; the decisions worth testing are about membership.
Folds are stored as ordinary runs, so an index that does not know them is 90%
folds. A run that will not read back must be visible as broken rather than
absent. And a link must never point at a page that is not there.
"""

import json
from pathlib import Path

import pytest

from trading_system.analytics.run_index import (
    RunIndex,
    RunRow,
    build_index,
    filter_rows,
    fold_run_ids,
    render,
    run_directories,
)
from trading_system.backtest.reproducibility import MANIFEST_FILE


def _run_dir(root: Path, run_id: str, *, strategy: str = "demo", stream: str = "EURUSD@H4") -> Path:
    """A directory that looks like a run to the walker but cannot be read as one."""
    directory = root / run_id
    directory.mkdir(parents=True)
    (directory / MANIFEST_FILE).write_text(
        json.dumps({"strategies": {strategy: "d"}, "data": {stream: "d"}})
    )
    return directory


class TestWhichDirectoriesAreRuns:
    def test_walkforward_and_calibration_are_not_runs(self, tmp_path: Path) -> None:
        _run_dir(tmp_path, "aaa")
        (tmp_path / "walkforward").mkdir()
        (tmp_path / "calibration").mkdir()
        assert [d.name for d in run_directories(tmp_path)] == ["aaa"]

    def test_a_directory_without_a_manifest_is_not_a_run(self, tmp_path: Path) -> None:
        (tmp_path / "not-a-run").mkdir()
        assert run_directories(tmp_path) == []

    def test_a_missing_runs_directory_is_empty_rather_than_an_error(self, tmp_path: Path) -> None:
        assert run_directories(tmp_path / "nope") == []


class TestFoldsAreExcludedByDefault:
    """A fold is one slice of a procedure, not a result standing on its own."""

    def _walkforward(self, root: Path, *, is_id: str, oos_id: str) -> None:
        directory = root / "walkforward" / "wf1"
        directory.mkdir(parents=True)
        (directory / "manifest.json").write_text(
            json.dumps(
                {
                    "wf_id": "wf1",
                    "folds": [
                        {
                            "index": 0,
                            "is_run": {"run_id": is_id, "digest": "x", "path": f"runs/{is_id}"},
                            "oos_run": {"run_id": oos_id, "digest": "y", "path": f"runs/{oos_id}"},
                        }
                    ],
                }
            )
        )

    def test_fold_ids_come_off_the_walkforward_manifests(self, tmp_path: Path) -> None:
        self._walkforward(tmp_path, is_id="fold-is", oos_id="fold-oos")
        assert fold_run_ids(tmp_path) == frozenset({"fold-is", "fold-oos"})

    def test_excluding_them_leaves_only_the_standalone_run(self, tmp_path: Path) -> None:
        for run_id in ("fold-is", "fold-oos", "standalone"):
            _run_dir(tmp_path, run_id)
        self._walkforward(tmp_path, is_id="fold-is", oos_id="fold-oos")
        kept = run_directories(tmp_path, exclude=fold_run_ids(tmp_path))
        assert [d.name for d in kept] == ["standalone"]

    def test_no_walkforward_directory_excludes_nothing(self, tmp_path: Path) -> None:
        assert fold_run_ids(tmp_path) == frozenset()


class TestUnreadableRunsAreListedNotSkipped:
    def test_a_run_whose_tables_are_missing_appears_with_its_reason(self, tmp_path: Path) -> None:
        # Absent from the index would mean both "never ran" and "cannot be
        # read", which are opposite problems.
        _run_dir(tmp_path, "broken")
        index = build_index(run_directories(tmp_path), root=tmp_path)
        assert index.rows == ()
        assert len(index.broken) == 1
        assert index.broken[0].run_id == "broken"
        assert index.broken[0].error
        assert not index.broken[0].readable
        assert index.total == 1


class TestLinks:
    def _index(self, rows: tuple[RunRow, ...]) -> RunIndex:
        from datetime import UTC, datetime

        return RunIndex(rows=rows, broken=(), generated=datetime.now(UTC), root=Path("runs"))

    def test_a_run_with_no_rendered_page_is_not_linked(self, tmp_path: Path) -> None:
        _run_dir(tmp_path, "broken")
        index = build_index(run_directories(tmp_path), root=tmp_path, detail_dir=tmp_path)
        assert index.broken[0].detail_href is None

    def test_a_rendered_page_next_to_the_index_is_linked_relatively(self, tmp_path: Path) -> None:
        _run_dir(tmp_path, "broken")
        (tmp_path / "broken.html").write_text("<html></html>")
        index = build_index(run_directories(tmp_path), root=tmp_path, detail_dir=tmp_path)
        # Relative, so the folder can be moved or shared whole.
        assert index.broken[0].detail_href == "broken.html"


class TestFiltering:
    def test_filtering_reads_the_manifest_not_the_directory_name(self, tmp_path: Path) -> None:
        # A run id is a digest and carries nothing a human would filter on.
        _run_dir(tmp_path, "aaa", strategy="ema-pullback-h1", stream="EURUSD@H1")
        _run_dir(tmp_path, "bbb", strategy="donchian-breakout-h4", stream="GBPUSD@H4")
        directories = run_directories(tmp_path)
        assert [d.name for d in filter_rows(directories, strategy="donchian", stream=None)] == [
            "bbb"
        ]
        assert [d.name for d in filter_rows(directories, strategy=None, stream="EURUSD")] == ["aaa"]
        assert len(filter_rows(directories, strategy=None, stream=None)) == 2


class TestRendering:
    def test_the_page_renders_with_no_runs_at_all(self, tmp_path: Path) -> None:
        html = render(build_index([], root=tmp_path))
        assert "Stored runs" in html
        assert "0 run(s)" in html

    def test_the_page_escapes_what_comes_off_disk(self, tmp_path: Path) -> None:
        _run_dir(tmp_path, "broken", strategy="<script>alert(1)</script>")
        html = render(build_index(run_directories(tmp_path), root=tmp_path))
        assert "<script>alert(1)</script>" not in html

    def test_a_broken_run_names_itself_on_the_page(self, tmp_path: Path) -> None:
        _run_dir(tmp_path, "broken")
        html = render(build_index(run_directories(tmp_path), root=tmp_path))
        assert "Could not be read" in html
        assert "broken"[:12] in html


class TestSorting:
    def test_rows_open_on_the_best_expectancy_and_unscored_rows_sink(self) -> None:
        rows = [
            RunRow("a", "s", "x", 10, -0.2, None, None, None, None, None, None),
            RunRow("b", "s", "x", 10, None, None, None, None, None, None, None),
            RunRow("c", "s", "x", 10, +0.3, None, None, None, None, None, None),
        ]
        index = RunIndex(rows=tuple(rows), broken=(), generated=None, root=Path())  # type: ignore[arg-type]
        ordered = sorted(
            index.rows, key=lambda row: (row.expectancy_r is None, -(row.expectancy_r or 0.0))
        )
        assert [row.run_id for row in ordered] == ["c", "a", "b"]


@pytest.mark.parametrize("name", ["walkforward", "calibration", "optimization"])
def test_the_non_run_directory_names_are_the_ones_the_store_actually_uses(name: str) -> None:
    from trading_system.analytics.run_index import NON_RUN_DIRECTORIES

    assert name in NON_RUN_DIRECTORIES
