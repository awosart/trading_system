"""Logging emits human-readable console output and JSON to file."""

import json
import logging
from collections.abc import Iterator
from pathlib import Path

import pytest

from trading_system.core.logging import get_logger, setup_logging


@pytest.fixture(autouse=True)
def _restore_root_logger() -> Iterator[None]:
    """Keep handler mutation from leaking into other tests."""
    root = logging.getLogger()
    saved_handlers, saved_level = list(root.handlers), root.level
    yield
    root.handlers = saved_handlers
    root.setLevel(saved_level)


def test_writes_json_lines_to_file(tmp_path: Path) -> None:
    log_file = tmp_path / "nested" / "app.log"
    setup_logging(level="INFO", log_file=log_file)

    get_logger("test").info("order_placed", symbol="EURUSD", size=2)

    record = json.loads(log_file.read_text(encoding="utf-8").strip())
    assert record["event"] == "order_placed"
    assert record["symbol"] == "EURUSD"
    assert record["level"] == "info"
    assert record["timestamp"].endswith("Z")


def test_level_threshold_is_applied(tmp_path: Path) -> None:
    log_file = tmp_path / "app.log"
    setup_logging(level="WARNING", log_file=log_file)

    logger = get_logger("test")
    logger.info("suppressed")
    logger.warning("kept")

    events = [json.loads(line)["event"] for line in log_file.read_text().splitlines()]
    assert events == ["kept"]


def test_repeated_setup_does_not_duplicate_handlers(tmp_path: Path) -> None:
    setup_logging(level="INFO", log_file=tmp_path / "a.log")
    setup_logging(level="INFO", log_file=tmp_path / "b.log")
    assert len(logging.getLogger().handlers) == 2  # console + one file
