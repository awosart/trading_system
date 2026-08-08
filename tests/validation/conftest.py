"""Fixtures for the validation layer.

Declared here rather than inherited from ``tests/backtest/conftest.py``, which
is not visible from this directory — the same reason
``tests/test_no_lookahead.py`` redeclares ``registry``/``library`` at its own
top level.
"""

from pathlib import Path

import pytest

from tests.backtest.conftest import LIBRARY_PATH, REGISTRY_PATH
from trading_system.core.instruments import InstrumentRegistry, load_instruments
from trading_system.exit.library import ExitLibrarySpec

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="session")
def registry() -> InstrumentRegistry:
    """The bundled instrument registry."""
    return load_instruments(REGISTRY_PATH)


@pytest.fixture(scope="session")
def library() -> ExitLibrarySpec:
    """The bundled exit library, as specs."""
    return ExitLibrarySpec.model_validate_json(LIBRARY_PATH.read_text())
