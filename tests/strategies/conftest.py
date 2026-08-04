"""Shared fixtures for strategy spec tests."""

from pathlib import Path
from typing import Any

import pytest

from trading_system.strategies import schema as schema_module

#: Where the three worked examples live, resolved via the package rather than
#: a path relative to this test file so it survives test-directory moves.
EXAMPLES_DIR = Path(schema_module.__file__).parent / "examples"
EXAMPLE_FILES = sorted(EXAMPLES_DIR.glob("*.json"))


def leaf(op: str, left: Any = None, right: Any = None) -> dict[str, Any]:
    """Build a leaf-condition dict, omitting operands left unset."""
    condition: dict[str, Any] = {"type": "leaf", "op": op}
    if left is not None:
        condition["left"] = left
    if right is not None:
        condition["right"] = right
    return condition


def feature_ref(
    indicator: str, params: dict[str, Any] | None = None, channel: str | None = None
) -> dict[str, Any]:
    """Build a structural feature-reference dict for a leaf operand."""
    ref: dict[str, Any] = {"indicator": indicator, "params": params or {}}
    if channel is not None:
        ref["channel"] = channel
    return ref


@pytest.fixture
def minimal_spec_dict() -> dict[str, Any]:
    """The smallest strategy spec dict satisfying every required field.

    References ``ema`` with ``period=50``, a registered single-output
    indicator, so this spec passes
    :func:`~trading_system.strategies.validator.validate_spec` unmodified.
    """
    return {
        "id": "test-strategy",
        "name": "Test Strategy",
        "version": "1.0.0",
        "author": "tester",
        "type": "SWING",
        "timeframes": {"signal_tf": "H4", "entry_tf": "H1"},
        "instruments": {"allowed_classes": ["FX"]},
        "entry": {
            "direction": "LONG",
            "trigger": leaf("gt", "price:close", feature_ref("ema", {"period": 50})),
            "entry_order": {"order": {"type": "MARKET"}},
        },
        "exit_ref": "test-exit",
        "risk_profile": {
            "base_quality": 0.5,
            "stop_reference": {"kind": "ATR"},
        },
    }
