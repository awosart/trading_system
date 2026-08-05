"""The import discipline, asserted rather than described.

Execution must not depend on the feature pipeline, on the Entry Engine or on the
Risk Engine. That is the whole reason volatility arrives as a scalar on
:class:`~trading_system.execution.market_state.MarketState` and the reason the
stop-buffer cross-check takes a bare float instead of P10's config object. A
comment saying so survives exactly until someone needs an ATR and reaches for the
obvious import, so the rule is a test.
"""

import ast
from pathlib import Path

import pytest

PACKAGE = Path(__file__).resolve().parents[2] / "src" / "trading_system" / "execution"

#: Packages Execution is not allowed to reach into, and why.
FORBIDDEN = {
    "trading_system.features": "volatility arrives as a scalar on MarketState",
    "trading_system.entry": "Execution prices orders, it does not recognise setups",
    "trading_system.risk": "the stop-buffer check takes a float, not P10's config",
}

#: The one Exit import allowed, and the only symbol it may bring.
ALLOWED_EXIT_SYMBOL = "approaches_from_below"


def _imported_modules(path: Path) -> list[tuple[str, tuple[str, ...]]]:
    """Every module a source file imports, with the names taken from it.

    Args:
        path: Python file to parse.

    Returns:
        Pairs of module name and imported symbol names.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found: list[tuple[str, tuple[str, ...]]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            found.append((node.module, tuple(alias.name for alias in node.names)))
        elif isinstance(node, ast.Import):
            found.extend((alias.name, ()) for alias in node.names)
    return found


@pytest.mark.parametrize("path", sorted(PACKAGE.glob("*.py")), ids=lambda p: p.name)
def test_execution_does_not_import_the_layers_above_it(path: Path) -> None:
    """No module in this package reaches into features, entry or risk."""
    for module, _ in _imported_modules(path):
        for forbidden, reason in FORBIDDEN.items():
            assert not module.startswith(forbidden), f"{path.name} imports {module}; {reason}"


@pytest.mark.parametrize("path", sorted(PACKAGE.glob("*.py")), ids=lambda p: p.name)
def test_the_only_exit_import_is_the_order_geometry(path: Path) -> None:
    """``exit.fills.approaches_from_below`` is allowed; nothing else from Exit is.

    It states which side of a level an order is reached from, which is a fact
    about order types rather than about exits. Duplicating it here would give the
    system two tables that can disagree about which way a stop fills — the defect
    P07 avoided by making ``order_type`` a load-bearing field in the first place.
    """
    for module, names in _imported_modules(path):
        if not module.startswith("trading_system.exit"):
            continue
        assert module == "trading_system.exit.fills", f"{path.name} imports {module}"
        assert set(names) == {ALLOWED_EXIT_SYMBOL}, f"{path.name} imports {names} from Exit"
