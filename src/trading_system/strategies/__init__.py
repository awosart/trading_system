"""The strategy contract shared by Entry Engine, Exit Engine and the strategy database.

Also exports the semantic validator layered above it.
"""

from trading_system.strategies.schema import StrategySpec, strategy_json_schema
from trading_system.strategies.validator import (
    ParsedStrategy,
    Severity,
    ValidationIssue,
    check_unique_ids,
    load_spec,
    validate_paths,
    validate_spec,
)

__all__ = [
    "ParsedStrategy",
    "Severity",
    "StrategySpec",
    "ValidationIssue",
    "check_unique_ids",
    "load_spec",
    "strategy_json_schema",
    "validate_paths",
    "validate_spec",
]
