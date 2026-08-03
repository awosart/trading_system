#!/bin/bash

# Создаём структуру папок
mkdir -p src/trading_system/core
mkdir -p src/trading_system/data
mkdir -p src/trading_system/indicators
mkdir -p src/trading_system/regime
mkdir -p src/trading_system/strategies
mkdir -p src/trading_system/entries
mkdir -p src/trading_system/exits
mkdir -p src/trading_system/risk
mkdir -p src/trading_system/backtest
mkdir -p src/trading_system/analytics
mkdir -p configs
mkdir -p tests

# pyproject.toml
cat > pyproject.toml << 'PYEOF'
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "trading-system"
version = "0.1.0"
description = "Modular trading system with risk management"
requires-python = ">=3.12"

dependencies = [
    "pydantic>=2.0",
    "polars>=0.20",
    "duckdb>=0.9",
    "sqlalchemy>=2.0",
    "structlog>=24.0",
    "typer>=0.9",
    "streamlit>=1.28",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0",
    "pytest-cov>=4.0",
    "hypothesis>=6.0",
    "ruff>=0.1",
    "mypy>=1.0",
]

[tool.ruff]
line-length = 100
target-version = "py312"

[tool.mypy]
strict = true
python_version = "3.12"

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = "test_*.py"
PYEOF

# .gitignore
cat > .gitignore << 'GIEOF'
__pycache__/
*.py[cod]
*$py.class
.pytest_cache/
.mypy_cache/
.ruff_cache/
.venv/
venv/
ENV/
*.egg-info/
dist/
build/
.DS_Store
*.parquet
data/
.env
*.log
.claude/
GIEOF

# Пустые init файлы
touch src/trading_system/__init__.py
touch src/trading_system/core/__init__.py
touch tests/__init__.py

# core/types.py
cat > src/trading_system/core/types.py << 'TYPEEOF'
"""Core domain types."""

from enum import Enum
from dataclasses import dataclass
from typing import NewType

Price = NewType("Price", float)
Volume = NewType("Volume", float)


class Side(str, Enum):
    """Trade direction."""
    BUY = "BUY"
    SELL = "SELL"


class OrderType(str, Enum):
    """Order type."""
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"


class Timeframe(str, Enum):
    """Trading timeframe."""
    M1 = "M1"
    M5 = "M5"
    M15 = "M15"
    H1 = "H1"
    H4 = "H4"
    D1 = "D1"


@dataclass(frozen=True)
class Bar:
    """OHLCV bar."""
    timestamp: str
    open: Price
    high: Price
    low: Price
    close: Price
    volume: Volume


@dataclass
class Signal:
    """Trading signal."""
    strategy_id: str
    symbol: str
    direction: Side
    quality: float  # 0..1


@dataclass
class Order:
    """Order to place."""
    symbol: str
    side: Side
    size: float
    price: Price | None = None


@dataclass
class Position:
    """Open position."""
    symbol: str
    side: Side
    size: float
    entry_price: Price
TYPEEOF

# core/exceptions.py
cat > src/trading_system/core/exceptions.py << 'EXCEOF'
"""Custom exceptions."""


class TradingSystemError(Exception):
    """Base exception."""
    pass


class DataError(TradingSystemError):
    """Data loading error."""
    pass


class ValidationError(TradingSystemError):
    """Validation error."""
    pass


class RiskLimitError(TradingSystemError):
    """Risk limit exceeded."""
    pass


class ExecutionError(TradingSystemError):
    """Order execution error."""
    pass
EXCEOF

# core/config.py
cat > src/trading_system/core/config.py << 'CFGEOF'
"""Configuration management."""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings."""
    
    debug: bool = False
    log_level: str = "INFO"
    
    class Config:
        env_file = ".env"
        case_sensitive = False


settings = Settings()
CFGEOF

# core/logging.py
cat > src/trading_system/core/logging.py << 'LOGEOF'
"""Logging setup."""

import logging


def setup_logging(level: str = "INFO") -> None:
    """Setup JSON logging."""
    logging.basicConfig(level=getattr(logging, level))
LOGEOF

# cli.py
cat > src/trading_system/cli.py << 'CLIEOF'
"""Command-line interface."""

import typer

app = typer.Typer()


@app.command()
def data():
    """Data management."""
    typer.echo("Data commands")


@app.command()
def backtest():
    """Run backtest."""
    typer.echo("Backtest")


@app.command()
def validate():
    """Validate strategies."""
    typer.echo("Validate")


@app.command()
def ui():
    """Launch UI."""
    typer.echo("UI")


if __name__ == "__main__":
    app()
CLIEOF

# tests/conftest.py
cat > tests/conftest.py << 'CONFEOF'
"""Test configuration."""

import pytest


@pytest.fixture
def sample_bar():
    """Sample OHLCV bar."""
    from trading_system.core.types import Bar, Price, Volume
    return Bar(
        timestamp="2024-01-01T00:00:00Z",
        open=Price(1.0),
        high=Price(1.1),
        low=Price(0.9),
        close=Price(1.05),
        volume=Volume(1000),
    )
CONFEOF

# Makefile
cat > Makefile << 'MAKEOF'
.PHONY: install lint typecheck test format all

install:
	python3 -m venv venv
	. venv/bin/activate && pip install -U pip && pip install uv && uv pip install -e .[dev]

lint:
	. venv/bin/activate && ruff check src/ tests/

typecheck:
	. venv/bin/activate && mypy --strict src/

test:
	. venv/bin/activate && pytest tests/ -v

format:
	. venv/bin/activate && ruff format src/ tests/

all: lint typecheck test
	@echo "✓ All checks passed"
MAKEOF

echo "✓ Project structure created"
