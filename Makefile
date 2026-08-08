.PHONY: install lint typecheck test bench format all

# pyproject pins requires-python to >=3.12,<3.13, so the interpreter must be
# 3.12. Override if python3.12 is not on PATH:
#   make install PYTHON=~/.local/share/uv/python/cpython-3.12.11-*/bin/python3.12
PYTHON ?= python3.12
# .venv, not venv: uv is this project's dependency manager, and every uv command
# that touches an environment (`uv sync`, `uv run`) targets .venv unconditionally.
# A Makefile pointing anywhere else means a bare `uv sync` silently builds a
# second environment that `make all` never looks at.
VENV := $(CURDIR)/.venv
BIN := $(VENV)/bin

# Every extra except `ui`, which is deliberately left out: it pulls streamlit,
# and streamlit pulls pandas, which this project does not use (P02).
# `optimization` is installed even though the runtime does without it — OptunaSearch
# is production code and mypy --strict has to see optuna's types to check it. The
# rule that a plain backtest imports without extras is a runtime property, and it
# is enforced where it belongs, by the try/except ImportError around the import.
install:
	$(PYTHON) -m venv $(VENV)
	$(BIN)/python -m pip install -qU pip uv
	VIRTUAL_ENV=$(VENV) $(BIN)/uv pip install -e ".[dev,analytics,optimization]"

lint:
	$(BIN)/ruff check src/ tests/
	$(BIN)/ruff format --check src/ tests/

typecheck:
	$(BIN)/mypy --strict src/

test:
	$(BIN)/pytest tests/ -v

# Performance budgets. Deselected from `make test` because building a
# million-bar frame costs more than the rest of the suite put together.
bench:
	$(BIN)/pytest tests/ -v -m benchmark

format:
	$(BIN)/ruff format src/ tests/

all: lint typecheck test
	@echo "✓ All checks passed"
