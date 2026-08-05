.PHONY: install lint typecheck test bench format all

# pyproject pins requires-python to >=3.12,<3.13, so the interpreter must be
# 3.12. Override if python3.12 is not on PATH:
#   make install PYTHON=~/.local/share/uv/python/cpython-3.12.11-*/bin/python3.12
PYTHON ?= python3.12
VENV := $(CURDIR)/venv
BIN := $(VENV)/bin

install:
	$(PYTHON) -m venv $(VENV)
	$(BIN)/python -m pip install -qU pip uv
	VIRTUAL_ENV=$(VENV) $(BIN)/uv pip install -e ".[dev,analytics]"

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
