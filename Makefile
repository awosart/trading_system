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

# `uv sync`, with no extra flags, so that this target and the command a developer
# types by hand are the same operation against the same lockfile — the set of
# packages to install is pyproject's business (see [tool.uv] default-groups), not
# something spelled out a second time here, where the two spellings could drift.
# pip is used once, only to bootstrap a uv to run that sync with; the dev group
# declares uv itself, so every later sync keeps it.
install:
	$(PYTHON) -m venv $(VENV)
	$(BIN)/python -m pip install -qU pip uv
	$(BIN)/uv sync

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
