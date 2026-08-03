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
