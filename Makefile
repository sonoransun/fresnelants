PY ?= .venv/bin/python
PIP ?= .venv/bin/pip
RUFF ?= .venv/bin/ruff
MYPY ?= .venv/bin/mypy
PYTEST ?= .venv/bin/pytest

.PHONY: help install dev test test-fast test-fullwave lint fmt mypy gallery docs all clean

help:
	@echo "Common targets:"
	@echo "  make install      — install package + base deps"
	@echo "  make dev          — install with [dev] extras"
	@echo "  make test         — run pytest (skips fullwave)"
	@echo "  make test-fast    — pytest with -x --tb=short"
	@echo "  make test-fullwave— pytest -m fullwave (Meep / NEC)"
	@echo "  make lint         — ruff check"
	@echo "  make fmt          — ruff format"
	@echo "  make mypy         — type check src/"
	@echo "  make gallery      — regenerate docs/img/ from the library"
	@echo "  make docs         — build the MkDocs site"
	@echo "  make all          — lint + fmt --check + mypy + tests + gallery"

install:
	$(PIP) install -e .

dev:
	$(PIP) install -e ".[dev]"

test:
	$(PYTEST) -m "not fullwave"

test-fast:
	$(PYTEST) -m "not fullwave" -x --tb=short

test-fullwave:
	$(PYTEST) -m fullwave

lint:
	$(RUFF) check .

fmt:
	$(RUFF) format .

mypy:
	$(MYPY) src

gallery:
	$(PY) docs/generate_figures.py

docs:
	mkdocs build --strict

all: lint
	$(RUFF) format --check .
	$(MYPY) src || true
	$(PYTEST) -m "not fullwave"
	$(PY) docs/generate_figures.py

clean:
	rm -rf .pytest_cache .ruff_cache .mypy_cache .hypothesis
	rm -rf build dist *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
