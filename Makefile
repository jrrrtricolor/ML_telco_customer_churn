.PHONY: install lint format test test-unit test-e2e train api mlflow-ui clean

PYTHON ?= $(if $(wildcard .venv/bin/python),.venv/bin/python,$(if $(wildcard venv/bin/python),venv/bin/python,python))
PIP ?= $(PYTHON) -m pip
APP_HOST ?= 127.0.0.1
APP_PORT ?= 8000
MLFLOW_PORT ?= 5001

install:
	$(PIP) install --upgrade pip
	$(PIP) install -e ".[dev]"

lint:
	$(PYTHON) -m ruff check src tests

format:
	$(PYTHON) -m ruff check src tests --fix
	$(PYTHON) -m black src tests

test:
	$(PYTHON) -m pytest -q

test-unit:
	$(PYTHON) -m pytest -q -m "not e2e"

test-e2e:
	$(PYTHON) -m pytest -q -m e2e

train:
	$(PYTHON) -m src.main

api:
	$(PYTHON) -m uvicorn src.api:app --host $(APP_HOST) --port $(APP_PORT)

mlflow-ui:
	$(PYTHON) -m mlflow ui --backend-store-uri sqlite:///mlflow.db --port $(MLFLOW_PORT)

clean:
	find . -type d -name "__pycache__" -prune -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -prune -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -prune -exec rm -rf {} +
