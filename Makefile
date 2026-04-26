.PHONY: install train test test-fast lint api mlflow

install:
	pip install -e ".[dev]"

train:
	python -m src.main

test:
	pytest -q

test-fast:
	pytest -q -m "not e2e"

lint:
	ruff check src tests

api:
	uvicorn src.api:app --host 127.0.0.1 --port 8000

mlflow:
	mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5001
