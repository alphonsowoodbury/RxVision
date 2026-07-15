.PHONY: install test lint format

install:
	python -m pip install -e ".[test,api,data]"

test:
	pytest

lint:
	ruff check .
	ruff format --check .

format:
	ruff check --fix .
	ruff format .
