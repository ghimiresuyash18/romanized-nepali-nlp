.PHONY: install test lint format clean build

install:
	python3 -m pip install -e ".[dev]"

test:
	pytest

lint:
	ruff check .

format:
	ruff format .

clean:
	rm -rf build/ dist/ *.egg-info .pytest_cache src/*.egg-info

build: clean
	python3 -m pip install --upgrade build && python3 -m build
