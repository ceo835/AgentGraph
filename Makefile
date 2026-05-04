.PHONY: test lint format

TEST ?= tests

test:
	uv run pytest $(TEST)

lint:
	uv run ruff check .
	uv run ruff format . --diff

format:
	uv run ruff check --fix .
	uv run ruff format .
