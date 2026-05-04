set shell := ["powershell", "-NoProfile", "-Command"]

test:
  uv run pytest tests

lint:
  uv run ruff check .
  uv run ruff format . --check

format:
  uv run ruff check --fix .
  uv run ruff format .

build:
  uv build
