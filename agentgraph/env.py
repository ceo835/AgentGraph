"""Environment loading helpers for local AgentGraph runs."""

from __future__ import annotations

import os
from pathlib import Path


def load_env_file(
    path: str | os.PathLike[str] = ".env",
    *,
    override: bool = False,
) -> dict[str, str]:
    """Load simple `KEY=VALUE` pairs from a local `.env` file.

    This parser intentionally stays minimal:

    - empty lines and comment lines are ignored;
    - inline comments are not supported;
    - single and double quotes around values are stripped;
    - existing process variables are preserved unless `override=True`.
    """

    env_path = Path(path)
    if not env_path.exists():
        return {}

    loaded: dict[str, str] = {}
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        normalized_key = key.strip()
        normalized_value = _strip_quotes(value.strip())
        if override or normalized_key not in os.environ:
            os.environ[normalized_key] = normalized_value
        loaded[normalized_key] = os.environ.get(normalized_key, normalized_value)
    return loaded


def _strip_quotes(value: str) -> str:
    if len(value) >= 2 and value[:1] == value[-1:] and value[:1] in {'"', "'"}:
        return value[1:-1]
    return value
