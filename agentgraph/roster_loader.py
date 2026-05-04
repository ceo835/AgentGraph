"""YAML-backed loading for starter agent rosters."""

from __future__ import annotations

from importlib.resources import files
from pathlib import Path
from typing import Any

from agentgraph.contracts import AgentConfig

try:
    import yaml  # type: ignore[import-untyped]
except ImportError:  # pragma: no cover - exercised only without optional deps
    yaml = None


DEFAULT_AGENTS_YAML = "agents.yaml"
RESEARCH_TOOL_PLACEHOLDER = "__RESEARCH_TOOL_REF__"
VALIDATOR_TOOL_PLACEHOLDER = "__VALIDATOR_TOOL_REF__"


def default_agents_yaml_path() -> Path:
    """Return the packaged default YAML roster path."""
    return Path(str(files("agentgraph.config").joinpath(DEFAULT_AGENTS_YAML)))


def load_core_agent_configs(
    *,
    include_critic: bool = False,
    research_tool_ref: str = "research.search",
    path: str | Path | None = None,
) -> list[AgentConfig]:
    """Load core agent profiles from YAML and resolve runtime placeholders."""
    payload = _load_yaml_payload(path)
    agents = payload.get("core_agents", [])
    if not include_critic:
        agents = [agent for agent in agents if agent.get("agent_id") != "critic"]
    resolved = [
        _resolve_placeholders(
            agent,
            {
                RESEARCH_TOOL_PLACEHOLDER: research_tool_ref,
            },
        )
        for agent in agents
    ]
    return [AgentConfig.model_validate(agent) for agent in resolved]


def load_specialist_agent_config(
    specialist_name: str,
    *,
    validator_tool_ref: str = "validator.logic",
    path: str | Path | None = None,
) -> AgentConfig:
    """Load one specialist profile from YAML and resolve tool placeholders."""
    payload = _load_yaml_payload(path)
    specialists = payload.get("specialists", {})
    if specialist_name not in specialists:
        raise KeyError(f"unknown specialist profile: {specialist_name}")
    resolved = _resolve_placeholders(
        specialists[specialist_name],
        {
            VALIDATOR_TOOL_PLACEHOLDER: validator_tool_ref,
        },
    )
    return AgentConfig.model_validate(resolved)


def _load_yaml_payload(path: str | Path | None) -> dict[str, Any]:
    if yaml is None:
        raise ImportError(
            "PyYAML is required to load agent rosters. Install `pip install -e .[config]`."
        )
    source = Path(path) if path is not None else default_agents_yaml_path()
    with source.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError("agents YAML must contain a top-level mapping")
    return payload


def _resolve_placeholders(value: Any, replacements: dict[str, str]) -> Any:
    if isinstance(value, str):
        return replacements.get(value, value)
    if isinstance(value, list):
        return [_resolve_placeholders(item, replacements) for item in value]
    if isinstance(value, dict):
        return {
            key: _resolve_placeholders(item, replacements)
            for key, item in value.items()
        }
    return value
