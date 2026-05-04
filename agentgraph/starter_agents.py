"""Starter agent profiles for a practical multi-agent AgentGraph setup."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from agentgraph.contracts import (
    AgentConfig,
    ToolResult,
)
from agentgraph.roster_loader import (
    load_core_agent_configs,
    load_specialist_agent_config,
)
from agentgraph.runtime import AgentRegistry, ToolRegistry
from agentgraph.validator_adapter import register_validator_agent


def build_starter_agent_configs(
    *,
    research_tool_ref: str = "research.search",
    include_critic: bool = False,
    roster_path: str | Path | None = None,
) -> list[AgentConfig]:
    """Return the recommended starter set of AgentGraph personas.

    The returned profiles are data-only `AgentConfig` declarations. They provide
    the non-tooling core of the stack:

    - `coordinator`
    - `researcher`
    - `synthesizer`
    - `memory_curator`
    - optional `critic`

    Planner and validator are registered separately through
    `register_starter_specialists()` because they may require runtime tool
    handlers.
    """

    return load_core_agent_configs(
        include_critic=include_critic,
        research_tool_ref=research_tool_ref,
        path=roster_path,
    )


def describe_agent_configs(agents: list[AgentConfig]) -> list[dict[str, Any]]:
    """Return a stable JSON-serializable summary of agent profiles."""
    return [
        {
            "agent_id": agent.agent_id,
            "role": agent.role,
            "goal": agent.goal,
            "backstory": agent.backstory,
            "tools": [binding.tool_ref for binding in agent.tools],
            "capabilities": [
                {
                    "name": capability.name,
                    "summary": capability.summary,
                    "keywords": capability.keywords,
                    "domains": capability.domains,
                }
                for capability in agent.capabilities
            ],
            "delegation_policy": {
                "confidence_threshold": agent.delegation_policy.confidence_threshold,
                "fallback_strategy": agent.delegation_policy.fallback_strategy,
                "policy_weight": agent.delegation_policy.policy_weight,
            },
            "memory_policy": {
                "auto_sync": agent.memory_policy.auto_sync,
                "link_generation": agent.memory_policy.link_generation,
                "versioning": agent.memory_policy.versioning,
                "graph_linking": agent.memory_policy.graph_linking,
            },
        }
        for agent in agents
    ]


def register_starter_specialists(
    *,
    agent_registry: AgentRegistry,
    tool_registry: ToolRegistry,
    include_planner: bool = True,
    include_validator: bool = True,
    validator_tool_ref: str = "validator.logic",
    roster_path: str | Path | None = None,
) -> list[AgentConfig]:
    """Register runtime specialists for the starter setup.

    This adds the tool-aware roles that are easier to wire at runtime:

    - `planner`
    - `validator`
    """

    registered: list[AgentConfig] = []
    if include_planner:
        planner = load_specialist_agent_config(
            "planner",
            validator_tool_ref=validator_tool_ref,
            path=roster_path,
        )
        agent_registry.register(planner)
        registered.append(planner)
    if include_validator:
        register_validator_agent(
            agent_registry=agent_registry,
            tool_registry=tool_registry,
            tool_ref=validator_tool_ref,
            validator_handler=_starter_validator_handler,
        )
        validator = load_specialist_agent_config(
            "validator",
            validator_tool_ref=validator_tool_ref,
            path=roster_path,
        )
        agent_registry.register(validator)
        registered.append(validator)
    return registered


def _starter_validator_handler(args: dict[str, Any]) -> ToolResult:
    """Default validator specialist for the starter setup."""
    output = args.get("output_candidate", {}) or {}
    summary = str(output.get("summary", "")).strip()
    citations = output.get("citations", [])
    errors: list[str] = []
    if not summary:
        errors.append("summary is empty")
    if not citations:
        errors.append("citations are required")
    return ToolResult(
        success=True,
        data={
            "valid": not errors,
            "errors": errors,
            "source": "starter-validator",
        },
    )
