"""Schema-version wrappers and checkpoint migration helpers."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, model_validator

from agentgraph.contracts import AgentState, ThreadStatus

CURRENT_AGENT_STATE_SCHEMA_VERSION = 2


class VersionedAgentState(BaseModel):
    """Backward-compatible envelope for serialized agent state."""

    schema_version: int = Field(default=CURRENT_AGENT_STATE_SCHEMA_VERSION)
    state: AgentState

    @model_validator(mode="before")
    @classmethod
    def migrate_legacy_payload(cls, value: Any) -> Any:
        if isinstance(value, dict) and "state" not in value:
            return {
                "schema_version": value.get(
                    "schema_version", CURRENT_AGENT_STATE_SCHEMA_VERSION
                ),
                "state": migrate_agent_state_payload(value),
            }
        return value


def migrate_agent_state_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Upgrade legacy serialized state payloads to the current defaults."""
    migrated = dict(payload)
    migrated.setdefault("schema_version", CURRENT_AGENT_STATE_SCHEMA_VERSION)
    migrated.setdefault("protocol_messages", [])
    migrated.setdefault("route_candidates", [])
    migrated.setdefault("shared_context", {})
    migrated.setdefault("artifacts", [])
    migrated.setdefault("output_candidate", None)
    migrated.setdefault("schema_validation", None)
    migrated.setdefault("logic_validation", None)
    migrated.setdefault("memory_refs", [])
    migrated.setdefault("sync_ticket_id", None)
    migrated.setdefault("human_checkpoint", None)
    migrated.setdefault(
        "retry_counters", {"schema": 0, "logic": 0, "tool": 0, "repair": 0}
    )
    migrated.setdefault("errors", [])
    migrated.setdefault("audit_log", [])
    migrated.setdefault("current_agent", None)
    migrated.setdefault("status", ThreadStatus.INIT)
    migrated.setdefault("messages", [])
    if "task_spec" in migrated and "task" not in migrated:
        migrated["task"] = migrated.pop("task_spec")
    return migrated


def load_versioned_agent_state(payload: dict[str, Any]) -> VersionedAgentState:
    """Validate a serialized state payload after migration."""
    migrated = migrate_agent_state_payload(payload)
    return VersionedAgentState(
        schema_version=migrated["schema_version"],
        state=AgentState.model_validate(migrated),
    )


def migrate_checkpoint_agent_state(
    checkpoint: dict[str, Any],
    *,
    channel_key: str = "agent_state",
) -> VersionedAgentState | None:
    """Load a versioned state from a raw checkpoint payload."""
    channel_values = checkpoint.get("channel_values", {})
    payload = channel_values.get(channel_key)
    if payload is None:
        return None
    return load_versioned_agent_state(payload)
