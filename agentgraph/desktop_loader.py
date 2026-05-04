"""YAML-backed loading for the desktop-assistant v1.2 configuration layer."""

from __future__ import annotations

from importlib.resources import files
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from agentgraph.contracts import AgentConfig
from agentgraph.roster_loader import _load_yaml_payload
from agentgraph.runtime import AgentRegistry

DESKTOP_EXECUTOR_YAML = ("agents", "desktop_executor.yaml")
DESKTOP_TOOLS_YAML = ("tools", "desktop_tools.yaml")
DESKTOP_POLICIES_YAML = ("policies", "desktop_policies.yaml")
DESKTOP_WORKFLOW_YAML = ("workflows", "autonomous_desktop_loop.yaml")


class DesktopToolSpec(BaseModel):
    """Declarative tool metadata for the desktop-assistant layer."""

    description: str
    input_schema: dict[str, Any] = Field(default_factory=dict)
    side_effect_level: str
    requires_hitl: bool
    sandbox_scope: str
    dry_run_preview: bool
    is_idempotent: bool
    rollback_strategy: str
    verification_policy: str
    confirmation_required: str | None = None
    supported_managers: list[str] = Field(default_factory=list)
    allowed_sources: list[str] = Field(default_factory=list)
    allowed_extensions: list[str] = Field(default_factory=list)
    max_file_size_mb: int | None = None
    allowed_apps: list[str] = Field(default_factory=list)
    artifact_defaults: dict[str, str] = Field(default_factory=dict)


class DesktopToolsCatalog(BaseModel):
    """Loaded desktop-tool definitions keyed by tool ref."""

    version: int
    tools: dict[str, DesktopToolSpec]
    artifact_mapping: dict[str, dict[str, str]] = Field(default_factory=dict)


class SafetyPolicy(BaseModel):
    """Safety and sandbox controls for desktop automation."""

    risk_assessment: dict[str, list[str]]
    hitl_rules: list[dict[str, Any]]
    sandbox_policy: dict[str, Any]
    path_and_app_resolution: dict[str, Any]
    planning_policy: dict[str, Any] = Field(default_factory=dict)
    path_inference_policy: dict[str, Any] = Field(default_factory=dict)


class QuarantinePolicy(BaseModel):
    """Quarantine behavior for staged downloads and edits."""

    enabled: bool
    quarantine_root: str
    staged_downloads: bool
    staged_edits: bool
    auto_scan: bool
    scan_required_for_export: bool
    cleanup_timeout_hours: int
    cleanup_on_failed_verification: bool


class ExportPolicy(BaseModel):
    """Sandbox export policy for moving artifacts into user paths."""

    sandbox_only_default: bool
    allowed_export_modes: list[str]
    rules: list[dict[str, Any]]


class TrustLifecycle(BaseModel):
    """Trust-score update policy derived from audit outcomes."""

    initial_value: float
    update_rule: str
    cap_rules: list[Any]


class DesktopPolicyBundle(BaseModel):
    """Complete desktop policy bundle loaded from YAML."""

    version: int
    safety_policy: SafetyPolicy
    verification_policy: dict[str, dict[str, Any]]
    rollback_policy: dict[str, dict[str, Any]]
    quarantine_policy: QuarantinePolicy
    export_policy: ExportPolicy
    trust_lifecycle: TrustLifecycle


class WorkflowStateUsage(BaseModel):
    """How the workflow reads and writes existing runtime state containers."""

    task_metadata_reads: list[str] = Field(default_factory=list)
    shared_context_reads: list[str] = Field(default_factory=list)
    shared_context_writes: list[str] = Field(default_factory=list)
    audit_log_writes: list[str] = Field(default_factory=list)
    memory_refs_writes: list[str] = Field(default_factory=list)


class WorkflowMatrixEntry(BaseModel):
    """Intent-to-tool mapping row for UI and planner consumption."""

    request: str
    capability: str
    tool: str
    risk_level: Literal["low", "medium", "high", "destructive"]
    hitl_required: bool
    operational_check: str


class DesktopWorkflowSpec(BaseModel):
    """Loaded workflow specification for the desktop-assistant loop."""

    name: str
    entrypoint: str
    state_usage: WorkflowStateUsage
    state_diagram_mermaid: str
    routing_predicates: dict[str, dict[str, Any]]
    hitl_points: dict[str, Any]
    fallback_chain: dict[str, list[str]]
    intent_tool_policy_matrix: list[WorkflowMatrixEntry]


class DesktopWorkflowBundle(BaseModel):
    """Top-level workflow YAML payload."""

    version: int
    workflow: DesktopWorkflowSpec


def default_desktop_executor_yaml_path() -> Path:
    """Return the packaged desktop executor YAML path."""
    return _config_resource_path(*DESKTOP_EXECUTOR_YAML)


def default_desktop_tools_yaml_path() -> Path:
    """Return the packaged desktop tools YAML path."""
    return _config_resource_path(*DESKTOP_TOOLS_YAML)


def default_desktop_policies_yaml_path() -> Path:
    """Return the packaged desktop policies YAML path."""
    return _config_resource_path(*DESKTOP_POLICIES_YAML)


def default_desktop_workflow_yaml_path() -> Path:
    """Return the packaged desktop workflow YAML path."""
    return _config_resource_path(*DESKTOP_WORKFLOW_YAML)


def load_desktop_executor_config(path: str | Path | None = None) -> AgentConfig:
    """Load the desktop executor profile as a runtime-ready `AgentConfig`."""
    payload = _load_yaml_payload(path or default_desktop_executor_yaml_path())
    return AgentConfig.model_validate(payload["agent"])


def register_desktop_executor(
    agent_registry: AgentRegistry,
    *,
    path: str | Path | None = None,
) -> AgentConfig:
    """Load and register the desktop executor into an existing registry."""
    agent = load_desktop_executor_config(path)
    agent_registry.register(agent)
    return agent


def load_desktop_tools_catalog(
    path: str | Path | None = None,
) -> DesktopToolsCatalog:
    """Load desktop tool metadata from YAML."""
    payload = _load_yaml_payload(path or default_desktop_tools_yaml_path())
    return DesktopToolsCatalog.model_validate(payload)


def load_desktop_policy_bundle(
    path: str | Path | None = None,
) -> DesktopPolicyBundle:
    """Load desktop safety, verification, rollback, and trust policies."""
    payload = _load_yaml_payload(path or default_desktop_policies_yaml_path())
    return DesktopPolicyBundle.model_validate(payload)


def load_desktop_workflow_bundle(
    path: str | Path | None = None,
) -> DesktopWorkflowBundle:
    """Load the autonomous desktop workflow definition from YAML."""
    payload = _load_yaml_payload(path or default_desktop_workflow_yaml_path())
    return DesktopWorkflowBundle.model_validate(payload)


def _config_resource_path(*parts: str) -> Path:
    return Path(str(files("agentgraph.config").joinpath(*parts)))
