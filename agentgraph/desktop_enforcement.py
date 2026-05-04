"""Adapter-layer enforcement for the desktop assistant."""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agentgraph.contracts import AgentState, ToolResult
from agentgraph.desktop_loader import DesktopPolicyBundle, DesktopToolsCatalog
from agentgraph.runtime import ToolRegistry


@dataclass
class RollbackManager:
    """Stateless rollback helper driven by desktop policy configuration."""

    policy_bundle: DesktopPolicyBundle

    def apply(
        self,
        *,
        tool_ref: str,
        result: ToolResult,
    ) -> dict[str, Any]:
        if not isinstance(result.data, dict):
            return {"applied": False, "reason": "no_result_data"}

        strategy = self._strategy_for_tool(tool_ref)
        data = result.data
        if strategy == "move_back_to_source":
            source = data.get("source_path")
            target = data.get("target_path")
            if (
                source
                and target
                and Path(target).exists()
                and not Path(source).exists()
            ):
                Path(source).parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(target), str(source))
                return {"applied": True, "strategy": strategy}
            return {"applied": False, "strategy": strategy}

        if strategy == "delete_quarantined_file":
            target = data.get("path") or data.get("quarantine_path")
            if target and Path(target).exists():
                Path(target).unlink(missing_ok=True)
                return {"applied": True, "strategy": strategy}
            return {"applied": False, "strategy": strategy}

        if strategy == "delete_created_dir_if_empty":
            target = data.get("path")
            if target:
                path = Path(target)
                if path.exists() and path.is_dir() and not any(path.iterdir()):
                    path.rmdir()
                    return {"applied": True, "strategy": strategy}
            return {"applied": False, "strategy": strategy}

        if strategy == "restore_from_snapshot_or_quarantine":
            source = data.get("source_path")
            quarantine = data.get("quarantine_path")
            if (
                source
                and quarantine
                and Path(quarantine).exists()
                and not Path(source).exists()
            ):
                Path(source).parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(quarantine), str(source))
                return {"applied": True, "strategy": strategy}
            return {"applied": False, "strategy": strategy}

        return {"applied": False, "strategy": strategy, "reason": "no_runtime_rollback"}

    def _strategy_for_tool(self, tool_ref: str) -> str:
        rollback_policy = self.policy_bundle.rollback_policy
        if tool_ref == "fs.create_dir":
            return "delete_created_dir_if_empty"
        if tool_ref == "fs.move":
            return "move_back_to_source"
        if tool_ref == "fs.delete":
            return "restore_from_snapshot_or_quarantine"
        if tool_ref == "web.download":
            return "delete_quarantined_file"
        if tool_ref == "app.launch":
            return "close_launched_app_if_requested"
        if tool_ref == "package.install":
            return "uninstall_installed_package"
        if tool_ref == "package.update":
            return "reinstall_previous_version"
        if "no_op" in rollback_policy:
            return "no_op"
        return "no_runtime_rollback"


class DesktopEnforcedToolRegistry(ToolRegistry):
    """Tool registry with path guard, verification, rollback, and per-thread ledger."""

    def __init__(
        self,
        *,
        tools_catalog: DesktopToolsCatalog,
        policy_bundle: DesktopPolicyBundle,
    ) -> None:
        super().__init__()
        self.tools_catalog = tools_catalog
        self.policy_bundle = policy_bundle
        self.rollback_manager = RollbackManager(policy_bundle=policy_bundle)
        self._thread_events: dict[str, list[dict[str, Any]]] = {}

    def invoke(self, tool_ref: str, args: dict[str, Any]) -> ToolResult:
        thread_id = str(args.get("thread_id") or "thread")
        guarded_args = dict(args)
        path_error = self._guard_paths(tool_ref, guarded_args)
        risk_level = self._risk_level_for_tool(tool_ref)
        if path_error is not None:
            self._record_event(
                thread_id=thread_id,
                tool_ref=tool_ref,
                phase="path_guard",
                verification_passed=False,
                rollback_applied=False,
                risk_level=risk_level,
                details=[path_error],
            )
            return ToolResult(
                success=False,
                error=path_error,
                error_type="path_escape_blocked",
            )

        result = super().invoke(tool_ref, guarded_args)
        if not result.success:
            self._record_event(
                thread_id=thread_id,
                tool_ref=tool_ref,
                phase="tool_execution",
                verification_passed=False,
                rollback_applied=False,
                risk_level=risk_level,
                details=[result.error or "tool execution failed"],
            )
            return result

        verification = self._verify(tool_ref=tool_ref, result=result)
        if not verification["passed"]:
            rollback = self.rollback_manager.apply(tool_ref=tool_ref, result=result)
            self._record_event(
                thread_id=thread_id,
                tool_ref=tool_ref,
                phase="verification",
                verification_passed=False,
                rollback_applied=bool(rollback.get("applied")),
                risk_level=risk_level,
                details=verification["details"],
                rollback=rollback,
            )
            return ToolResult(
                success=False,
                error="; ".join(verification["details"]),
                error_type="verification_failed",
            )

        self._record_event(
            thread_id=thread_id,
            tool_ref=tool_ref,
            phase="verification",
            verification_passed=True,
            rollback_applied=False,
            risk_level=risk_level,
            details=verification["details"],
        )
        data = dict(result.data) if isinstance(result.data, dict) else result.data
        if isinstance(data, dict):
            data["_desktop_enforcement"] = {
                "verification_passed": True,
                "risk_level": risk_level,
                "details": verification["details"],
            }
        return ToolResult(
            success=True,
            data=data,
            error=result.error,
            error_type=result.error_type,
            metadata=result.metadata,
        )

    def pop_events(self, thread_id: str) -> list[dict[str, Any]]:
        return self._thread_events.pop(thread_id, [])

    def _guard_paths(self, tool_ref: str, args: dict[str, Any]) -> str | None:
        candidate_fields = {
            "fs.create_dir": ["path"],
            "fs.write_file": ["path"],
            "fs.read_file": ["path"],
            "fs.list_dir": ["path"],
            "fs.move": ["source_path", "target_path"],
            "fs.delete": ["path"],
            "web.download": ["target_filename"],
        }.get(tool_ref, [])
        if not candidate_fields:
            return None

        task = args.get("task") or {}
        metadata = task.get("metadata", {}) if isinstance(task, dict) else {}
        allowed_paths = metadata.get("allowed_paths") or []
        blocked_roots = self.policy_bundle.safety_policy.sandbox_policy.get(
            "blocked_roots", []
        )

        for field in candidate_fields:
            raw_value = str(args.get(field) or metadata.get(field) or "").strip()
            if not raw_value:
                continue
            normalized = self._normalize_path(raw_value)
            if self._has_escape_segments(raw_value):
                return f"path escape blocked for {field}: {raw_value}"
            if self._matches_blocked_root(normalized, blocked_roots):
                return f"path blocked by sandbox policy for {field}: {raw_value}"
            if self._looks_absolute_or_home(raw_value) and allowed_paths:
                if not any(
                    self._normalize_path(root) in normalized for root in allowed_paths
                ):
                    return f"path outside allowed_paths for {field}: {raw_value}"
        return None

    def _verify(self, *, tool_ref: str, result: ToolResult) -> dict[str, Any]:
        data = result.data if isinstance(result.data, dict) else {}
        details: list[str] = []
        if tool_ref == "fs.create_dir":
            path = Path(str(data.get("path", "")))
            if not path.exists():
                details.append("target path does not exist")
            if not path.is_dir():
                details.append("target path is not a directory")
        elif tool_ref == "fs.move":
            source = Path(str(data.get("source_path", "")))
            target = Path(str(data.get("target_path", "")))
            if source.exists():
                details.append("source path still exists after move")
            if not target.exists():
                details.append("target path missing after move")
        elif tool_ref == "fs.delete":
            source = Path(str(data.get("source_path", "")))
            quarantine = Path(str(data.get("quarantine_path", "")))
            if source.exists():
                details.append("source path still exists after delete")
            if not quarantine.exists():
                details.append("quarantine path missing after delete")
        elif tool_ref == "web.download":
            path = Path(str(data.get("path", "")))
            if not path.exists():
                details.append("downloaded file missing")
            allowed_extensions = self.tools_catalog.tools[tool_ref].allowed_extensions
            if path.suffix.lower() not in {ext.lower() for ext in allowed_extensions}:
                details.append("downloaded file extension is not allowed")
            max_size_mb = self.tools_catalog.tools[tool_ref].max_file_size_mb or 100
            if path.exists() and path.stat().st_size > max_size_mb * 1024 * 1024:
                details.append("downloaded file exceeds size limit")
        elif tool_ref in {"package.install", "package.update"}:
            if data.get("dry_run") is False and not data.get("installed_after", {}).get(
                "installed", True
            ):
                details.append(
                    "package manager verification did not confirm installation"
                )
        elif tool_ref == "app.launch":
            if data.get("dry_run") is False and not data.get("pid"):
                details.append("application launch did not return a pid")
        return {"passed": not details, "details": details or ["verification_passed"]}

    def _record_event(
        self,
        *,
        thread_id: str,
        tool_ref: str,
        phase: str,
        verification_passed: bool,
        rollback_applied: bool,
        risk_level: str,
        details: list[str],
        rollback: dict[str, Any] | None = None,
    ) -> None:
        self._thread_events.setdefault(thread_id, []).append(
            {
                "event": "desktop_enforcement",
                "tool_ref": tool_ref,
                "phase": phase,
                "verification_passed": verification_passed,
                "rollback_applied": rollback_applied,
                "risk_level": risk_level,
                "details": details,
                "rollback": rollback or {},
            }
        )

    def _risk_level_for_tool(self, tool_ref: str) -> str:
        assessment = self.policy_bundle.safety_policy.risk_assessment
        for level, tools in assessment.items():
            if tool_ref in tools:
                return level
        return "medium"

    @staticmethod
    def _has_escape_segments(raw_path: str) -> bool:
        return ".." in Path(raw_path).parts

    def _normalize_path(self, raw_path: str) -> str:
        normalized = raw_path
        if raw_path.startswith("~"):
            normalized = os.path.expanduser(raw_path)
        normalized = os.path.normcase(os.path.normpath(normalized))
        return normalized

    @staticmethod
    def _looks_absolute_or_home(raw_path: str) -> bool:
        return raw_path.startswith("~") or Path(raw_path).is_absolute()

    def _matches_blocked_root(
        self, normalized_path: str, blocked_roots: list[str]
    ) -> bool:
        for root in blocked_roots:
            normalized_root = self._normalize_path(root)
            if normalized_path.startswith(normalized_root):
                return True
        return False


def apply_desktop_enforcement_updates(graph: Any, *, thread_id: str) -> AgentState:
    """Flush registry enforcement events into graph state and update trust score."""
    registry = getattr(graph, "_desktop_tool_registry", None)
    config = {"configurable": {"thread_id": thread_id}}
    snapshot = graph.get_state(config)
    values = getattr(snapshot, "values", snapshot)
    state = AgentState.model_validate(values)
    if registry is None or not hasattr(registry, "pop_events"):
        return state

    events = registry.pop_events(thread_id)

    audit_log = list(state.audit_log)
    shared_context = dict(state.shared_context)
    artifacts = [
        dict(item) if isinstance(item, dict) else item for item in state.artifacts
    ]
    output_candidate = (
        dict(state.output_candidate)
        if isinstance(state.output_candidate, dict)
        else state.output_candidate
    )
    desktop_context = dict(shared_context.get("desktop_context") or {})
    trust_lifecycle = getattr(
        getattr(registry, "policy_bundle", None), "trust_lifecycle", None
    )
    default_trust = (
        trust_lifecycle.initial_value if trust_lifecycle is not None else 0.5
    )
    trust_score = float(desktop_context.get("trust_score", default_trust))
    last_actions = list(desktop_context.get("last_actions") or [])

    for event in events:
        audit_log.append(event)
        if len(audit_log) > 50:
            audit_log = audit_log[-50:]
        last_actions.append(event)
        if len(last_actions) > 20:
            last_actions = last_actions[-20:]
        if event.get("verification_passed") is True:
            trust_score = min(1.0, trust_score + 0.1)
        else:
            trust_score = max(0.0, trust_score - 0.2)

    artifact_mapping = getattr(
        getattr(registry, "tools_catalog", None), "artifact_mapping", {}
    )
    latest_domain_tag: str | None = None
    latest_tool_ref: str | None = None
    for artifact in artifacts:
        if not isinstance(artifact, dict):
            continue
        tool_ref = str(artifact.get("tool_ref") or "")
        latest_tool_ref = tool_ref or latest_tool_ref
        mapped = _artifact_defaults_for_tool(tool_ref, artifact_mapping)
        if not mapped:
            continue
        artifact.update(mapped)
        latest_domain_tag = mapped.get("domain_tag", latest_domain_tag)

    if isinstance(output_candidate, dict):
        if latest_domain_tag:
            output_candidate["domain_tag"] = latest_domain_tag
        if latest_tool_ref:
            mapped = _artifact_defaults_for_tool(latest_tool_ref, artifact_mapping)
            if mapped.get("artifact_type"):
                output_candidate["artifact_type"] = mapped["artifact_type"]

    desktop_context["trust_score"] = trust_score
    desktop_context["last_actions"] = last_actions
    shared_context["desktop_context"] = desktop_context
    if (
        not events
        and artifacts == state.artifacts
        and output_candidate == state.output_candidate
    ):
        return state

    graph.update_state(
        config,
        {
            "audit_log": audit_log,
            "shared_context": shared_context,
            "artifacts": artifacts,
            "output_candidate": output_candidate,
        },
        as_node="interface",
    )
    updated = graph.get_state(config)
    return AgentState.model_validate(getattr(updated, "values", updated))


def _artifact_defaults_for_tool(
    tool_ref: str, artifact_mapping: dict[str, dict[str, str]]
) -> dict[str, str]:
    for prefix, defaults in artifact_mapping.items():
        normalized_prefix = prefix[:-1] if prefix.endswith("*") else prefix
        if tool_ref.startswith(normalized_prefix):
            return dict(defaults)
    return {}
