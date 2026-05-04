"""Helpers for bootstrapping the desktop-assistant v1.2 layer."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from pydantic import BaseModel, Field

from agentgraph.autonomy_tools import register_autonomy_toolkit
from agentgraph.contracts import (
    AgentConfig,
    AgentState,
    FactLogicValidationMode,
    HITLPoint,
    SideEffectLevel,
    TaskSpec,
    ThreadStatus,
    ToolResult,
)
from agentgraph.demo import (
    FileSystemActionReport,
    ResearchReport,
    build_demo_task,
)
from agentgraph.desktop_enforcement import (
    DesktopEnforcedToolRegistry,
    apply_desktop_enforcement_updates,
)
from agentgraph.desktop_loader import (
    DesktopPolicyBundle,
    DesktopToolsCatalog,
    DesktopWorkflowBundle,
    load_desktop_executor_config,
    load_desktop_policy_bundle,
    load_desktop_tools_catalog,
    load_desktop_workflow_bundle,
)
from agentgraph.env import load_env_file
from agentgraph.openai_adapter import register_openai_research_tool
from agentgraph.runtime import Crew, ToolRegistry, stream_envelopes
from agentgraph.starter_agents import (
    build_starter_agent_configs,
    register_starter_specialists,
)


class DesktopActionReport(BaseModel):
    """Generic structured output for non-filesystem desktop actions."""

    task_id: str
    summary: str = Field(min_length=8)
    citations: list[str]
    artifacts_used: int = Field(ge=1)
    confidence: float = Field(ge=0.3)


@dataclass
class DesktopCrewBundle:
    """Resolved desktop-assistant assembly built from YAML plus runtime adapters."""

    crew: Crew
    desktop_executor: AgentConfig
    tools_catalog: DesktopToolsCatalog
    policy_bundle: DesktopPolicyBundle
    workflow_bundle: DesktopWorkflowBundle
    registered_tool_refs: list[str]
    stubbed_tool_refs: list[str]


@dataclass
class DesktopPlanExecution:
    """Result of experimental sequential execution for desktop subtasks."""

    root_thread_id: str
    step_states: list[AgentState]
    step_events: list[list[dict[str, Any]]]
    stopped_reason: str | None = None


def build_desktop_demo_crew(
    *,
    env_file: str = ".env",
    workspace_root: str = ".",
    include_starter_agents: bool = True,
    include_critic: bool = False,
    include_planner: bool = True,
    include_validator: bool = True,
) -> DesktopCrewBundle:
    """Assemble a desktop-capable crew from the v1.2 YAML configuration layer."""
    load_env_file(env_file)
    tools_catalog = load_desktop_tools_catalog()
    policy_bundle = load_desktop_policy_bundle()
    workflow_bundle = load_desktop_workflow_bundle()

    tool_registry = DesktopEnforcedToolRegistry(
        tools_catalog=tools_catalog,
        policy_bundle=policy_bundle,
    )
    registered_tool_refs: list[str] = []
    stubbed_tool_refs: list[str] = []

    if include_starter_agents:
        _register_openai_or_stub(
            tool_registry,
            tool_ref="research.search",
            env_file=env_file,
            registered_tool_refs=registered_tool_refs,
            stubbed_tool_refs=stubbed_tool_refs,
        )

    register_autonomy_toolkit(tool_registry, workspace_root=workspace_root)
    for implemented in (
        "fs.create_dir",
        "fs.write_file",
        "fs.read_file",
        "fs.list_dir",
        "package.install",
        "package.update",
        "app.launch",
        "web.download",
        "web.fetch",
        "web.crawl",
        "parse.extract",
        "citation.verify",
    ):
        if implemented not in registered_tool_refs:
            registered_tool_refs.append(implemented)

    _register_openai_or_stub(
        tool_registry,
        tool_ref="web.search",
        env_file=env_file,
        registered_tool_refs=registered_tool_refs,
        stubbed_tool_refs=stubbed_tool_refs,
    )
    _ensure_desktop_tool_specs(
        tool_registry,
        tools_catalog=tools_catalog,
        registered_tool_refs=registered_tool_refs,
        stubbed_tool_refs=stubbed_tool_refs,
    )

    agents = []
    if include_starter_agents:
        agents.extend(
            build_starter_agent_configs(
                research_tool_ref="research.search",
                include_critic=include_critic,
            )
        )

    desktop_executor = load_desktop_executor_config()
    agents.append(desktop_executor)

    crew = Crew(
        name="desktop-demo",
        agents=agents,
        tool_registry=tool_registry,
        schema_registry={
            "ResearchReport": ResearchReport,
            "FileSystemActionReport": FileSystemActionReport,
            "DesktopActionReport": DesktopActionReport,
        },
    )

    if include_starter_agents:
        register_starter_specialists(
            agent_registry=crew.agent_registry,
            tool_registry=crew.tool_registry,
            include_planner=include_planner,
            include_validator=include_validator,
        )

    return DesktopCrewBundle(
        crew=crew,
        desktop_executor=desktop_executor,
        tools_catalog=tools_catalog,
        policy_bundle=policy_bundle,
        workflow_bundle=workflow_bundle,
        registered_tool_refs=sorted(set(registered_tool_refs)),
        stubbed_tool_refs=sorted(set(stubbed_tool_refs)),
    )


def build_desktop_demo_graph(
    *,
    env_file: str = ".env",
    workspace_root: str = ".",
    include_starter_agents: bool = True,
    include_critic: bool = False,
    include_planner: bool = True,
    include_validator: bool = True,
    checkpointer: BaseCheckpointSaver[Any] | None = None,
) -> tuple[DesktopCrewBundle, Any]:
    """Compile a desktop-capable crew into a runnable LangGraph graph."""
    bundle = build_desktop_demo_crew(
        env_file=env_file,
        workspace_root=workspace_root,
        include_starter_agents=include_starter_agents,
        include_critic=include_critic,
        include_planner=include_planner,
        include_validator=include_validator,
    )
    graph = bundle.crew.compile(checkpointer=checkpointer or InMemorySaver())
    graph._desktop_tool_registry = bundle.crew.tool_registry
    graph._desktop_policy_bundle = bundle.policy_bundle
    return bundle, graph


def build_desktop_task(
    *,
    query: str,
    allowed_paths: list[str] | None = None,
    blocked_commands: list[str] | None = None,
    locale: str = "ru",
    hitl_points: list[HITLPoint] | None = None,
) -> TaskSpec:
    """Build a desktop-oriented task with tool and safety metadata pre-filled."""
    policy_bundle = load_desktop_policy_bundle()
    planning_policy = policy_bundle.safety_policy.planning_policy
    if _should_plan_desktop_task_v2(query, planning_policy):
        subtasks = _decompose_desktop_query_v2(
            query,
            allowed_paths=allowed_paths,
            blocked_commands=blocked_commands,
            locale=locale,
            hitl_points=hitl_points,
        )
        if len(subtasks) > 1:
            return TaskSpec(
                task_id=f"task-{uuid4()}",
                description=query,
                output_schema="DesktopActionReport",
                assignee="planner",
                auto_route=False,
                fact_logic_validation=FactLogicValidationMode.NONE,
                hitl_points=list(hitl_points or []),
                metadata={
                    "enable_planner": True,
                    "planner_domain": "desktop",
                    "locale": locale,
                    "allowed_paths": allowed_paths or [],
                    "blocked_commands": blocked_commands or [],
                    "subtasks": [
                        {
                            "description": subtask.description,
                            "metadata": subtask.metadata,
                            "assignee": subtask.assignee,
                            "output_schema": subtask.output_schema,
                            "fact_logic_validation": subtask.fact_logic_validation.value,
                            "hitl_points": [
                                point.value for point in subtask.hitl_points
                            ],
                        }
                        for subtask in subtasks
                    ],
                },
            )

    desktop_plan = _classify_desktop_intent_v2(query)
    if desktop_plan is None:
        return build_demo_task(
            query=query,
            hitl_points=hitl_points,
            locale=locale,
        )

    normalized_hitl_points = list(hitl_points or [])
    if (
        desktop_plan["requires_hitl"]
        and HITLPoint.BEFORE_TOOL_CALL not in normalized_hitl_points
    ):
        normalized_hitl_points.append(HITLPoint.BEFORE_TOOL_CALL)

    metadata = {
        "tool_ref": desktop_plan["tool_ref"],
        "desktop_intent": desktop_plan["capability"],
        "allowed_paths": allowed_paths or [],
        "blocked_commands": blocked_commands or [],
        "locale": locale,
        "risk_level": desktop_plan.get("risk_level", "medium"),
    }
    if desktop_plan["tool_ref"] in {"fs.create_dir", "fs.move", "fs.delete"}:
        if desktop_plan["tool_ref"] == "fs.create_dir":
            resolved = _infer_desktop_target_path_v2(
                query,
                aliases=policy_bundle.safety_policy.path_inference_policy.get(
                    "aliases", {}
                ),
            )
            metadata["requested_path"] = resolved["requested_path"]
            metadata["target_path"] = resolved["sandbox_path"]
            metadata["required_terms"] = _required_terms_for_locale(
                locale, "created", "создан"
            )
            metadata["dry_run_preview"] = (
                f".agentgraph_cache/workspace/{resolved['sandbox_path']}"
            )
        elif desktop_plan["tool_ref"] == "fs.move":
            metadata["source_path"] = desktop_plan.get(
                "source_path", "{thread_id}/MEMORY"
            )
            metadata["target_path"] = desktop_plan.get(
                "target_path", "{thread_id}/ARCHIVE"
            )
            metadata["required_terms"] = _required_terms_for_locale(
                locale, "moved", "перемещ"
            )
            metadata["dry_run_preview"] = (
                f"move {metadata['source_path']} -> {metadata['target_path']}"
            )
        else:
            metadata["target_path"] = desktop_plan.get(
                "target_path", "{thread_id}/MEMORY"
            )
            metadata["required_terms"] = _required_terms_for_locale(
                locale, "deleted", "карантин"
            )
            metadata["dry_run_preview"] = (
                f"delete {metadata['target_path']} via quarantine"
            )
        metadata["require_citations"] = True
        output_schema = "FileSystemActionReport"
        fact_logic_validation = FactLogicValidationMode.POLICY
    elif desktop_plan["tool_ref"] == "web.download":
        metadata["url"] = desktop_plan.get("url", "")
        metadata["target_filename"] = desktop_plan.get("target_filename", "")
        if _query_mentions_destination_path_v2(query):
            resolved = _infer_desktop_target_path_v2(
                query,
                aliases=policy_bundle.safety_policy.path_inference_policy.get(
                    "aliases", {}
                ),
            )
            metadata["requested_path"] = resolved["requested_path"]
        metadata["required_terms"] = _required_terms_for_locale(
            locale, "downloaded", "загруж"
        )
        metadata["require_citations"] = True
        metadata["quarantine_status"] = "required"
        metadata["dry_run_preview"] = (
            f"download {metadata['url']} -> quarantine/{metadata['target_filename']}"
        )
        output_schema = "DesktopActionReport"
        fact_logic_validation = FactLogicValidationMode.POLICY
    elif desktop_plan["tool_ref"] in {"package.install", "package.update"}:
        metadata["manager"] = desktop_plan.get("manager", "pip")
        metadata["package_name"] = desktop_plan.get("package_name", "")
        metadata["required_terms"] = _required_terms_for_locale(
            locale, "prepared", "подготов"
        )
        metadata["dry_run_preview"] = (
            f"{metadata['manager']} "
            f"{'install' if desktop_plan['tool_ref'] == 'package.install' else 'update'} "
            f"{metadata['package_name']}".strip()
        )
        output_schema = "DesktopActionReport"
        fact_logic_validation = FactLogicValidationMode.POLICY
    elif desktop_plan["tool_ref"] == "app.launch":
        metadata["app_name"] = desktop_plan.get("app_name", "browser")
        metadata["required_terms"] = _required_terms_for_locale(
            locale, "launch", "запущ"
        )
        metadata["dry_run_preview"] = f"launch {metadata['app_name']}"
        output_schema = "DesktopActionReport"
        fact_logic_validation = FactLogicValidationMode.POLICY
    else:
        output_schema = "DesktopActionReport"
        fact_logic_validation = FactLogicValidationMode.NONE

    return TaskSpec(
        task_id=f"task-{uuid4()}",
        description=query,
        output_schema=output_schema,
        assignee="desktop_executor",
        auto_route=False,
        fact_logic_validation=fact_logic_validation,
        hitl_points=normalized_hitl_points,
        metadata=metadata,
    )


def run_desktop_task(
    graph: Any,
    *,
    task: TaskSpec,
    desktop_context: dict[str, Any] | None = None,
    shared_context_overrides: dict[str, Any] | None = None,
    thread_id: str | None = None,
    stream: bool = True,
) -> tuple[AgentState, list[dict[str, Any]]]:
    """Run a desktop task seeded with `shared_context.desktop_context`."""
    resolved_thread_id = thread_id or f"desktop-{uuid4()}"
    shared_context = dict(shared_context_overrides or {})
    shared_context["desktop_context"] = desktop_context or {
        "current_path": ".",
        "installed_packages": [],
        "trust_score": 0.5,
        "last_actions": [],
    }
    initial_state = AgentState(
        thread_id=resolved_thread_id,
        task=task,
        shared_context=shared_context,
    )
    config = {"configurable": {"thread_id": resolved_thread_id}}

    if stream:
        events = list(
            stream_envelopes(
                graph,
                initial_state.model_dump(by_alias=True),
                config,
                stream_mode="updates",
            )
        )
        state = apply_desktop_enforcement_updates(graph, thread_id=resolved_thread_id)
        return state, events

    result = graph.invoke(initial_state.model_dump(by_alias=True), config)
    _ = result
    return apply_desktop_enforcement_updates(graph, thread_id=resolved_thread_id), []


def resume_desktop_task(
    graph: Any,
    *,
    thread_id: str,
    human_feedback: dict[str, Any],
    stream: bool = True,
) -> tuple[AgentState, list[dict[str, Any]]]:
    """Resume a desktop task that was paused for HITL approval."""
    config = {"configurable": {"thread_id": thread_id}}
    if stream:
        events = list(
            stream_envelopes(
                graph,
                Command(resume=human_feedback),
                config,
                stream_mode="updates",
            )
        )
        state = apply_desktop_enforcement_updates(graph, thread_id=thread_id)
        return state, events

    result = graph.invoke(Command(resume=human_feedback), config)
    _ = result
    return apply_desktop_enforcement_updates(graph, thread_id=thread_id), []


def export_desktop_artifact(
    graph: Any,
    *,
    state: AgentState,
    destination_path: str | None = None,
) -> ToolResult:
    """Export the last desktop filesystem artifact from sandbox into an approved user path."""
    tool_registry = getattr(graph, "_desktop_tool_registry", None)
    if tool_registry is None:
        return ToolResult(
            success=False,
            error="desktop tool registry is not attached to the graph",
            error_type="missing_registry",
        )

    metadata = state.task.metadata
    source_path = _resolve_export_source_path(state)
    if not source_path:
        return ToolResult(
            success=False,
            error="task does not contain an exportable sandbox artifact",
            error_type="missing_source",
        )

    destination = _resolve_export_destination_path(
        state,
        explicit_destination=destination_path,
    )
    if not destination:
        return ToolResult(
            success=False,
            error="destination path is required for export",
            error_type="missing_destination",
        )

    return tool_registry.invoke(
        "fs.export",
        {
            "source_path": source_path,
            "destination_path": destination,
            "allowed_paths": metadata.get("allowed_paths", []),
            "task": state.task.model_dump(mode="json"),
            "thread_id": state.thread_id,
            "shared_context": state.shared_context,
        },
    )


def run_desktop_plan(
    graph: Any,
    *,
    task: TaskSpec,
    desktop_context: dict[str, Any] | None = None,
    thread_id: str | None = None,
    stream: bool = True,
    auto_approve_tools: list[str] | None = None,
    max_steps: int = 5,
) -> DesktopPlanExecution:
    """Run a limited desktop plan sequentially without changing the core runtime."""
    root_thread_id = thread_id or f"desktop-plan-{uuid4()}"
    subtasks_raw = task.metadata.get("subtasks", [])
    if not isinstance(subtasks_raw, list) or not subtasks_raw:
        state, events = run_desktop_task(
            graph,
            task=task,
            desktop_context=desktop_context,
            thread_id=root_thread_id,
            stream=stream,
        )
        return DesktopPlanExecution(
            root_thread_id=root_thread_id,
            step_states=[state],
            step_events=[events],
            stopped_reason=None
            if state.status == ThreadStatus.COMPLETED
            else state.status.value,
        )

    context = dict(desktop_context or {})
    context.setdefault("current_path", "~/Desktop")
    context.setdefault("installed_packages", [])
    context.setdefault("trust_score", 0.5)
    context.setdefault("last_actions", [])
    shared_context_overrides: dict[str, Any] = {}
    approved = set(auto_approve_tools or [])
    step_states: list[AgentState] = []
    step_events: list[list[dict[str, Any]]] = []

    for index, subtask_payload in enumerate(subtasks_raw[:max_steps]):
        if not isinstance(subtask_payload, dict):
            continue
        subtask = TaskSpec(
            task_id=f"{task.task_id}-step-{index}",
            description=str(subtask_payload.get("description") or task.description),
            output_schema=str(
                subtask_payload.get("output_schema") or task.output_schema
            ),
            assignee=subtask_payload.get("assignee"),
            auto_route=False,
            fact_logic_validation=FactLogicValidationMode(
                str(
                    subtask_payload.get("fact_logic_validation")
                    or task.fact_logic_validation.value
                )
            ),
            hitl_points=[
                point if isinstance(point, HITLPoint) else HITLPoint(str(point))
                for point in (
                    subtask_payload.get("hitl_points")
                    or [point.value for point in task.hitl_points]
                )
            ],
            metadata=dict(subtask_payload.get("metadata") or {}),
        )
        step_thread_id = f"{root_thread_id}-step-{index}"
        state, events = run_desktop_task(
            graph,
            task=subtask,
            desktop_context=context,
            shared_context_overrides=shared_context_overrides,
            thread_id=step_thread_id,
            stream=stream,
        )
        combined_events = list(events)
        tool_ref = str(subtask.metadata.get("tool_ref") or "")
        if state.status == ThreadStatus.HITL_WAIT and tool_ref in approved:
            state, resume_events = resume_desktop_task(
                graph,
                thread_id=state.thread_id,
                human_feedback={"decision": "approve"},
                stream=stream,
            )
            combined_events.extend(resume_events)
        step_states.append(state)
        step_events.append(combined_events)
        shared_context_overrides = dict(state.shared_context)
        shared_context_overrides["previous_step_output"] = state.output_candidate
        shared_context_overrides["previous_step_artifacts"] = state.artifacts
        desktop_ctx = shared_context_overrides.get("desktop_context")
        if isinstance(desktop_ctx, dict):
            context = dict(desktop_ctx)
        if state.status != ThreadStatus.COMPLETED:
            return DesktopPlanExecution(
                root_thread_id=root_thread_id,
                step_states=step_states,
                step_events=step_events,
                stopped_reason=state.status.value,
            )

    stopped_reason = None
    if len(subtasks_raw) > max_steps:
        stopped_reason = "max_steps_reached"
    return DesktopPlanExecution(
        root_thread_id=root_thread_id,
        step_states=step_states,
        step_events=step_events,
        stopped_reason=stopped_reason,
    )


def resume_desktop_plan(
    graph: Any,
    *,
    task: TaskSpec,
    paused_state: AgentState,
    human_feedback: dict[str, Any],
    desktop_context: dict[str, Any] | None = None,
    root_thread_id: str | None = None,
    auto_approve_tools: list[str] | None = None,
    max_steps: int = 5,
    stream: bool = True,
    prior_step_states: list[AgentState] | None = None,
    prior_step_events: list[list[dict[str, Any]]] | None = None,
) -> DesktopPlanExecution:
    """Resume a paused desktop plan step, then continue remaining subtasks."""
    resumed_state, resume_events = resume_desktop_task(
        graph,
        thread_id=paused_state.thread_id,
        human_feedback=human_feedback,
        stream=stream,
    )
    resolved_root_thread_id = root_thread_id or re.sub(
        r"-step-\d+$", "", resumed_state.thread_id
    )
    step_states = list(prior_step_states or [])
    step_events = [list(events) for events in (prior_step_events or [])]
    if step_states and step_states[-1].thread_id == paused_state.thread_id:
        step_states = step_states[:-1]
    if step_events and len(step_events) >= len(step_states) + 1:
        step_events = step_events[:-1]

    step_states.append(resumed_state)
    step_events.append(list(resume_events))
    if resumed_state.status != ThreadStatus.COMPLETED:
        return DesktopPlanExecution(
            root_thread_id=resolved_root_thread_id,
            step_states=step_states,
            step_events=step_events,
            stopped_reason=resumed_state.status.value,
        )

    subtasks_raw = task.metadata.get("subtasks", [])
    if not isinstance(subtasks_raw, list) or not subtasks_raw:
        return DesktopPlanExecution(
            root_thread_id=resolved_root_thread_id,
            step_states=step_states,
            step_events=step_events,
            stopped_reason=None,
        )

    match = re.search(r"-step-(\d+)$", resumed_state.task.task_id)
    current_step_index = (
        int(match.group(1)) if match is not None else len(step_states) - 1
    )
    context = dict(desktop_context or {})
    desktop_ctx = resumed_state.shared_context.get("desktop_context")
    if isinstance(desktop_ctx, dict):
        context.update(desktop_ctx)
    context.setdefault("current_path", "~/Desktop")
    context.setdefault("installed_packages", [])
    context.setdefault("trust_score", 0.5)
    context.setdefault("last_actions", [])
    shared_context_overrides = dict(resumed_state.shared_context)
    shared_context_overrides["previous_step_output"] = resumed_state.output_candidate
    shared_context_overrides["previous_step_artifacts"] = resumed_state.artifacts
    approved = set(auto_approve_tools or [])

    for index, subtask_payload in enumerate(
        subtasks_raw[current_step_index + 1 : max_steps], start=current_step_index + 1
    ):
        if not isinstance(subtask_payload, dict):
            continue
        subtask = TaskSpec(
            task_id=f"{task.task_id}-step-{index}",
            description=str(subtask_payload.get("description") or task.description),
            output_schema=str(
                subtask_payload.get("output_schema") or task.output_schema
            ),
            assignee=subtask_payload.get("assignee"),
            auto_route=False,
            fact_logic_validation=FactLogicValidationMode(
                str(
                    subtask_payload.get("fact_logic_validation")
                    or task.fact_logic_validation.value
                )
            ),
            hitl_points=[
                point if isinstance(point, HITLPoint) else HITLPoint(str(point))
                for point in (
                    subtask_payload.get("hitl_points")
                    or [point.value for point in task.hitl_points]
                )
            ],
            metadata=dict(subtask_payload.get("metadata") or {}),
        )
        step_thread_id = f"{resolved_root_thread_id}-step-{index}"
        state, events = run_desktop_task(
            graph,
            task=subtask,
            desktop_context=context,
            shared_context_overrides=shared_context_overrides,
            thread_id=step_thread_id,
            stream=stream,
        )
        combined_events = list(events)
        tool_ref = str(subtask.metadata.get("tool_ref") or "")
        if state.status == ThreadStatus.HITL_WAIT and tool_ref in approved:
            state, resume_events = resume_desktop_task(
                graph,
                thread_id=state.thread_id,
                human_feedback={"decision": "approve"},
                stream=stream,
            )
            combined_events.extend(resume_events)
        step_states.append(state)
        step_events.append(combined_events)
        shared_context_overrides = dict(state.shared_context)
        shared_context_overrides["previous_step_output"] = state.output_candidate
        shared_context_overrides["previous_step_artifacts"] = state.artifacts
        desktop_ctx = shared_context_overrides.get("desktop_context")
        if isinstance(desktop_ctx, dict):
            context = dict(desktop_ctx)
        if state.status != ThreadStatus.COMPLETED:
            return DesktopPlanExecution(
                root_thread_id=resolved_root_thread_id,
                step_states=step_states,
                step_events=step_events,
                stopped_reason=state.status.value,
            )

    stopped_reason = None
    if len(subtasks_raw) > max_steps:
        stopped_reason = "max_steps_reached"
    return DesktopPlanExecution(
        root_thread_id=resolved_root_thread_id,
        step_states=step_states,
        step_events=step_events,
        stopped_reason=stopped_reason,
    )


def _register_openai_or_stub(
    tool_registry: ToolRegistry,
    *,
    tool_ref: str,
    env_file: str,
    registered_tool_refs: list[str],
    stubbed_tool_refs: list[str],
) -> None:
    try:
        register_openai_research_tool(
            tool_registry,
            tool_ref=tool_ref,
            env_path=env_file,
        )
        registered_tool_refs.append(tool_ref)
    except ValueError:
        tool_registry.register(
            tool_ref=tool_ref,
            description="Desktop search stub used when OpenAI is not configured.",
            side_effect_level=SideEffectLevel.READ_ONLY,
            handler=_not_implemented_handler(
                tool_ref, "OpenAI search is not configured."
            ),
        )
        registered_tool_refs.append(tool_ref)
        stubbed_tool_refs.append(tool_ref)


def _ensure_desktop_tool_specs(
    tool_registry: ToolRegistry,
    *,
    tools_catalog: DesktopToolsCatalog,
    registered_tool_refs: list[str],
    stubbed_tool_refs: list[str],
) -> None:
    for tool_ref, spec in tools_catalog.tools.items():
        try:
            tool_registry.get(tool_ref)
            registered_tool_refs.append(tool_ref)
            continue
        except KeyError:
            pass

        tool_registry.register(
            tool_ref=tool_ref,
            description=spec.description,
            side_effect_level=_parse_side_effect_level(spec.side_effect_level),
            handler=_not_implemented_handler(
                tool_ref,
                f"{tool_ref} is declared in desktop_tools.yaml but not implemented yet.",
            ),
        )
        registered_tool_refs.append(tool_ref)
        stubbed_tool_refs.append(tool_ref)


def _not_implemented_handler(tool_ref: str, message: str) -> Any:
    def handler(args: dict[str, Any]) -> ToolResult:
        return ToolResult(
            success=False,
            error=message,
            error_type="not_implemented",
            metadata={"tool_ref": tool_ref, "args": args},
        )

    return handler


def _parse_side_effect_level(raw: str) -> SideEffectLevel:
    mapping = {
        "none": SideEffectLevel.NONE,
        "read_only": SideEffectLevel.READ_ONLY,
        "external_write": SideEffectLevel.EXTERNAL_WRITE,
        "destructive": SideEffectLevel.DESTRUCTIVE,
    }
    return mapping[raw]


def _classify_desktop_intent_v2(query: str) -> dict[str, Any] | None:
    lowered = query.lower()
    if any(
        token in lowered
        for token in (
            "\u043f\u0435\u0440\u0435\u043c\u0435\u0441\u0442\u0438",
            "\u043f\u0435\u0440\u0435\u043d\u0435\u0441\u0438",
            "move ",
        )
    ) and not any(
        token in lowered
        for token in ("\u0441\u043a\u0430\u0447\u0430\u0439", "download")
    ):
        source_path, target_path = _infer_move_paths_v2(query)
        return {
            "capability": "fs_operations",
            "tool_ref": "fs.move",
            "requires_hitl": True,
            "source_path": source_path,
            "target_path": target_path,
            "risk_level": "medium",
        }
    if any(
        token in lowered for token in ("\u0443\u0434\u0430\u043b\u0438", "delete ")
    ) and not any(
        token in lowered for token in ("package", "pip", "npm", "winget", "brew", "apt")
    ):
        target_path = _infer_delete_path_v2(query)
        return {
            "capability": "fs_operations",
            "tool_ref": "fs.delete",
            "requires_hitl": True,
            "target_path": target_path,
            "risk_level": "destructive",
        }
    if any(
        token in lowered
        for token in (
            "\u0443\u0441\u0442\u0430\u043d\u043e\u0432\u0438",
            "install ",
            "pip ",
            "npm ",
            "winget ",
            "brew ",
            "apt ",
        )
    ):
        manager, package_name = _infer_package_request_v2(query)
        return {
            "capability": "software_install",
            "tool_ref": "package.install",
            "requires_hitl": True,
            "manager": manager,
            "package_name": package_name,
            "risk_level": "high",
        }
    if any(
        token in lowered
        for token in ("\u043e\u0431\u043d\u043e\u0432\u0438", "update ")
    ):
        manager, package_name = _infer_package_request_v2(query)
        return {
            "capability": "software_install",
            "tool_ref": "package.update",
            "requires_hitl": True,
            "manager": manager,
            "package_name": package_name,
            "risk_level": "medium",
        }
    if any(
        token in lowered
        for token in (
            "\u0441\u043a\u0430\u0447\u0430\u0439",
            "download ",
            ".pdf",
            ".zip",
            ".docx",
            ".xlsx",
        )
    ):
        url, target_filename = _infer_download_request_v2(query)
        return {
            "capability": "web_research",
            "tool_ref": "web.download",
            "requires_hitl": True,
            "url": url,
            "target_filename": target_filename,
            "risk_level": "medium",
        }
    if any(
        token in lowered
        for token in (
            "\u043d\u0430\u0439\u0434\u0438",
            "\u043f\u043e\u0438\u0441\u043a",
            "\u0434\u043e\u043a\u0443\u043c\u0435\u043d\u0442\u0430\u0446",
            "search ",
            "find ",
        )
    ):
        return {
            "capability": "web_research",
            "tool_ref": "web.search",
            "requires_hitl": False,
            "risk_level": "low",
        }
    if any(
        token in lowered
        for token in (
            "\u043e\u0442\u043a\u0440\u043e\u0439",
            "\u0437\u0430\u043f\u0443\u0441\u0442\u0438",
            "\u0431\u0440\u0430\u0443\u0437\u0435\u0440",
            "browser",
            "calendar",
            "mail",
            "\u043f\u043e\u0447\u0442\u0430",
            "terminal",
            "\u0442\u0435\u0440\u043c\u0438\u043d\u0430\u043b",
            "text_editor",
            "\u0440\u0435\u0434\u0430\u043a\u0442\u043e\u0440",
        )
    ):
        return {
            "capability": "app_control",
            "tool_ref": "app.launch",
            "requires_hitl": False,
            "app_name": _infer_app_name_v2(query),
            "risk_level": "low",
        }
    if _contains_create_dir_request_v2(lowered):
        return {
            "capability": "fs_operations",
            "tool_ref": "fs.create_dir",
            "requires_hitl": True,
            "risk_level": "medium",
        }
    return None


def _contains_create_dir_request_v2(lowered: str) -> bool:
    if any(
        phrase in lowered
        for phrase in ("create folder", "create directory", "create dir")
    ):
        return True
    create_verbs = (
        "\u0441\u043e\u0437\u0434\u0430\u0439",
        "\u0441\u043e\u0437\u0434\u0430\u0442\u044c",
    )
    folder_terms = (
        "\u043f\u0430\u043f\u043a",
        "\u043a\u0430\u0442\u0430\u043b\u043e\u0433",
        "\u0434\u0438\u0440\u0435\u043a\u0442\u043e\u0440",
        " folder",
        " directory",
        " dir",
    )
    return any(token in lowered for token in create_verbs) and any(
        token in lowered for token in folder_terms
    )


def _query_mentions_destination_path_v2(query: str) -> bool:
    lowered = query.lower()
    return any(
        token in lowered
        for token in (
            "\u043f\u0430\u043f\u043a",
            "\u043f\u0443\u0442\u044c",
            "\u0440\u0430\u0431\u043e\u0447",
            "\u0434\u043e\u043a\u0443\u043c\u0435\u043d\u0442",
            "\u0437\u0430\u0433\u0440\u0443\u0437\u043a",
            "folder",
            "path",
            "desktop",
            "documents",
            "downloads",
        )
    )


def _infer_move_paths(query: str) -> tuple[str, str]:
    match = re.search(
        r"(?:move|перемести|перенеси)\s+(?:file|folder|directory|файл|папк\w*|каталог\w*|директор\w*)?\s*([A-Za-z0-9._/\-]+)\s+(?:to|в)\s+([A-Za-z0-9._/\-]+)",
        query,
        flags=re.IGNORECASE,
    )
    if match is not None:
        return match.group(1), match.group(2)
    return "{thread_id}/MEMORY", "{thread_id}/ARCHIVE"


def _infer_delete_path(query: str) -> str:
    match = re.search(
        r"(?:delete|удали)\s+(?:file|folder|directory|файл|папк\w*|каталог\w*|директор\w*)?\s*([A-Za-z0-9._/\-]+)",
        query,
        flags=re.IGNORECASE,
    )
    if match is not None:
        return match.group(1)
    return "{thread_id}/MEMORY"


def _infer_download_request(query: str) -> tuple[str, str]:
    url_match = re.search(r"https?://[^\s\"']+", query, flags=re.IGNORECASE)
    url = url_match.group(0) if url_match is not None else ""
    if url:
        inferred = url.rstrip("/").rsplit("/", 1)[-1]
        return url, inferred or "download.bin"

    filename_match = re.search(
        r"([A-Za-z0-9._-]+\.(?:pdf|zip|docx|xlsx|txt|csv))",
        query,
        flags=re.IGNORECASE,
    )
    if filename_match is not None:
        return "", filename_match.group(1)
    return "", "download.bin"


def _infer_package_request(query: str) -> tuple[str, str]:
    lowered = query.lower()
    manager = "pip"
    for candidate in ("pip", "npm", "brew", "winget", "apt"):
        if candidate in lowered:
            manager = candidate
            break
    match = re.search(
        r"(?:install|update|установи|обнови)\s+([A-Za-z0-9._-]+)",
        query,
        flags=re.IGNORECASE,
    )
    package_name = match.group(1) if match is not None else ""
    if package_name.lower() in {"pip", "npm", "brew", "winget", "apt"}:
        package_name = ""
    return manager, package_name


def _infer_app_name(query: str) -> str:
    lowered = query.lower()
    if "брауз" in lowered or "browser" in lowered:
        return "browser"
    if "почт" in lowered or "mail" in lowered:
        return "mail"
    if "календар" in lowered or "calendar" in lowered:
        return "calendar"
    if "терминал" in lowered or "terminal" in lowered:
        return "terminal"
    if "редактор" in lowered or "блокнот" in lowered or "text" in lowered:
        return "text_editor"
    return "browser"


def _classify_desktop_intent(query: str) -> dict[str, Any] | None:
    lowered = query.lower()
    if any(
        token in lowered
        for token in (
            "create folder",
            "create directory",
            "create dir",
            "создай папк",
            "создай каталог",
            "создай директор",
        )
    ):
        return {
            "capability": "fs_operations",
            "tool_ref": "fs.create_dir",
            "requires_hitl": True,
            "risk_level": "medium",
        }
    if any(
        token in lowered for token in ("move ", "перемести", "перенеси")
    ) and not any(token in lowered for token in ("download", "скачай")):
        source_path, target_path = _infer_move_paths(query)
        return {
            "capability": "fs_operations",
            "tool_ref": "fs.move",
            "requires_hitl": True,
            "source_path": source_path,
            "target_path": target_path,
            "risk_level": "medium",
        }
    if any(token in lowered for token in ("delete ", "удали")) and not any(
        token in lowered
        for token in ("package", "pip", "npm", "winget", "brew", "apt", "пакет")
    ):
        target_path = _infer_delete_path(query)
        return {
            "capability": "fs_operations",
            "tool_ref": "fs.delete",
            "requires_hitl": True,
            "target_path": target_path,
            "risk_level": "destructive",
        }
    if any(
        token in lowered
        for token in (
            "install ",
            "pip ",
            "npm ",
            "winget ",
            "brew ",
            "apt ",
            "установи",
        )
    ):
        manager, package_name = _infer_package_request(query)
        return {
            "capability": "software_install",
            "tool_ref": "package.install",
            "requires_hitl": True,
            "manager": manager,
            "package_name": package_name,
            "risk_level": "high",
        }
    if any(token in lowered for token in ("update ", "обнови")):
        manager, package_name = _infer_package_request(query)
        return {
            "capability": "software_install",
            "tool_ref": "package.update",
            "requires_hitl": True,
            "manager": manager,
            "package_name": package_name,
            "risk_level": "medium",
        }
    if any(
        token in lowered
        for token in ("download ", "скачай", ".pdf", ".zip", ".docx", ".xlsx")
    ):
        url, target_filename = _infer_download_request(query)
        return {
            "capability": "web_research",
            "tool_ref": "web.download",
            "requires_hitl": True,
            "url": url,
            "target_filename": target_filename,
            "risk_level": "medium",
        }
    if any(
        token in lowered
        for token in ("search ", "find ", "найди", "поиск", "документац", "информац")
    ):
        return {
            "capability": "web_research",
            "tool_ref": "web.search",
            "requires_hitl": False,
            "risk_level": "low",
        }
    if any(
        token in lowered
        for token in (
            "browser",
            "calendar",
            "mail",
            "terminal",
            "text_editor",
            "открой",
            "запусти",
            "браузер",
            "почту",
            "календар",
            "терминал",
            "редактор",
        )
    ):
        return {
            "capability": "app_control",
            "tool_ref": "app.launch",
            "requires_hitl": False,
            "app_name": _infer_app_name(query),
            "risk_level": "low",
        }
    return None


def _should_plan_desktop_task_v2(query: str, planning_policy: dict[str, Any]) -> bool:
    if not planning_policy.get("enable_for_multistep_tasks", False):
        return False
    if re.search(r"(?:^|\n)\s*(?:\d+[.)]|[-*])\s+", query, flags=re.IGNORECASE):
        return True
    lowered = query.lower()
    keywords = [
        str(value).lower()
        for value in planning_policy.get("multistep_signals", {}).get("keywords", [])
    ]
    distinct_matches = {
        keyword for keyword in keywords if keyword and keyword in lowered
    }
    min_actions = int(
        planning_policy.get("multistep_signals", {}).get("min_distinct_actions", 2)
    )
    return len(distinct_matches) >= min_actions or _contains_order_markers_v2(lowered)


def _contains_order_markers_v2(lowered: str) -> bool:
    return any(
        marker in lowered
        for marker in (
            " then ",
            " after ",
            " \u0437\u0430\u0442\u0435\u043c ",
            " \u043f\u043e\u0442\u043e\u043c ",
            " \u043f\u043e\u0441\u043b\u0435 ",
        )
    )


def _decompose_desktop_query_v2(
    query: str,
    *,
    allowed_paths: list[str] | None,
    blocked_commands: list[str] | None,
    locale: str,
    hitl_points: list[HITLPoint] | None,
) -> list[TaskSpec]:
    line_parts = [
        part.strip(" ,.") for part in query.splitlines() if part.strip(" ,.\n")
    ]
    if len(line_parts) > 1:
        return _apply_desktop_subtask_dependencies(
            [
                build_desktop_task(
                    query=part,
                    allowed_paths=allowed_paths,
                    blocked_commands=blocked_commands,
                    locale=locale,
                    hitl_points=hitl_points,
                )
                for part in line_parts
            ]
        )
    numbered_parts = [
        part.strip(" ,.")
        for part in re.split(
            r"(?:^|\n)\s*(?:\d+[.)]|[-*])\s+",
            query,
            flags=re.IGNORECASE,
        )
        if part.strip(" ,.\n")
    ]
    if len(numbered_parts) > 1:
        return _apply_desktop_subtask_dependencies(
            [
                build_desktop_task(
                    query=part,
                    allowed_paths=allowed_paths,
                    blocked_commands=blocked_commands,
                    locale=locale,
                    hitl_points=hitl_points,
                )
                for part in numbered_parts
            ]
        )
    normalized = re.sub(
        r"\s+(?:then|after that|after|\u0437\u0430\u0442\u0435\u043c|\u043f\u043e\u0442\u043e\u043c|\u043f\u043e\u0441\u043b\u0435 \u044d\u0442\u043e\u0433\u043e)\s+",
        " || ",
        query,
        flags=re.IGNORECASE,
    )
    normalized = re.sub(r"\s+\u0438\s+", " || ", normalized, flags=re.IGNORECASE)
    parts = [part.strip(" ,.") for part in normalized.split("||") if part.strip(" ,.")]
    if len(parts) <= 1:
        return []
    subtasks = [
        build_desktop_task(
            query=part,
            allowed_paths=allowed_paths,
            blocked_commands=blocked_commands,
            locale=locale,
            hitl_points=hitl_points,
        )
        for part in parts
    ]
    return _apply_desktop_subtask_dependencies(subtasks)


def _apply_desktop_subtask_dependencies(subtasks: list[TaskSpec]) -> list[TaskSpec]:
    adjusted: list[TaskSpec] = []
    for index, subtask in enumerate(subtasks):
        metadata = dict(subtask.metadata)
        tool_ref = str(metadata.get("tool_ref") or "")
        if tool_ref == "web.search" and any(
            _download_requests_pdf(candidate) for candidate in subtasks[index + 1 :]
        ):
            locale = str(
                metadata.get("locale") or _infer_locale_v2(subtask.description)
            )
            pdf_hint = (
                " Верни прямые ссылки на PDF-файлы, если они доступны."
                if locale.lower().startswith("ru")
                else " Return direct PDF file links when available."
            )
            metadata["search_constraints"] = {
                "prefer_file_type": "pdf",
                "direct_links_only": True,
            }
            adjusted.append(
                subtask.model_copy(
                    update={
                        "description": f"{subtask.description}{pdf_hint}",
                        "metadata": metadata,
                    }
                )
            )
            continue
        adjusted.append(subtask)
    return adjusted


def _download_requests_pdf(task: TaskSpec) -> bool:
    metadata = task.metadata
    if str(metadata.get("tool_ref") or "") != "web.download":
        return False
    return "pdf" in task.description.lower()


def _infer_desktop_target_path_v2(
    query: str, *, aliases: dict[str, Any]
) -> dict[str, str]:
    alias_lookup = {
        str(key).replace("_", " ").lower(): str(value) for key, value in aliases.items()
    }
    lowered = query.lower()
    display_root = "~/Desktop"
    location_patterns = (
        (
            r"(?:^|\s)(?:\u043d\u0430\s+)?\u0440\u0430\u0431\u043e\u0447\w*\s+\u0441\u0442\u043e\u043b\w*(?:$|\s)",
            "~/Desktop",
        ),
        (
            r"(?:^|\s)(?:\u0432\s+)?\u0434\u043e\u043a\u0443\u043c\u0435\u043d\u0442\w*(?:$|\s)",
            "~/Documents",
        ),
        (
            r"(?:^|\s)(?:\u0432\s+)?\u0437\u0430\u0433\u0440\u0443\u0437\u043a\w*(?:$|\s)",
            "~/Downloads",
        ),
        (r"(?:^|\s)(?:on\s+)?desktop(?:$|\s)", "~/Desktop"),
        (r"(?:^|\s)(?:in\s+)?documents?(?:$|\s)", "~/Documents"),
        (r"(?:^|\s)(?:in\s+)?downloads?(?:$|\s)", "~/Downloads"),
    )
    for pattern, mapped in location_patterns:
        if re.search(pattern, lowered, flags=re.IGNORECASE):
            display_root = mapped
            break
    else:
        for alias_name, mapped in alias_lookup.items():
            if alias_name in lowered:
                display_root = mapped
                break
    entity_name = _infer_desktop_entity_name_v2(query)
    parent_name = _infer_desktop_parent_name_v2(query)
    sandbox_root = {
        "~/desktop": "Desktop",
        "~/documents": "Documents",
        "~/downloads": "Downloads",
    }.get(display_root.lower(), "Desktop")
    safe_name = (
        re.sub(
            r"\s+(?:\u043d\u0430|\u0432|on|in)\s+"
            r"(?:\u0440\u0430\u0431\u043e\u0447\w*\s+\u0441\u0442\u043e\u043b\w*|desktop|"
            r"\u0434\u043e\u043a\u0443\u043c\u0435\u043d\u0442\w*|documents|"
            r"\u0437\u0430\u0433\u0440\u0443\u0437\u043a\w*|downloads)$",
            "",
            entity_name,
            flags=re.IGNORECASE,
        ).strip("/\\ ")
        or "DESKTOP_ACTION"
    )
    safe_parent = parent_name.strip("/\\") if parent_name else ""
    requested_parts = [display_root.rstrip("/")]
    sandbox_parts = ["{thread_id}", sandbox_root]
    if safe_parent:
        requested_parts.append(safe_parent)
        sandbox_parts.append(safe_parent)
    requested_parts.append(safe_name)
    sandbox_parts.append(safe_name)
    return {
        "requested_path": "/".join(requested_parts),
        "sandbox_path": "/".join(sandbox_parts),
    }


def _infer_desktop_entity_name_v2(query: str) -> str:
    patterns = [
        (
            r"(?:\u0441\u043e\u0437\u0434\u0430\u0439|create)\s+"
            r"(?:\u043f\u0430\u043f\u043a\w*|folder|directory)\s+"
            r'["«]?([^\n"»,.;:]+?)["»]?'
            r"(?=\s+(?:\u0432\u043d\u0443\u0442\u0440\u0438|inside)\s+"
            r"(?:\u043f\u0430\u043f\u043a\w*|folder|directory)\s+[^\n,.;:]+|$|[,.])"
        ),
        (
            r"(?:\u0441\u043e\u0437\u0434\u0430\u0439|create)\s+"
            r"(?:\u043f\u0430\u043f\u043a\w*|folder|directory)\s+"
            r'["«]?([^\n"»,.;:]+?)["»]?'
            r"(?=\s+(?:\u043d\u0430|\u0432|on|in)\s+(?:"
            r"\u0440\u0430\u0431\u043e\u0447\w*\s+\u0441\u0442\u043e\u043b\w*|desktop|"
            r"\u0434\u043e\u043a\u0443\u043c\u0435\u043d\u0442\w*|documents|"
            r"\u0437\u0430\u0433\u0440\u0443\u0437\u043a\w*|downloads)|$|[,.])"
        ),
        (
            r"(?:\u0432\u043d\u0443\u0442\u0440\u0438|inside)\s+"
            r"(?:\u043f\u0430\u043f\u043a\w*|folder|directory)\s+[^\n,.;:]+?\s+"
            r"(?:\u0441\u043e\u0437\u0434\u0430\u0439|create)\s+"
            r"(?:\u043f\u0430\u043f\u043a\w*|folder|directory)\s+"
            r'["?]?([^\n"?,.;:]+?)["?]?(?:$|[,.])'
        ),
        (
            r"(?:\u0441\u043e\u0437\u0434\u0430\u0439|create)\s+"
            r"(?:\u043f\u0430\u043f\u043a\w*|folder|directory)\s+"
            r'["?]?([^\n"?,.;:]+?)["?]?'
            r"(?=\s+(?:\u0432\u043d\u0443\u0442\u0440\u0438|inside)\s+"
            r"(?:\u043f\u0430\u043f\u043a\w*|folder|directory)\s+[^\n,.;:]+|$|[,.])"
        ),
        (
            r"(?:\u043f\u0430\u043f\u043a\w*|\u043a\u0430\u0442\u0430\u043b\u043e\u0433\w*|"
            r"\u0434\u0438\u0440\u0435\u043a\u0442\u043e\u0440\w*|folder|directory)"
            r"(?:\s+(?:\u0441\s+\u0438\u043c\u0435\u043d\u0435\u043c|named|called))?"
            r'\s+["?]?([^\n"?,.;:]+?)["?]?'
            r"(?=\s+(?:\u043d\u0430|\u0432|on|in)\s+(?:"
            r"\u0440\u0430\u0431\u043e\u0447\w*\s+\u0441\u0442\u043e\u043b\w*|desktop|"
            r"\u0434\u043e\u043a\u0443\u043c\u0435\u043d\u0442\w*|documents|"
            r"\u0437\u0430\u0433\u0440\u0443\u0437\u043a\w*|downloads)|$|[,.])"
        ),
        r'(?:named|called)\s+["?]?([^\s,.;:"?]+)["?]?',
    ]
    for pattern in patterns:
        match = re.search(pattern, query, flags=re.IGNORECASE)
        if match is not None:
            return match.group(1).strip()
    return "DESKTOP_ACTION"


def _infer_desktop_parent_name_v2(query: str) -> str | None:
    patterns = [
        (
            r"(?:\u0432\u043d\u0443\u0442\u0440\u0438|inside)\s+"
            r"(?:\u043f\u0430\u043f\u043a\w*|folder|directory)\s+"
            r'["?]?([^\n"?,.;:]+?)["?]?'
            r"(?=\s+(?:\u0441\u043e\u0437\u0434\u0430\u0439|create)|$|[,.])"
        ),
    ]
    for pattern in patterns:
        match = re.search(pattern, query, flags=re.IGNORECASE)
        if match is not None:
            return match.group(1).strip()
    return None


def _infer_move_paths_v2(query: str) -> tuple[str, str]:
    match = re.search(
        r"(?:move|перемести|перенеси)\s+(?:file|folder|directory|файл|папк\w*|каталог\w*|директор\w*)?\s*([^\s,;:]+)\s+(?:to|в)\s+([^\s,;:]+)",
        query,
        flags=re.IGNORECASE,
    )
    if match is not None:
        return match.group(1), match.group(2)
    return "{thread_id}/MEMORY", "{thread_id}/ARCHIVE"


def _infer_delete_path_v2(query: str) -> str:
    match = re.search(
        r"(?:delete|удали)\s+(?:file|folder|directory|файл|папк\w*|каталог\w*|директор\w*)?\s*([^\s,;:]+)",
        query,
        flags=re.IGNORECASE,
    )
    if match is not None:
        return match.group(1)
    return "{thread_id}/MEMORY"


def _infer_download_request_v2(query: str) -> tuple[str, str]:
    url_match = re.search(r"https?://[^\s\"']+", query, flags=re.IGNORECASE)
    url = url_match.group(0) if url_match is not None else ""
    if url:
        inferred = url.rstrip("/").rsplit("/", 1)[-1]
        return url, inferred or "download.bin"
    filename_match = re.search(
        r"([A-Za-z0-9._-]+\.(?:pdf|zip|docx|xlsx|txt|csv))",
        query,
        flags=re.IGNORECASE,
    )
    if filename_match is not None:
        return "", filename_match.group(1)
    return "", ""


def _infer_package_request_v2(query: str) -> tuple[str, str]:
    lowered = query.lower()
    manager = "pip"
    for candidate in ("pip", "npm", "brew", "winget", "apt"):
        if candidate in lowered:
            manager = candidate
            break
    match = re.search(
        r"(?:install|update|установи|обнови)\s+([A-Za-z0-9._-]+)",
        query,
        flags=re.IGNORECASE,
    )
    package_name = match.group(1) if match is not None else ""
    if package_name.lower() in {"pip", "npm", "brew", "winget", "apt"}:
        package_name = ""
    return manager, package_name


def _infer_app_name_v2(query: str) -> str:
    lowered = query.lower()
    if "брауз" in lowered or "browser" in lowered:
        return "browser"
    if "почт" in lowered or "mail" in lowered:
        return "mail"
    if "календар" in lowered or "calendar" in lowered:
        return "calendar"
    if "терминал" in lowered or "terminal" in lowered:
        return "terminal"
    if "редактор" in lowered or "блокнот" in lowered or "text" in lowered:
        return "text_editor"
    return "browser"


def _infer_locale_v2(query: str) -> str:
    return "ru" if any("\u0400" <= ch <= "\u04ff" for ch in query) else "en"


def _resolve_export_destination_path(
    state: AgentState, *, explicit_destination: str | None = None
) -> str:
    if explicit_destination:
        return explicit_destination

    metadata = state.task.metadata
    requested_path = str(metadata.get("requested_path") or "").strip()
    if metadata.get("tool_ref") == "web.download" and requested_path:
        filename = _resolve_export_filename(state)
        if filename:
            return str(Path(requested_path) / filename)
    target_path = str(metadata.get("target_path") or "").replace(
        "{thread_id}", state.thread_id
    )
    if target_path:
        parts = [part for part in target_path.split("/") if part]
        if len(parts) >= 3:
            root_map = {
                "desktop": "~/Desktop",
                "documents": "~/Documents",
                "downloads": "~/Downloads",
            }
            root = root_map.get(parts[1].lower())
            if root is not None:
                rebuilt = "/".join([root, *parts[2:]])
                if not requested_path:
                    return rebuilt
                requested_parts = [part for part in requested_path.split("/") if part]
                rebuilt_parts = [part for part in rebuilt.split("/") if part]
                if len(rebuilt_parts) > len(requested_parts):
                    return rebuilt
    return requested_path


def _resolve_export_filename(state: AgentState) -> str:
    metadata = state.task.metadata
    explicit_name = str(metadata.get("target_filename") or "").strip()
    if explicit_name:
        return Path(explicit_name).name

    for artifact in reversed(state.artifacts):
        if not isinstance(artifact, dict):
            continue
        data = artifact.get("data")
        if not isinstance(data, dict):
            continue
        filename = str(data.get("filename") or "").strip()
        if filename:
            return Path(filename).name
        url = str(data.get("url") or "").strip()
        if url:
            candidate = Path(url.split("?", 1)[0].rstrip("/")).name
            if candidate:
                return candidate
    return ""


def _resolve_export_source_path(state: AgentState) -> str:
    metadata = state.task.metadata
    target_path = str(metadata.get("target_path") or "").replace(
        "{thread_id}", state.thread_id
    )
    if target_path:
        return target_path

    for artifact in reversed(state.artifacts):
        if not isinstance(artifact, dict):
            continue
        data = artifact.get("data")
        if not isinstance(data, dict):
            continue
        path = str(data.get("path") or data.get("exported_path") or "").strip()
        if path:
            return path

    research_result = state.shared_context.get("research_result")
    if isinstance(research_result, dict):
        path = str(
            research_result.get("path") or research_result.get("exported_path") or ""
        ).strip()
        if path:
            return path
    return ""


def _required_terms_for_locale(
    locale: str, english_term: str, russian_term: str
) -> list[str]:
    return [russian_term] if locale.lower().startswith("ru") else [english_term]
