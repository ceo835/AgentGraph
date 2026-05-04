"""Shared helpers for local CLI and Streamlit AgentGraph demos."""

from __future__ import annotations

import re
from typing import Any
from uuid import uuid4

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from pydantic import BaseModel, Field, model_validator

from agentgraph.autonomy_tools import register_autonomy_toolkit
from agentgraph.contracts import (
    AgentState,
    FactLogicValidationMode,
    HITLPoint,
    TaskSpec,
)
from agentgraph.env import load_env_file
from agentgraph.openai_adapter import register_openai_research_tool
from agentgraph.runtime import Crew, ToolRegistry, stream_envelopes
from agentgraph.starter_agents import (
    build_starter_agent_configs,
    register_starter_specialists,
)


class ResearchReport(BaseModel):
    """Structured output returned by the demo crew."""

    task_id: str
    summary: str
    citations: list[str]
    artifacts_used: int
    confidence: float


class FileSystemActionReport(BaseModel):
    """Strict output contract for local filesystem action tasks."""

    task_id: str
    summary: str = Field(min_length=8)
    citations: list[str] = Field(min_length=1)
    artifacts_used: int = Field(ge=1)
    confidence: float = Field(ge=0.5)

    @model_validator(mode="after")
    def validate_business_rules(self) -> FileSystemActionReport:
        lowered = self.summary.lower()
        if not any(
            token in lowered
            for token in (
                "created",
                "wrote",
                "directory",
                "folder",
                "file",
                "создан",
                "записан",
                "папк",
                "файл",
            )
        ):
            raise ValueError(
                "summary must describe a concrete file or directory action"
            )
        if not any(ref.strip() for ref in self.citations):
            raise ValueError("citations must contain at least one non-empty reference")
        return self


def build_demo_crew(
    *,
    env_file: str = ".env",
    workspace_root: str = ".",
    include_critic: bool = False,
) -> Crew:
    """Assemble the starter multi-agent demo crew with OpenAI and local tools."""
    load_env_file(env_file)
    tool_registry = ToolRegistry()
    register_openai_research_tool(tool_registry, env_path=env_file)
    register_autonomy_toolkit(tool_registry, workspace_root=workspace_root)
    crew = Crew(
        name="openai-demo",
        agents=build_starter_agent_configs(include_critic=include_critic),
        tool_registry=tool_registry,
        schema_registry={
            "ResearchReport": ResearchReport,
            "FileSystemActionReport": FileSystemActionReport,
        },
    )
    register_starter_specialists(
        agent_registry=crew.agent_registry,
        tool_registry=crew.tool_registry,
    )
    return crew


def build_demo_graph(
    *,
    env_file: str = ".env",
    workspace_root: str = ".",
    include_critic: bool = False,
    checkpointer: BaseCheckpointSaver[Any] | None = None,
) -> tuple[Crew, Any]:
    """Compile the demo crew into a LangGraph runnable graph."""
    crew = build_demo_crew(
        env_file=env_file,
        workspace_root=workspace_root,
        include_critic=include_critic,
    )
    graph = crew.compile(checkpointer=checkpointer or InMemorySaver())
    return crew, graph


def build_demo_task(
    *,
    query: str,
    output_schema: str = "ResearchReport",
    enable_planner: bool = False,
    require_citations: bool = False,
    fact_logic_validation: FactLogicValidationMode = FactLogicValidationMode.NONE,
    hitl_points: list[HITLPoint] | None = None,
    locale: str | None = None,
) -> TaskSpec:
    """Build a default `TaskSpec` for local demo runs."""
    normalized = _normalize_demo_task(
        query=query,
        output_schema=output_schema,
        fact_logic_validation=fact_logic_validation,
        hitl_points=hitl_points or [],
    )
    return TaskSpec(
        task_id=f"task-{uuid4()}",
        description=normalized["description"],
        output_schema=normalized["output_schema"],
        assignee=normalized["assignee"],
        auto_route=normalized["auto_route"],
        fact_logic_validation=normalized["fact_logic_validation"],
        hitl_points=normalized["hitl_points"],
        metadata={
            "enable_planner": enable_planner,
            "require_citations": require_citations,
            "locale": (locale or _infer_locale(query)).lower(),
            **normalized["metadata"],
        },
    )


def run_task(
    graph: Any,
    *,
    task: TaskSpec,
    thread_id: str | None = None,
    stream: bool = True,
) -> tuple[AgentState, list[dict[str, Any]]]:
    """Run a single task and optionally collect stream envelopes."""
    resolved_thread_id = thread_id or f"demo-{uuid4()}"
    initial_state = AgentState(thread_id=resolved_thread_id, task=task)
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
        snapshot = graph.get_state(config)
        return AgentState.model_validate(snapshot.values), events

    result = graph.invoke(initial_state.model_dump(by_alias=True), config)
    return AgentState.model_validate(result), []


def resume_task(
    graph: Any,
    *,
    thread_id: str,
    human_feedback: dict[str, Any],
    stream: bool = True,
) -> tuple[AgentState, list[dict[str, Any]]]:
    """Resume a HITL-paused thread and optionally collect stream envelopes."""
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
        snapshot = graph.get_state(config)
        return AgentState.model_validate(snapshot.values), events

    result = graph.invoke(Command(resume=human_feedback), config)
    return AgentState.model_validate(result), []


def render_state_summary(state: AgentState) -> dict[str, Any]:
    """Render a compact, UI-friendly summary of the current thread state."""
    return {
        "thread_id": state.thread_id,
        "status": state.status.value,
        "current_agent": state.current_agent,
        "output_candidate": state.output_candidate,
        "memory_refs": state.memory_refs,
        "errors": state.errors,
        "human_checkpoint": (
            state.human_checkpoint.model_dump(by_alias=True)
            if state.human_checkpoint is not None
            else None
        ),
        "retry_counters": state.retry_counters,
    }


def _normalize_demo_task(
    *,
    query: str,
    output_schema: str,
    fact_logic_validation: FactLogicValidationMode,
    hitl_points: list[HITLPoint],
) -> dict[str, Any]:
    lowered = query.lower()
    if not _is_filesystem_task(lowered):
        return {
            "description": query,
            "output_schema": output_schema,
            "assignee": None,
            "auto_route": True,
            "fact_logic_validation": fact_logic_validation,
            "hitl_points": hitl_points,
            "metadata": {},
        }

    action = "create_dir" if _is_directory_task(lowered) else "write_file"
    target_name = _infer_target_path(query)
    target_path = f"{{thread_id}}/{target_name}"
    preview_path = f".agentgraph_cache/workspace/{target_path}"
    required_terms = ["created" if action == "create_dir" else "wrote", target_name]
    normalized_hitl_points = list(
        dict.fromkeys([*hitl_points, HITLPoint.BEFORE_TOOL_CALL])
    )
    return {
        "description": query,
        "output_schema": "FileSystemActionReport",
        "assignee": "file_executor",
        "auto_route": False,
        "fact_logic_validation": FactLogicValidationMode.POLICY,
        "hitl_points": normalized_hitl_points,
        "metadata": {
            "tool_ref": "fs.create_dir" if action == "create_dir" else "fs.write_file",
            "target_path": target_path,
            "dry_run_preview": preview_path,
            "required_terms": required_terms,
            "require_citations": True,
        },
    }


def _is_filesystem_task(lowered_query: str) -> bool:
    filesystem_terms = (
        "folder",
        "directory",
        "mkdir",
        "create dir",
        "create directory",
        "create folder",
        "file",
        "write file",
        "save file",
        "create file",
        "папк",
        "каталог",
        "директор",
        "файл",
        "создай",
        "создать",
        "запиши",
        "записать",
        "сохрани",
        "сохранить",
        "прочитай",
        "прочитать",
        "путь",
    )
    matches = sum(1 for term in filesystem_terms if term in lowered_query)
    return matches >= 2


def _is_directory_task(lowered_query: str) -> bool:
    directory_terms = (
        "folder",
        "directory",
        "mkdir",
        "create dir",
        "create folder",
        "папк",
        "каталог",
        "директор",
    )
    return any(term in lowered_query for term in directory_terms)


def _infer_target_path(query: str) -> str:
    patterns = [
        r"(?:named|called|с именем|под именем|названием)\s+([A-Za-z0-9._/-]+)",
        r"(?:folder|directory|file|папк\w*|каталог\w*|директор\w*|файл\w*)\s+([A-Za-z0-9._/-]+)",
    ]
    target = "MEMORY"
    for pattern in patterns:
        match = re.search(pattern, query, flags=re.IGNORECASE)
        if match is not None:
            target = match.group(1)
            break
    return target.strip("/\\")


def _infer_locale(query: str) -> str:
    if any("а" <= ch.lower() <= "я" or ch.lower() == "ё" for ch in query):
        return "ru"
    return "en"
