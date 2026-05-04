"""Integration tests for the AgentGraph vertical slice."""

from __future__ import annotations

import json
import os
import subprocess
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Event, Thread, Timer
from time import monotonic
from typing import Any
from unittest.mock import patch
from urllib import error as urlerror
from uuid import uuid4

from langgraph.checkpoint.memory import InMemorySaver
from pydantic import BaseModel

from agentgraph import (
    AgentConfig,
    AgentState,
    AutonomyToolkitConfig,
    CapabilityDescriptor,
    Crew,
    DelegationPolicy,
    DesktopCrewBundle,
    DesktopEnforcedToolRegistry,
    DesktopPolicyBundle,
    DesktopToolsCatalog,
    DesktopWorkflowBundle,
    FileSystemActionReport,
    HITLPoint,
    IdempotentToolRegistry,
    InMemoryVectorBackend,
    MemoryPolicy,
    MemorySyncService,
    NetworkXGraphBackend,
    OpenAISettings,
    RetryPolicy,
    RollbackManager,
    SQLiteCheckpointBackend,
    TaskSpec,
    ToolBinding,
    ToolRegistry,
    ToolResult,
    apply_desktop_enforcement_updates,
    build_demo_task,
    build_desktop_demo_crew,
    build_desktop_demo_graph,
    build_desktop_task,
    build_starter_agent_configs,
    compile_with_observability,
    default_agents_yaml_path,
    default_desktop_executor_yaml_path,
    default_desktop_policies_yaml_path,
    default_desktop_tools_yaml_path,
    default_desktop_workflow_yaml_path,
    describe_agent_configs,
    export_desktop_artifact,
    load_core_agent_configs,
    load_desktop_executor_config,
    load_desktop_policy_bundle,
    load_desktop_tools_catalog,
    load_desktop_workflow_bundle,
    load_env_file,
    load_specialist_agent_config,
    register_autonomy_toolkit,
    register_desktop_executor,
    register_openai_research_tool,
    register_starter_specialists,
    resume_desktop_plan,
    resume_desktop_task,
    resume_task,
    run_desktop_plan,
    run_desktop_task,
    run_task,
    stream_envelopes,
)
from agentgraph.async_memory_sync import AsyncMemorySyncBackend
from agentgraph.contracts import (
    FactLogicValidationMode,
    PermissionLevel,
    SideEffectLevel,
    ThreadStatus,
)
from agentgraph.planner_adapter import PlannerAdapter, register_planner_agent
from agentgraph.runtime import AgentRegistry, resume_from_human_feedback
from agentgraph.schema_evolution import (
    CURRENT_AGENT_STATE_SCHEMA_VERSION,
    migrate_checkpoint_agent_state,
)
from agentgraph.validator_adapter import (
    DynamicValidatorAdapter,
    register_validator_agent,
)


class ResearchReport(BaseModel):
    """Output schema used by the vertical slice tests."""

    task_id: str
    summary: str
    citations: list[str]
    artifacts_used: int
    confidence: float


def _test_artifacts_dir() -> Path:
    path = Path(__file__).resolve().parent / "artifacts"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _build_crew(
    *,
    requires_hitl: bool = False,
    tool_registry: ToolRegistry | None = None,
    memory_backend: Any | None = None,
) -> Crew:
    registry = tool_registry or ToolRegistry()

    def search_tool(args: dict[str, Any]) -> ToolResult:
        query = str(args["query"])
        return ToolResult(
            success=True,
            data={
                "results": [
                    {
                        "title": "LangGraph",
                        "snippet": f"LangGraph orchestrates durable stateful agents for {query}.",
                    }
                ]
            },
        )

    registry.register(
        tool_ref="research.search",
        description="Research knowledge lookup.",
        side_effect_level=SideEffectLevel.READ_ONLY,
        handler=search_tool,
    )

    researcher = AgentConfig(
        agent_id="researcher",
        role="researcher",
        goal="Collect relevant evidence",
        backstory="Finds focused technical evidence for downstream synthesis.",
        tools=[
            ToolBinding(
                tool_ref="research.search",
                permission_level=PermissionLevel.READ,
                requires_hitl=requires_hitl,
                allowed_side_effect_level=SideEffectLevel.READ_ONLY,
            )
        ],
        delegation_policy=DelegationPolicy(
            confidence_threshold=0.10, policy_weight=1.0
        ),
        memory_policy=MemoryPolicy(
            link_generation=True, versioning=True, auto_sync=True
        ),
        capabilities=[
            CapabilityDescriptor(
                capability_id="cap-research",
                name="Technical Research",
                summary="Finds documentation and evidence for LangGraph questions.",
                keywords=["research", "langgraph", "evidence"],
                domains=["agents", "orchestration"],
                tool_affinity=["research.search"],
                embedding_text="langgraph research documentation evidence orchestration",
            )
        ],
    )

    return Crew(
        name="core-mvp",
        agents=[researcher],
        tool_registry=registry,
        schema_registry={"ResearchReport": ResearchReport},
        memory_backend=memory_backend,
    )


def _initial_input(task: TaskSpec, *, thread_id: str) -> dict[str, Any]:
    return AgentState(thread_id=thread_id, task=task).model_dump(by_alias=True)


def test_happy_path_stream_envelope() -> None:
    """A task routes to Researcher, validates, syncs memory, and completes."""
    crew = _build_crew()
    graph = crew.compile(checkpointer=InMemorySaver())
    thread_id = "happy-path"
    task = TaskSpec(
        task_id="task-1",
        description="Summarize LangGraph",
        output_schema="ResearchReport",
        auto_route=True,
        retry_policy=RetryPolicy(max_attempts=2),
        fact_logic_validation=FactLogicValidationMode.POLICY,
        metadata={"require_citations": True, "required_terms": ["LangGraph"]},
    )

    result = graph.invoke(
        _initial_input(task, thread_id=thread_id),
        {"configurable": {"thread_id": thread_id}},
    )
    state = AgentState.model_validate(result)

    assert state.status.value == "COMPLETED"
    assert state.output_candidate["citations"] == ["LangGraph"]
    assert state.memory_refs
    assert any(
        message.metadata.get("event_name") == "context_updated"
        for message in state.protocol_messages
    )

    events = list(
        stream_envelopes(
            graph,
            _initial_input(task, thread_id=f"{thread_id}-stream"),
            {"configurable": {"thread_id": f"{thread_id}-stream"}},
            stream_mode="updates",
        )
    )
    assert events
    assert {"event_type", "thread_id", "timestamp", "payload", "metadata"} <= events[
        0
    ].keys()


def test_hitl_interrupt_and_resume() -> None:
    """Tool execution blocks on HITL and resumes from human feedback."""
    crew = _build_crew(requires_hitl=True)
    graph = crew.compile(checkpointer=InMemorySaver())
    thread_id = "hitl-thread"
    task = TaskSpec(
        task_id="task-2",
        description="Research LangGraph with approval",
        output_schema="ResearchReport",
        auto_route=True,
        hitl_points=[HITLPoint.BEFORE_TOOL_CALL],
        fact_logic_validation=FactLogicValidationMode.NONE,
    )

    chunks = list(
        graph.stream(
            _initial_input(task, thread_id=thread_id),
            {"configurable": {"thread_id": thread_id}},
            stream_mode="updates",
        )
    )
    assert chunks[-1]["__interrupt__"]

    resumed = resume_from_human_feedback(
        graph,
        thread_id=thread_id,
        human_feedback={"decision": "approve", "resume_to": "specialist_executor"},
    )
    state = AgentState.model_validate(resumed)

    assert state.status.value == "COMPLETED"
    assert "research.search" in state.shared_context["approved_tools"]
    assert state.output_candidate["summary"]


def test_validation_failure_repair_loop() -> None:
    """A schema-validation failure repairs through the synthesizer and then completes."""
    crew = _build_crew()
    graph = crew.compile(checkpointer=InMemorySaver())
    thread_id = "repair-thread"
    task = TaskSpec(
        task_id="task-3",
        description="Summarize LangGraph with one forced repair",
        output_schema="ResearchReport",
        auto_route=True,
        retry_policy=RetryPolicy(schema_repair_attempts=1, max_attempts=2),
        fact_logic_validation=FactLogicValidationMode.NONE,
        metadata={"force_schema_failure_once": True},
    )

    result = graph.invoke(
        _initial_input(task, thread_id=thread_id),
        {"configurable": {"thread_id": thread_id}},
    )
    state = AgentState.model_validate(result)

    assert state.status.value == "COMPLETED"
    assert state.retry_counters["schema"] == 1
    assert state.schema_validation is not None
    assert state.schema_validation["valid"] is True


def test_sqlite_checkpoint_backend_resume_and_serialization() -> None:
    """SQLite checkpoints survive graph recreation and resume cleanly."""
    tool_registry = IdempotentToolRegistry(ttl_seconds=60.0)
    crew = _build_crew(requires_hitl=True, tool_registry=tool_registry)
    checkpoint_dir = Path(__file__).resolve().parent / "artifacts"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / "sqlite-hitl-thread.sqlite"
    if checkpoint_path.exists():
        checkpoint_path.unlink()

    thread_id = "sqlite-hitl-thread"
    task = TaskSpec(
        task_id="task-sqlite",
        description="Research LangGraph with sqlite persistence",
        output_schema="ResearchReport",
        auto_route=True,
        hitl_points=[HITLPoint.BEFORE_TOOL_CALL],
        fact_logic_validation=FactLogicValidationMode.NONE,
    )

    try:
        with SQLiteCheckpointBackend(str(checkpoint_path)) as backend:
            observed_graph = compile_with_observability(
                crew,
                checkpointer=backend.as_langgraph_checkpointer(),
            )
            events = list(
                observed_graph.stream(
                    _initial_input(task, thread_id=thread_id),
                    {"configurable": {"thread_id": thread_id}},
                )
            )
            assert events[-1]["event_type"] == "interrupt"
            assert backend.load(thread_id=thread_id) is not None
            assert backend.list(thread_id=thread_id)

        with SQLiteCheckpointBackend(str(checkpoint_path)) as backend:
            observed_graph = compile_with_observability(
                crew,
                checkpointer=backend.as_langgraph_checkpointer(),
            )
            resumed = resume_from_human_feedback(
                observed_graph.graph,
                thread_id=thread_id,
                human_feedback={
                    "decision": "approve",
                    "resume_to": "specialist_executor",
                },
            )
            state = AgentState.model_validate(resumed)
            payload = state.model_dump_json()

            assert state.status.value == "COMPLETED"
            assert '"thread_id":"sqlite-hitl-thread"' in payload
            assert backend.load(thread_id=thread_id) is not None
    finally:
        if checkpoint_path.exists():
            checkpoint_path.unlink()


def test_memory_sync_with_mock_vector_and_graph_backends() -> None:
    """Async memory sync propagates note updates without blocking completion."""

    class MockVectorBackend(InMemoryVectorBackend):
        def __init__(self) -> None:
            super().__init__()
            self.upsert_calls = 0

        def upsert(
            self,
            *,
            thread_id: str,
            document_id: str,
            text: str,
            metadata: dict[str, Any],
        ) -> dict[str, Any]:
            self.upsert_calls += 1
            return super().upsert(
                thread_id=thread_id,
                document_id=document_id,
                text=text,
                metadata=metadata,
            )

    class MockGraphBackend(NetworkXGraphBackend):
        def __init__(self) -> None:
            super().__init__()
            self.upsert_calls = 0

        def upsert_links(
            self,
            *,
            thread_id: str,
            source: str,
            links: list[str],
            metadata: dict[str, Any],
        ) -> list[dict[str, Any]]:
            self.upsert_calls += 1
            return super().upsert_links(
                thread_id=thread_id,
                source=source,
                links=links,
                metadata=metadata,
            )

    vector_backend = MockVectorBackend()
    graph_backend = MockGraphBackend()
    memory_sync = MemorySyncService(
        vector_backend=vector_backend,
        graph_backend=graph_backend,
    )
    async_memory_backend = AsyncMemorySyncBackend(memory_sync_service=memory_sync)
    crew = _build_crew(memory_backend=async_memory_backend)
    graph = crew.compile(checkpointer=InMemorySaver())
    thread_id = "memory-sync-thread"
    task = TaskSpec(
        task_id="task-memory",
        description="Summarize LangGraph with synced memory",
        output_schema="ResearchReport",
        auto_route=True,
        fact_logic_validation=FactLogicValidationMode.NONE,
    )

    result = graph.invoke(
        _initial_input(task, thread_id=thread_id),
        {"configurable": {"thread_id": thread_id}},
    )
    state = AgentState.model_validate(result)

    assert state.status.value == "COMPLETED"
    assert state.sync_ticket_id is not None
    ticket_ref = next(
        ref for ref in state.memory_refs if ref["ref_type"] == "async_memory_ticket"
    )
    assert state.sync_ticket_id == ticket_ref["ticket_id"]
    async_memory_backend.wait(ticket_ref["ticket_id"], timeout=5.0)
    assert vector_backend.upsert_calls == 1
    assert graph_backend.upsert_calls == 1
    drained = async_memory_backend.drain(timeout=5.0)
    refs = [ref for item in drained for ref in item["refs"]]
    assert any(ref["ref_type"] == "vector" for ref in refs)
    assert any(ref["ref_type"] == "graph" for ref in refs)


def test_idempotent_tool_registry_cache() -> None:
    """Repeated read-only tool calls are served from the idempotency cache."""
    call_count = {"value": 0}
    registry = IdempotentToolRegistry(ttl_seconds=60.0)

    def cached_tool(args: dict[str, Any]) -> ToolResult:
        call_count["value"] += 1
        return ToolResult(success=True, data={"echo": args["query"]})

    registry.register(
        tool_ref="research.search",
        description="Cacheable research tool.",
        side_effect_level=SideEffectLevel.READ_ONLY,
        handler=cached_tool,
    )

    first = registry.invoke("research.search", {"query": "langgraph"})
    second = registry.invoke("research.search", {"query": "langgraph"})

    assert call_count["value"] == 1
    assert first.success is True
    assert second.metadata["cache_hit"] is True


def test_validator_dynamic_lookup_and_policy_routing() -> None:
    """Policy validation falls back to a registered validator specialist."""
    crew = _build_crew()

    def validator_handler(args: dict[str, Any]) -> ToolResult:
        output = args["output_candidate"]
        summary = str(output.get("summary", ""))
        if "LangGraph" in summary:
            return ToolResult(
                success=True, data={"valid": True, "source": "specialist"}
            )
        return ToolResult(
            success=True,
            data={"valid": False, "errors": ["summary missing LangGraph"]},
        )

    register_validator_agent(
        agent_registry=crew.agent_registry,
        tool_registry=crew.tool_registry,
        tool_ref="validator.logic",
        validator_handler=validator_handler,
    )

    graph = crew.compile(checkpointer=InMemorySaver())
    thread_id = "validator-thread"
    task = TaskSpec(
        task_id="task-validator",
        description="Summarize agent orchestration",
        output_schema="ResearchReport",
        auto_route=True,
        fact_logic_validation=FactLogicValidationMode.SPECIALIST,
        retry_policy=RetryPolicy(logic_repair_attempts=1),
        metadata={"required_terms": ["NonexistentTerm"]},
    )
    result = graph.invoke(
        _initial_input(task, thread_id=thread_id),
        {"configurable": {"thread_id": thread_id}},
    )
    state = AgentState.model_validate(result)
    adapter = DynamicValidatorAdapter(
        agent_registry=crew.agent_registry,
        tool_registry=crew.tool_registry,
    )

    validated = adapter.validate(state)

    assert validated.logic_validation is not None
    assert validated.logic_validation["valid"] is True
    assert validated.logic_validation["mode"] == "specialist"
    assert validated.status.value == "COMPLETED"


def test_planner_subtask_decomposition_and_send_aggregation() -> None:
    """Planner decomposes a task and uses Send-based parallel subtask execution."""
    crew = _build_crew()
    register_planner_agent(agent_registry=crew.agent_registry)
    planner = PlannerAdapter(crew=crew)
    task = TaskSpec(
        task_id="task-planner",
        description="Research LangGraph and summarize memory",
        output_schema="ResearchReport",
        auto_route=True,
        metadata={
            "enable_planner": True,
            "subtasks": ["Research LangGraph", "Summarize memory"],
        },
    )
    state = AgentState(thread_id="planner-thread", task=task)

    planned = planner.invoke(state)

    assert planned.status.value == "COMPLETED"
    assert len(planned.shared_context["subtask_results"]) == 2
    assert planned.output_candidate["subtasks_completed"] == 2
    assert "summary" in planned.output_candidate


def test_planner_partial_results_apply_fallback_strategy() -> None:
    """A failing subtask yields partial aggregation and reroute-compatible state."""
    crew = _build_crew()
    register_planner_agent(agent_registry=crew.agent_registry)
    planner = PlannerAdapter(crew=crew)
    task = TaskSpec(
        task_id="task-planner-failure",
        description="Research LangGraph and summarize memory",
        output_schema="ResearchReport",
        auto_route=True,
        retry_policy=RetryPolicy(fallback_strategy="reroute"),
        metadata={
            "enable_planner": True,
            "subtasks": [
                {"description": "Research LangGraph"},
                {
                    "description": "Summarize memory",
                    "metadata": {"force_failure": True},
                },
            ],
        },
    )
    state = AgentState(thread_id="planner-failure-thread", task=task)

    planned = planner.invoke(state)

    assert planned.status.value == "REPAIR"
    assert planned.shared_context["failed_subtask_ids"] == [
        "task-planner-failure-subtask-1"
    ]
    assert planned.output_candidate["partial_results"] is True
    assert any(error["type"] == "planner_subtask" for error in planned.errors)


def test_schema_version_migration_on_old_checkpoint_load() -> None:
    """Legacy checkpoint payloads are migrated into a versioned agent state."""
    old_state = {
        "thread_id": "legacy-thread",
        "task_spec": {
            "task_id": "legacy-task",
            "description": "Legacy task",
            "output_schema": "ResearchReport",
        },
        "messages": [],
    }
    checkpoint = {
        "v": 1,
        "ts": "2026-05-01T00:00:00+00:00",
        "id": "legacy-checkpoint",
        "channel_values": {"agent_state": old_state},
        "channel_versions": {},
        "versions_seen": {},
    }

    migrated = migrate_checkpoint_agent_state(checkpoint)

    assert migrated is not None
    assert migrated.schema_version == CURRENT_AGENT_STATE_SCHEMA_VERSION
    assert migrated.state.thread_id == "legacy-thread"
    assert migrated.state.task.task_id == "legacy-task"
    assert migrated.state.retry_counters["logic"] == 0
    assert migrated.state.sync_ticket_id is None


def test_resume_waits_for_pending_async_memory_ticket() -> None:
    """Resume safety waits for pending async memory sync unless stale refs are allowed."""

    class BlockingMemorySyncService:
        def __init__(self) -> None:
            self.released = Event()

        def sync(
            self,
            *,
            thread_id: str,
            content: dict[str, Any],
            auto_sync: bool,
            link_generation: bool,
            versioning: bool,
        ) -> list[dict[str, Any]]:
            self.released.wait(timeout=5.0)
            return [{"ref_type": "note", "thread_id": thread_id, "links": []}]

    class FakeStateSnapshot:
        def __init__(self, values: dict[str, Any]) -> None:
            self.values = values

    class FakeGraph:
        def __init__(self, state: AgentState) -> None:
            self._state = state

        def get_state(self, config: dict[str, Any]) -> FakeStateSnapshot:
            return FakeStateSnapshot(self._state.model_dump(by_alias=True))

    from agentgraph.async_memory_sync import ensure_resume_safe

    memory_backend = AsyncMemorySyncBackend(
        memory_sync_service=BlockingMemorySyncService()
    )
    refs = memory_backend.sync(
        thread_id="resume-safety-thread",
        content={"summary": "LangGraph"},
        auto_sync=True,
        link_generation=True,
        versioning=True,
    )
    ticket_id = refs[0]["ticket_id"]
    state = AgentState(
        thread_id="resume-safety-thread",
        task=TaskSpec(
            task_id="resume-safety",
            description="resume safety",
            output_schema="ResearchReport",
        ),
        memory_refs=refs,
        sync_ticket_id=ticket_id,
    )
    graph = FakeGraph(state)
    timer = Timer(0.2, memory_backend.memory_sync_service.released.set)
    timer.start()
    started = monotonic()
    try:
        result = ensure_resume_safe(
            graph, thread_id="resume-safety-thread", timeout=5.0
        )
    finally:
        timer.cancel()
    elapsed = monotonic() - started

    assert result["status"] == "completed"
    assert elapsed >= 0.15


def test_load_env_file_populates_process_environment() -> None:
    """Local `.env` files are loaded into the process environment."""
    env_path = _test_artifacts_dir() / "test-openai.env"
    env_path.write_text(
        "\n".join(
            [
                "# test env",
                "OPENAI_API_KEY=test-key",
                'OPENAI_MODEL="gpt-4.1-mini"',
            ]
        ),
        encoding="utf-8",
    )
    os.environ.pop("OPENAI_API_KEY", None)
    os.environ.pop("OPENAI_MODEL", None)

    try:
        loaded = load_env_file(env_path)

        assert loaded["OPENAI_API_KEY"] == "test-key"
        assert os.environ["OPENAI_API_KEY"] == "test-key"
        assert os.environ["OPENAI_MODEL"] == "gpt-4.1-mini"
    finally:
        if env_path.exists():
            env_path.unlink()


def test_register_openai_research_tool_invokes_http_adapter() -> None:
    """The OpenAI adapter loads `.env` and registers a real HTTP-backed tool."""

    class FakeHTTPResponse:
        def __enter__(self) -> FakeHTTPResponse:
            return self

        def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
            return None

        def read(self) -> bytes:
            payload = {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "results": [
                                        {
                                            "title": "LangGraph",
                                            "snippet": "Durable agent orchestration.",
                                        }
                                    ]
                                }
                            )
                        }
                    }
                ]
            }
            return json.dumps(payload).encode("utf-8")

    env_path = _test_artifacts_dir() / "test-openai-http.env"
    env_path.write_text("OPENAI_API_KEY=test-key\n", encoding="utf-8")
    tool_registry = ToolRegistry()
    captured: dict[str, Any] = {}

    def fake_urlopen(req: Any, timeout: float) -> FakeHTTPResponse:
        captured["url"] = req.full_url
        captured["authorization"] = req.headers.get("Authorization")
        captured["timeout"] = timeout
        captured["body"] = json.loads(req.data.decode("utf-8"))
        return FakeHTTPResponse()

    try:
        with patch(
            "agentgraph.openai_adapter.request.urlopen", side_effect=fake_urlopen
        ):
            settings = register_openai_research_tool(
                tool_registry,
                env_path=env_path,
                override_env=True,
            )
            result = tool_registry.invoke(
                "research.search",
                {
                    "query": "Summarize LangGraph",
                    "context_refs": [],
                    "shared_context": {},
                },
            )

        assert isinstance(settings, OpenAISettings)
        assert settings.api_key == "test-key"
        assert captured["url"].endswith("/chat/completions")
        assert captured["authorization"] == "Bearer test-key"
        assert captured["body"]["model"] == "gpt-4.1-mini"
        assert result.success is True
        assert result.data["results"][0]["title"] == "LangGraph"
    finally:
        if env_path.exists():
            env_path.unlink()


def test_openai_settings_sanitize_non_ascii_api_key() -> None:
    """The adapter should ignore accidental non-ASCII noise in OPENAI_API_KEY."""
    env_path = _test_artifacts_dir() / "test-openai-sanitize.env"
    env_path.write_text('OPENAI_API_KEY="test-keyя"\n', encoding="utf-8")

    try:
        settings = OpenAISettings.from_env(env_path=env_path, override=True)
        assert settings.api_key == "test-key"
    finally:
        if env_path.exists():
            env_path.unlink()


def test_openai_research_tool_includes_russian_locale_instruction() -> None:
    """The OpenAI adapter should request Russian output when locale is `ru`."""

    class FakeHTTPResponse:
        def __enter__(self) -> FakeHTTPResponse:
            return self

        def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
            return None

        def read(self) -> bytes:
            payload = {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "results": [
                                        {
                                            "title": "LangGraph",
                                            "snippet": "LangGraph используется для оркестрации stateful agent workflow.",
                                        }
                                    ]
                                }
                            )
                        }
                    }
                ]
            }
            return json.dumps(payload).encode("utf-8")

    env_path = _test_artifacts_dir() / "test-openai-locale.env"
    env_path.write_text("OPENAI_API_KEY=test-key\n", encoding="utf-8")
    tool_registry = ToolRegistry()
    captured: dict[str, Any] = {}

    def fake_urlopen(req: Any, timeout: float) -> FakeHTTPResponse:
        captured["body"] = json.loads(req.data.decode("utf-8"))
        return FakeHTTPResponse()

    try:
        with patch(
            "agentgraph.openai_adapter.request.urlopen",
            side_effect=fake_urlopen,
        ):
            register_openai_research_tool(
                tool_registry,
                env_path=env_path,
                override_env=True,
            )
            result = tool_registry.invoke(
                "research.search",
                {
                    "query": "Собери краткое описание LangGraph",
                    "context_refs": [],
                    "shared_context": {},
                    "task": {"metadata": {"locale": "ru"}},
                },
            )

        assert "strictly in Russian" in captured["body"]["messages"][0]["content"]
        assert "Output language: Russian" in captured["body"]["messages"][1]["content"]
        assert result.success is True
    finally:
        if env_path.exists():
            env_path.unlink()


def test_openai_research_tool_normalizes_noncanonical_json_payload() -> None:
    """The OpenAI adapter coerces varied JSON shapes into stable research results."""

    class FakeHTTPResponse:
        def __enter__(self) -> FakeHTTPResponse:
            return self

        def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
            return None

        def read(self) -> bytes:
            payload = {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "summary": "LangGraph is used to orchestrate durable, stateful agent workflows.",
                                    "citations": ["LangGraph overview"],
                                }
                            )
                        }
                    }
                ]
            }
            return json.dumps(payload).encode("utf-8")

    env_path = _test_artifacts_dir() / "test-openai-normalize.env"
    env_path.write_text("OPENAI_API_KEY=test-key\n", encoding="utf-8")
    tool_registry = ToolRegistry()

    try:
        with patch(
            "agentgraph.openai_adapter.request.urlopen",
            side_effect=lambda req, timeout: FakeHTTPResponse(),
        ):
            register_openai_research_tool(
                tool_registry,
                env_path=env_path,
                override_env=True,
            )
            result = tool_registry.invoke(
                "research.search",
                {
                    "query": "Что такое LangGraph?",
                    "context_refs": [],
                    "shared_context": {},
                },
            )

        assert result.success is True
        assert result.data["results"]
        assert result.data["results"][0]["snippet"].startswith("LangGraph is used")
    finally:
        if env_path.exists():
            env_path.unlink()


def test_openai_research_tool_filters_unverified_direct_pdf_results() -> None:
    """Direct PDF search should keep only verified URLs before downstream download."""

    class FakeResponsesAPIResponse:
        def __enter__(self) -> FakeResponsesAPIResponse:
            return self

        def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
            return None

        def read(self) -> bytes:
            payload = {
                "output": [
                    {
                        "type": "web_search_call",
                        "action": {
                            "type": "search",
                            "sources": [
                                {
                                    "type": "url",
                                    "url": "https://docs.example.test/langgraph-manual.pdf",
                                },
                                {
                                    "type": "url",
                                    "url": "https://docs.example.test/langgraph-api.pdf",
                                },
                            ],
                        },
                    },
                    {
                        "type": "message",
                        "content": [
                            {
                                "type": "output_text",
                                "text": json.dumps(
                                    {
                                        "results": [
                                            {
                                                "title": "Broken PDF",
                                                "snippet": "Missing file",
                                                "url": "https://example.test/missing.pdf",
                                            },
                                            {
                                                "title": "Working PDF",
                                                "snippet": "Available file",
                                                "url": "https://example.test/manual.pdf",
                                            },
                                        ]
                                    }
                                ),
                            }
                        ],
                    },
                ]
            }
            return json.dumps(payload).encode("utf-8")

    class FakeProbeResponse:
        status = 200

        def __enter__(self) -> FakeProbeResponse:
            return self

        def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
            return None

        @property
        def headers(self) -> dict[str, str]:
            return {"Content-Type": "application/pdf"}

        def read(self) -> bytes:
            return b""

    env_path = _test_artifacts_dir() / "test-openai-verified-pdf.env"
    env_path.write_text("OPENAI_API_KEY=test-key\n", encoding="utf-8")
    tool_registry = ToolRegistry()
    captured: dict[str, Any] = {}

    def fake_urlopen(req: Any, timeout: float) -> Any:
        url = req.full_url
        if url.endswith("/responses"):
            captured["url"] = url
            captured["body"] = json.loads(req.data.decode("utf-8"))
            return FakeResponsesAPIResponse()
        return FakeProbeResponse()

    try:
        with patch(
            "agentgraph.openai_adapter.request.urlopen",
            side_effect=fake_urlopen,
        ):
            register_openai_research_tool(
                tool_registry,
                env_path=env_path,
                override_env=True,
                tool_ref="web.search",
            )
            result = tool_registry.invoke(
                "web.search",
                {
                    "query": "find LangGraph PDF docs",
                    "shared_context": {},
                    "task": {
                        "metadata": {
                            "search_constraints": {
                                "prefer_file_type": "pdf",
                                "direct_links_only": True,
                            }
                        }
                    },
                },
            )

        assert result.success is True
        assert captured["url"].endswith("/responses")
        assert captured["body"]["tools"][0]["type"] == "web_search"
        assert result.data["results"] == [
            {
                "title": "Broken PDF",
                "snippet": "Missing file",
                "url": "https://docs.example.test/langgraph-manual.pdf",
            },
            {
                "title": "Working PDF",
                "snippet": "Available file",
                "url": "https://docs.example.test/langgraph-api.pdf",
            },
        ]
    finally:
        if env_path.exists():
            env_path.unlink()


def test_openai_research_tool_fails_when_no_verified_direct_pdf_results() -> None:
    """Direct PDF search should fail early when all generated URLs are dead."""

    class FakeResponsesAPIResponse:
        def __enter__(self) -> FakeResponsesAPIResponse:
            return self

        def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
            return None

        def read(self) -> bytes:
            payload = {
                "output": [
                    {
                        "type": "web_search_call",
                        "action": {
                            "type": "search",
                            "sources": [
                                {
                                    "type": "url",
                                    "url": "https://docs.example.test/missing-1.pdf",
                                },
                                {
                                    "type": "url",
                                    "url": "https://docs.example.test/missing-2.pdf",
                                },
                            ],
                        },
                    },
                    {
                        "type": "message",
                        "content": [
                            {
                                "type": "output_text",
                                "text": json.dumps(
                                    {
                                        "results": [
                                            {
                                                "title": "Broken PDF 1",
                                                "snippet": "Missing file",
                                                "url": "https://example.test/missing-1.pdf",
                                            },
                                            {
                                                "title": "Broken PDF 2",
                                                "snippet": "Missing file",
                                                "url": "https://example.test/missing-2.pdf",
                                            },
                                        ]
                                    }
                                ),
                            }
                        ],
                    },
                ]
            }
            return json.dumps(payload).encode("utf-8")

    env_path = _test_artifacts_dir() / "test-openai-no-verified-pdf.env"
    env_path.write_text("OPENAI_API_KEY=test-key\n", encoding="utf-8")
    tool_registry = ToolRegistry()

    def fake_urlopen(req: Any, timeout: float) -> Any:
        url = req.full_url
        if url.endswith("/responses"):
            return FakeResponsesAPIResponse()
        raise urlerror.HTTPError(url, 404, "Not Found", hdrs=None, fp=None)

    try:
        with patch(
            "agentgraph.openai_adapter.request.urlopen",
            side_effect=fake_urlopen,
        ):
            register_openai_research_tool(
                tool_registry,
                env_path=env_path,
                override_env=True,
                tool_ref="web.search",
            )
            result = tool_registry.invoke(
                "web.search",
                {
                    "query": "find LangGraph PDF docs",
                    "shared_context": {},
                    "task": {
                        "metadata": {
                            "search_constraints": {
                                "prefer_file_type": "pdf",
                                "direct_links_only": True,
                            }
                        }
                    },
                },
            )

        assert result.success is False
        assert result.error_type == "no_results"
        assert result.metadata["retriable"] is False
        assert result.metadata["attempted_urls"] == [
            "https://docs.example.test/missing-1.pdf",
            "https://docs.example.test/missing-2.pdf",
        ]
    finally:
        if env_path.exists():
            env_path.unlink()


def test_openai_web_search_uses_sources_when_output_text_is_not_json() -> None:
    """Web search should fall back to tool sources when Responses text is plain prose."""

    class FakeResponsesAPIResponse:
        def __enter__(self) -> FakeResponsesAPIResponse:
            return self

        def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
            return None

        def read(self) -> bytes:
            payload = {
                "output": [
                    {
                        "type": "web_search_call",
                        "action": {
                            "type": "search",
                            "sources": [
                                {
                                    "type": "url",
                                    "url": "https://docs.langchain.com/oss/python/langgraph/overview",
                                },
                                {
                                    "type": "url",
                                    "url": "https://langchain-ai.github.io/langgraph/",
                                },
                            ],
                        },
                    },
                    {
                        "type": "message",
                        "content": [
                            {
                                "type": "output_text",
                                "text": "LangGraph documentation overview and reference links.",
                            }
                        ],
                    },
                ]
            }
            return json.dumps(payload).encode("utf-8")

    env_path = _test_artifacts_dir() / "test-openai-web-search-text.env"
    env_path.write_text("OPENAI_API_KEY=test-key\n", encoding="utf-8")
    tool_registry = ToolRegistry()

    try:
        with patch(
            "agentgraph.openai_adapter.request.urlopen",
            side_effect=lambda req, timeout: FakeResponsesAPIResponse(),
        ):
            register_openai_research_tool(
                tool_registry,
                env_path=env_path,
                override_env=True,
                tool_ref="web.search",
            )
            result = tool_registry.invoke(
                "web.search",
                {
                    "query": "find LangGraph docs",
                    "shared_context": {},
                    "task": {"metadata": {}},
                },
            )

        assert result.success is True
        assert result.data["results"] == [
            {
                "title": "overview",
                "snippet": "LangGraph documentation overview and reference links.",
                "url": "https://docs.langchain.com/oss/python/langgraph/overview",
            },
            {
                "title": "langgraph",
                "snippet": "LangGraph documentation overview and reference links.",
                "url": "https://langchain-ai.github.io/langgraph/",
            },
        ]
    finally:
        if env_path.exists():
            env_path.unlink()


def test_openai_web_search_requests_sources_and_uses_annotations_fallback() -> None:
    """Web search should request explicit sources and recover URLs from message annotations."""

    class FakeResponsesAPIResponse:
        def __enter__(self) -> FakeResponsesAPIResponse:
            return self

        def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
            return None

        def read(self) -> bytes:
            payload = {
                "output": [
                    {
                        "type": "message",
                        "content": [
                            {
                                "type": "output_text",
                                "text": "LangGraph documentation links.",
                                "annotations": [
                                    {
                                        "type": "url_citation",
                                        "url": "https://docs.langchain.com/oss/python/langgraph/overview",
                                    },
                                    {
                                        "type": "url_citation",
                                        "url": "https://langchain-ai.github.io/langgraph/",
                                    },
                                ],
                            }
                        ],
                    }
                ]
            }
            return json.dumps(payload).encode("utf-8")

    env_path = _test_artifacts_dir() / "test-openai-web-search-annotations.env"
    env_path.write_text("OPENAI_API_KEY=test-key\n", encoding="utf-8")
    tool_registry = ToolRegistry()
    captured: dict[str, Any] = {}

    def fake_urlopen(req: Any, timeout: float) -> Any:
        if req.full_url.endswith("/responses"):
            captured["body"] = json.loads(req.data.decode("utf-8"))
        return FakeResponsesAPIResponse()

    try:
        with patch(
            "agentgraph.openai_adapter.request.urlopen",
            side_effect=fake_urlopen,
        ):
            register_openai_research_tool(
                tool_registry,
                env_path=env_path,
                override_env=True,
                tool_ref="web.search",
            )
            result = tool_registry.invoke(
                "web.search",
                {
                    "query": "find LangGraph docs",
                    "shared_context": {},
                    "task": {"metadata": {}},
                },
            )

        assert captured["body"]["include"] == ["web_search_call.action.sources"]
        assert result.success is True
        assert result.data["results"] == [
            {
                "title": "overview",
                "snippet": "LangGraph documentation links.",
                "url": "https://docs.langchain.com/oss/python/langgraph/overview",
            },
            {
                "title": "langgraph",
                "snippet": "LangGraph documentation links.",
                "url": "https://langchain-ai.github.io/langgraph/",
            },
        ]
    finally:
        if env_path.exists():
            env_path.unlink()


def test_openai_web_search_extracts_urls_from_plain_text_when_sources_missing() -> None:
    """Web search should recover direct URLs from plain text when sources are absent."""

    class FakeResponsesAPIResponse:
        def __enter__(self) -> FakeResponsesAPIResponse:
            return self

        def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
            return None

        def read(self) -> bytes:
            payload = {
                "output": [
                    {
                        "type": "message",
                        "content": [
                            {
                                "type": "output_text",
                                "text": (
                                    "Useful PDFs: "
                                    "https://example.test/langgraph-manual.pdf "
                                    "and https://example.test/langgraph-api.pdf"
                                ),
                            }
                        ],
                    }
                ]
            }
            return json.dumps(payload).encode("utf-8")

    class FakeProbeResponse:
        status = 200

        def __enter__(self) -> FakeProbeResponse:
            return self

        def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
            return None

        @property
        def headers(self) -> dict[str, str]:
            return {"Content-Type": "application/pdf"}

        def read(self) -> bytes:
            return b""

    env_path = _test_artifacts_dir() / "test-openai-web-search-text-urls.env"
    env_path.write_text("OPENAI_API_KEY=test-key\n", encoding="utf-8")
    tool_registry = ToolRegistry()

    def fake_urlopen(req: Any, timeout: float) -> Any:
        if req.full_url.endswith("/responses"):
            return FakeResponsesAPIResponse()
        return FakeProbeResponse()

    try:
        with patch(
            "agentgraph.openai_adapter.request.urlopen",
            side_effect=fake_urlopen,
        ):
            register_openai_research_tool(
                tool_registry,
                env_path=env_path,
                override_env=True,
                tool_ref="web.search",
            )
            result = tool_registry.invoke(
                "web.search",
                {
                    "query": "find LangGraph PDFs",
                    "shared_context": {},
                    "task": {
                        "metadata": {
                            "search_constraints": {
                                "prefer_file_type": "pdf",
                                "direct_links_only": True,
                            }
                        }
                    },
                },
            )

        assert result.success is True
        assert result.data["results"] == [
            {
                "title": "langgraph-manual.pdf",
                "snippet": (
                    "Useful PDFs: https://example.test/langgraph-manual.pdf "
                    "and https://example.test/langgraph-api.pdf"
                ),
                "url": "https://example.test/langgraph-manual.pdf",
            },
            {
                "title": "langgraph-api.pdf",
                "snippet": (
                    "Useful PDFs: https://example.test/langgraph-manual.pdf "
                    "and https://example.test/langgraph-api.pdf"
                ),
                "url": "https://example.test/langgraph-api.pdf",
            },
        ]
    finally:
        if env_path.exists():
            env_path.unlink()


def test_openai_web_search_normalizes_nonbreaking_hyphen_in_query() -> None:
    """Web search should normalize U+2011 in user queries before sending the request."""

    class FakeResponsesAPIResponse:
        def __enter__(self) -> FakeResponsesAPIResponse:
            return self

        def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
            return None

        def read(self) -> bytes:
            payload = {
                "output": [
                    {
                        "type": "message",
                        "content": [
                            {
                                "type": "output_text",
                                "text": "https://example.test/langgraph-manual.pdf",
                            }
                        ],
                    }
                ]
            }
            return json.dumps(payload).encode("utf-8")

    class FakeProbeResponse:
        status = 200

        def __enter__(self) -> FakeProbeResponse:
            return self

        def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
            return None

        @property
        def headers(self) -> dict[str, str]:
            return {"Content-Type": "application/pdf"}

        def read(self) -> bytes:
            return b""

    env_path = _test_artifacts_dir() / "test-openai-web-search-hyphen.env"
    env_path.write_text("OPENAI_API_KEY=test-key\n", encoding="utf-8")
    tool_registry = ToolRegistry()
    captured: dict[str, Any] = {}

    def fake_urlopen(req: Any, timeout: float) -> Any:
        if req.full_url.endswith("/responses"):
            captured["body"] = json.loads(req.data.decode("utf-8"))
            return FakeResponsesAPIResponse()
        return FakeProbeResponse()

    try:
        with patch(
            "agentgraph.openai_adapter.request.urlopen",
            side_effect=fake_urlopen,
        ):
            register_openai_research_tool(
                tool_registry,
                env_path=env_path,
                override_env=True,
                tool_ref="web.search",
            )
            result = tool_registry.invoke(
                "web.search",
                {
                    "query": "скачай первый PDF‑результат",
                    "shared_context": {},
                    "task": {
                        "metadata": {
                            "search_constraints": {
                                "prefer_file_type": "pdf",
                                "direct_links_only": True,
                            }
                        }
                    },
                },
            )

        assert "PDF-результат" in captured["body"]["input"]
        assert result.success is True
    finally:
        if env_path.exists():
            env_path.unlink()


def test_openai_web_search_falls_back_when_json_results_are_empty() -> None:
    """Web search should not propagate an empty results list into an empty summary."""

    class FakeResponsesAPIResponse:
        def __enter__(self) -> FakeResponsesAPIResponse:
            return self

        def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
            return None

        def read(self) -> bytes:
            payload = {
                "output": [
                    {
                        "type": "web_search_call",
                        "action": {
                            "type": "search",
                            "sources": [
                                {
                                    "type": "url",
                                    "url": "https://example.test/flights/almaty-moscow",
                                }
                            ],
                        },
                    },
                    {
                        "type": "message",
                        "content": [
                            {
                                "type": "output_text",
                                "text": json.dumps({"results": []}),
                            }
                        ],
                    },
                ]
            }
            return json.dumps(payload).encode("utf-8")

    env_path = _test_artifacts_dir() / "test-openai-web-search-empty-results.env"
    env_path.write_text("OPENAI_API_KEY=test-key\n", encoding="utf-8")
    tool_registry = ToolRegistry()

    try:
        with patch(
            "agentgraph.openai_adapter.request.urlopen",
            side_effect=lambda req, timeout: FakeResponsesAPIResponse(),
        ):
            register_openai_research_tool(
                tool_registry,
                env_path=env_path,
                override_env=True,
                tool_ref="web.search",
            )
            result = tool_registry.invoke(
                "web.search",
                {
                    "query": "find cheapest flights Almaty Moscow next week",
                    "shared_context": {},
                    "task": {"metadata": {}},
                },
            )

        assert result.success is True
        assert result.data["results"] == [
            {
                "title": "almaty-moscow",
                "snippet": '{"results": []}',
                "url": "https://example.test/flights/almaty-moscow",
            }
        ]
    finally:
        if env_path.exists():
            env_path.unlink()


def test_build_starter_agent_configs_returns_expected_roles() -> None:
    """Starter profiles expose the agreed baseline agent personas."""
    agents = build_starter_agent_configs()

    assert len(agents) == 5
    roles = {agent.agent_id: agent.role for agent in agents}
    assert roles == {
        "coordinator": "orchestration lead",
        "researcher": "evidence collector",
        "file_executor": "filesystem operator",
        "synthesizer": "answer composer",
        "memory_curator": "memory manager",
    }
    researcher = next(agent for agent in agents if agent.agent_id == "researcher")
    file_executor = next(agent for agent in agents if agent.agent_id == "file_executor")
    synthesizer = next(agent for agent in agents if agent.agent_id == "synthesizer")
    coordinator = next(agent for agent in agents if agent.agent_id == "coordinator")
    memory_curator = next(
        agent for agent in agents if agent.agent_id == "memory_curator"
    )
    assert researcher.tools[0].tool_ref == "research.search"
    assert "state.inspect" in [binding.tool_ref for binding in coordinator.tools]
    assert coordinator.memory_policy.graph_linking is True
    assert coordinator.memory_policy.versioning is True
    assert "fs.write_file" in [binding.tool_ref for binding in memory_curator.tools]
    assert researcher.memory_policy.graph_linking is True
    assert "fs.create_dir" in [binding.tool_ref for binding in file_executor.tools]
    assert synthesizer.memory_policy.auto_sync is True
    assert synthesizer.memory_policy.graph_linking is True


def test_russian_research_query_prefers_researcher() -> None:
    """Semantic lookup should route Russian research prompts to the researcher."""
    registry = AgentRegistry()
    for agent in build_starter_agent_configs():
        registry.register(agent)

    candidates = registry.lookup(
        "Собери краткое описание того, для чего используется LangGraph."
    )

    assert candidates
    assert candidates[0].agent_id == "researcher"
    assert (
        candidates[0].confidence
        >= registry.get("researcher").delegation_policy.confidence_threshold
    )


def test_default_agents_yaml_path_exists_and_loads() -> None:
    """The packaged YAML roster is available as a runtime-editable config source."""
    path = default_agents_yaml_path()
    agents = load_core_agent_configs(research_tool_ref="research.search")

    assert path.exists()
    assert path.suffix == ".yaml"
    assert agents[0].agent_id == "coordinator"
    assert agents[1].tools[0].tool_ref == "research.search"


def test_desktop_config_paths_exist() -> None:
    """Packaged desktop-assistant YAML resources should be discoverable."""
    assert default_desktop_executor_yaml_path().exists()
    assert default_desktop_tools_yaml_path().exists()
    assert default_desktop_policies_yaml_path().exists()
    assert default_desktop_workflow_yaml_path().exists()


def test_load_desktop_executor_config_returns_agent() -> None:
    """Desktop executor should load as a valid AgentConfig from YAML."""
    agent = load_desktop_executor_config()

    assert isinstance(agent, AgentConfig)
    assert agent.agent_id == "desktop_executor"
    assert "fs.create_dir" in [binding.tool_ref for binding in agent.tools]
    assert any(
        capability.name == "Работа с файлами и папками"
        for capability in agent.capabilities
    )


def test_register_desktop_executor_supports_russian_lookup() -> None:
    """Registering the desktop executor should make it discoverable for desktop intents."""
    registry = AgentRegistry()
    agent = register_desktop_executor(registry)
    candidates = registry.lookup("Создай папку REPORTS на рабочем столе")

    assert agent.agent_id == "desktop_executor"
    assert candidates
    assert candidates[0].agent_id == "desktop_executor"


def test_load_desktop_tools_catalog_returns_idempotency_metadata() -> None:
    """Desktop tool metadata should expose idempotency and rollback config."""
    catalog = load_desktop_tools_catalog()

    assert isinstance(catalog, DesktopToolsCatalog)
    assert catalog.tools["fs.create_dir"].is_idempotent is True
    assert catalog.tools["fs.move"].is_idempotent is False
    assert catalog.tools["web.download"].rollback_strategy == "delete_quarantined_file"


def test_load_desktop_policy_bundle_returns_os_and_trust_rules() -> None:
    """Desktop policies should load OS-aware path normalization and trust lifecycle."""
    bundle = load_desktop_policy_bundle()

    assert isinstance(bundle, DesktopPolicyBundle)
    assert bundle.safety_policy.path_and_app_resolution["normalize_paths"] is True
    assert "windows" in bundle.safety_policy.path_and_app_resolution["os_mapping"]
    assert "fs_create_dir_exists_and_stat" in bundle.verification_policy
    assert bundle.trust_lifecycle.initial_value == 0.5
    assert any(rule == "never_exceeds_1.0" for rule in bundle.trust_lifecycle.cap_rules)


def test_load_desktop_workflow_bundle_returns_matrix_and_predicates() -> None:
    """Desktop workflow YAML should expose predicates and intent-tool matrix."""
    bundle = load_desktop_workflow_bundle()

    assert isinstance(bundle, DesktopWorkflowBundle)
    assert bundle.workflow.name == "AutonomousDesktopLoop"
    assert "coordinator_to_desktop_executor" in bundle.workflow.routing_predicates
    assert any(
        row.tool == "package.install"
        for row in bundle.workflow.intent_tool_policy_matrix
    )


def test_desktop_enforced_registry_blocks_path_escape() -> None:
    """Path guard should block escape paths before tool execution."""
    registry = DesktopEnforcedToolRegistry(
        tools_catalog=load_desktop_tools_catalog(),
        policy_bundle=load_desktop_policy_bundle(),
    )
    register_autonomy_toolkit(registry, workspace_root=_test_artifacts_dir())

    result = registry.invoke(
        "fs.create_dir",
        {
            "thread_id": "guard-thread",
            "path": "../../Windows/System32",
            "task": {"metadata": {"allowed_paths": ["~/Desktop"]}},
        },
    )

    assert result.success is False
    assert result.error_type == "path_escape_blocked"
    events = registry.pop_events("guard-thread")
    assert events[-1]["phase"] == "path_guard"


def test_rollback_manager_restores_move_target() -> None:
    """Rollback manager should restore moved files when asked to undo `fs.move`."""
    workspace = _test_artifacts_dir() / "rollback-fixtures"
    workspace.mkdir(parents=True, exist_ok=True)
    source_path = workspace / f"source-{uuid4().hex[:8]}.txt"
    target_path = workspace / f"target-{uuid4().hex[:8]}.txt"
    target_path.write_text("rollback", encoding="utf-8")

    manager = RollbackManager(policy_bundle=load_desktop_policy_bundle())
    rollback = manager.apply(
        tool_ref="fs.move",
        result=ToolResult(
            success=True,
            data={
                "source_path": str(source_path),
                "target_path": str(target_path),
            },
        ),
    )

    assert rollback["applied"] is True
    assert source_path.exists() is True
    assert target_path.exists() is False


def test_apply_desktop_enforcement_updates_appends_audit_and_trust() -> None:
    """Post-run enforcement hook should write audit entries and update trust score."""

    class FakeStateSnapshot:
        def __init__(self, values: dict[str, Any]) -> None:
            self.values = values

    class FakeRegistry:
        def __init__(self) -> None:
            self.events = [
                {
                    "event": "desktop_enforcement",
                    "tool_ref": "web.download",
                    "phase": "verification",
                    "verification_passed": True,
                    "rollback_applied": False,
                    "risk_level": "medium",
                    "details": ["verification_passed"],
                    "rollback": {},
                }
            ]

        def pop_events(self, thread_id: str) -> list[dict[str, Any]]:
            _ = thread_id
            events, self.events = self.events, []
            return events

    class FakeGraph:
        def __init__(self, state: AgentState) -> None:
            self._state = state
            self._desktop_tool_registry = FakeRegistry()

        def get_state(self, config: dict[str, Any]) -> FakeStateSnapshot:
            _ = config
            return FakeStateSnapshot(self._state.model_dump(by_alias=True))

        def update_state(
            self,
            config: dict[str, Any],
            values: dict[str, Any],
            as_node: str | None = None,
        ) -> dict[str, Any]:
            _ = config
            _ = as_node
            payload = self._state.model_dump(by_alias=True)
            payload.update(values)
            self._state = AgentState.model_validate(payload)
            return config

    state = AgentState(
        thread_id="trust-thread",
        task=TaskSpec(
            task_id="trust-task",
            description="desktop trust",
            output_schema="DesktopActionReport",
            metadata={"locale": "ru"},
        ),
        shared_context={
            "desktop_context": {
                "current_path": ".",
                "installed_packages": [],
                "trust_score": 0.5,
                "last_actions": [],
            }
        },
    )
    graph = FakeGraph(state)

    updated = apply_desktop_enforcement_updates(graph, thread_id="trust-thread")

    assert updated.shared_context["desktop_context"]["trust_score"] == 0.6
    assert updated.audit_log[-1]["event"] == "desktop_enforcement"


def test_apply_desktop_enforcement_updates_maps_artifact_domain() -> None:
    """Post-run enforcement hook should remap artifact type/domain from tool_ref."""

    class FakeStateSnapshot:
        def __init__(self, values: dict[str, Any]) -> None:
            self.values = values

    class FakeRegistry:
        def __init__(self) -> None:
            self.events: list[dict[str, Any]] = []
            self.tools_catalog = load_desktop_tools_catalog()
            self.policy_bundle = load_desktop_policy_bundle()

        def pop_events(self, thread_id: str) -> list[dict[str, Any]]:
            _ = thread_id
            return []

    class FakeGraph:
        def __init__(self, state: AgentState) -> None:
            self._state = state
            self._desktop_tool_registry = FakeRegistry()

        def get_state(self, config: dict[str, Any]) -> FakeStateSnapshot:
            _ = config
            return FakeStateSnapshot(self._state.model_dump(by_alias=True))

        def update_state(
            self,
            config: dict[str, Any],
            values: dict[str, Any],
            as_node: str | None = None,
        ) -> dict[str, Any]:
            _ = config
            _ = as_node
            payload = self._state.model_dump(by_alias=True)
            payload.update(values)
            self._state = AgentState.model_validate(payload)
            return config

    state = AgentState(
        thread_id="artifact-thread",
        task=TaskSpec(
            task_id="artifact-task",
            description="создай папку проекты",
            output_schema="FileSystemActionReport",
            metadata={"locale": "ru"},
        ),
        artifacts=[
            {
                "artifact_type": "research",
                "tool_ref": "fs.create_dir",
                "data": {"path": "x"},
            }
        ],
        output_candidate={
            "summary": "Папка создана",
            "citations": [],
            "confidence": 0.8,
        },
    )
    graph = FakeGraph(state)

    updated = apply_desktop_enforcement_updates(graph, thread_id="artifact-thread")

    assert updated.artifacts[0]["artifact_type"] == "filesystem_action"
    assert updated.artifacts[0]["artifact_domain"] == "filesystem_action"
    assert updated.output_candidate["domain_tag"] == "filesystem_action"


def test_build_desktop_demo_crew_returns_bundle_with_stubbed_tools() -> None:
    """Desktop bootstrap should return a bundle with runtime metadata and stubs."""
    bundle = build_desktop_demo_crew(
        env_file=str(_test_artifacts_dir() / "missing-openai.env"),
        workspace_root=_test_artifacts_dir(),
        include_starter_agents=False,
    )

    assert isinstance(bundle, DesktopCrewBundle)
    assert bundle.desktop_executor.agent_id == "desktop_executor"
    assert "fs.create_dir" in bundle.registered_tool_refs
    assert "package.install" in bundle.registered_tool_refs
    assert "package.update" in bundle.registered_tool_refs
    assert "app.launch" in bundle.registered_tool_refs
    assert "web.download" in bundle.registered_tool_refs
    assert "web.search" in bundle.registered_tool_refs
    assert "package.install" not in bundle.stubbed_tool_refs
    assert "package.update" not in bundle.stubbed_tool_refs
    assert "app.launch" not in bundle.stubbed_tool_refs
    assert "web.download" not in bundle.stubbed_tool_refs


def test_build_desktop_task_assigns_desktop_executor() -> None:
    """Desktop task builder should map intents to desktop_executor tools."""
    task = build_desktop_task(
        query="Создай папку REPORTS на рабочем столе",
        allowed_paths=["~/Desktop"],
    )

    assert task.assignee == "desktop_executor"
    assert task.metadata["tool_ref"] == "fs.create_dir"
    assert task.metadata["allowed_paths"] == ["~/Desktop"]
    assert task.output_schema == "FileSystemActionReport"


def test_build_desktop_task_maps_move_intent() -> None:
    """Desktop task builder should capture move source and target metadata."""
    task = build_desktop_task(query="move file report.txt to ARCHIVE", locale="en")

    assert task.assignee == "desktop_executor"
    assert task.metadata["tool_ref"] == "fs.move"
    assert task.metadata["source_path"] == "report.txt"
    assert task.metadata["target_path"] == "ARCHIVE"
    assert task.metadata["required_terms"] == ["moved"]


def test_build_desktop_task_maps_delete_intent() -> None:
    """Desktop task builder should capture delete target metadata."""
    task = build_desktop_task(query="delete file report.txt", locale="en")

    assert task.assignee == "desktop_executor"
    assert task.metadata["tool_ref"] == "fs.delete"
    assert task.metadata["target_path"] == "report.txt"
    assert task.metadata["required_terms"] == ["deleted"]


def test_run_desktop_task_seeds_desktop_context_and_resumes_hitl() -> None:
    """Desktop runner should seed shared_context.desktop_context and complete after approval."""
    bundle, graph = build_desktop_demo_graph(
        env_file=str(_test_artifacts_dir() / "missing-openai.env"),
        workspace_root=_test_artifacts_dir(),
        include_starter_agents=False,
    )
    task = build_desktop_task(query="create folder REPORTS on desktop", locale="en")

    state, _ = run_desktop_task(
        graph,
        task=task,
        thread_id="desktop-create-dir-thread",
        desktop_context={
            "current_path": "~/Desktop",
            "installed_packages": [],
            "trust_score": 0.5,
            "last_actions": [],
        },
    )
    resumed_state, _ = resume_desktop_task(
        graph,
        thread_id=state.thread_id,
        human_feedback={"decision": "approve"},
    )

    assert bundle.desktop_executor.agent_id == "desktop_executor"
    assert state.shared_context["desktop_context"]["current_path"] == "~/Desktop"
    assert state.status == ThreadStatus.HITL_WAIT
    assert resumed_state.status == ThreadStatus.COMPLETED


def test_export_desktop_artifact_exports_last_sandbox_result() -> None:
    """Desktop demo helper exports the last sandbox artifact into an approved path."""
    bundle, graph = build_desktop_demo_graph(
        env_file=str(_test_artifacts_dir() / "missing-openai.env"),
        workspace_root=_test_artifacts_dir(),
        include_starter_agents=False,
    )
    thread_id = f"desktop-export-{uuid4().hex[:8]}"
    export_root = _test_artifacts_dir() / f"desktop-export-target-{thread_id}"
    export_root.mkdir(parents=True, exist_ok=True)

    create_result = bundle.crew.tool_registry.invoke(
        "fs.create_dir",
        {"path": f"{thread_id}/Desktop/Проекты"},
    )
    state = AgentState(
        thread_id=thread_id,
        task=TaskSpec(
            task_id="desktop-export-task",
            description="Создай папку Проекты на рабочем столе",
            output_schema="FileSystemActionReport",
            assignee="desktop_executor",
            auto_route=False,
            metadata={
                "tool_ref": "fs.create_dir",
                "target_path": "{thread_id}/Desktop/Проекты",
                "requested_path": str(export_root / "Проекты"),
                "allowed_paths": [str(export_root)],
                "locale": "ru",
            },
        ),
    )

    export_result = export_desktop_artifact(
        graph,
        state=state,
        destination_path=str(export_root / "Ïðîåêòû"),
    )

    assert create_result.success is True
    assert export_result.success is True
    assert Path(export_result.data["exported_path"]).exists() is True
    assert Path(export_result.data["exported_path"]).is_dir() is True


def test_export_desktop_artifact_prefers_nested_target_path_when_requested_path_is_stale() -> (
    None
):
    """Nested exports should be rebuilt from sandbox target_path when requested_path is stale."""
    bundle, graph = build_desktop_demo_graph(
        env_file=str(_test_artifacts_dir() / "missing-openai.env"),
        workspace_root=_test_artifacts_dir(),
        include_starter_agents=False,
    )
    export_root = _test_artifacts_dir() / "desktop-export-nested"
    export_root.mkdir(parents=True, exist_ok=True)
    thread_id = f"desktop-nested-export-{uuid4().hex[:8]}"

    create_result = bundle.crew.tool_registry.invoke(
        "fs.create_dir",
        {
            "path": f"{thread_id}/Desktop/Projects/AI",
        },
    )
    state = AgentState(
        thread_id=thread_id,
        task=TaskSpec(
            task_id="desktop-export-stale-requested-path",
            description="Create nested AI folder inside Projects",
            output_schema="FileSystemActionReport",
            assignee="desktop_executor",
            auto_route=False,
            metadata={
                "tool_ref": "fs.create_dir",
                "target_path": "{thread_id}/Desktop/Projects/AI",
                "requested_path": str(export_root / "AI"),
                "allowed_paths": [str(export_root)],
                "locale": "en",
            },
        ),
    )

    with patch.object(bundle.crew.tool_registry, "invoke") as invoke_mock:
        invoke_mock.return_value = ToolResult(
            success=True,
            data={"exported_path": str(export_root / "Projects" / "AI")},
        )
        export_result = export_desktop_artifact(graph, state=state)

    assert create_result.success is True
    assert export_result.success is True
    invoke_mock.assert_called_once()
    assert invoke_mock.call_args.args[0] == "fs.export"
    assert invoke_mock.call_args.args[1]["destination_path"] == "~/Desktop/Projects/AI"


def test_export_desktop_artifact_uses_download_artifact_path() -> None:
    """Download exports should use the produced sandbox file when no target_path exists."""
    bundle, graph = build_desktop_demo_graph(
        env_file=str(_test_artifacts_dir() / "missing-openai.env"),
        workspace_root=_test_artifacts_dir(),
        include_starter_agents=False,
    )
    thread_id = f"desktop-download-export-{uuid4().hex[:8]}"
    export_root = _test_artifacts_dir() / f"desktop-download-export-{thread_id}"
    export_root.mkdir(parents=True, exist_ok=True)

    source_file = (
        _test_artifacts_dir()
        / ".agentgraph_cache"
        / "workspace"
        / "quarantine"
        / f"{thread_id}-manual.pdf"
    )
    source_file.parent.mkdir(parents=True, exist_ok=True)
    source_file.write_bytes(b"%PDF-1.4\nLangGraph export helper\n")

    state = AgentState(
        thread_id=thread_id,
        task=TaskSpec(
            task_id="desktop-download-export",
            description="Download manual.pdf",
            output_schema="DesktopActionReport",
            assignee="desktop_executor",
            auto_route=False,
            metadata={
                "tool_ref": "web.download",
                "requested_path": str(export_root),
                "allowed_paths": [str(export_root)],
                "locale": "en",
            },
        ),
        artifacts=[
            {
                "artifact_type": "research",
                "agent_id": "desktop_executor",
                "tool_ref": "web.download",
                "data": {"path": str(source_file), "filename": "manual.pdf"},
            }
        ],
    )

    export_result = export_desktop_artifact(graph, state=state)

    assert export_result.success is True
    assert Path(export_result.data["exported_path"]).exists() is True
    assert Path(export_result.data["exported_path"]).name == "manual.pdf"


def test_run_desktop_plan_executes_multiple_folder_steps_sequentially() -> None:
    """Experimental desktop plan runner should execute numbered folder steps sequentially."""
    _, graph = build_desktop_demo_graph(
        env_file=str(_test_artifacts_dir() / "missing-openai.env"),
        workspace_root=_test_artifacts_dir(),
        include_starter_agents=False,
    )
    task = build_desktop_task(
        query=(
            "1. Create folder Projects on Desktop\n"
            "2. Create folder AI inside folder Projects"
        ),
        allowed_paths=["~/Desktop"],
        locale="en",
    )

    execution = run_desktop_plan(
        graph,
        task=task,
        desktop_context={
            "current_path": "~/Desktop",
            "installed_packages": [],
            "trust_score": 0.5,
            "last_actions": [],
        },
        auto_approve_tools=["fs.create_dir"],
        max_steps=5,
    )

    assert task.assignee == "planner"
    assert len(execution.step_states) == 2
    assert all(
        state.status == ThreadStatus.COMPLETED for state in execution.step_states
    )
    assert execution.stopped_reason is None
    assert (
        execution.step_states[0]
        .task.metadata["target_path"]
        .endswith("/Desktop/Projects")
    )
    assert (
        execution.step_states[1]
        .task.metadata["target_path"]
        .endswith("/Desktop/Projects/AI")
    )


def test_resume_desktop_plan_continues_remaining_steps_after_human_approval() -> None:
    """Resuming a paused desktop plan should continue later subtasks."""
    _, graph = build_desktop_demo_graph(
        env_file=str(_test_artifacts_dir() / "missing-openai.env"),
        workspace_root=_test_artifacts_dir(),
        include_starter_agents=False,
    )
    task = build_desktop_task(
        query=(
            "\u0421\u043e\u0437\u0434\u0430\u0439 \u043f\u0430\u043f\u043a\u0443 "
            "\u041f\u0440\u043e\u0435\u043a\u0442\u044b \u043d\u0430 \u0440\u0430"
            "\u0431\u043e\u0447\u0435\u043c \u0441\u0442\u043e\u043b\u0435\n"
            "\u0421\u043e\u0437\u0434\u0430\u0439 \u043f\u0430\u043f\u043a\u0443 "
            "\u0418\u0418 \u043d\u0430 \u0440\u0430\u0431\u043e\u0447\u0435\u043c "
            "\u0441\u0442\u043e\u043b\u0435"
        ),
        allowed_paths=["~/Desktop"],
    )

    execution = run_desktop_plan(
        graph,
        task=task,
        desktop_context={
            "current_path": "~/Desktop",
            "installed_packages": [],
            "trust_score": 0.5,
            "last_actions": [],
        },
        auto_approve_tools=[],
        max_steps=5,
        stream=False,
    )

    assert len(execution.step_states) == 1
    assert execution.step_states[0].status == ThreadStatus.HITL_WAIT

    resumed = resume_desktop_plan(
        graph,
        task=task,
        paused_state=execution.step_states[0],
        human_feedback={"decision": "approve"},
        desktop_context={
            "current_path": "~/Desktop",
            "installed_packages": [],
            "trust_score": 0.5,
            "last_actions": [],
        },
        root_thread_id=execution.root_thread_id,
        auto_approve_tools=["fs.create_dir"],
        prior_step_states=execution.step_states,
        prior_step_events=execution.step_events,
        max_steps=5,
        stream=False,
    )

    assert len(resumed.step_states) == 2
    assert all(state.status == ThreadStatus.COMPLETED for state in resumed.step_states)
    assert resumed.stopped_reason is None


def test_build_desktop_task_maps_download_intent() -> None:
    """Desktop task builder should capture download URL and target filename."""
    task = build_desktop_task(
        query="download https://example.test/manual.pdf", locale="en"
    )

    assert task.assignee == "desktop_executor"
    assert task.metadata["tool_ref"] == "web.download"
    assert task.metadata["url"] == "https://example.test/manual.pdf"
    assert task.metadata["target_filename"] == "manual.pdf"
    assert task.metadata["required_terms"] == ["downloaded"]


def test_build_desktop_task_maps_package_install_intent() -> None:
    """Desktop task builder should capture package install manager and package name."""
    task = build_desktop_task(query="install requests with pip", locale="en")

    assert task.assignee == "desktop_executor"
    assert task.metadata["tool_ref"] == "package.install"
    assert task.metadata["manager"] == "pip"
    assert task.metadata["package_name"] == "requests"
    assert task.metadata["required_terms"] == ["prepared"]


def test_build_desktop_task_maps_app_launch_intent() -> None:
    """Desktop task builder should capture launch target for desktop app control."""
    task = build_desktop_task(query="open terminal", locale="en")

    assert task.assignee == "desktop_executor"
    assert task.metadata["tool_ref"] == "app.launch"
    assert task.metadata["app_name"] == "terminal"
    assert task.metadata["required_terms"] == ["launch"]


def test_build_desktop_task_routes_multistep_request_to_planner() -> None:
    """Multistep desktop tasks should route through planner with subtasks."""
    task = build_desktop_task(
        query="Создай папку Проекты на рабочем столе и затем скачай manual.pdf"
    )

    assert task.assignee == "planner"
    assert task.metadata["enable_planner"] is True
    assert len(task.metadata["subtasks"]) == 2
    assert task.metadata["subtasks"][0]["metadata"]["tool_ref"] == "fs.create_dir"
    assert task.metadata["subtasks"][1]["metadata"]["tool_ref"] == "web.download"


def test_build_desktop_task_routes_multiline_search_download_request_to_planner() -> (
    None
):
    """Multiline search and download requests should preserve both desktop steps."""
    task = build_desktop_task(
        query=(
            "\u043d\u0430\u0439\u0434\u0438 \u0434\u043e\u043a\u0443\u043c\u0435"
            "\u043d\u0442\u0430\u0446\u0438\u044e \u043f\u043e LangGraph\n"
            "\u0441\u043a\u0430\u0447\u0430\u0439 \u043f\u0435\u0440\u0432\u044b"
            "\u0439 PDF-\u0440\u0435\u0437\u0443\u043b\u044c\u0442\u0430\u0442 "
            "\u0432 \u043f\u0430\u043f\u043a\u0443 \u0418\u0418 \u0432 \u0434"
            "\u043e\u043a\u0443\u043c\u0435\u043d\u0442\u044b"
        )
    )

    assert task.assignee == "planner"
    assert len(task.metadata["subtasks"]) == 2
    assert task.metadata["subtasks"][0]["metadata"]["tool_ref"] == "web.search"
    assert task.metadata["subtasks"][0]["metadata"]["search_constraints"] == {
        "prefer_file_type": "pdf",
        "direct_links_only": True,
    }
    assert "PDF" in task.metadata["subtasks"][0]["description"]
    assert task.metadata["subtasks"][1]["metadata"]["tool_ref"] == "web.download"
    assert task.metadata["subtasks"][1]["metadata"]["requested_path"].endswith(
        "/Documents/\u0418\u0418"
    )


def test_build_desktop_task_keeps_download_intent_when_destination_folder_is_present() -> (
    None
):
    """Download commands that mention a destination folder should not collapse into create-dir."""
    task = build_desktop_task(
        query=(
            "\u0441\u043a\u0430\u0447\u0430\u0439 manual.pdf "
            "\u0432 \u043f\u0430\u043f\u043a\u0443 \u0418\u0418"
        )
    )

    assert task.assignee == "desktop_executor"
    assert task.metadata["tool_ref"] == "web.download"
    assert task.metadata["target_filename"] == "manual.pdf"


def test_build_desktop_task_infers_requested_desktop_path() -> None:
    """Desktop create-dir intent should preserve requested desktop path and sandbox path."""
    task = build_desktop_task(
        query="Создай папку Проекты на рабочем столе",
        allowed_paths=["~/Desktop"],
    )

    assert task.assignee == "desktop_executor"
    assert task.metadata["requested_path"] == "~/Desktop/Проекты"
    assert task.metadata["target_path"] == "{thread_id}/Desktop/Проекты"
    assert task.metadata["required_terms"] == ["создан"]
    assert (
        task.metadata["dry_run_preview"]
        == ".agentgraph_cache/workspace/{thread_id}/Desktop/Проекты"
    )


def test_build_desktop_task_infers_nested_desktop_path() -> None:
    """Nested folder commands should preserve parent and child folder names."""
    task = build_desktop_task(
        query="внутри папки Проекты создай папку ИИ",
        allowed_paths=["~/Desktop"],
    )

    assert task.assignee == "desktop_executor"
    assert task.metadata["requested_path"] == "~/Desktop/Проекты/ИИ"
    assert task.metadata["target_path"] == "{thread_id}/Desktop/Проекты/ИИ"


def test_build_desktop_task_infers_nested_desktop_path_reversed_order() -> None:
    """Nested folder parsing should also work when child action comes first."""
    task = build_desktop_task(
        query="создай папку ИИ внутри папки Проекты",
        allowed_paths=["~/Desktop"],
    )

    assert task.assignee == "desktop_executor"
    assert task.metadata["requested_path"] == "~/Desktop/Проекты/ИИ"
    assert task.metadata["target_path"] == "{thread_id}/Desktop/Проекты/ИИ"


def test_register_autonomy_toolkit_supports_fs_move() -> None:
    """Autonomy toolkit should move files inside the workspace sandbox."""
    tool_registry = ToolRegistry()
    workspace_root = _test_artifacts_dir()
    register_autonomy_toolkit(tool_registry, workspace_root=workspace_root)
    thread_id = f"move-thread-{uuid4().hex[:8]}"

    write_result = tool_registry.invoke(
        "fs.write_file",
        {
            "path": f"{thread_id}/source/report.txt",
            "content": "desktop-move-test",
        },
    )
    move_result = tool_registry.invoke(
        "fs.move",
        {
            "source_path": f"{thread_id}/source/report.txt",
            "target_path": f"{thread_id}/archive/report.txt",
        },
    )

    source_path = (
        _test_artifacts_dir()
        / ".agentgraph_cache"
        / "workspace"
        / thread_id
        / "source"
        / "report.txt"
    )
    target_path = (
        _test_artifacts_dir()
        / ".agentgraph_cache"
        / "workspace"
        / thread_id
        / "archive"
        / "report.txt"
    )

    assert write_result.success is True
    assert move_result.success is True
    assert source_path.exists() is False
    assert target_path.exists() is True


def test_register_autonomy_toolkit_localizes_fs_create_dir_for_russian_locale() -> None:
    """Filesystem actions should emit Russian snippets when locale is Russian."""
    tool_registry = ToolRegistry()
    workspace_root = _test_artifacts_dir()
    register_autonomy_toolkit(tool_registry, workspace_root=workspace_root)

    result = tool_registry.invoke(
        "fs.create_dir",
        {
            "task": {
                "metadata": {
                    "locale": "ru",
                    "target_path": f"{uuid4().hex[:8]}/Desktop/Проекты",
                }
            }
        },
    )

    assert result.success is True
    assert "Папка создана" in result.data["results"][0]["snippet"]


def test_register_autonomy_toolkit_supports_fs_delete_via_quarantine() -> None:
    """Autonomy toolkit should quarantine deleted files instead of removing them outright."""
    tool_registry = ToolRegistry()
    workspace_root = _test_artifacts_dir()
    register_autonomy_toolkit(tool_registry, workspace_root=workspace_root)
    thread_id = f"delete-thread-{uuid4().hex[:8]}"

    write_result = tool_registry.invoke(
        "fs.write_file",
        {
            "path": f"{thread_id}/source/report.txt",
            "content": "desktop-delete-test",
        },
    )
    delete_result = tool_registry.invoke(
        "fs.delete",
        {
            "path": f"{thread_id}/source/report.txt",
        },
    )

    source_path = (
        _test_artifacts_dir()
        / ".agentgraph_cache"
        / "workspace"
        / thread_id
        / "source"
        / "report.txt"
    )

    assert write_result.success is True
    assert delete_result.success is True
    assert delete_result.data["quarantined"] is True
    assert source_path.exists() is False
    assert Path(delete_result.data["quarantine_path"]).exists() is True


def test_register_autonomy_toolkit_supports_fs_export() -> None:
    """Sandbox directories can be exported into an approved user path."""
    tool_registry = ToolRegistry()
    workspace_root = _test_artifacts_dir()
    register_autonomy_toolkit(tool_registry, workspace_root=workspace_root)
    thread_id = f"export-thread-{uuid4().hex[:8]}"
    export_root = _test_artifacts_dir() / f"approved-export-{thread_id}"
    export_root.mkdir(parents=True, exist_ok=True)

    create_result = tool_registry.invoke(
        "fs.create_dir",
        {
            "path": f"{thread_id}/Desktop/Проекты",
        },
    )
    export_result = tool_registry.invoke(
        "fs.export",
        {
            "source_path": f"{thread_id}/Desktop/Проекты",
            "destination_path": str(export_root / "Проекты"),
            "allowed_paths": [str(export_root)],
        },
    )

    assert create_result.success is True
    assert export_result.success is True
    assert Path(export_result.data["exported_path"]).exists() is True
    assert Path(export_result.data["exported_path"]).is_dir() is True


def test_register_autonomy_toolkit_supports_fs_export_into_existing_directory_for_files() -> (
    None
):
    """File exports should land inside an approved existing directory."""
    tool_registry = ToolRegistry()
    workspace_root = _test_artifacts_dir()
    register_autonomy_toolkit(tool_registry, workspace_root=workspace_root)
    thread_id = f"export-file-thread-{uuid4().hex[:8]}"
    export_root = _test_artifacts_dir() / f"approved-export-file-{thread_id}"
    export_root.mkdir(parents=True, exist_ok=True)

    write_result = tool_registry.invoke(
        "fs.write_file",
        {
            "path": f"{thread_id}/Downloads/manual.pdf",
            "content": "%PDF-1.4\nAgentGraph export file test\n",
        },
    )
    export_result = tool_registry.invoke(
        "fs.export",
        {
            "source_path": f"{thread_id}/Downloads/manual.pdf",
            "destination_path": str(export_root),
            "allowed_paths": [str(export_root)],
        },
    )

    assert write_result.success is True
    assert export_result.success is True
    assert Path(export_result.data["exported_path"]).exists() is True
    assert Path(export_result.data["exported_path"]).name == "manual.pdf"


def test_register_autonomy_toolkit_supports_web_download_via_quarantine() -> None:
    """Autonomy toolkit should download files into quarantine via a local HTTP server."""
    serve_root = _test_artifacts_dir() / "http-fixtures"
    serve_root.mkdir(parents=True, exist_ok=True)
    payload_path = serve_root / "manual.pdf"
    payload_bytes = b"%PDF-1.4\nAgentGraph desktop download test\n"
    payload_path.write_bytes(payload_bytes)

    handler = partial(SimpleHTTPRequestHandler, directory=str(serve_root))
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()

    try:
        tool_registry = ToolRegistry()
        workspace_root = _test_artifacts_dir()
        register_autonomy_toolkit(tool_registry, workspace_root=workspace_root)

        result = tool_registry.invoke(
            "web.download",
            {
                "url": f"http://127.0.0.1:{server.server_port}/manual.pdf",
                "target_filename": "manual.pdf",
            },
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

    assert result.success is True
    assert result.data["quarantined"] is True
    assert result.data["bytes_downloaded"] == len(payload_bytes)
    quarantined_path = Path(result.data["path"])
    assert quarantined_path.exists() is True
    assert quarantined_path.read_bytes() == payload_bytes


def test_register_autonomy_toolkit_web_download_uses_previous_pdf_result_url() -> None:
    """Download requests can reuse the first PDF URL from prior search context."""

    class FakeHTTPResponse:
        status = 200

        def __enter__(self) -> FakeHTTPResponse:
            return self

        def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
            return None

        @property
        def headers(self) -> dict[str, str]:
            return {"Content-Type": "application/pdf"}

        def read(self) -> bytes:
            return b"%PDF-1.4\nLangGraph\n"

    tool_registry = ToolRegistry()
    workspace_root = _test_artifacts_dir()
    register_autonomy_toolkit(tool_registry, workspace_root=workspace_root)

    with patch(
        "agentgraph.autonomy_tools.request.urlopen",
        side_effect=lambda req, timeout: FakeHTTPResponse(),
    ):
        result = tool_registry.invoke(
            "web.download",
            {
                "query": (
                    "\u0441\u043a\u0430\u0447\u0430\u0439 \u043f\u0435\u0440\u0432"
                    "\u044b\u0439 PDF-\u0440\u0435\u0437\u0443\u043b\u044c\u0442"
                    "\u0430\u0442"
                ),
                "shared_context": {
                    "research_result": {
                        "results": [
                            {
                                "title": "LangGraph PDF",
                                "snippet": "Official documentation",
                                "url": "https://example.test/langgraph-manual.pdf",
                            }
                        ]
                    }
                },
            },
        )

    assert result.success is True
    assert result.data["url"] == "https://example.test/langgraph-manual.pdf"
    assert result.data["filename"] == "langgraph-manual.pdf"


def test_register_autonomy_toolkit_web_download_falls_back_after_404() -> None:
    """Download should try the next candidate URL when the first one returns 404."""

    class FakeHTTPResponse:
        status = 200

        def __enter__(self) -> FakeHTTPResponse:
            return self

        def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
            return None

        @property
        def headers(self) -> dict[str, str]:
            return {"Content-Type": "application/pdf"}

        def read(self) -> bytes:
            return b"%PDF-1.4\nLangGraph fallback\n"

    tool_registry = ToolRegistry()
    workspace_root = _test_artifacts_dir()
    register_autonomy_toolkit(tool_registry, workspace_root=workspace_root)

    def fake_urlopen(req: Any, timeout: float) -> FakeHTTPResponse:
        url = req.full_url
        if "missing.pdf" in url:
            raise urlerror.HTTPError(url, 404, "Not Found", hdrs=None, fp=None)
        return FakeHTTPResponse()

    with patch("agentgraph.autonomy_tools.request.urlopen", side_effect=fake_urlopen):
        result = tool_registry.invoke(
            "web.download",
            {
                "query": (
                    "\u0441\u043a\u0430\u0447\u0430\u0439 \u043f\u0435\u0440\u0432"
                    "\u044b\u0439 PDF-\u0440\u0435\u0437\u0443\u043b\u044c\u0442"
                    "\u0430\u0442"
                ),
                "shared_context": {
                    "research_result": {
                        "results": [
                            {
                                "title": "Broken PDF",
                                "snippet": "Missing file",
                                "url": "https://example.test/missing.pdf",
                            },
                            {
                                "title": "Working PDF",
                                "snippet": "Available file",
                                "url": "https://example.test/working.pdf",
                            },
                        ]
                    }
                },
            },
        )

    assert result.success is True
    assert result.data["url"] == "https://example.test/working.pdf"
    assert result.data["attempted_urls"] == [
        "https://example.test/missing.pdf",
        "https://example.test/working.pdf",
    ]


def test_run_desktop_task_does_not_retry_non_retriable_download_404() -> None:
    """Desktop runtime should surface a final 404 once instead of duplicating it via retry."""
    _, graph = build_desktop_demo_graph(
        env_file=str(_test_artifacts_dir() / "missing-openai.env"),
        workspace_root=_test_artifacts_dir(),
        include_starter_agents=False,
    )
    task = build_desktop_task(query="download https://example.test/missing.pdf")

    def fake_urlopen(req: Any, timeout: float) -> Any:
        url = req.full_url
        raise urlerror.HTTPError(url, 404, "Not Found", hdrs=None, fp=None)

    with patch("agentgraph.autonomy_tools.request.urlopen", side_effect=fake_urlopen):
        state, _ = run_desktop_task(
            graph,
            task=task,
            thread_id="desktop-download-404-thread",
            desktop_context={
                "current_path": "~/Desktop",
                "installed_packages": [],
                "trust_score": 0.5,
                "last_actions": [],
            },
        )
        resumed_state, _ = resume_desktop_task(
            graph,
            thread_id=state.thread_id,
            human_feedback={"decision": "approve"},
        )

    assert state.status == ThreadStatus.HITL_WAIT
    assert resumed_state.status == ThreadStatus.FAILED
    assert resumed_state.retry_counters["tool"] == 1
    assert len(resumed_state.errors) == 1
    assert resumed_state.errors[0]["tool_ref"] == "web.download"
    assert resumed_state.errors[0]["error_type"] == "http_error"
    assert resumed_state.errors[0]["metadata"]["status_code"] == 404
    assert resumed_state.errors[0]["metadata"]["retriable"] is False
    assert resumed_state.errors[0]["metadata"]["attempted_urls"] == [
        "https://example.test/missing.pdf"
    ]


def test_register_autonomy_toolkit_supports_package_install_dry_run() -> None:
    """Autonomy toolkit should expose a safe dry-run for package installation."""
    tool_registry = ToolRegistry()
    register_autonomy_toolkit(tool_registry, workspace_root=_test_artifacts_dir())

    result = tool_registry.invoke(
        "package.install",
        {
            "manager": "pip",
            "package_name": "pip",
        },
    )

    assert result.success is True
    assert result.data["dry_run"] is True
    assert result.data["applied"] is False
    assert result.data["manager"] == "pip"
    assert result.data["package_name"] == "pip"
    assert "pip install" in result.data["command_preview"]
    assert result.data["installed_before"]["installed"] is True


def test_register_autonomy_toolkit_supports_package_update_dry_run() -> None:
    """Autonomy toolkit should expose a safe dry-run for package updates."""
    tool_registry = ToolRegistry()
    register_autonomy_toolkit(tool_registry, workspace_root=_test_artifacts_dir())

    result = tool_registry.invoke(
        "package.update",
        {
            "manager": "pip",
            "package_name": "pip",
        },
    )

    assert result.success is True
    assert result.data["dry_run"] is True
    assert result.data["applied"] is False
    assert result.data["manager"] == "pip"
    assert result.data["package_name"] == "pip"
    assert "--upgrade" in result.data["command_preview"]


def test_register_autonomy_toolkit_applies_package_install_after_approval() -> None:
    """Approved package actions should switch from preview to real execution."""
    tool_registry = ToolRegistry()
    register_autonomy_toolkit(tool_registry, workspace_root=_test_artifacts_dir())

    with patch("agentgraph.autonomy_tools.subprocess.run") as run_mock:
        run_mock.return_value = subprocess.CompletedProcess(
            args=["python", "-m", "pip", "install", "pip"],
            returncode=0,
            stdout="installed",
            stderr="",
        )
        result = tool_registry.invoke(
            "package.install",
            {
                "manager": "pip",
                "package_name": "pip",
                "shared_context": {"approved_tools": ["package.install"]},
            },
        )

    assert result.success is True
    assert result.data["dry_run"] is False
    assert result.data["applied"] is True
    run_mock.assert_called_once()


def test_register_autonomy_toolkit_supports_app_launch_dry_run() -> None:
    """Autonomy toolkit should resolve an allowed app alias without launching it."""
    tool_registry = ToolRegistry()
    register_autonomy_toolkit(tool_registry, workspace_root=_test_artifacts_dir())

    result = tool_registry.invoke(
        "app.launch",
        {
            "app_name": "terminal",
        },
    )

    assert result.success is True
    assert result.data["dry_run"] is True
    assert result.data["launched"] is False
    assert result.data["app_name"] == "terminal"
    assert result.data["executable"]


def test_register_autonomy_toolkit_launches_app_after_approval() -> None:
    """Approved app launches should call the platform launcher."""
    tool_registry = ToolRegistry()
    register_autonomy_toolkit(tool_registry, workspace_root=_test_artifacts_dir())

    with patch("agentgraph.autonomy_tools.subprocess.Popen") as popen_mock:
        popen_mock.return_value.pid = 4321
        result = tool_registry.invoke(
            "app.launch",
            {
                "app_name": "terminal",
                "shared_context": {"approved_tools": ["app.launch"]},
            },
        )

    assert result.success is True
    assert result.data["dry_run"] is False
    assert result.data["launched"] is True
    assert result.data["pid"] == 4321
    popen_mock.assert_called_once()


def test_load_specialist_agent_config_resolves_placeholders() -> None:
    """Specialist profiles resolve runtime tool refs from the YAML template."""
    validator = load_specialist_agent_config(
        "validator",
        validator_tool_ref="validator.logic",
    )

    assert validator.agent_id == "validator"
    assert validator.tools[0].tool_ref == "validator.logic"


def test_register_starter_specialists_adds_planner_and_validator() -> None:
    """Starter runtime specialists are registered on top of the core personas."""
    tool_registry = ToolRegistry()
    tool_registry.register(
        tool_ref="research.search",
        description="Research tool.",
        side_effect_level=SideEffectLevel.READ_ONLY,
        handler=lambda args: ToolResult(success=True, data={"results": []}),
    )
    crew = Crew(
        name="starter-agents",
        agents=build_starter_agent_configs(),
        tool_registry=tool_registry,
        schema_registry={"ResearchReport": ResearchReport},
    )

    registered = register_starter_specialists(
        agent_registry=crew.agent_registry,
        tool_registry=crew.tool_registry,
    )

    assert {agent.agent_id for agent in registered} == {"planner", "validator"}
    assert crew.agent_registry.get("planner").role == "decomposition specialist"
    assert crew.agent_registry.get("validator").role == "quality gate"
    assert crew.agent_registry.get("validator").tools[0].tool_ref == "validator.logic"


def test_describe_agent_configs_returns_serializable_roster() -> None:
    """Starter-agent descriptions are stable and easy to inspect."""
    description = describe_agent_configs(build_starter_agent_configs())

    assert len(description) == 5
    assert description[0]["agent_id"] == "coordinator"
    assert "research.search" in description[1]["tools"]
    assert description[1]["capabilities"][0]["name"] == "Technical Research"


def test_register_autonomy_toolkit_supports_sandbox_file_tools() -> None:
    """Autonomy toolkit exposes safe filesystem tools inside the workspace sandbox."""
    tool_registry = ToolRegistry()
    workspace_root = _test_artifacts_dir()
    config = register_autonomy_toolkit(tool_registry, workspace_root=workspace_root)

    assert isinstance(config, AutonomyToolkitConfig)
    create_dir_result = tool_registry.invoke(
        "fs.create_dir",
        {"path": "demo-thread/MEMORY"},
    )
    write_result = tool_registry.invoke(
        "fs.write_file",
        {
            "path": "demo-thread/MEMORY/demo.txt",
            "content": "hello agentgraph",
        },
    )
    read_result = tool_registry.invoke(
        "fs.read_file",
        {"path": ".agentgraph_cache/workspace/demo-thread/MEMORY/demo.txt"},
    )
    list_result = tool_registry.invoke(
        "fs.list_dir", {"path": ".agentgraph_cache/workspace/demo-thread/MEMORY"}
    )

    assert create_dir_result.success is True
    assert write_result.success is True
    assert read_result.data["content"] == "hello agentgraph"
    assert any(entry["name"] == "demo.txt" for entry in list_result.data["entries"])


def test_register_autonomy_toolkit_supports_fetch_and_extract() -> None:
    """Autonomy toolkit can fetch and parse simple HTML responses."""

    class FakeHTTPResponse:
        status = 200

        def __init__(self, body: str) -> None:
            self.body = body
            self.headers = {"Content-Type": "text/html; charset=utf-8"}

        def __enter__(self) -> FakeHTTPResponse:
            return self

        def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
            return None

        def read(self) -> bytes:
            return self.body.encode("utf-8")

    tool_registry = ToolRegistry()
    register_autonomy_toolkit(tool_registry, workspace_root=_test_artifacts_dir())
    html = (
        "<html><head><title>Example</title></head>"
        "<body><a href='/docs'>Docs</a><p>AgentGraph demo page</p></body></html>"
    )

    def fake_urlopen(req: Any, timeout: float) -> FakeHTTPResponse:
        return FakeHTTPResponse(html)

    with patch("agentgraph.autonomy_tools.request.urlopen", side_effect=fake_urlopen):
        fetch_result = tool_registry.invoke("web.fetch", {"url": "https://example.com"})
        parse_result = tool_registry.invoke(
            "parse.extract",
            {"html": fetch_result.data["content"]},
        )

    assert fetch_result.success is True
    assert parse_result.data["title"] == "Example"
    assert "/docs" in parse_result.data["links"]


def test_build_demo_task_applies_ui_flags() -> None:
    """Demo task builder maps UI options into a stable task contract."""
    task = build_demo_task(
        query="Summarize LangGraph",
        enable_planner=True,
        require_citations=True,
        fact_logic_validation=FactLogicValidationMode.POLICY,
        hitl_points=[HITLPoint.BEFORE_TOOL_CALL, HITLPoint.ON_LOW_CONFIDENCE],
    )

    assert task.description == "Summarize LangGraph"
    assert task.metadata["enable_planner"] is True
    assert task.metadata["require_citations"] is True
    assert task.metadata["locale"] == "en"
    assert task.fact_logic_validation == FactLogicValidationMode.POLICY
    assert task.hitl_points == [
        HITLPoint.BEFORE_TOOL_CALL,
        HITLPoint.ON_LOW_CONFIDENCE,
    ]


def test_build_demo_task_sets_russian_locale_for_cyrillic_query() -> None:
    """Cyrillic prompts should default to a Russian output locale."""
    task = build_demo_task(query="Собери краткое описание LangGraph")

    assert task.metadata["locale"] == "ru"


def test_build_demo_task_normalizes_filesystem_intent() -> None:
    """File-system intents get routed to the sandboxed file executor with strict schema."""
    task = build_demo_task(query="Create folder named MEMORY")

    assert task.assignee == "file_executor"
    assert task.auto_route is False
    assert task.output_schema == "FileSystemActionReport"
    assert task.fact_logic_validation == FactLogicValidationMode.POLICY
    assert task.metadata["tool_ref"] == "fs.create_dir"
    assert task.metadata["target_path"] == "{thread_id}/MEMORY"
    assert task.metadata["require_citations"] is True
    assert "dry_run_preview" in task.metadata


def test_synthesizer_uses_title_when_search_results_have_no_snippet() -> None:
    """Research results without snippets should still produce a non-empty summary."""
    registry = ToolRegistry()

    def search_tool(args: dict[str, Any]) -> ToolResult:
        return ToolResult(
            success=True,
            data={
                "results": [
                    {
                        "title": "Cheap flights Almaty to Moscow",
                        "url": "https://example.test/flights",
                    }
                ]
            },
        )

    registry.register(
        tool_ref="research.search",
        description="Research knowledge lookup.",
        side_effect_level=SideEffectLevel.READ_ONLY,
        handler=search_tool,
    )
    crew = _build_crew(tool_registry=registry)
    registry.register(
        tool_ref="research.search",
        description="Research knowledge lookup.",
        side_effect_level=SideEffectLevel.READ_ONLY,
        handler=search_tool,
    )
    graph = crew.compile(checkpointer=InMemorySaver())
    task = TaskSpec(
        task_id="travel-search-title-only",
        description="Find the cheapest flights from Almaty to Moscow next week",
        output_schema="ResearchReport",
        auto_route=True,
        fact_logic_validation=FactLogicValidationMode.POLICY,
        metadata={"require_citations": True},
    )

    state, _ = run_task(graph, task=task, thread_id="travel-title-only-thread")

    assert state.status == ThreadStatus.COMPLETED
    assert state.output_candidate is not None
    assert state.output_candidate["summary"] == "Cheap flights Almaty to Moscow"
    assert state.output_candidate["citations"] == ["https://example.test/flights"]


def test_run_task_collects_stream_events_without_double_invocation() -> None:
    """The shared demo runner executes once and returns both state and envelopes."""
    crew = _build_crew()
    graph = crew.compile(checkpointer=InMemorySaver())
    task = build_demo_task(query="Summarize LangGraph")

    state, events = run_task(graph, task=task, thread_id="demo-runner-thread")

    assert state.status == ThreadStatus.COMPLETED
    assert state.output_candidate is not None
    assert events
    assert all(event["thread_id"] == "demo-runner-thread" for event in events)


def test_resume_task_handles_hitl_threads() -> None:
    """The shared demo resume helper advances a paused HITL thread."""
    crew = _build_crew(requires_hitl=True)
    graph = crew.compile(checkpointer=InMemorySaver())
    task = build_demo_task(
        query="Summarize LangGraph",
        hitl_points=[HITLPoint.BEFORE_TOOL_CALL],
    )

    state, _ = run_task(graph, task=task, thread_id="demo-hitl-thread")
    resumed_state, resume_events = resume_task(
        graph,
        thread_id=state.thread_id,
        human_feedback={"decision": "approve"},
    )

    assert state.status == ThreadStatus.HITL_WAIT
    assert resumed_state.status == ThreadStatus.COMPLETED
    assert resume_events


def test_filesystem_task_routes_to_file_executor_and_requires_hitl() -> None:
    """Filesystem actions use the dedicated executor, pause for approval, then complete."""
    tool_registry = ToolRegistry()
    workspace_root = _test_artifacts_dir()
    register_autonomy_toolkit(tool_registry, workspace_root=workspace_root)
    crew = Crew(
        name="filesystem-demo",
        agents=build_starter_agent_configs(),
        tool_registry=tool_registry,
        schema_registry={"FileSystemActionReport": FileSystemActionReport},
    )
    graph = crew.compile(checkpointer=InMemorySaver())
    task = build_demo_task(query="Create folder named MEMORY")

    state, _ = run_task(graph, task=task, thread_id="file-task-thread")
    resumed_state, _ = resume_task(
        graph,
        thread_id=state.thread_id,
        human_feedback={"decision": "approve"},
    )

    created_path = (
        _test_artifacts_dir()
        / ".agentgraph_cache"
        / "workspace"
        / "file-task-thread"
        / "MEMORY"
    )
    assert task.assignee == "file_executor"
    assert state.status == ThreadStatus.HITL_WAIT
    assert resumed_state.status == ThreadStatus.COMPLETED
    assert created_path.exists()
