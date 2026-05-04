"""DSL compiler, registries, runtime nodes, and stream utilities."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from pydantic import BaseModel, TypeAdapter, ValidationError

from agentgraph.contracts import (
    AgentConfig,
    AgentState,
    FactLogicValidationMode,
    HITLPoint,
    HumanCheckpoint,
    LookupCandidate,
    MessageProtocol,
    MessageType,
    SideEffectLevel,
    ThreadStatus,
    ToolBinding,
    ToolResult,
)

MAX_STATE_LIST_ITEMS = 50


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _enum_rank(level: SideEffectLevel) -> int:
    ranks = {
        SideEffectLevel.NONE: 0,
        SideEffectLevel.READ_ONLY: 1,
        SideEffectLevel.EXTERNAL_WRITE: 2,
        SideEffectLevel.DESTRUCTIVE: 3,
    }
    return ranks[level]


def _is_tool_result_retriable(result: ToolResult) -> bool:
    metadata = result.metadata or {}
    if "retriable" in metadata:
        return bool(metadata["retriable"])
    status_code = metadata.get("status_code")
    if isinstance(status_code, int):
        return status_code >= 500 or status_code in {408, 409, 425, 429}
    if result.error_type in {"input_error", "extension_blocked", "hash_mismatch"}:
        return False
    return True


def _tokens(value: str) -> set[str]:
    cleaned = "".join(ch.lower() if ch.isalnum() else " " for ch in value)
    return {token for token in cleaned.split() if token}


def _bounded_append(
    items: list[dict[str, Any]],
    new_item: dict[str, Any],
    *,
    bucket: str,
    storage: ExternalStorage,
    thread_id: str,
) -> list[dict[str, Any]]:
    updated = [*items, new_item]
    if len(updated) <= MAX_STATE_LIST_ITEMS:
        return updated
    overflow = updated[: len(updated) - MAX_STATE_LIST_ITEMS]
    ref = storage.store(bucket, {"thread_id": thread_id, "items": overflow})
    tail = updated[-MAX_STATE_LIST_ITEMS:]
    tail[0] = {
        **tail[0],
        "external_storage_ref": ref["ref_id"],
    }
    return tail


def _append_protocol_message(
    state: AgentState,
    protocol_message: MessageProtocol,
) -> dict[str, Any]:
    return {
        "messages": [protocol_message.to_graph_message()],
        "protocol_messages": [*state.protocol_messages, protocol_message],
    }


@dataclass
class ToolDefinition:
    """Executable tool registered by the DSL layer."""

    tool_ref: str
    description: str
    side_effect_level: SideEffectLevel
    handler: Callable[[dict[str, Any]], ToolResult]


class ToolRegistry:
    """Registry and invocation facade for all tools."""

    def __init__(self) -> None:
        self._tools: dict[str, ToolDefinition] = {}

    def register(
        self,
        *,
        tool_ref: str,
        description: str,
        side_effect_level: SideEffectLevel,
        handler: Callable[[dict[str, Any]], ToolResult],
    ) -> None:
        self._tools[tool_ref] = ToolDefinition(
            tool_ref=tool_ref,
            description=description,
            side_effect_level=side_effect_level,
            handler=handler,
        )

    def get(self, tool_ref: str) -> ToolDefinition:
        return self._tools[tool_ref]

    def invoke(self, tool_ref: str, args: dict[str, Any]) -> ToolResult:
        definition = self.get(tool_ref)
        return definition.handler(args)


class AgentRegistry:
    """Semantic registry used by the coordinator and validator routing."""

    def __init__(self) -> None:
        self._agents: dict[str, AgentConfig] = {}

    def register(self, agent: AgentConfig) -> None:
        self._agents[agent.agent_id] = agent

    def get(self, agent_id: str) -> AgentConfig:
        return self._agents[agent_id]

    def all(self) -> list[AgentConfig]:
        return list(self._agents.values())

    def lookup(
        self,
        query: str,
        *,
        validator_hint: str | None = None,
        top_k: int = 3,
    ) -> list[LookupCandidate]:
        candidates: list[LookupCandidate] = []
        query_tokens = _tokens(query)
        for agent in self._agents.values():
            if validator_hint and validator_hint not in {agent.role, agent.agent_id}:
                if all(
                    validator_hint not in descriptor.embedding_text
                    for descriptor in agent.capabilities
                ):
                    continue
            capability_tokens = set()
            for descriptor in agent.capabilities:
                capability_tokens |= _tokens(descriptor.embedding_text)
                capability_tokens |= {
                    keyword.lower() for keyword in descriptor.keywords
                }
                capability_tokens |= {domain.lower() for domain in descriptor.domains}
            if not capability_tokens:
                semantic_score = 0.0
            else:
                overlap = query_tokens & capability_tokens
                semantic_score = len(overlap) / len(query_tokens | capability_tokens)
            policy_weight = agent.delegation_policy.policy_weight
            availability_factor = agent.availability_factor
            confidence = semantic_score * policy_weight * availability_factor
            candidates.append(
                LookupCandidate(
                    agent_id=agent.agent_id,
                    semantic_score=semantic_score,
                    policy_weight=policy_weight,
                    availability_factor=availability_factor,
                    confidence=confidence,
                    rationale=f"semantic={semantic_score:.3f}",
                )
            )
        candidates.sort(key=lambda item: item.confidence, reverse=True)
        return candidates[:top_k]


class ExternalStorage:
    """Tiny in-memory storage for overflow references."""

    def __init__(self) -> None:
        self._storage: dict[str, dict[str, Any]] = {}

    def store(self, bucket: str, value: dict[str, Any]) -> dict[str, Any]:
        ref_id = f"{bucket}:{uuid4()}"
        self._storage[ref_id] = {"bucket": bucket, "value": value}
        return {"ref_id": ref_id, "bucket": bucket}

    def get(self, ref_id: str) -> dict[str, Any]:
        return self._storage[ref_id]


class MemoryBackend:
    """Active memory backend that versions notes and emits refs."""

    def __init__(self) -> None:
        self._versions: defaultdict[str, int] = defaultdict(int)
        self._notes: dict[str, list[dict[str, Any]]] = defaultdict(list)

    def sync(
        self,
        *,
        thread_id: str,
        content: dict[str, Any],
        auto_sync: bool,
        link_generation: bool,
        versioning: bool,
    ) -> list[dict[str, Any]]:
        if not auto_sync:
            return []
        version = (
            self._versions[thread_id] + 1 if versioning else self._versions[thread_id]
        )
        self._versions[thread_id] = version
        text = str(content)
        links = self._generate_links(text) if link_generation else []
        note = {
            "thread_id": thread_id,
            "version": version,
            "content": content,
            "links": links,
            "timestamp": _utcnow(),
        }
        self._notes[thread_id].append(note)
        return [
            {
                "ref_type": "note",
                "thread_id": thread_id,
                "version": version,
                "links": links,
            }
        ]

    @staticmethod
    def _generate_links(text: str) -> list[str]:
        seen: set[str] = set()
        links: list[str] = []
        for word in text.replace("{", " ").replace("}", " ").replace(",", " ").split():
            normalized = word.strip("[]().'\":")
            if len(normalized) < 4 or not normalized[:1].isalpha():
                continue
            token = normalized.title()
            if token in seen:
                continue
            seen.add(token)
            links.append(f"[[{token}]]")
        return links[:10]


@dataclass
class RuntimeServices:
    """Bound services injected into runtime nodes."""

    tool_registry: ToolRegistry
    agent_registry: AgentRegistry
    schema_registry: dict[str, type[BaseModel]] = field(default_factory=dict)
    memory_backend: MemoryBackend = field(default_factory=MemoryBackend)
    external_storage: ExternalStorage = field(default_factory=ExternalStorage)


def _make_protocol_message(
    *,
    thread_id: str,
    from_agent: str,
    to_agent: str,
    message_type: MessageType,
    payload: dict[str, Any] | list[Any] | str,
    metadata: dict[str, Any] | None = None,
) -> MessageProtocol:
    return MessageProtocol(
        thread_id=thread_id,
        from_agent=from_agent,
        to_agent=to_agent,
        type=message_type,
        payload=payload,
        metadata=metadata or {},
    )


class Crew:
    """Declarative crew that compiles into a LangGraph runtime."""

    def __init__(
        self,
        *,
        name: str,
        agents: Iterable[AgentConfig],
        tool_registry: ToolRegistry,
        schema_registry: dict[str, type[BaseModel]] | None = None,
        memory_backend: MemoryBackend | None = None,
        external_storage: ExternalStorage | None = None,
    ) -> None:
        self.name = name
        self.tool_registry = tool_registry
        self.agent_registry = AgentRegistry()
        for agent in agents:
            self.agent_registry.register(agent)
        self.schema_registry = schema_registry or {}
        self.memory_backend = memory_backend or MemoryBackend()
        self.external_storage = external_storage or ExternalStorage()

    def compile(
        self,
        *,
        checkpointer: BaseCheckpointSaver[Any] | None = None,
        interrupt_before: list[str] | None = None,
    ) -> Any:
        """Compile the DSL declaration into a runnable LangGraph graph."""
        services = RuntimeServices(
            tool_registry=self.tool_registry,
            agent_registry=self.agent_registry,
            schema_registry=self.schema_registry,
            memory_backend=self.memory_backend,
            external_storage=self.external_storage,
        )
        builder = StateGraph(AgentState)
        builder.add_node("interface", self._interface_node(services))
        builder.add_node("task_validation", self._task_validation_node(services))
        builder.add_node("coordinator", self._coordinator_node(services))
        builder.add_node("registry_lookup", self._registry_lookup_node(services))
        builder.add_node(
            "specialist_executor", self._specialist_executor_node(services)
        )
        builder.add_node("synthesizer", self._synthesizer_node(services))
        builder.add_node(
            "output_validation_schema", self._schema_validation_node(services)
        )
        builder.add_node(
            "output_validation_logic", self._logic_validation_node(services)
        )
        builder.add_node("memory", self._memory_node(services))
        builder.add_node("human_gate", self._human_gate_node(services))

        builder.add_edge(START, "interface")
        builder.add_conditional_edges(
            "interface",
            self._interface_router,
            {
                "task_validation": "task_validation",
                "end": END,
            },
        )
        builder.add_conditional_edges(
            "task_validation",
            self._task_validation_router,
            {
                "coordinator": "coordinator",
                "end": END,
            },
        )
        builder.add_conditional_edges(
            "coordinator",
            self._coordinator_router,
            {
                "registry_lookup": "registry_lookup",
                "specialist_executor": "specialist_executor",
                "synthesizer": "synthesizer",
                "human_gate": "human_gate",
                "end": END,
            },
        )
        builder.add_conditional_edges(
            "registry_lookup",
            self._registry_lookup_router,
            {
                "specialist_executor": "specialist_executor",
                "human_gate": "human_gate",
                "coordinator": "coordinator",
                "end": END,
            },
        )
        builder.add_conditional_edges(
            "specialist_executor",
            self._specialist_router,
            {
                "coordinator": "coordinator",
                "synthesizer": "synthesizer",
                "human_gate": "human_gate",
                "end": END,
            },
        )
        builder.add_edge("synthesizer", "output_validation_schema")
        builder.add_conditional_edges(
            "output_validation_schema",
            self._schema_validation_router,
            {
                "output_validation_logic": "output_validation_logic",
                "memory": "memory",
                "coordinator": "coordinator",
                "human_gate": "human_gate",
                "end": END,
            },
        )
        builder.add_conditional_edges(
            "output_validation_logic",
            self._logic_validation_router,
            {
                "memory": "memory",
                "coordinator": "coordinator",
                "human_gate": "human_gate",
                "end": END,
            },
        )
        builder.add_edge("memory", "interface")
        builder.add_conditional_edges(
            "human_gate",
            self._human_gate_router,
            {
                "specialist_executor": "specialist_executor",
                "coordinator": "coordinator",
                "output_validation_schema": "output_validation_schema",
                "output_validation_logic": "output_validation_logic",
                "end": END,
            },
        )

        return builder.compile(
            checkpointer=checkpointer,
            interrupt_before=interrupt_before or [],
        )

    @staticmethod
    def _interface_router(state: AgentState) -> str:
        return "end" if state.status == ThreadStatus.COMPLETED else "task_validation"

    @staticmethod
    def _task_validation_router(state: AgentState) -> str:
        return "end" if state.status == ThreadStatus.FAILED else "coordinator"

    @staticmethod
    def _coordinator_router(state: AgentState) -> str:
        if state.status == ThreadStatus.FAILED:
            return "end"
        resume_target = state.shared_context.get("resume_target")
        if resume_target == "synthesizer":
            return "synthesizer"
        if state.human_checkpoint and not state.human_checkpoint.resolved:
            return "human_gate"
        if state.task.assignee:
            return "specialist_executor"
        if state.task.auto_route:
            return "registry_lookup"
        return "end"

    @staticmethod
    def _registry_lookup_router(state: AgentState) -> str:
        if state.status == ThreadStatus.HITL_WAIT:
            return "human_gate"
        if state.status == ThreadStatus.FAILED:
            return "end"
        if state.current_agent:
            return "specialist_executor"
        return "coordinator"

    @staticmethod
    def _specialist_router(state: AgentState) -> str:
        if state.status == ThreadStatus.HITL_WAIT:
            return "human_gate"
        if state.status == ThreadStatus.REPAIR:
            return "coordinator"
        if state.status == ThreadStatus.FAILED:
            return "end"
        return "synthesizer"

    @staticmethod
    def _schema_validation_router(state: AgentState) -> str:
        validation = state.schema_validation or {}
        if not validation.get("valid"):
            if state.status == ThreadStatus.HITL_WAIT:
                return "human_gate"
            if state.status == ThreadStatus.REPAIR:
                return "coordinator"
            return "end"
        if state.task.fact_logic_validation == FactLogicValidationMode.NONE:
            return "memory"
        return "output_validation_logic"

    @staticmethod
    def _logic_validation_router(state: AgentState) -> str:
        validation = state.logic_validation or {}
        if not validation.get("valid"):
            if state.status == ThreadStatus.HITL_WAIT:
                return "human_gate"
            if state.status == ThreadStatus.REPAIR:
                return "coordinator"
            return "end"
        return "memory"

    @staticmethod
    def _human_gate_router(state: AgentState) -> str:
        checkpoint = state.human_checkpoint
        if checkpoint is None:
            return "end"
        if checkpoint.decision == "reject":
            return "end"
        return checkpoint.resume_to

    def _interface_node(
        self, services: RuntimeServices
    ) -> Callable[[AgentState], dict[str, Any]]:
        def node(state: AgentState) -> dict[str, Any]:
            if state.status == ThreadStatus.MEMORY_SYNC:
                protocol_message = _make_protocol_message(
                    thread_id=state.thread_id,
                    from_agent="memory",
                    to_agent="interface",
                    message_type=MessageType.DATA,
                    payload={
                        "result": state.output_candidate,
                        "memory_refs": state.memory_refs,
                    },
                    metadata={"event_name": "completed"},
                )
                update = _append_protocol_message(state, protocol_message)
                update["status"] = ThreadStatus.COMPLETED
                return update
            protocol_message = _make_protocol_message(
                thread_id=state.thread_id,
                from_agent="interface",
                to_agent="coordinator",
                message_type=MessageType.DATA,
                payload={
                    "task_id": state.task.task_id,
                    "description": state.task.description,
                },
                metadata={"event_name": "task_received"},
            )
            update = _append_protocol_message(state, protocol_message)
            update["status"] = ThreadStatus.INIT
            return update

        return node

    def _task_validation_node(
        self, services: RuntimeServices
    ) -> Callable[[AgentState], dict[str, Any]]:
        def node(state: AgentState) -> dict[str, Any]:
            errors = list(state.errors)
            audit_log = list(state.audit_log)
            if not state.task.description.strip():
                errors.append(
                    {"type": "task_validation", "error": "description is required"}
                )
                status = ThreadStatus.FAILED
            else:
                status = ThreadStatus.ROUTED
            audit_log = _bounded_append(
                audit_log,
                {
                    "event": "task_validation",
                    "status": status.value,
                    "timestamp": _utcnow(),
                },
                bucket="audit_log",
                storage=services.external_storage,
                thread_id=state.thread_id,
            )
            return {"status": status, "errors": errors, "audit_log": audit_log}

        return node

    def _coordinator_node(
        self, services: RuntimeServices
    ) -> Callable[[AgentState], dict[str, Any]]:
        def node(state: AgentState) -> dict[str, Any]:
            shared_context = dict(state.shared_context)
            audit_log = list(state.audit_log)
            current_agent = state.task.assignee
            if state.status == ThreadStatus.REPAIR:
                shared_context["resume_target"] = state.shared_context.get(
                    "repair_target", "synthesizer"
                )
            else:
                shared_context.pop("resume_target", None)
            if (
                HITLPoint.BEFORE_DELEGATION in state.task.hitl_points
                and not shared_context.get("delegation_approved")
            ):
                checkpoint = HumanCheckpoint(
                    reason="before_delegation",
                    resume_to="coordinator"
                    if state.task.assignee
                    else "registry_lookup",
                    request={"description": state.task.description},
                )
                status = ThreadStatus.HITL_WAIT
            else:
                checkpoint = (
                    state.human_checkpoint
                    if state.human_checkpoint and not state.human_checkpoint.resolved
                    else None
                )
                status = ThreadStatus.ROUTED
            protocol_message = _make_protocol_message(
                thread_id=state.thread_id,
                from_agent="coordinator",
                to_agent=current_agent or "registry_lookup",
                message_type=MessageType.CONTROL,
                payload={
                    "assignee": current_agent,
                    "auto_route": state.task.auto_route,
                    "status": status.value,
                },
                metadata={"event_name": "routing_started"},
            )
            update = _append_protocol_message(state, protocol_message)
            audit_log = _bounded_append(
                audit_log,
                {
                    "event": "coordinator",
                    "status": status.value,
                    "timestamp": _utcnow(),
                },
                bucket="audit_log",
                storage=services.external_storage,
                thread_id=state.thread_id,
            )
            update.update(
                {
                    "status": status,
                    "current_agent": current_agent,
                    "human_checkpoint": checkpoint,
                    "shared_context": shared_context,
                    "audit_log": audit_log,
                }
            )
            return update

        return node

    def _registry_lookup_node(
        self, services: RuntimeServices
    ) -> Callable[[AgentState], dict[str, Any]]:
        def node(state: AgentState) -> dict[str, Any]:
            candidates = services.agent_registry.lookup(
                state.task.description,
                validator_hint=state.task.validator_hint,
            )
            top = candidates[0] if candidates else None
            threshold = 0.0
            if top:
                threshold = services.agent_registry.get(
                    top.agent_id
                ).delegation_policy.confidence_threshold
            low_confidence = top is None or top.confidence < threshold
            if low_confidence and HITLPoint.ON_LOW_CONFIDENCE in state.task.hitl_points:
                checkpoint = HumanCheckpoint(
                    reason="low_confidence",
                    resume_to="specialist_executor",
                    request={
                        "candidates": [
                            candidate.model_dump() for candidate in candidates
                        ],
                        "threshold": threshold,
                    },
                )
                status = ThreadStatus.HITL_WAIT
                current_agent = top.agent_id if top else None
            elif top is None:
                current_agent = None
                if state.task.retry_policy.fallback_strategy == "fail":
                    status = ThreadStatus.FAILED
                else:
                    status = ThreadStatus.REPAIR
                checkpoint = None
            else:
                current_agent = top.agent_id
                status = ThreadStatus.ROUTED
                checkpoint = None
            protocol_message = _make_protocol_message(
                thread_id=state.thread_id,
                from_agent="registry_lookup",
                to_agent=current_agent or "coordinator",
                message_type=MessageType.CONTROL,
                payload={
                    "candidates": [candidate.model_dump() for candidate in candidates]
                },
                metadata={"event_name": "lookup_completed"},
            )
            update = _append_protocol_message(state, protocol_message)
            update.update(
                {
                    "route_candidates": candidates,
                    "current_agent": current_agent,
                    "status": status,
                    "human_checkpoint": checkpoint,
                }
            )
            return update

        return node

    def _specialist_executor_node(
        self, services: RuntimeServices
    ) -> Callable[[AgentState], dict[str, Any]]:
        def node(state: AgentState) -> dict[str, Any]:
            if not state.current_agent:
                return {
                    "status": ThreadStatus.REPAIR,
                    "shared_context": {
                        **state.shared_context,
                        "repair_target": "coordinator",
                    },
                }
            agent = services.agent_registry.get(state.current_agent)
            binding = _select_tool_binding(
                agent.tools, state.task.metadata.get("tool_ref")
            )
            if binding is None:
                return {
                    "status": ThreadStatus.REPAIR,
                    "errors": [
                        *state.errors,
                        {
                            "type": "tool_binding",
                            "error": f"no tool binding for {agent.agent_id}",
                        },
                    ],
                    "shared_context": {
                        **state.shared_context,
                        "repair_target": "coordinator",
                    },
                }

            tool_definition = services.tool_registry.get(binding.tool_ref)
            approved_tools = set(state.shared_context.get("approved_tools", []))
            requires_gate = (
                binding.requires_hitl
                or HITLPoint.BEFORE_TOOL_CALL in state.task.hitl_points
                or _enum_rank(tool_definition.side_effect_level)
                > _enum_rank(binding.allowed_side_effect_level)
            )
            if requires_gate and binding.tool_ref not in approved_tools:
                checkpoint = HumanCheckpoint(
                    reason="tool_approval",
                    resume_to="specialist_executor",
                    request={
                        "tool_ref": binding.tool_ref,
                        "permission_level": binding.permission_level.value,
                        "side_effect_level": tool_definition.side_effect_level.value,
                    },
                )
                return {
                    "status": ThreadStatus.HITL_WAIT,
                    "human_checkpoint": checkpoint,
                }

            tool_args = {
                "query": state.task.description,
                "task_id": state.task.task_id,
                "context_refs": state.task.context_refs,
                "task": state.task.model_dump(),
                "thread_id": state.thread_id,
                "current_agent": state.current_agent,
                "status": state.status.value,
                "shared_context": state.shared_context,
                "route_candidates": [
                    candidate.model_dump() for candidate in state.route_candidates
                ],
                "artifacts": state.artifacts,
                "output_candidate": state.output_candidate,
                "errors": state.errors,
            }
            result = services.tool_registry.invoke(binding.tool_ref, tool_args)
            audit_log = _bounded_append(
                list(state.audit_log),
                {
                    "event": "tool_invoked",
                    "tool_ref": binding.tool_ref,
                    "success": result.success,
                    "timestamp": _utcnow(),
                },
                bucket="audit_log",
                storage=services.external_storage,
                thread_id=state.thread_id,
            )
            if not result.success:
                retry_count = state.retry_counters.get("tool", 0) + 1
                errors = [
                    *state.errors,
                    {
                        "type": "tool_execution",
                        "tool_ref": binding.tool_ref,
                        "error": result.error,
                        "error_type": result.error_type,
                        "metadata": result.metadata,
                    },
                ]
                if (
                    _is_tool_result_retriable(result)
                    and retry_count <= state.task.retry_policy.tool_retry_attempts
                ):
                    return {
                        "status": ThreadStatus.REPAIR,
                        "retry_counters": {**state.retry_counters, "tool": retry_count},
                        "errors": errors,
                        "audit_log": audit_log,
                        "shared_context": {
                            **state.shared_context,
                            "repair_target": "specialist_executor",
                        },
                    }
                return {
                    "status": ThreadStatus.FAILED,
                    "retry_counters": {**state.retry_counters, "tool": retry_count},
                    "errors": errors,
                    "audit_log": audit_log,
                }

            artifact = {
                "artifact_type": "research",
                "agent_id": agent.agent_id,
                "tool_ref": binding.tool_ref,
                "data": result.data,
                "timestamp": _utcnow(),
            }
            artifacts = _bounded_append(
                list(state.artifacts),
                artifact,
                bucket="artifacts",
                storage=services.external_storage,
                thread_id=state.thread_id,
            )
            protocol_message = _make_protocol_message(
                thread_id=state.thread_id,
                from_agent=agent.agent_id,
                to_agent="synthesizer",
                message_type=MessageType.DATA,
                payload={"artifact_type": "research", "tool_ref": binding.tool_ref},
                metadata={"event_name": "specialist_completed"},
            )
            update = _append_protocol_message(state, protocol_message)
            update.update(
                {
                    "status": ThreadStatus.EXECUTING,
                    "artifacts": artifacts,
                    "audit_log": audit_log,
                    "shared_context": {
                        **state.shared_context,
                        "research_result": result.data,
                    },
                }
            )
            return update

        return node

    def _synthesizer_node(
        self, services: RuntimeServices
    ) -> Callable[[AgentState], dict[str, Any]]:
        def node(state: AgentState) -> dict[str, Any]:
            research = state.shared_context.get("research_result", {})
            results = research.get("results", [])
            summary_parts = []
            for item in results:
                if not isinstance(item, dict):
                    continue
                snippet = str(item.get("snippet", "")).strip()
                title = str(item.get("title", "")).strip()
                url = str(item.get("url", "")).strip()
                if snippet:
                    summary_parts.append(snippet)
                elif title:
                    summary_parts.append(title)
                elif url:
                    summary_parts.append(url)
            citations = [
                item.get("url") or item.get("title", "")
                for item in results
                if item.get("url") or item.get("title")
            ]
            output_candidate: dict[str, Any] = {
                "task_id": state.task.task_id,
                "summary": " ".join(summary_parts).strip(),
                "citations": citations,
                "artifacts_used": len(state.artifacts),
            }
            if not (
                state.task.metadata.get("force_schema_failure_once")
                and state.retry_counters.get("schema", 0) == 0
            ):
                output_candidate["confidence"] = 0.8 if citations else 0.3
            protocol_message = _make_protocol_message(
                thread_id=state.thread_id,
                from_agent="synthesizer",
                to_agent="output_validation_schema",
                message_type=MessageType.DATA,
                payload=output_candidate,
                metadata={"event_name": "synthesis_ready"},
            )
            update = _append_protocol_message(state, protocol_message)
            update.update(
                {
                    "status": ThreadStatus.VALIDATING,
                    "output_candidate": output_candidate,
                }
            )
            return update

        return node

    def _schema_validation_node(
        self, services: RuntimeServices
    ) -> Callable[[AgentState], dict[str, Any]]:
        def node(state: AgentState) -> dict[str, Any]:
            validation = _validate_output_schema(
                output_candidate=state.output_candidate,
                output_schema=state.task.output_schema,
                schema_registry=services.schema_registry,
            )
            if validation["valid"]:
                return {
                    "schema_validation": validation,
                    "status": ThreadStatus.VALIDATING,
                }

            retry_count = state.retry_counters.get("schema", 0) + 1
            errors = [
                *state.errors,
                {"type": "schema_validation", "details": validation["errors"]},
            ]
            if HITLPoint.AFTER_SCHEMA_VALIDATION_FAILURE in state.task.hitl_points:
                checkpoint = HumanCheckpoint(
                    reason="schema_validation_failed",
                    resume_to="output_validation_schema",
                    request={"errors": validation["errors"]},
                )
                return {
                    "schema_validation": validation,
                    "status": ThreadStatus.HITL_WAIT,
                    "retry_counters": {**state.retry_counters, "schema": retry_count},
                    "errors": errors,
                    "human_checkpoint": checkpoint,
                }
            if retry_count <= state.task.retry_policy.schema_repair_attempts:
                return {
                    "schema_validation": validation,
                    "status": ThreadStatus.REPAIR,
                    "retry_counters": {**state.retry_counters, "schema": retry_count},
                    "errors": errors,
                    "shared_context": {
                        **state.shared_context,
                        "repair_target": "synthesizer",
                    },
                }
            return {
                "schema_validation": validation,
                "status": ThreadStatus.FAILED,
                "retry_counters": {**state.retry_counters, "schema": retry_count},
                "errors": errors,
            }

        return node

    def _logic_validation_node(
        self, services: RuntimeServices
    ) -> Callable[[AgentState], dict[str, Any]]:
        def node(state: AgentState) -> dict[str, Any]:
            validation = _validate_facts_and_logic(state)
            if validation["valid"]:
                return {
                    "logic_validation": validation,
                    "status": ThreadStatus.MEMORY_SYNC,
                }

            retry_count = state.retry_counters.get("logic", 0) + 1
            errors = [
                *state.errors,
                {"type": "logic_validation", "details": validation["errors"]},
            ]
            if HITLPoint.AFTER_LOGIC_VALIDATION_FAILURE in state.task.hitl_points:
                checkpoint = HumanCheckpoint(
                    reason="logic_validation_failed",
                    resume_to="output_validation_logic",
                    request={"errors": validation["errors"]},
                )
                return {
                    "logic_validation": validation,
                    "status": ThreadStatus.HITL_WAIT,
                    "retry_counters": {**state.retry_counters, "logic": retry_count},
                    "errors": errors,
                    "human_checkpoint": checkpoint,
                }
            if retry_count <= state.task.retry_policy.logic_repair_attempts:
                return {
                    "logic_validation": validation,
                    "status": ThreadStatus.REPAIR,
                    "retry_counters": {**state.retry_counters, "logic": retry_count},
                    "errors": errors,
                    "shared_context": {
                        **state.shared_context,
                        "repair_target": "synthesizer",
                    },
                }
            return {
                "logic_validation": validation,
                "status": ThreadStatus.FAILED,
                "retry_counters": {**state.retry_counters, "logic": retry_count},
                "errors": errors,
            }

        return node

    def _memory_node(
        self, services: RuntimeServices
    ) -> Callable[[AgentState], dict[str, Any]]:
        def node(state: AgentState) -> dict[str, Any]:
            agent_id = state.current_agent
            if agent_id is None and state.route_candidates:
                agent_id = state.route_candidates[0].agent_id
            if agent_id is None:
                all_agents = services.agent_registry.all()
                agent_id = all_agents[0].agent_id if all_agents else None
            if agent_id is None:
                return {"status": ThreadStatus.FAILED}
            agent = services.agent_registry.get(agent_id)
            refs = services.memory_backend.sync(
                thread_id=state.thread_id,
                content=state.output_candidate or {},
                auto_sync=agent.memory_policy.auto_sync,
                link_generation=agent.memory_policy.link_generation,
                versioning=agent.memory_policy.versioning,
            )
            protocol_message = _make_protocol_message(
                thread_id=state.thread_id,
                from_agent="memory",
                to_agent="interface",
                message_type=MessageType.CONTROL,
                payload={"memory_refs": refs},
                metadata={"event_name": "context_updated"},
            )
            update = _append_protocol_message(state, protocol_message)
            sync_ticket_id = None
            for ref in refs:
                if ref.get("ref_type") == "async_memory_ticket":
                    sync_ticket_id = ref.get("ticket_id")
                    break
            update.update(
                {
                    "memory_refs": [*state.memory_refs, *refs],
                    "status": ThreadStatus.MEMORY_SYNC,
                    "sync_ticket_id": sync_ticket_id,
                }
            )
            return update

        return node

    def _human_gate_node(
        self, services: RuntimeServices
    ) -> Callable[[AgentState], dict[str, Any]]:
        def node(state: AgentState) -> dict[str, Any]:
            _wait_for_pending_sync_ticket(state)
            checkpoint = state.human_checkpoint
            if checkpoint is None:
                return {"status": ThreadStatus.FAILED}
            feedback = interrupt(
                {
                    "checkpoint_id": checkpoint.checkpoint_id,
                    "reason": checkpoint.reason,
                    "resume_to": checkpoint.resume_to,
                    "request": checkpoint.request,
                }
            )
            decision = feedback.get("decision", "approve")
            payload = dict(feedback)
            shared_context = dict(state.shared_context)
            if checkpoint.reason == "tool_approval" and decision == "approve":
                approved_tools = set(shared_context.get("approved_tools", []))
                tool_ref = checkpoint.request.get("tool_ref")
                if tool_ref:
                    approved_tools.add(tool_ref)
                shared_context["approved_tools"] = sorted(approved_tools)
            if checkpoint.reason == "before_delegation" and decision == "approve":
                shared_context["delegation_approved"] = True
            resolved_checkpoint = checkpoint.model_copy(
                update={
                    "resolved": True,
                    "decision": decision,
                    "payload": payload,
                    "resume_to": payload.get("resume_to", checkpoint.resume_to),
                }
            )
            status = (
                ThreadStatus.FAILED if decision == "reject" else ThreadStatus.ROUTED
            )
            protocol_message = _make_protocol_message(
                thread_id=state.thread_id,
                from_agent="human",
                to_agent=resolved_checkpoint.resume_to,
                message_type=MessageType.FEEDBACK,
                payload=payload,
                metadata={"event_name": "human_feedback"},
            )
            update = _append_protocol_message(state, protocol_message)
            update.update(
                {
                    "status": status,
                    "human_checkpoint": resolved_checkpoint,
                    "shared_context": shared_context,
                }
            )
            return update

        return node


def _select_tool_binding(
    bindings: list[ToolBinding],
    requested_tool_ref: str | None,
) -> ToolBinding | None:
    if requested_tool_ref:
        for binding in bindings:
            if binding.tool_ref == requested_tool_ref:
                return binding
    return bindings[0] if bindings else None


def _validate_output_schema(
    *,
    output_candidate: Any,
    output_schema: str | dict[str, Any],
    schema_registry: dict[str, type[BaseModel]],
) -> dict[str, Any]:
    if isinstance(output_schema, str):
        model = schema_registry.get(output_schema)
        if model is None:
            return {"valid": False, "errors": [f"unknown schema: {output_schema}"]}
        try:
            TypeAdapter(model).validate_python(output_candidate)
        except ValidationError as exc:
            return {"valid": False, "errors": exc.errors()}
        return {"valid": True, "errors": []}

    required_fields = output_schema.get("required", [])
    errors = []
    for field_name in required_fields:
        if field_name not in (output_candidate or {}):
            errors.append(f"missing field: {field_name}")
    return {"valid": not errors, "errors": errors}


def _validate_facts_and_logic(state: AgentState) -> dict[str, Any]:
    if state.task.fact_logic_validation == FactLogicValidationMode.NONE:
        return {"valid": True, "errors": []}

    output = state.output_candidate or {}
    summary = str(output.get("summary", ""))
    citations = output.get("citations", [])
    errors: list[str] = []
    if not summary.strip():
        errors.append("summary is empty")
    if state.task.metadata.get("require_citations") and not citations:
        errors.append("citations are required")
    required_terms = state.task.metadata.get("required_terms", [])
    for term in required_terms:
        if term.lower() not in summary.lower():
            errors.append(f"missing required term: {term}")
    return {"valid": not errors, "errors": errors}


def _wait_for_pending_sync_ticket(state: AgentState, *, timeout: float = 5.0) -> None:
    ticket_id = state.sync_ticket_id
    if ticket_id is None:
        return
    if state.task.metadata.get("allow_stale_memory_refs", False):
        return
    try:
        from agentgraph.async_memory_sync import AsyncMemorySyncBackend
    except ImportError:
        return
    status = AsyncMemorySyncBackend.status_for_ticket(ticket_id)
    if status["status"] == "pending":
        AsyncMemorySyncBackend.wait_for_ticket(ticket_id, timeout=timeout)


def stream_envelopes(
    graph: Any,
    input_value: Any,
    config: dict[str, Any],
    *,
    stream_mode: str | list[str] = "updates",
    **kwargs: Any,
) -> Iterator[dict[str, Any]]:
    """Wrap raw LangGraph stream chunks in a topology-agnostic envelope."""
    thread_id = config.get("configurable", {}).get("thread_id", "unknown-thread")
    for chunk in graph.stream(input_value, config, stream_mode=stream_mode, **kwargs):
        event_type = "state"
        payload: Any = chunk
        metadata: dict[str, Any] = {}
        if isinstance(chunk, tuple) and len(chunk) == 2 and isinstance(chunk[0], str):
            event_type = chunk[0]
            payload = chunk[1]
        elif isinstance(chunk, dict) and "__interrupt__" in chunk:
            event_type = "interrupt"
            payload = {"interrupts": [repr(item) for item in chunk["__interrupt__"]]}
        yield {
            "event_type": event_type,
            "thread_id": thread_id,
            "timestamp": _utcnow(),
            "payload": payload,
            "metadata": metadata,
        }


def resume_from_human_feedback(
    graph: Any,
    *,
    thread_id: str,
    human_feedback: dict[str, Any],
) -> Any:
    """Resume a blocked thread from human feedback."""
    try:
        from agentgraph.async_memory_sync import ensure_resume_safe
    except ImportError:
        pass
    else:
        ensure_resume_safe(graph, thread_id=thread_id, timeout=5.0)
    return graph.invoke(
        Command(resume=human_feedback),
        {"configurable": {"thread_id": thread_id}},
    )
