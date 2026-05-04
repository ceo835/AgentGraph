"""Public DSL and runtime contracts for the AgentGraph vertical slice."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Annotated, Any
from uuid import uuid4

from langgraph.graph.message import add_messages
from pydantic import BaseModel, ConfigDict, Field


class MessageType(str, Enum):
    """Message categories flowing through the inter-agent protocol."""

    DATA = "data"
    CONTROL = "control"
    FEEDBACK = "feedback"


class ThreadStatus(str, Enum):
    """Lifecycle states for a single thread."""

    INIT = "INIT"
    ROUTED = "ROUTED"
    EXECUTING = "EXECUTING"
    VALIDATING = "VALIDATING"
    MEMORY_SYNC = "MEMORY_SYNC"
    COMPLETED = "COMPLETED"
    HITL_WAIT = "HITL_WAIT"
    REPAIR = "REPAIR"
    FAILED = "FAILED"


class HITLPoint(str, Enum):
    """Human-in-the-loop checkpoints declared at the task layer."""

    BEFORE_DELEGATION = "before_delegation"
    BEFORE_TOOL_CALL = "before_tool_call"
    BEFORE_FINALIZE = "before_finalize"
    AFTER_SCHEMA_VALIDATION_FAILURE = "after_schema_validation_failure"
    AFTER_LOGIC_VALIDATION_FAILURE = "after_logic_validation_failure"
    ON_LOW_CONFIDENCE = "on_low_confidence"


class PermissionLevel(str, Enum):
    """Coarse-grained tool permission levels."""

    READ = "read"
    WRITE = "write"
    ADMIN = "admin"


class SideEffectLevel(str, Enum):
    """Expected side-effect envelope for a tool."""

    NONE = "none"
    READ_ONLY = "read_only"
    EXTERNAL_WRITE = "external_write"
    DESTRUCTIVE = "destructive"


class FactLogicValidationMode(str, Enum):
    """How post-schema validation should be executed."""

    NONE = "none"
    POLICY = "policy"
    SPECIALIST = "specialist"


class CapabilityDescriptor(BaseModel):
    """Semantic profile used by the agent registry for routing."""

    capability_id: str
    name: str
    summary: str
    keywords: list[str] = Field(default_factory=list)
    domains: list[str] = Field(default_factory=list)
    tool_affinity: list[str] = Field(default_factory=list)
    embedding_text: str


class DelegationPolicy(BaseModel):
    """Routing thresholds and fallback behaviour for an agent."""

    can_delegate: bool = True
    selection_mode: str = "semantic"
    max_depth: int = 3
    confidence_threshold: float = 0.45
    fallback_strategy: str = "reroute"
    approval_required: bool = False
    policy_weight: float = 1.0


class MemoryPolicy(BaseModel):
    """Controls how an agent reads and writes thread memory."""

    read_scopes: list[str] = Field(default_factory=lambda: ["thread"])
    write_scopes: list[str] = Field(default_factory=lambda: ["thread"])
    summarization: bool = True
    vectorize: bool = False
    graph_linking: bool = True
    link_generation: bool = True
    versioning: bool = True
    auto_sync: bool = True


class ToolBinding(BaseModel):
    """Associates an agent with an allowed tool and guardrails."""

    tool_ref: str
    permission_level: PermissionLevel = PermissionLevel.READ
    requires_hitl: bool = False
    allowed_side_effect_level: SideEffectLevel = SideEffectLevel.READ_ONLY
    doc_ref: str | None = None


class RetryPolicy(BaseModel):
    """Repair and escalation policy for a task."""

    max_attempts: int = 2
    schema_repair_attempts: int = 1
    logic_repair_attempts: int = 1
    tool_retry_attempts: int = 1
    escalate_to_hitl_after: int = 1
    fallback_strategy: str = "reroute"
    fallback_agent: str | None = None


class AgentConfig(BaseModel):
    """Declarative definition of an agent registered with the crew."""

    agent_id: str
    role: str
    goal: str
    backstory: str
    tools: list[ToolBinding] = Field(default_factory=list)
    delegation_policy: DelegationPolicy = Field(default_factory=DelegationPolicy)
    memory_policy: MemoryPolicy = Field(default_factory=MemoryPolicy)
    capabilities: list[CapabilityDescriptor] = Field(default_factory=list)
    constraints: dict[str, Any] | None = None
    availability_factor: float = 1.0


class TaskSpec(BaseModel):
    """Declarative task contract passed into the runtime graph."""

    task_id: str
    description: str
    output_schema: str | dict[str, Any]
    assignee: str | None = None
    auto_route: bool = True
    retry_policy: RetryPolicy = Field(default_factory=RetryPolicy)
    hitl_points: list[HITLPoint] = Field(default_factory=list)
    context_refs: list[str] = Field(default_factory=list)
    success_criteria: list[str] = Field(default_factory=list)
    fact_logic_validation: FactLogicValidationMode = FactLogicValidationMode.NONE
    validator_hint: str | None = None
    priority: str | None = "normal"
    metadata: dict[str, Any] = Field(default_factory=dict)


class MessageProtocol(BaseModel):
    """Structured message envelope exchanged between runtime participants."""

    message_id: str = Field(default_factory=lambda: str(uuid4()))
    thread_id: str
    from_agent: str = Field(alias="from")
    to_agent: str = Field(alias="to")
    type: MessageType
    payload: dict[str, Any] | list[Any] | str
    metadata: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    model_config = ConfigDict(populate_by_name=True)

    def to_graph_message(self) -> dict[str, Any]:
        """Convert the protocol envelope into a LangGraph-compatible message."""
        role_map = {
            MessageType.DATA: "assistant",
            MessageType.CONTROL: "system",
            MessageType.FEEDBACK: "user",
        }
        return {
            "id": self.message_id,
            "role": role_map[self.type],
            "content": self.render_content(),
        }

    def render_content(self) -> str:
        """Render a stable text projection for the state message reducer."""
        return f"{self.from_agent}->{self.to_agent} [{self.type.value}] {self.payload}"


class ToolResult(BaseModel):
    """Canonical tool execution response."""

    success: bool
    data: Any = None
    error: str | None = None
    error_type: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class LookupCandidate(BaseModel):
    """Ranked routing candidate produced by the agent registry."""

    agent_id: str
    semantic_score: float
    policy_weight: float
    availability_factor: float
    confidence: float
    rationale: str


class HumanCheckpoint(BaseModel):
    """Pending or resolved human-review state."""

    checkpoint_id: str = Field(default_factory=lambda: str(uuid4()))
    reason: str
    resume_to: str
    request: dict[str, Any] = Field(default_factory=dict)
    resolved: bool = False
    decision: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)


class AgentState(BaseModel):
    """Shared runtime state passed through the LangGraph StateGraph."""

    schema_version: int = 2
    thread_id: str
    status: ThreadStatus = ThreadStatus.INIT
    task: TaskSpec
    messages: Annotated[list[Any], add_messages] = Field(default_factory=list)
    protocol_messages: list[MessageProtocol] = Field(default_factory=list)
    current_agent: str | None = None
    route_candidates: list[LookupCandidate] = Field(default_factory=list)
    shared_context: dict[str, Any] = Field(default_factory=dict)
    artifacts: list[dict[str, Any]] = Field(default_factory=list, max_length=50)
    output_candidate: Any = None
    schema_validation: dict[str, Any] | None = None
    logic_validation: dict[str, Any] | None = None
    memory_refs: list[dict[str, Any]] = Field(default_factory=list)
    sync_ticket_id: str | None = None
    human_checkpoint: HumanCheckpoint | None = None
    retry_counters: dict[str, int] = Field(
        default_factory=lambda: {"schema": 0, "logic": 0, "tool": 0, "repair": 0}
    )
    errors: list[dict[str, Any]] = Field(default_factory=list)
    audit_log: list[dict[str, Any]] = Field(default_factory=list, max_length=50)
