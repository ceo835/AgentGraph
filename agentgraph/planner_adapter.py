"""Planner specialists and Send-based subtask execution."""

from __future__ import annotations

import operator
from typing import Annotated, Any
from uuid import uuid4

from langgraph.graph import START, StateGraph
from langgraph.types import Command, Send
from pydantic import BaseModel, Field

from agentgraph.contracts import (
    AgentConfig,
    AgentState,
    CapabilityDescriptor,
    DelegationPolicy,
    MemoryPolicy,
    TaskSpec,
    ThreadStatus,
)
from agentgraph.runtime import AgentRegistry, Crew


def register_planner_agent(
    *,
    agent_registry: AgentRegistry,
    agent_id: str = "planner",
) -> AgentConfig:
    """Register a planner specialist in the shared agent registry."""
    agent = AgentConfig(
        agent_id=agent_id,
        role="planner",
        goal="Decompose complex tasks into executable subtasks.",
        backstory="Optimizes complex tasks by planning parallel specialist work.",
        tools=[],
        delegation_policy=DelegationPolicy(
            confidence_threshold=0.05,
            fallback_strategy="reroute",
            policy_weight=1.1,
        ),
        memory_policy=MemoryPolicy(auto_sync=False),
        capabilities=[
            CapabilityDescriptor(
                capability_id="planner-capability",
                name="Planner",
                summary="Breaks tasks into parallelizable subtasks.",
                keywords=["planner", "subtasks", "decompose", "parallel"],
                domains=["planning", "orchestration"],
                tool_affinity=[],
                embedding_text="planner task decomposition subtask parallel execution send",
            )
        ],
    )
    agent_registry.register(agent)
    return agent


class PlannerState(BaseModel):
    """State for the Send-based planner graph."""

    thread_id: str
    root_task: dict[str, Any]
    subtasks: list[dict[str, Any]] = Field(default_factory=list)
    subtask_results: Annotated[list[dict[str, Any]], operator.add] = Field(
        default_factory=list
    )
    failed_subtask_ids: Annotated[list[str], operator.add] = Field(default_factory=list)
    subtask_errors: Annotated[list[dict[str, Any]], operator.add] = Field(
        default_factory=list
    )
    shared_context: dict[str, Any] = Field(default_factory=dict)
    status: str = "INIT"


class SubtaskExecutionState(BaseModel):
    """Input state for a single planner-dispatched subtask."""

    subtask: dict[str, Any]
    planner_thread_id: str


class PlannerAdapter:
    """Planner wrapper that can decompose tasks and execute them in parallel."""

    def __init__(self, *, crew: Crew) -> None:
        self.crew = crew
        self.worker_graph = crew.compile(checkpointer=None)

    def decompose(self, task: TaskSpec) -> list[TaskSpec]:
        """Split a complex task into subtasks using task metadata or text heuristics."""
        if isinstance(task.metadata.get("subtasks"), list):
            subtasks = [
                TaskSpec(
                    task_id=f"{task.task_id}-subtask-{index}",
                    description=(
                        str(description["description"])
                        if isinstance(description, dict)
                        else str(description)
                    ),
                    output_schema=task.output_schema,
                    assignee=task.assignee,
                    auto_route=task.auto_route,
                    retry_policy=task.retry_policy,
                    hitl_points=task.hitl_points,
                    context_refs=task.context_refs,
                    success_criteria=task.success_criteria,
                    fact_logic_validation=task.fact_logic_validation,
                    validator_hint=task.validator_hint,
                    priority=task.priority,
                    metadata={
                        **task.metadata,
                        **(
                            description.get("metadata", {})
                            if isinstance(description, dict)
                            else {}
                        ),
                        "parent_task_id": task.task_id,
                    },
                )
                for index, description in enumerate(task.metadata["subtasks"])
            ]
            return subtasks

        parts = [
            part.strip() for part in task.description.split(" and ") if part.strip()
        ]
        if len(parts) <= 1:
            return [task]
        return [
            task.model_copy(
                update={
                    "task_id": f"{task.task_id}-subtask-{index}",
                    "description": description,
                    "metadata": {**task.metadata, "parent_task_id": task.task_id},
                }
            )
            for index, description in enumerate(parts)
        ]

    def should_plan(self, task: TaskSpec) -> bool:
        """Route to the planner if there are multiple decomposition candidates."""
        if task.metadata.get("enable_planner"):
            return True
        return len(self.decompose(task)) > 1

    def invoke(self, state: AgentState) -> AgentState:
        """Execute subtasks in parallel using `Send`, then aggregate results."""
        subtasks = self.decompose(state.task)
        if len(subtasks) <= 1:
            return state

        planner_graph = self._compile_planner_graph()
        planner_state = PlannerState(
            thread_id=state.thread_id,
            root_task=state.task.model_dump(),
        )
        result = planner_graph.invoke(planner_state.model_dump())
        final_planner_state = PlannerState.model_validate(result)
        aggregated = self.aggregate(final_planner_state.subtask_results)
        failed_ids = final_planner_state.failed_subtask_ids
        if failed_ids:
            fallback = state.task.retry_policy.fallback_strategy
            if fallback == "human":
                next_status = ThreadStatus.HITL_WAIT
            elif fallback == "reroute":
                next_status = ThreadStatus.REPAIR
            else:
                next_status = ThreadStatus.FAILED
        else:
            next_status = ThreadStatus.COMPLETED
        return state.model_copy(
            update={
                "status": next_status,
                "shared_context": {
                    **state.shared_context,
                    "subtask_results": final_planner_state.subtask_results,
                    "planner_aggregation": aggregated,
                    "failed_subtask_ids": failed_ids,
                },
                "output_candidate": aggregated,
                "errors": [
                    *state.errors,
                    *final_planner_state.subtask_errors,
                ],
            }
        )

    def aggregate(self, subtask_results: list[dict[str, Any]]) -> dict[str, Any]:
        """Aggregate completed subtask payloads into a single root result."""
        summaries = [
            item["output_candidate"]["summary"]
            for item in subtask_results
            if item.get("output_candidate")
        ]
        citations: list[str] = []
        for item in subtask_results:
            citations.extend(item.get("output_candidate", {}).get("citations", []))
        return {
            "summary": " ".join(summaries),
            "citations": citations,
            "subtasks_completed": len(subtask_results),
            "partial_results": len(subtask_results) > 0,
        }

    def _compile_planner_graph(self) -> Any:
        builder = StateGraph(PlannerState)
        builder.add_node("plan", self._plan_node)
        builder.add_node(
            "execute_subtask",
            self._execute_subtask_node,
            input_schema=SubtaskExecutionState,
        )
        builder.add_edge(START, "plan")
        return builder.compile(checkpointer=None)

    def _plan_node(self, state: PlannerState) -> Command[str]:
        task = TaskSpec.model_validate(state.root_task)
        subtasks = self.decompose(task)
        sends = [
            Send(
                "execute_subtask",
                {"subtask": subtask.model_dump(), "planner_thread_id": state.thread_id},
            )
            for subtask in subtasks
        ]
        return Command(
            goto=sends,
            update={
                "subtasks": [subtask.model_dump() for subtask in subtasks],
                "status": "PLANNED",
            },
        )

    def _execute_subtask_node(self, state: Any) -> dict[str, Any]:
        payload = SubtaskExecutionState.model_validate(state)
        subtask = TaskSpec.model_validate(payload.subtask)
        planner_thread_id = payload.planner_thread_id
        thread_id = f"{planner_thread_id}:{subtask.task_id}:{uuid4()}"
        try:
            if subtask.metadata.get("force_failure"):
                raise RuntimeError(f"forced failure for {subtask.task_id}")
            result = self.worker_graph.invoke(
                AgentState(thread_id=thread_id, task=subtask).model_dump(by_alias=True),
                {"configurable": {"thread_id": thread_id}},
            )
            subtask_state = AgentState.model_validate(result)
            return {
                "subtask_results": [
                    {
                        "task_id": subtask.task_id,
                        "output_candidate": subtask_state.output_candidate,
                        "status": subtask_state.status.value,
                    }
                ]
            }
        except Exception as exc:
            return {
                "failed_subtask_ids": [subtask.task_id],
                "subtask_errors": [
                    {
                        "type": "planner_subtask",
                        "task_id": subtask.task_id,
                        "error": str(exc),
                    }
                ],
            }
