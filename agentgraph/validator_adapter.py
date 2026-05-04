"""Dynamic validator specialists and fallback orchestration."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from agentgraph.contracts import (
    AgentConfig,
    AgentState,
    CapabilityDescriptor,
    DelegationPolicy,
    FactLogicValidationMode,
    MemoryPolicy,
    MessageType,
    SideEffectLevel,
    ThreadStatus,
    ToolBinding,
    ToolResult,
)
from agentgraph.runtime import (
    AgentRegistry,
    ToolRegistry,
    _append_protocol_message,
    _make_protocol_message,
)

PolicyRule = Callable[[AgentState], list[str]]


def register_validator_agent(
    *,
    agent_registry: AgentRegistry,
    tool_registry: ToolRegistry,
    tool_ref: str,
    validator_handler: Callable[[dict[str, Any]], ToolResult],
    agent_id: str = "validator",
) -> AgentConfig:
    """Register a validator specialist through the existing registries."""
    tool_registry.register(
        tool_ref=tool_ref,
        description="Fact and logic validation specialist.",
        side_effect_level=SideEffectLevel.READ_ONLY,
        handler=validator_handler,
    )
    agent = AgentConfig(
        agent_id=agent_id,
        role="validator",
        goal="Validate facts, structure, and reasoning quality.",
        backstory="Acts as an external specialist for validation fallback.",
        tools=[ToolBinding(tool_ref=tool_ref)],
        delegation_policy=DelegationPolicy(
            confidence_threshold=0.05,
            fallback_strategy="reroute",
            policy_weight=1.2,
        ),
        memory_policy=MemoryPolicy(auto_sync=False),
        capabilities=[
            CapabilityDescriptor(
                capability_id="validator-capability",
                name="Validator",
                summary="Validates factuality and logical coherence.",
                keywords=["validator", "fact-check", "logic", "schema"],
                domains=["qa", "validation"],
                tool_affinity=[tool_ref],
                embedding_text="validator fact logic specialist quality assurance",
            )
        ],
    )
    agent_registry.register(agent)
    return agent


class DynamicValidatorAdapter:
    """Two-phase validator that runs policy rules then specialist fallback."""

    def __init__(
        self,
        *,
        agent_registry: AgentRegistry,
        tool_registry: ToolRegistry,
        policy_rules: list[PolicyRule] | None = None,
    ) -> None:
        self.agent_registry = agent_registry
        self.tool_registry = tool_registry
        self.policy_rules = policy_rules or [self._required_terms_rule]

    def validate(self, state: AgentState) -> AgentState:
        """Validate a completed state without changing the core runtime graph."""
        if state.task.fact_logic_validation == FactLogicValidationMode.NONE:
            return state

        policy_errors = self._run_policy_rules(state)
        if not policy_errors:
            return state.model_copy(
                update={
                    "logic_validation": {"valid": True, "errors": [], "mode": "policy"},
                    "status": ThreadStatus.COMPLETED,
                }
            )

        specialist_result = self._run_specialist(state, policy_errors)
        if (
            specialist_result
            and specialist_result.success
            and specialist_result.data.get("valid", False)
        ):
            update = {
                "logic_validation": {
                    "valid": True,
                    "errors": [],
                    "mode": "specialist",
                    "specialist": specialist_result.data,
                },
                "status": ThreadStatus.COMPLETED,
            }
            protocol_message = _make_protocol_message(
                thread_id=state.thread_id,
                from_agent="validator",
                to_agent="interface",
                message_type=MessageType.CONTROL,
                payload={"result": "validated"},
                metadata={"event_name": "validator_fallback_success"},
            )
            message_update = _append_protocol_message(state, protocol_message)
            update.update(message_update)
            return state.model_copy(update=update)

        retry_count = state.retry_counters.get("logic", 0) + 1
        errors = list(policy_errors)
        if specialist_result and specialist_result.error:
            errors.append(specialist_result.error)
        can_retry = retry_count <= state.task.retry_policy.logic_repair_attempts
        return state.model_copy(
            update={
                "logic_validation": {
                    "valid": False,
                    "errors": errors,
                    "mode": "policy",
                },
                "retry_counters": {**state.retry_counters, "logic": retry_count},
                "status": ThreadStatus.REPAIR if can_retry else ThreadStatus.FAILED,
                "errors": [
                    *state.errors,
                    {
                        "type": "validator",
                        "errors": errors,
                    },
                ],
            }
        )

    def _run_policy_rules(self, state: AgentState) -> list[str]:
        errors: list[str] = []
        for rule in self.policy_rules:
            errors.extend(rule(state))
        return errors

    def _run_specialist(
        self,
        state: AgentState,
        policy_errors: list[str],
    ) -> ToolResult | None:
        query = state.task.validator_hint or f"validator {state.task.description}"
        candidates = self.agent_registry.lookup(
            query, validator_hint="validator", top_k=1
        )
        if not candidates:
            return None
        validator_agent = self.agent_registry.get(candidates[0].agent_id)
        if not validator_agent.tools:
            return None
        binding = validator_agent.tools[0]
        return self.tool_registry.invoke(
            binding.tool_ref,
            {
                "task": state.task.model_dump(),
                "output_candidate": state.output_candidate,
                "policy_errors": policy_errors,
                "shared_context": state.shared_context,
            },
        )

    @staticmethod
    def _required_terms_rule(state: AgentState) -> list[str]:
        output = state.output_candidate or {}
        summary = str(output.get("summary", ""))
        required_terms = state.task.metadata.get("required_terms", [])
        errors: list[str] = []
        for term in required_terms:
            if term.lower() not in summary.lower():
                errors.append(f"missing required term: {term}")
        return errors
