"""Public exports for the AgentGraph vertical slice."""

from importlib.util import find_spec

from agentgraph.autonomy_tools import AutonomyToolkitConfig, register_autonomy_toolkit
from agentgraph.checkpointing import CheckpointBackend
from agentgraph.contracts import (
    AgentConfig,
    AgentState,
    CapabilityDescriptor,
    DelegationPolicy,
    HITLPoint,
    MemoryPolicy,
    MessageProtocol,
    RetryPolicy,
    TaskSpec,
    ToolBinding,
    ToolResult,
)
from agentgraph.demo import (
    FileSystemActionReport,
    ResearchReport,
    build_demo_crew,
    build_demo_graph,
    build_demo_task,
    render_state_summary,
    resume_task,
    run_task,
)
from agentgraph.desktop_demo import (
    DesktopActionReport,
    DesktopCrewBundle,
    DesktopPlanExecution,
    build_desktop_demo_crew,
    build_desktop_demo_graph,
    build_desktop_task,
    export_desktop_artifact,
    resume_desktop_plan,
    resume_desktop_task,
    run_desktop_plan,
    run_desktop_task,
)
from agentgraph.desktop_enforcement import (
    DesktopEnforcedToolRegistry,
    RollbackManager,
    apply_desktop_enforcement_updates,
)
from agentgraph.desktop_loader import (
    DesktopPolicyBundle,
    DesktopToolsCatalog,
    DesktopToolSpec,
    DesktopWorkflowBundle,
    DesktopWorkflowSpec,
    default_desktop_executor_yaml_path,
    default_desktop_policies_yaml_path,
    default_desktop_tools_yaml_path,
    default_desktop_workflow_yaml_path,
    load_desktop_executor_config,
    load_desktop_policy_bundle,
    load_desktop_tools_catalog,
    load_desktop_workflow_bundle,
    register_desktop_executor,
)
from agentgraph.env import load_env_file
from agentgraph.openai_adapter import (
    OpenAIResearchTool,
    OpenAISettings,
    register_openai_research_tool,
)
from agentgraph.roster_loader import (
    default_agents_yaml_path,
    load_core_agent_configs,
    load_specialist_agent_config,
)
from agentgraph.runtime import (
    AgentRegistry,
    Crew,
    ExternalStorage,
    MemoryBackend,
    ToolRegistry,
    stream_envelopes,
)
from agentgraph.starter_agents import (
    build_starter_agent_configs,
    describe_agent_configs,
    register_starter_specialists,
)
from agentgraph.tool_caching import IdempotentToolRegistry

if find_spec("langgraph.checkpoint.sqlite") is not None:
    from agentgraph.checkpointing import SQLiteCheckpointBackend
else:  # pragma: no cover - exercised only without optional extras
    SQLiteCheckpointBackend = None

if find_spec("networkx") is not None:
    from agentgraph.memory_adapters import (
        GraphBackend,
        InMemoryVectorBackend,
        MemorySyncService,
        NetworkXGraphBackend,
        VectorBackend,
    )
else:  # pragma: no cover - exercised only without optional extras
    GraphBackend = None
    InMemoryVectorBackend = None
    MemorySyncService = None
    NetworkXGraphBackend = None
    VectorBackend = None

if find_spec("langsmith") is not None:
    from agentgraph.observability import (
        JSONStructuredLogger,
        ObservedGraph,
        compile_with_observability,
    )
else:  # pragma: no cover - exercised only without optional extras
    JSONStructuredLogger = None
    ObservedGraph = None
    compile_with_observability = None

__all__ = [
    "AgentConfig",
    "AgentRegistry",
    "AgentState",
    "AutonomyToolkitConfig",
    "CapabilityDescriptor",
    "CheckpointBackend",
    "Crew",
    "DesktopActionReport",
    "DesktopCrewBundle",
    "DesktopEnforcedToolRegistry",
    "DesktopPolicyBundle",
    "DesktopToolSpec",
    "DesktopToolsCatalog",
    "DesktopWorkflowBundle",
    "DesktopWorkflowSpec",
    "DelegationPolicy",
    "ExternalStorage",
    "FileSystemActionReport",
    "GraphBackend",
    "HITLPoint",
    "IdempotentToolRegistry",
    "InMemoryVectorBackend",
    "JSONStructuredLogger",
    "MemoryBackend",
    "MemorySyncService",
    "MemoryPolicy",
    "MessageProtocol",
    "NetworkXGraphBackend",
    "ObservedGraph",
    "OpenAIResearchTool",
    "OpenAISettings",
    "ResearchReport",
    "RollbackManager",
    "RetryPolicy",
    "SQLiteCheckpointBackend",
    "TaskSpec",
    "ToolBinding",
    "ToolRegistry",
    "ToolResult",
    "VectorBackend",
    "build_demo_crew",
    "build_demo_graph",
    "build_demo_task",
    "build_desktop_demo_crew",
    "build_desktop_demo_graph",
    "build_desktop_task",
    "DesktopPlanExecution",
    "export_desktop_artifact",
    "apply_desktop_enforcement_updates",
    "build_starter_agent_configs",
    "compile_with_observability",
    "default_agents_yaml_path",
    "default_desktop_executor_yaml_path",
    "default_desktop_policies_yaml_path",
    "default_desktop_tools_yaml_path",
    "default_desktop_workflow_yaml_path",
    "describe_agent_configs",
    "load_core_agent_configs",
    "load_desktop_executor_config",
    "load_desktop_policy_bundle",
    "load_desktop_tools_catalog",
    "load_desktop_workflow_bundle",
    "load_env_file",
    "load_specialist_agent_config",
    "render_state_summary",
    "register_autonomy_toolkit",
    "register_desktop_executor",
    "register_starter_specialists",
    "register_openai_research_tool",
    "resume_desktop_plan",
    "resume_desktop_task",
    "resume_task",
    "run_desktop_plan",
    "run_desktop_task",
    "run_task",
    "stream_envelopes",
]
