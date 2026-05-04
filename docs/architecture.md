# Architecture

This page describes the standalone `agentgraph` package architecture.

## Layers

- DSL layer
  Holds `AgentConfig`, `TaskSpec`, `ToolBinding`, `RetryPolicy`,
  `MemoryPolicy`, `ToolRegistry`, `AgentRegistry`, and `Crew`.
- Compilation boundary
  `Crew.compile()` turns declarations into runtime nodes, routing predicates,
  checkpoint hooks, interrupt gates, and stream wrappers.
- LangGraph engine
  Executes the compiled `StateGraph[AgentState]` using only public LangGraph
  primitives.

## Stable runtime path

```text
Interface
  -> TaskValidation
  -> Coordinator
  -> RegistryLookup
  -> SpecialistExecutor
  -> Synthesizer
  -> OutputValidation.Schema
  -> OutputValidation.Logic or Validator
  -> Memory
  -> Interface
```

## Extension points

- Dynamic specialists through `AgentRegistry`
- Tool execution through `ToolRegistry.invoke()`
- Planner fan-out through `Send`
- Checkpoint persistence through `CheckpointBackend`
- Memory persistence through pluggable vector, graph, and async sync backends
- Observability through the structured logger and tracing wrapper

## Release guarantees

- `AgentState` persists `schema_version=2`
- `sync_ticket_id` protects resume safety with async memory sync
- `artifacts` and `audit_log` are bounded to avoid state bloat
- optional integrations are guarded by lazy import safety
