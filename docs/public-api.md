# Public API

This reference covers the public surface intended for external consumers in the
`v1.0.0` release.

## Contracts

### `AgentConfig`

Docstring summary:

> Declarative definition of an agent registered with the crew.

Use `AgentConfig` to describe:

- agent identity
- role and goal
- backstory and semantic capabilities
- tool bindings
- delegation and memory policies

### `TaskSpec`

Docstring summary:

> Declarative task contract passed into the runtime graph.

Use `TaskSpec` to declare:

- the task description
- the output schema
- HITL points
- retry behavior
- validation mode
- routing hints and metadata

### `ToolBinding`

Docstring summary:

> Associates an agent with an allowed tool and guardrails.

`ToolBinding` is where external consumers define:

- the tool reference
- permission level
- HITL requirements
- allowed side-effect boundary

### `MessageProtocol`

Docstring summary:

> Structured message envelope exchanged between runtime participants.

This is the stable envelope for inter-agent and human-feedback messages.

### `ToolResult`

Docstring summary:

> Canonical tool execution response.

All tools invoked through `ToolRegistry.invoke()` must return this shape.

### `AgentState`

Docstring summary:

> Shared runtime state passed through the LangGraph StateGraph.

External consumers should treat `AgentState` as:

- the checkpoint payload
- the replay payload
- the debugging payload

Important fields:

- `schema_version`
- `thread_id`
- `status`
- `task`
- `messages`
- `protocol_messages`
- `memory_refs`
- `sync_ticket_id`
- `errors`

## Runtime entry points

### `Crew`

`Crew` is the declarative root object for a runnable graph.

Typical responsibilities:

- hold registered agents
- hold tool and schema registries
- bind memory backends
- expose `compile()`

### `Crew.compile()`

Docstring summary:

> Compile the DSL declaration into a runnable LangGraph graph.

`Crew.compile()` accepts:

- an optional public LangGraph checkpointer
- an optional list of `interrupt_before` node names

It returns a compiled graph that can:

- `invoke()`
- `stream()`
- `get_state()`

## Public registries and adapters

- `AgentRegistry`
- `ToolRegistry`
- `IdempotentToolRegistry`
- `CheckpointBackend`
- `SQLiteCheckpointBackend`
- `MemorySyncService`
- `AsyncMemorySyncBackend`
- `DynamicValidatorAdapter`
- `PlannerAdapter`

## Stability notes

- Public consumers should depend on `agentgraph.__init__` exports.
- Runtime internals and private helper functions are not part of the stable API.
- Optional integrations may export `None` when the matching extra is not
  installed.
