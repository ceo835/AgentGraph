# AgentGraph

AgentGraph is a declarative multi-agent DSL that compiles onto the public
LangGraph runtime API.

This documentation set is intentionally small and release-focused:

- `Architecture` explains the layered design, lifecycle, and extension points.
- `Public API` documents the stable contracts expected by external consumers.
- `Release Artifacts` points to the reproducibility bundle for `v1.0.0`.

For a fast orientation path:

1. Read `Architecture`.
2. Read `Public API`.
3. Inspect `artifacts/v1.0.0/` before changing runtime or checkpoint behavior.
