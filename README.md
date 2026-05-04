# agentgraph

Declarative multi-agent DSL and desktop workflow layer built on top of
LangGraph.

## What is in this repository

This package contains:

- the `agentgraph` Python package
- starter agent configs in `agentgraph/config/`
- local demos in `examples/`
- tests in `tests/`
- lightweight package docs in `docs/`

It is intended to be published as its own repository and installed against
released Python packages, not against sibling folders from a larger monorepo.

## Installation

```bash
pip install -U agentgraph
```

For local development:

```bash
pip install -e .[ui]
```

## Quick Start

Run the local demo:

```bash
python examples/run_openai_demo.py "Summarize what LangGraph is used for."
```

Launch the Streamlit UI:

```bash
streamlit run examples/streamlit_app.py
```

## Local `.env`

`agentgraph` includes a minimal `.env` loader plus an OpenAI-backed research
tool adapter:

- `load_env_file(".env")`
- `register_openai_research_tool(tool_registry, env_path=".env")`

Supported variables:

- `OPENAI_API_KEY`
- `OPENAI_MODEL` optional, default `gpt-4.1-mini`
- `OPENAI_BASE_URL` optional, default `https://api.openai.com/v1`

Optional demo flags:

- `--env-file .env`
- `--stream`

The UI lets you:

- submit a task into the starter multi-agent crew
- inspect stream envelopes and the current `AgentState`
- resume HITL checkpoints with human feedback
- toggle planner, validation, and review settings without editing code

## Starter agent roster

The editable source of truth for the bundled agents is:

```txt
agentgraph/config/agents.yaml
```

That file defines:

- core agents: `coordinator`, `researcher`, `synthesizer`, `memory_curator`, `critic`
- specialists: `planner`, `validator`

The Python helpers `build_starter_agent_configs()` and
`register_starter_specialists()` load that YAML and resolve runtime placeholders
such as `research.search` and `validator.logic`.

## Publishing to GitHub

This directory is ready to be used as the root of a standalone repository.
Before pushing, initialize git here and set the final repository URL in
`pyproject.toml` if you want package metadata to point to GitHub.
