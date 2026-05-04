"""Run a minimal end-to-end AgentGraph demo backed by OpenAI.

Usage:
    python examples/run_openai_demo.py "Summarize LangGraph"

Environment:
    OPENAI_API_KEY is required and may be loaded from `.env`.
"""

from __future__ import annotations

import argparse
import json

from agentgraph import (
    build_demo_graph,
    build_demo_task,
    render_state_summary,
    run_task,
)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the demo script."""
    parser = argparse.ArgumentParser(description="Run the AgentGraph OpenAI demo.")
    parser.add_argument(
        "query",
        nargs="?",
        default="Summarize what LangGraph is used for.",
        help="Task description passed into AgentGraph.",
    )
    parser.add_argument(
        "--env-file",
        default=".env",
        help="Path to the local .env file containing OPENAI_API_KEY.",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Print stream envelopes before the final result.",
    )
    return parser.parse_args()


def main() -> int:
    """Run the CLI demo and print the final state summary."""
    args = parse_args()
    try:
        _, graph = build_demo_graph(env_file=args.env_file, workspace_root=".")
        task = build_demo_task(query=args.query)
        state, events = run_task(graph, task=task, stream=args.stream)
    except Exception as exc:
        print(f"Demo failed: {exc}")
        return 1

    if args.stream:
        for event in events:
            print(json.dumps(event, ensure_ascii=False, default=str))

    payload = render_state_summary(state)
    print(json.dumps(payload, indent=2, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
