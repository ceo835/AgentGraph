"""Print the current starter-agent roster for AgentGraph.

Usage:
    python examples/show_agents.py
    python examples/show_agents.py --include-critic
"""

from __future__ import annotations

import argparse
import json

from agentgraph import build_starter_agent_configs, describe_agent_configs


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the roster printer."""
    parser = argparse.ArgumentParser(
        description="Show the current AgentGraph starter-agent roster."
    )
    parser.add_argument(
        "--include-critic",
        action="store_true",
        help="Include the optional critic profile in the output.",
    )
    return parser.parse_args()


def main() -> int:
    """Render the starter-agent configuration as pretty JSON."""
    args = parse_args()
    agents = build_starter_agent_configs(include_critic=args.include_critic)
    description = describe_agent_configs(agents)
    print(json.dumps(description, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
