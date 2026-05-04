"""Observability wrappers for AgentGraph crews and graphs."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any

from agentgraph.runtime import Crew, stream_envelopes


def _noop_traceable(
    *args: Any, **kwargs: Any
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        return func

    return decorator


try:
    import langsmith as _langsmith
except ImportError:  # pragma: no cover - optional tracing dependency
    _traceable = _noop_traceable
else:  # pragma: no cover - exercised only when langsmith is installed
    _traceable = _langsmith.traceable


class JSONStructuredLogger:
    """JSON logger for audit and stream events."""

    def __init__(self, name: str = "agentgraph") -> None:
        self._logger = logging.getLogger(name)

    def emit(
        self,
        *,
        event_type: str,
        payload: Any,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self._logger.info(
            json.dumps(
                {
                    "event_type": event_type,
                    "payload": payload,
                    "metadata": metadata or {},
                },
                ensure_ascii=False,
                default=str,
            )
        )

    def emit_audit_log(self, audit_log: list[dict[str, Any]]) -> None:
        for item in audit_log:
            self.emit(event_type="audit_log", payload=item)


@dataclass
class ObservedGraph:
    """Wrapper that adds tracing and structured logging to a compiled graph."""

    graph: Any
    logger: JSONStructuredLogger | None = None

    @_traceable(name="agentgraph.invoke")
    def invoke(self, input_value: Any, config: dict[str, Any], **kwargs: Any) -> Any:
        result = self.graph.invoke(input_value, config, **kwargs)
        if self.logger and isinstance(result, dict):
            self.logger.emit(
                event_type="invoke_result", payload=result, metadata=config
            )
            audit_log = result.get("audit_log")
            if isinstance(audit_log, list):
                self.logger.emit_audit_log(audit_log)
        return result

    @_traceable(name="agentgraph.stream")
    def stream(
        self,
        input_value: Any,
        config: dict[str, Any],
        *,
        stream_mode: str | list[str] = "updates",
        **kwargs: Any,
    ) -> Iterator[dict[str, Any]]:
        for envelope in stream_envelopes(
            self.graph,
            input_value,
            config,
            stream_mode=stream_mode,
            **kwargs,
        ):
            if self.logger:
                self.logger.emit(
                    event_type="stream_envelope",
                    payload=envelope["payload"],
                    metadata={
                        "thread_id": envelope["thread_id"],
                        "timestamp": envelope["timestamp"],
                        "event_type": envelope["event_type"],
                    },
                )
            yield envelope

    def get_state(self, *args: Any, **kwargs: Any) -> Any:
        return self.graph.get_state(*args, **kwargs)


def compile_with_observability(
    crew: Crew,
    *,
    checkpointer: Any = None,
    interrupt_before: list[str] | None = None,
    logger: JSONStructuredLogger | None = None,
) -> ObservedGraph:
    """Compile a crew and wrap the result with tracing and structured logging."""
    graph = crew.compile(checkpointer=checkpointer, interrupt_before=interrupt_before)
    return ObservedGraph(graph=graph, logger=logger)
