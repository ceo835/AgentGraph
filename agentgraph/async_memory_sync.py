"""Deferred memory sync backend for non-blocking finalization."""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from threading import Lock
from typing import Any
from uuid import uuid4

from agentgraph.contracts import AgentState
from agentgraph.memory_adapters import MemorySyncService
from agentgraph.schema_evolution import migrate_agent_state_payload

_GLOBAL_TICKETS: dict[str, Future[list[dict[str, Any]]]] = {}
_GLOBAL_TICKETS_LOCK = Lock()


class AsyncMemorySyncBackend:
    """Memory backend facade that schedules persistence in the background."""

    def __init__(
        self,
        *,
        memory_sync_service: MemorySyncService,
        max_workers: int = 2,
        max_retries: int = 2,
    ) -> None:
        self.memory_sync_service = memory_sync_service
        self.max_retries = max_retries
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._futures: dict[str, Future[list[dict[str, Any]]]] = {}
        self._lock = Lock()

    def sync(
        self,
        *,
        thread_id: str,
        content: dict[str, Any],
        auto_sync: bool,
        link_generation: bool,
        versioning: bool,
    ) -> list[dict[str, Any]]:
        """Queue memory sync and return immediately with a ticket ref."""
        if not auto_sync:
            return []
        ticket_id = str(uuid4())
        future = self._executor.submit(
            self._run_with_retry,
            thread_id=thread_id,
            content=content,
            auto_sync=auto_sync,
            link_generation=link_generation,
            versioning=versioning,
        )
        with self._lock:
            self._futures[ticket_id] = future
        with _GLOBAL_TICKETS_LOCK:
            _GLOBAL_TICKETS[ticket_id] = future
        return [
            {
                "ref_type": "async_memory_ticket",
                "thread_id": thread_id,
                "ticket_id": ticket_id,
                "status": "pending",
            }
        ]

    def status(self, ticket_id: str) -> dict[str, Any]:
        """Inspect the background memory-sync ticket."""
        future = self._get_future(ticket_id)
        if not future.done():
            return {"ticket_id": ticket_id, "status": "pending"}
        try:
            refs = future.result()
        except Exception as exc:  # pragma: no cover - surfaced in tests via status
            return {"ticket_id": ticket_id, "status": "failed", "error": str(exc)}
        return {"ticket_id": ticket_id, "status": "completed", "refs": refs}

    def wait(self, ticket_id: str, timeout: float | None = None) -> dict[str, Any]:
        """Block until a given ticket completes."""
        future = self._get_future(ticket_id)
        refs = future.result(timeout=timeout)
        return {"ticket_id": ticket_id, "status": "completed", "refs": refs}

    def drain(self, timeout: float | None = None) -> list[dict[str, Any]]:
        """Wait for all queued sync operations."""
        results: list[dict[str, Any]] = []
        with self._lock:
            items = list(self._futures.items())
        for ticket_id, future in items:
            refs = future.result(timeout=timeout)
            results.append(
                {"ticket_id": ticket_id, "status": "completed", "refs": refs}
            )
        return results

    @staticmethod
    def status_for_ticket(ticket_id: str) -> dict[str, Any]:
        """Inspect a ticket without access to the creating backend instance."""
        with _GLOBAL_TICKETS_LOCK:
            future = _GLOBAL_TICKETS.get(ticket_id)
        if future is None:
            return {"ticket_id": ticket_id, "status": "unknown"}
        if not future.done():
            return {"ticket_id": ticket_id, "status": "pending"}
        try:
            refs = future.result()
        except Exception as exc:  # pragma: no cover - surfaced through status
            return {"ticket_id": ticket_id, "status": "failed", "error": str(exc)}
        return {"ticket_id": ticket_id, "status": "completed", "refs": refs}

    @classmethod
    def wait_for_ticket(
        cls,
        ticket_id: str,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        """Block on a globally tracked ticket."""
        with _GLOBAL_TICKETS_LOCK:
            future = _GLOBAL_TICKETS[ticket_id]
        refs = future.result(timeout=timeout)
        return {"ticket_id": ticket_id, "status": "completed", "refs": refs}

    def _get_future(self, ticket_id: str) -> Future[list[dict[str, Any]]]:
        if ticket_id in self._futures:
            return self._futures[ticket_id]
        with _GLOBAL_TICKETS_LOCK:
            return _GLOBAL_TICKETS[ticket_id]

    def _run_with_retry(
        self,
        *,
        thread_id: str,
        content: dict[str, Any],
        auto_sync: bool,
        link_generation: bool,
        versioning: bool,
    ) -> list[dict[str, Any]]:
        attempts = 0
        while True:
            attempts += 1
            try:
                return self.memory_sync_service.sync(
                    thread_id=thread_id,
                    content=content,
                    auto_sync=auto_sync,
                    link_generation=link_generation,
                    versioning=versioning,
                )
            except Exception:
                if attempts > self.max_retries:
                    raise


def ensure_resume_safe(
    graph: Any,
    *,
    thread_id: str,
    timeout: float | None = None,
) -> dict[str, Any]:
    """Ensure pending background memory sync is resolved before a resume."""
    state_snapshot = graph.get_state({"configurable": {"thread_id": thread_id}})
    values = getattr(state_snapshot, "values", state_snapshot)
    state = AgentState.model_validate(migrate_agent_state_payload(dict(values)))
    ticket_id = state.sync_ticket_id or _find_sync_ticket_id(state.memory_refs)
    if ticket_id is None:
        return {"status": "no_ticket"}
    status = AsyncMemorySyncBackend.status_for_ticket(ticket_id)
    if status["status"] != "pending":
        return status
    allow_stale = bool(state.task.metadata.get("allow_stale_memory_refs", False))
    if allow_stale:
        return {"ticket_id": ticket_id, "status": "stale_allowed"}
    return AsyncMemorySyncBackend.wait_for_ticket(ticket_id, timeout=timeout)


def _find_sync_ticket_id(memory_refs: list[dict[str, Any]]) -> str | None:
    for ref in reversed(memory_refs):
        if ref.get("ref_type") == "async_memory_ticket":
            return ref.get("ticket_id")
    return None
