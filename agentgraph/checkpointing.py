"""Checkpoint backend adapters for AgentGraph."""

from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import AbstractContextManager
from typing import Any

from langgraph.checkpoint.base import BaseCheckpointSaver, CheckpointTuple

try:
    from langgraph.checkpoint.sqlite import SqliteSaver
except ImportError:  # pragma: no cover - optional adapter dependency
    SqliteSaver = None  # type: ignore[assignment]


class CheckpointBackend(ABC):
    """Abstract backend for checkpoint persistence."""

    @abstractmethod
    def as_langgraph_checkpointer(self) -> BaseCheckpointSaver[Any]:
        """Return a checkpointer object compatible with `StateGraph.compile()`."""

    @abstractmethod
    def save(
        self,
        *,
        thread_id: str,
        checkpoint: dict[str, Any],
        checkpoint_ns: str = "",
        checkpoint_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        new_versions: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Persist a checkpoint via the public checkpointer API."""

    @abstractmethod
    def load(
        self,
        *,
        thread_id: str,
        checkpoint_id: str | None = None,
        checkpoint_ns: str = "",
    ) -> CheckpointTuple | None:
        """Load a checkpoint tuple for a thread."""

    @abstractmethod
    def list(
        self,
        *,
        thread_id: str,
        checkpoint_ns: str = "",
        limit: int | None = None,
    ) -> list[CheckpointTuple]:
        """List persisted checkpoints for a thread."""

    @abstractmethod
    def delete(self, *, thread_id: str) -> None:
        """Delete all checkpoints for a thread."""


class SQLiteCheckpointBackend(AbstractContextManager, CheckpointBackend):
    """SQLite-backed checkpointer built on the public `SqliteSaver` API."""

    def __init__(self, conn_string: str) -> None:
        self.conn_string = conn_string
        self._manager: AbstractContextManager[BaseCheckpointSaver[Any]] | None = None
        self._saver: BaseCheckpointSaver[Any] | None = None

    def __enter__(self) -> SQLiteCheckpointBackend:
        if SqliteSaver is None:  # pragma: no cover - runtime dependency check
            raise ImportError(
                "langgraph.checkpoint.sqlite.SqliteSaver is not available. "
                "Add `libs/checkpoint-sqlite` to PYTHONPATH or install langgraph-checkpoint-sqlite."
            )
        self._manager = SqliteSaver.from_conn_string(self.conn_string)
        self._saver = self._manager.__enter__()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> bool | None:
        if self._manager is None:
            return None
        result = self._manager.__exit__(exc_type, exc_value, traceback)
        self._manager = None
        self._saver = None
        return result

    def as_langgraph_checkpointer(self) -> BaseCheckpointSaver[Any]:
        if self._saver is None:
            raise RuntimeError("SQLiteCheckpointBackend must be entered before use.")
        return self._saver

    def save(
        self,
        *,
        thread_id: str,
        checkpoint: dict[str, Any],
        checkpoint_ns: str = "",
        checkpoint_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        new_versions: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        saver = self.as_langgraph_checkpointer()
        config: dict[str, Any] = {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
            }
        }
        if checkpoint_id is not None:
            config["configurable"]["checkpoint_id"] = checkpoint_id
        return saver.put(
            config,
            checkpoint,
            metadata or {},
            new_versions or {},
        )

    def load(
        self,
        *,
        thread_id: str,
        checkpoint_id: str | None = None,
        checkpoint_ns: str = "",
    ) -> CheckpointTuple | None:
        saver = self.as_langgraph_checkpointer()
        config: dict[str, Any] = {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
            }
        }
        if checkpoint_id is not None:
            config["configurable"]["checkpoint_id"] = checkpoint_id
        return saver.get_tuple(config)

    def list(
        self,
        *,
        thread_id: str,
        checkpoint_ns: str = "",
        limit: int | None = None,
    ) -> list[CheckpointTuple]:
        saver = self.as_langgraph_checkpointer()
        config = {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
            }
        }
        return list(saver.list(config, limit=limit))

    def delete(self, *, thread_id: str) -> None:
        saver = self.as_langgraph_checkpointer()
        saver.delete_thread(thread_id)
