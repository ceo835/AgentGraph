"""Vector, graph, and memory-sync adapters for AgentGraph."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Any
from uuid import uuid4

try:
    import networkx as nx
except ImportError:  # pragma: no cover - optional memory dependency
    nx = None

from agentgraph.runtime import MemoryBackend


def _token_counts(text: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    cleaned = "".join(ch.lower() if ch.isalnum() else " " for ch in text)
    for token in cleaned.split():
        counts[token] = counts.get(token, 0) + 1
    return counts


def _cosine_similarity(left: dict[str, int], right: dict[str, int]) -> float:
    if not left or not right:
        return 0.0
    numerator = sum(left[token] * right.get(token, 0) for token in left)
    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return numerator / (left_norm * right_norm)


class VectorBackend(ABC):
    """Embeddings-like similarity interface."""

    @abstractmethod
    def upsert(
        self,
        *,
        thread_id: str,
        document_id: str,
        text: str,
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        """Persist vectorized content."""

    @abstractmethod
    def similarity_search(
        self,
        *,
        thread_id: str,
        query: str,
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        """Query documents by semantic similarity."""


class InMemoryVectorBackend(VectorBackend):
    """Simple token-vector backend for local MVP and tests."""

    def __init__(self) -> None:
        self._documents: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)

    def upsert(
        self,
        *,
        thread_id: str,
        document_id: str,
        text: str,
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        embedding = _token_counts(text)
        record = {
            "document_id": document_id,
            "text": text,
            "embedding": embedding,
            "metadata": metadata,
        }
        existing = self._documents[thread_id]
        self._documents[thread_id] = [
            item for item in existing if item["document_id"] != document_id
        ] + [record]
        return {
            "ref_type": "vector",
            "thread_id": thread_id,
            "document_id": document_id,
        }

    def similarity_search(
        self,
        *,
        thread_id: str,
        query: str,
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        query_embedding = _token_counts(query)
        results: list[dict[str, Any]] = []
        for record in self._documents[thread_id]:
            score = _cosine_similarity(query_embedding, record["embedding"])
            results.append(
                {
                    "document_id": record["document_id"],
                    "score": score,
                    "metadata": record["metadata"],
                }
            )
        results.sort(key=lambda item: item["score"], reverse=True)
        return results[:limit]


class GraphBackend(ABC):
    """Relationship graph interface for memory links."""

    @abstractmethod
    def upsert_links(
        self,
        *,
        thread_id: str,
        source: str,
        links: list[str],
        metadata: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Persist source-to-link edges."""

    @abstractmethod
    def neighbors(self, *, thread_id: str, node: str) -> list[str]:
        """Return neighboring nodes for a thread-local subgraph."""


class NetworkXGraphBackend(GraphBackend):
    """NetworkX-backed graph backend with a Neo4j-compatible interface shape."""

    def __init__(self) -> None:
        if nx is None:  # pragma: no cover - runtime dependency check
            raise ImportError(
                "networkx is not available. Install the `memory` extra to use "
                "NetworkXGraphBackend."
            )
        self._graphs: defaultdict[str, nx.MultiDiGraph] = defaultdict(nx.MultiDiGraph)

    def upsert_links(
        self,
        *,
        thread_id: str,
        source: str,
        links: list[str],
        metadata: dict[str, Any],
    ) -> list[dict[str, Any]]:
        graph = self._graphs[thread_id]
        refs: list[dict[str, Any]] = []
        graph.add_node(source, thread_id=thread_id)
        for link in links:
            graph.add_node(link, thread_id=thread_id)
            graph.add_edge(source, link, relation="references", **metadata)
            refs.append(
                {
                    "ref_type": "graph",
                    "thread_id": thread_id,
                    "source": source,
                    "target": link,
                }
            )
        return refs

    def neighbors(self, *, thread_id: str, node: str) -> list[str]:
        graph = self._graphs[thread_id]
        if node not in graph:
            return []
        return list(graph.neighbors(node))


class MemorySyncService:
    """Composes note versioning with optional vector and graph backends."""

    def __init__(
        self,
        *,
        base_memory_backend: MemoryBackend | None = None,
        vector_backend: VectorBackend | None = None,
        graph_backend: GraphBackend | None = None,
        enable_vector_sync: bool = True,
        enable_graph_sync: bool = True,
    ) -> None:
        self.base_memory_backend = base_memory_backend or MemoryBackend()
        self.vector_backend = vector_backend
        self.graph_backend = graph_backend
        self.enable_vector_sync = enable_vector_sync
        self.enable_graph_sync = enable_graph_sync

    def sync(
        self,
        *,
        thread_id: str,
        content: dict[str, Any],
        auto_sync: bool,
        link_generation: bool,
        versioning: bool,
    ) -> list[dict[str, Any]]:
        refs = self.base_memory_backend.sync(
            thread_id=thread_id,
            content=content,
            auto_sync=auto_sync,
            link_generation=link_generation,
            versioning=versioning,
        )
        if not auto_sync:
            return refs

        document_id = str(uuid4())
        text = str(content)
        if self.vector_backend is not None and self.enable_vector_sync:
            refs.append(
                self.vector_backend.upsert(
                    thread_id=thread_id,
                    document_id=document_id,
                    text=text,
                    metadata={"thread_id": thread_id},
                )
            )
        if self.graph_backend is not None and self.enable_graph_sync:
            links = _extract_links(refs) if link_generation else []
            refs.extend(
                self.graph_backend.upsert_links(
                    thread_id=thread_id,
                    source=document_id,
                    links=links,
                    metadata={"thread_id": thread_id},
                )
            )
        return refs


def _extract_links(refs: list[dict[str, Any]]) -> list[str]:
    links: list[str] = []
    for ref in refs:
        for link in ref.get("links", []):
            if link not in links:
                links.append(link)
    return links
