"""Idempotent tool-registry wrappers for AgentGraph."""

from __future__ import annotations

import copy
import hashlib
import json
import time
from dataclasses import dataclass
from typing import Any

from agentgraph.contracts import SideEffectLevel, ToolResult
from agentgraph.runtime import ToolRegistry


@dataclass
class _CachedToolResult:
    expires_at: float
    result: ToolResult


class IdempotentToolRegistry(ToolRegistry):
    """Tool registry with hash-based TTL caching for idempotent calls."""

    def __init__(
        self,
        *,
        ttl_seconds: float = 300.0,
        cache_write_tools: bool = False,
    ) -> None:
        super().__init__()
        self.ttl_seconds = ttl_seconds
        self.cache_write_tools = cache_write_tools
        self._cache: dict[str, _CachedToolResult] = {}

    def invoke(self, tool_ref: str, args: dict[str, Any]) -> ToolResult:
        self.purge_expired()
        definition = self.get(tool_ref)
        cacheable = self.cache_write_tools or definition.side_effect_level in {
            SideEffectLevel.NONE,
            SideEffectLevel.READ_ONLY,
        }
        cache_key = self._make_cache_key(tool_ref, args)
        if cacheable and cache_key in self._cache:
            cached = copy.deepcopy(self._cache[cache_key].result)
            cached.metadata = {**cached.metadata, "cache_hit": True}
            return cached

        result = super().invoke(tool_ref, args)
        if cacheable and result.success:
            cached_result = copy.deepcopy(result)
            cached_result.metadata = {**cached_result.metadata, "cache_hit": False}
            self._cache[cache_key] = _CachedToolResult(
                expires_at=time.time() + self.ttl_seconds,
                result=cached_result,
            )
        return result

    def purge_expired(self) -> None:
        now = time.time()
        expired = [key for key, value in self._cache.items() if value.expires_at <= now]
        for key in expired:
            self._cache.pop(key, None)

    @staticmethod
    def _make_cache_key(tool_ref: str, args: dict[str, Any]) -> str:
        serialized = json.dumps(args, sort_keys=True, default=str, ensure_ascii=False)
        digest = hashlib.sha256(f"{tool_ref}:{serialized}".encode()).hexdigest()
        return digest
