"""OpenAI-backed tool adapters for AgentGraph."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any, cast
from urllib import error, parse, request

from agentgraph.contracts import SideEffectLevel, ToolResult
from agentgraph.env import load_env_file
from agentgraph.runtime import ToolRegistry


@dataclass
class OpenAISettings:
    """Configuration resolved from environment variables for OpenAI calls."""

    api_key: str
    model: str = "gpt-4.1-mini"
    web_search_model: str | None = None
    base_url: str = "https://api.openai.com/v1"
    timeout_seconds: float = 30.0

    @classmethod
    def from_env(
        cls,
        *,
        env_path: str | os.PathLike[str] = ".env",
        override: bool = False,
    ) -> OpenAISettings:
        """Load settings from `.env` and process environment."""
        load_env_file(env_path, override=override)
        api_key = cls._sanitize_api_key(os.environ.get("OPENAI_API_KEY", ""))
        if not api_key:
            raise ValueError("OPENAI_API_KEY is required to use the OpenAI adapter.")
        model = os.environ.get("OPENAI_MODEL", "gpt-4.1-mini").strip()
        web_search_model = os.environ.get("OPENAI_WEB_SEARCH_MODEL", "").strip() or None
        base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
        timeout_raw = os.environ.get("OPENAI_TIMEOUT_SECONDS", "30").strip() or "30"
        return cls(
            api_key=api_key,
            model=model,
            web_search_model=web_search_model,
            base_url=base_url.rstrip("/"),
            timeout_seconds=float(timeout_raw),
        )

    @staticmethod
    def _sanitize_api_key(raw_value: str) -> str:
        """Trim and normalize accidental non-ASCII noise in API key values."""
        stripped = raw_value.strip().strip("'\"").strip()
        ascii_only = stripped.encode("ascii", errors="ignore").decode("ascii")
        cleaned = "".join(ch for ch in ascii_only if 33 <= ord(ch) <= 126)
        return cleaned


class OpenAIResearchTool:
    """OpenAI-backed read-only research tool returning AgentGraph `ToolResult`."""

    _DASH_TRANSLATION = str.maketrans(
        {
            "\u2010": "-",
            "\u2011": "-",
            "\u2012": "-",
            "\u2013": "-",
            "\u2014": "-",
            "\u2212": "-",
        }
    )

    def __init__(self, settings: OpenAISettings, *, tool_ref: str) -> None:
        self.settings = settings
        self.tool_ref = tool_ref

    def __call__(self, args: dict[str, Any]) -> ToolResult:
        """Invoke the configured OpenAI model and normalize the response."""
        try:
            if self.tool_ref == "web.search":
                response_payload = self._request_web_search(args)
                normalized = self._normalize_web_search_payload(response_payload, args)
            else:
                response_payload = self._request_chat_completions(args)
                content = self._extract_chat_content(response_payload)
                parsed = json.loads(content)
                normalized = self._normalize_research_payload(parsed)
        except (ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
            metadata: dict[str, Any] = {}
            if self.tool_ref == "web.search":
                metadata = {"retriable": False}
            return ToolResult(
                success=False,
                error=str(exc),
                error_type="parse_error",
                metadata=metadata,
            )
        except error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            return ToolResult(
                success=False,
                error=body or str(exc),
                error_type="http_error",
                metadata={"status_code": exc.code},
            )
        except error.URLError as exc:
            return ToolResult(
                success=False,
                error=str(exc.reason),
                error_type="network_error",
            )
        verified = self._verify_direct_results(args, normalized)
        if isinstance(verified, ToolResult):
            return verified
        return ToolResult(
            success=True,
            data=verified,
            metadata={
                "model": self._resolve_model_name(),
                "api_mode": "responses" if self.tool_ref == "web.search" else "chat",
            },
        )

    def _request_chat_completions(self, args: dict[str, Any]) -> dict[str, Any]:
        prompt = self._build_user_prompt(args)
        output_language = self._resolve_output_language(args)
        payload = {
            "model": self.settings.model,
            "temperature": 0,
            "response_format": {"type": "json_object"},
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a research tool for a multi-agent system. "
                        "Return compact JSON with shape "
                        '{"results":[{"title":"...","snippet":"...","url":"https://..."}]}. '
                        "Include a direct source URL whenever one is available. "
                        f"Write every title and snippet strictly in {output_language}. "
                        "Do not switch languages unless explicitly asked."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
        }
        req = request.Request(
            url=f"{self.settings.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.settings.api_key}",
                "Content-Type": "application/json; charset=utf-8",
            },
            method="POST",
        )
        with request.urlopen(req, timeout=self.settings.timeout_seconds) as response:
            payload = json.loads(response.read().decode("utf-8"))
        if not isinstance(payload, dict):
            raise TypeError("OpenAI response payload must be a JSON object.")
        return cast(dict[str, Any], payload)

    def _request_web_search(self, args: dict[str, Any]) -> dict[str, Any]:
        task = args.get("task", {}) or {}
        metadata = task.get("metadata", {}) if isinstance(task, dict) else {}
        search_constraints = metadata.get("search_constraints", {})
        allowed_domains = []
        if isinstance(search_constraints, dict):
            raw_domains = search_constraints.get("allowed_domains") or []
            if isinstance(raw_domains, list):
                allowed_domains = [
                    str(domain).strip() for domain in raw_domains if str(domain).strip()
                ]

        tool_spec: dict[str, Any] = {
            "type": "web_search",
            "search_context_size": "medium",
        }
        if allowed_domains:
            tool_spec["filters"] = {"allowed_domains": allowed_domains}

        payload = {
            "model": self._resolve_model_name(),
            "tools": [tool_spec],
            "include": ["web_search_call.action.sources"],
            "input": self._build_user_prompt(args),
        }
        req = request.Request(
            url=f"{self.settings.base_url}/responses",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.settings.api_key}",
                "Content-Type": "application/json; charset=utf-8",
            },
            method="POST",
        )
        with request.urlopen(req, timeout=self.settings.timeout_seconds) as response:
            response_payload = json.loads(response.read().decode("utf-8"))
        if not isinstance(response_payload, dict):
            raise TypeError("OpenAI response payload must be a JSON object.")
        return cast(dict[str, Any], response_payload)

    def _resolve_model_name(self) -> str:
        if self.tool_ref == "web.search":
            return self.settings.web_search_model or self.settings.model
        return self.settings.model

    @staticmethod
    def _build_user_prompt(args: dict[str, Any]) -> str:
        query = OpenAIResearchTool._normalize_transport_text(
            str(args.get("query", "")).strip()
        )
        context_refs = args.get("context_refs", [])
        shared_context = args.get("shared_context", {})
        task = args.get("task", {}) or {}
        metadata = task.get("metadata", {}) if isinstance(task, dict) else {}
        output_language = OpenAIResearchTool._resolve_output_language(args)
        search_constraints = metadata.get("search_constraints", {})
        file_type = ""
        direct_links_only = False
        if isinstance(search_constraints, dict):
            file_type = str(search_constraints.get("prefer_file_type") or "").strip()
            direct_links_only = bool(search_constraints.get("direct_links_only"))
        extra_instruction = ""
        if file_type:
            direct_fragment = (
                " Use direct file URLs only."
                if direct_links_only
                else " Prefer direct file URLs."
            )
            extra_instruction = (
                f"\nSearch constraint: prioritize {file_type.upper()} results."
                f"{direct_fragment}"
            )
        return (
            "Research the following task and respond with JSON only.\n"
            f"Query: {query}\n"
            f"Context refs: {context_refs}\n"
            f"Shared context: {shared_context}\n"
            f"Output language: {output_language}\n"
            f"{extra_instruction}"
            "Return 1-5 concise result items. "
            "Use only real source URLs discovered during search; never invent URLs."
        )

    def _normalize_web_search_payload(
        self,
        payload: dict[str, Any],
        args: dict[str, Any],
    ) -> dict[str, Any]:
        source_urls = self._extract_web_search_source_urls(payload)
        if not source_urls:
            source_urls = self._extract_web_search_annotation_urls(payload)
        content = self._extract_responses_content(payload)
        if content and not source_urls:
            source_urls = self._extract_urls_from_text(content)
        if content:
            query = str(args.get("query", "")).strip()
            try:
                parsed = json.loads(content)
            except json.JSONDecodeError:
                if source_urls:
                    return self._build_results_from_sources(
                        source_urls=source_urls,
                        query=query,
                        summary=content,
                    )
                return {
                    "results": [
                        {
                            "title": "Web search result",
                            "snippet": content,
                        }
                    ]
                }
            try:
                normalized = self._normalize_research_payload(parsed)
            except ValueError:
                if source_urls:
                    return self._build_results_from_sources(
                        source_urls=source_urls,
                        query=query,
                        summary=content,
                    )
                return {
                    "results": [
                        {
                            "title": "Web search result",
                            "snippet": content,
                        }
                    ]
                }
            if not normalized.get("results"):
                if source_urls:
                    return self._build_results_from_sources(
                        source_urls=source_urls,
                        query=query,
                        summary=content,
                    )
                return {
                    "results": [
                        {
                            "title": "Web search result",
                            "snippet": content,
                        }
                    ]
                }
            return self._reconcile_results_with_sources(
                normalized,
                source_urls=source_urls,
                query=query,
            )

        if source_urls:
            return self._build_results_from_sources(
                source_urls=source_urls,
                query=str(args.get("query", "")).strip(),
            )

        raise ValueError(
            "OpenAI web search response did not contain any usable results."
        )

    def _verify_direct_results(
        self,
        args: dict[str, Any],
        normalized: dict[str, Any],
    ) -> dict[str, Any] | ToolResult:
        task = args.get("task", {}) or {}
        metadata = task.get("metadata", {}) if isinstance(task, dict) else {}
        search_constraints = metadata.get("search_constraints", {})
        if not isinstance(search_constraints, dict):
            return normalized
        direct_links_only = bool(search_constraints.get("direct_links_only"))
        file_type = (
            str(search_constraints.get("prefer_file_type") or "").strip().lower()
        )
        if not direct_links_only and not file_type:
            return normalized

        verified_results: list[dict[str, str]] = []
        attempted_urls: list[str] = []
        for item in normalized.get("results", []):
            if not isinstance(item, dict):
                continue
            url = str(item.get("url", "")).strip()
            url = self._normalize_url(url)
            if not url.startswith(("http://", "https://")):
                continue
            attempted_urls.append(url)
            probe = self._probe_result_url(url)
            if not probe["ok"]:
                continue
            if file_type == "pdf":
                content_type = str(probe.get("content_type", "")).lower()
                if not (url.lower().endswith(".pdf") or "pdf" in content_type):
                    continue
            verified_results.append(item)

        if verified_results:
            return {"results": verified_results[:5]}

        if direct_links_only or file_type:
            target = file_type.upper() if file_type else "direct"
            return ToolResult(
                success=False,
                error=f"No verified {target} URLs found in search results.",
                error_type="no_results",
                metadata={
                    "attempted_urls": attempted_urls,
                    "retriable": False,
                },
            )
        return normalized

    @staticmethod
    def _extract_chat_content(payload: dict[str, Any]) -> str:
        choices = payload["choices"]
        message = choices[0]["message"]
        content = message["content"]
        if not isinstance(content, str):
            raise TypeError("OpenAI response content must be a string.")
        return content

    @staticmethod
    def _extract_responses_content(payload: dict[str, Any]) -> str:
        output = payload.get("output")
        if not isinstance(output, list):
            return ""
        for item in output:
            if not isinstance(item, dict) or item.get("type") != "message":
                continue
            content_parts = item.get("content")
            if not isinstance(content_parts, list):
                continue
            for part in content_parts:
                if (
                    isinstance(part, dict)
                    and part.get("type") == "output_text"
                    and isinstance(part.get("text"), str)
                ):
                    return cast(str, part["text"])
        return ""

    @staticmethod
    def _extract_web_search_source_urls(payload: dict[str, Any]) -> list[str]:
        output = payload.get("output")
        if not isinstance(output, list):
            return []
        urls: list[str] = []
        for item in output:
            if not isinstance(item, dict) or item.get("type") != "web_search_call":
                continue
            action = item.get("action")
            if not isinstance(action, dict):
                continue
            sources = action.get("sources")
            if not isinstance(sources, list):
                continue
            for source in sources:
                if not isinstance(source, dict):
                    continue
                url = str(source.get("url", "")).strip()
                url = OpenAIResearchTool._normalize_url(url)
                if url.startswith(("http://", "https://")) and url not in urls:
                    urls.append(url)
        return urls

    @staticmethod
    def _extract_web_search_annotation_urls(payload: dict[str, Any]) -> list[str]:
        output = payload.get("output")
        if not isinstance(output, list):
            return []
        urls: list[str] = []
        for item in output:
            if not isinstance(item, dict) or item.get("type") != "message":
                continue
            content_parts = item.get("content")
            if not isinstance(content_parts, list):
                continue
            for part in content_parts:
                if not isinstance(part, dict):
                    continue
                annotations = part.get("annotations")
                if not isinstance(annotations, list):
                    continue
                for annotation in annotations:
                    if not isinstance(annotation, dict):
                        continue
                    if annotation.get("type") == "url_citation":
                        url = str(annotation.get("url", "")).strip()
                        if not url and isinstance(annotation.get("url_citation"), dict):
                            url = str(annotation["url_citation"].get("url", "")).strip()
                        url = OpenAIResearchTool._normalize_url(url)
                        if url.startswith(("http://", "https://")) and url not in urls:
                            urls.append(url)
        return urls

    @staticmethod
    def _extract_urls_from_text(text: str) -> list[str]:
        urls: list[str] = []
        for match in re.findall(r"https?://[^\s\]\)>\",]+", text):
            url = OpenAIResearchTool._normalize_url(match.rstrip(".,;:"))
            if url.startswith(("http://", "https://")) and url not in urls:
                urls.append(url)
        return urls

    @classmethod
    def _normalize_transport_text(cls, text: str) -> str:
        return text.translate(cls._DASH_TRANSLATION)

    @classmethod
    def _normalize_url(cls, url: str) -> str:
        return cls._normalize_transport_text(url).strip()

    @classmethod
    def _reconcile_results_with_sources(
        cls,
        normalized: dict[str, Any],
        *,
        source_urls: list[str],
        query: str,
    ) -> dict[str, Any]:
        if not source_urls:
            return normalized
        remaining_urls = list(source_urls)
        reconciled: list[dict[str, str]] = []
        for item in normalized.get("results", []):
            if not isinstance(item, dict):
                continue
            result = dict(item)
            current_url = str(result.get("url", "")).strip()
            matched_url = cls._match_source_url(current_url, remaining_urls)
            if matched_url is None and remaining_urls:
                matched_url = remaining_urls.pop(0)
            elif matched_url is not None:
                remaining_urls.remove(matched_url)
            if matched_url:
                result["url"] = matched_url
            if not str(result.get("snippet", "")).strip():
                result["snippet"] = f"Web search result for query: {query}"
            if not str(result.get("title", "")).strip():
                result["title"] = cls._infer_title_from_url(result["url"])
            reconciled.append(result)

        if reconciled:
            return {"results": reconciled[:5]}

        return cls._build_results_from_sources(source_urls=source_urls, query=query)

    @classmethod
    def _build_results_from_sources(
        cls,
        *,
        source_urls: list[str],
        query: str,
        summary: str | None = None,
    ) -> dict[str, Any]:
        snippet = (
            summary.strip() if isinstance(summary, str) and summary.strip() else ""
        )
        return {
            "results": [
                {
                    "title": cls._infer_title_from_url(url),
                    "snippet": snippet or f"Web search result for query: {query}",
                    "url": url,
                }
                for url in source_urls[:5]
            ]
        }

    @staticmethod
    def _match_source_url(url: str, source_urls: list[str]) -> str | None:
        if not url:
            return None
        if url in source_urls:
            return url
        normalized_url = url.rstrip("/").lower()
        for source_url in source_urls:
            if source_url.rstrip("/").lower() == normalized_url:
                return source_url
        return None

    @staticmethod
    def _infer_title_from_url(url: str) -> str:
        parsed = parse.urlparse(url)
        normalized_path = parsed.path.rstrip("/")
        filename = normalized_path.rsplit("/", 1)[-1] if normalized_path else ""
        return filename or parsed.netloc or "Search result"

    def _probe_result_url(self, url: str) -> dict[str, Any]:
        head_request = request.Request(
            url,
            headers={"User-Agent": "AgentGraph/1.0"},
            method="HEAD",
        )
        try:
            with request.urlopen(
                head_request, timeout=self.settings.timeout_seconds
            ) as response:
                return {
                    "ok": True,
                    "status_code": getattr(response, "status", 200),
                    "content_type": response.headers.get("Content-Type", ""),
                }
        except error.HTTPError as exc:
            if exc.code not in {405, 501}:
                return {"ok": False, "status_code": exc.code}
        except error.URLError:
            return {"ok": False}

        get_request = request.Request(
            url,
            headers={"User-Agent": "AgentGraph/1.0"},
            method="GET",
        )
        try:
            with request.urlopen(
                get_request, timeout=self.settings.timeout_seconds
            ) as response:
                return {
                    "ok": True,
                    "status_code": getattr(response, "status", 200),
                    "content_type": response.headers.get("Content-Type", ""),
                }
        except error.HTTPError as exc:
            return {"ok": False, "status_code": exc.code}
        except error.URLError:
            return {"ok": False}

    @staticmethod
    def _resolve_output_language(args: dict[str, Any]) -> str:
        task = args.get("task", {}) or {}
        metadata = task.get("metadata", {}) if isinstance(task, dict) else {}
        locale = str(args.get("locale") or metadata.get("locale") or "").strip().lower()
        if locale in {"ru", "ru-ru"}:
            return "Russian"
        if locale in {"en", "en-us", "en-gb"}:
            return "English"
        query = str(args.get("query", ""))
        if any("а" <= ch.lower() <= "я" or ch.lower() == "ё" for ch in query):
            return "Russian"
        return "English"

    @classmethod
    def _normalize_research_payload(cls, payload: Any) -> dict[str, Any]:
        """Normalize model output to the stable `{"results": [...]}` shape."""
        if isinstance(payload, dict):
            direct_results = cls._normalize_results(payload.get("results"))
            if direct_results:
                return {"results": direct_results}

            result_like = cls._coerce_result_item(payload)
            if result_like is not None:
                return {"results": [result_like]}

            nested_results: list[dict[str, str]] = []
            for value in payload.values():
                nested_results.extend(cls._normalize_results(value))
            if nested_results:
                return {"results": nested_results[:5]}

        normalized_results = cls._normalize_results(payload)
        if normalized_results:
            return {"results": normalized_results[:5]}

        raise ValueError("OpenAI response did not contain any usable research results.")

    @classmethod
    def _normalize_results(cls, value: Any) -> list[dict[str, str]]:
        """Extract normalized result items from a list-like or scalar payload."""
        if value is None:
            return []
        if isinstance(value, list):
            normalized: list[dict[str, str]] = []
            for item in value:
                result = cls._coerce_result_item(item)
                if result is not None:
                    normalized.append(result)
            return normalized[:5]

        scalar_result = cls._coerce_result_item(value)
        if scalar_result is not None:
            return [scalar_result]
        return []

    @staticmethod
    def _coerce_result_item(item: Any) -> dict[str, str] | None:
        """Coerce loosely structured model output into `{title, snippet}`."""
        if isinstance(item, str):
            snippet = item.strip()
            if snippet:
                return {"title": "Generated result", "snippet": snippet}
            return None

        if not isinstance(item, dict):
            return None

        title = ""
        snippet = ""
        url = ""

        for key in ("title", "source", "name", "heading", "label"):
            candidate = str(item.get(key, "")).strip()
            if candidate:
                title = candidate
                break

        for key in ("snippet", "summary", "text", "content", "answer", "description"):
            candidate = str(item.get(key, "")).strip()
            if candidate:
                snippet = candidate
                break

        for key in ("url", "link", "href", "source_url", "sourceUrl"):
            candidate = str(item.get(key, "")).strip()
            if candidate:
                url = candidate
                break

        if not snippet:
            return None

        if not title:
            title = "Generated result"
        result = {"title": title, "snippet": snippet}
        if url:
            result["url"] = url
        return result


def register_openai_research_tool(
    tool_registry: ToolRegistry,
    *,
    tool_ref: str = "research.search",
    env_path: str | os.PathLike[str] = ".env",
    override_env: bool = False,
    settings: OpenAISettings | None = None,
) -> OpenAISettings:
    """Register an OpenAI-backed research tool in the shared `ToolRegistry`."""
    resolved_settings = settings or OpenAISettings.from_env(
        env_path=env_path,
        override=override_env,
    )
    tool_registry.register(
        tool_ref=tool_ref,
        description="OpenAI-backed research and synthesis lookup.",
        side_effect_level=SideEffectLevel.READ_ONLY,
        handler=OpenAIResearchTool(resolved_settings, tool_ref=tool_ref),
    )
    return resolved_settings
