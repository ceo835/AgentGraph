"""Autonomy-oriented tools for AgentGraph starter agents."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from collections import Counter, deque
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from hashlib import sha256
from html.parser import HTMLParser
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any
from urllib import error, parse, request

from agentgraph.contracts import SideEffectLevel, ToolResult
from agentgraph.runtime import ToolRegistry


@dataclass
class AutonomyToolkitConfig:
    """Local configuration and safety boundaries for autonomous tools."""

    workspace_root: Path
    cache_dir: Path
    workspace_actions_dir: Path
    drafts_dir: Path
    quarantine_dir: Path
    vector_index_path: Path
    allowed_read_roots: list[Path]
    allowed_write_roots: list[Path]
    timeout_seconds: float = 15.0

    @classmethod
    def from_workspace(
        cls,
        workspace_root: str | os.PathLike[str] = ".",
    ) -> AutonomyToolkitConfig:
        """Build a default sandbox rooted in the provided workspace."""
        root = Path(workspace_root).resolve()
        cache_dir = root / ".agentgraph_cache"
        workspace_actions_dir = cache_dir / "workspace"
        drafts_dir = cache_dir / "drafts"
        quarantine_dir = cache_dir / "quarantine"
        vector_index_path = cache_dir / "vector_index.jsonl"
        for directory in (cache_dir, workspace_actions_dir, drafts_dir, quarantine_dir):
            directory.mkdir(parents=True, exist_ok=True)
        return cls(
            workspace_root=root,
            cache_dir=cache_dir,
            workspace_actions_dir=workspace_actions_dir,
            drafts_dir=drafts_dir,
            quarantine_dir=quarantine_dir,
            vector_index_path=vector_index_path,
            allowed_read_roots=[root],
            allowed_write_roots=[
                cache_dir,
                workspace_actions_dir,
                drafts_dir,
                quarantine_dir,
            ],
        )


class _HTMLSummaryParser(HTMLParser):
    """Minimal HTML parser for fetch and extract tools."""

    def __init__(self) -> None:
        super().__init__()
        self.in_title = False
        self.title_chunks: list[str] = []
        self.text_chunks: list[str] = []
        self.links: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() == "title":
            self.in_title = True
        if tag.lower() == "a":
            href = dict(attrs).get("href")
            if href:
                self.links.append(href)

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() == "title":
            self.in_title = False

    def handle_data(self, data: str) -> None:
        cleaned = " ".join(data.split())
        if not cleaned:
            return
        if self.in_title:
            self.title_chunks.append(cleaned)
        self.text_chunks.append(cleaned)

    def as_dict(self) -> dict[str, Any]:
        text = " ".join(self.text_chunks).strip()
        return {
            "title": " ".join(self.title_chunks).strip(),
            "text": text,
            "links": self.links,
            "word_count": len(text.split()),
        }


def register_autonomy_toolkit(
    tool_registry: ToolRegistry,
    *,
    workspace_root: str | os.PathLike[str] = ".",
) -> AutonomyToolkitConfig:
    """Register the starter autonomy toolkit in the shared `ToolRegistry`."""
    config = AutonomyToolkitConfig.from_workspace(workspace_root)
    handlers = _AutonomyHandlers(config=config)
    for tool_ref, side_effect_level, handler, description in handlers.tool_specs():
        tool_registry.register(
            tool_ref=tool_ref,
            description=description,
            side_effect_level=side_effect_level,
            handler=handler,
        )
    return config


class _AutonomyHandlers:
    """Collection of simple, safe autonomous tool handlers."""

    def __init__(self, *, config: AutonomyToolkitConfig) -> None:
        self.config = config

    def tool_specs(self) -> list[tuple[str, SideEffectLevel, Any, str]]:
        return [
            (
                "web.fetch",
                SideEffectLevel.READ_ONLY,
                self.web_fetch,
                "Fetch a single web page.",
            ),
            (
                "web.crawl",
                SideEffectLevel.READ_ONLY,
                self.web_crawl,
                "Crawl a small set of linked pages.",
            ),
            (
                "web.download",
                SideEffectLevel.EXTERNAL_WRITE,
                self.web_download,
                "Download a file into quarantine inside the tool sandbox.",
            ),
            (
                "parse.extract",
                SideEffectLevel.NONE,
                self.parse_extract,
                "Extract title, text, and links from HTML or text.",
            ),
            (
                "citation.verify",
                SideEffectLevel.READ_ONLY,
                self.citation_verify,
                "Verify that citation URLs are reachable.",
            ),
            (
                "fs.list_dir",
                SideEffectLevel.READ_ONLY,
                self.fs_list_dir,
                "List files inside the allowed workspace.",
            ),
            (
                "fs.read_file",
                SideEffectLevel.READ_ONLY,
                self.fs_read_file,
                "Read a text file from the allowed workspace.",
            ),
            (
                "fs.create_dir",
                SideEffectLevel.EXTERNAL_WRITE,
                self.fs_create_dir,
                "Create a directory inside the tool sandbox.",
            ),
            (
                "fs.move",
                SideEffectLevel.EXTERNAL_WRITE,
                self.fs_move,
                "Move a file or directory inside the tool sandbox.",
            ),
            (
                "fs.delete",
                SideEffectLevel.DESTRUCTIVE,
                self.fs_delete,
                "Delete a file or directory by quarantining it inside the tool sandbox.",
            ),
            (
                "fs.write_file",
                SideEffectLevel.EXTERNAL_WRITE,
                self.fs_write_file,
                "Write a file inside the tool sandbox.",
            ),
            (
                "fs.export",
                SideEffectLevel.EXTERNAL_WRITE,
                self.fs_export,
                "Export a sandbox file or directory into an approved user path.",
            ),
            (
                "package.install",
                SideEffectLevel.EXTERNAL_WRITE,
                self.package_install,
                "Prepare or execute a package installation through a trusted manager.",
            ),
            (
                "package.update",
                SideEffectLevel.EXTERNAL_WRITE,
                self.package_update,
                "Prepare or execute a package update through a trusted manager.",
            ),
            (
                "app.launch",
                SideEffectLevel.READ_ONLY,
                self.app_launch,
                "Resolve and optionally launch an allowed desktop application.",
            ),
            (
                "cache.store",
                SideEffectLevel.EXTERNAL_WRITE,
                self.cache_store,
                "Persist structured intermediate data in the sandbox cache.",
            ),
            (
                "history.query",
                SideEffectLevel.READ_ONLY,
                self.history_query,
                "Search cached tool outputs for prior matches.",
            ),
            (
                "dependency.analyze",
                SideEffectLevel.NONE,
                self.dependency_analyze,
                "Infer dependencies and ordering constraints.",
            ),
            (
                "effort.estimate",
                SideEffectLevel.NONE,
                self.effort_estimate,
                "Estimate relative task effort from heuristics.",
            ),
            (
                "resource.check",
                SideEffectLevel.READ_ONLY,
                self.resource_check,
                "Inspect available local resources and credentials.",
            ),
            (
                "template.apply",
                SideEffectLevel.NONE,
                self.template_apply,
                "Apply a simple response template.",
            ),
            (
                "format.validate",
                SideEffectLevel.NONE,
                self.format_validate,
                "Validate required fields in structured data.",
            ),
            (
                "cross_ref.link",
                SideEffectLevel.NONE,
                self.cross_ref_link,
                "Cross-reference entities across artifacts.",
            ),
            (
                "draft.version",
                SideEffectLevel.EXTERNAL_WRITE,
                self.draft_version,
                "Persist a draft version in the sandbox.",
            ),
            (
                "policy.check",
                SideEffectLevel.NONE,
                self.policy_check,
                "Run rule-based validation checks.",
            ),
            (
                "fact.cross_reference",
                SideEffectLevel.NONE,
                self.fact_cross_reference,
                "Cross-check summary text against citations.",
            ),
            (
                "self.audit",
                SideEffectLevel.NONE,
                self.self_audit,
                "Produce a simple self-audit of quality risks.",
            ),
            (
                "confidence.calibrate",
                SideEffectLevel.NONE,
                self.confidence_calibrate,
                "Adjust confidence from evidence and errors.",
            ),
            (
                "vector.update",
                SideEffectLevel.EXTERNAL_WRITE,
                self.vector_update,
                "Append a local vector-like index record.",
            ),
            (
                "link.generate",
                SideEffectLevel.NONE,
                self.link_generate,
                "Generate wiki-style links from content.",
            ),
            (
                "stale.detect",
                SideEffectLevel.READ_ONLY,
                self.stale_detect,
                "Detect stale cache or draft artifacts.",
            ),
            (
                "index.optimize",
                SideEffectLevel.EXTERNAL_WRITE,
                self.index_optimize,
                "Compact the local vector index.",
            ),
            (
                "quarantine.merge",
                SideEffectLevel.EXTERNAL_WRITE,
                self.quarantine_merge,
                "Merge quarantined files back into drafts.",
            ),
            (
                "state.inspect",
                SideEffectLevel.NONE,
                self.state_inspect,
                "Inspect the current tool-state payload.",
            ),
            (
                "routing.simulate",
                SideEffectLevel.NONE,
                self.routing_simulate,
                "Simulate which role should handle a task.",
            ),
            (
                "quota.check",
                SideEffectLevel.READ_ONLY,
                self.quota_check,
                "Check environment-level quotas and readiness.",
            ),
            (
                "conflict.resolve",
                SideEffectLevel.NONE,
                self.conflict_resolve,
                "Resolve competing options with a simple heuristic.",
            ),
            (
                "gap.analyze",
                SideEffectLevel.NONE,
                self.gap_analyze,
                "Find missing evidence or sections in a draft.",
            ),
            (
                "alternative.generate",
                SideEffectLevel.NONE,
                self.alternative_generate,
                "Generate alternative approaches.",
            ),
            (
                "bias.detect",
                SideEffectLevel.NONE,
                self.bias_detect,
                "Flag potentially biased or absolute language.",
            ),
            (
                "tradeoff.evaluate",
                SideEffectLevel.NONE,
                self.tradeoff_evaluate,
                "Summarize pros and cons for options.",
            ),
        ]

    def web_fetch(self, args: dict[str, Any]) -> ToolResult:
        url = str(args.get("url") or args.get("query") or "").strip()
        if not url.startswith(("http://", "https://")):
            return ToolResult(
                success=False,
                error="url must start with http:// or https://",
                error_type="input_error",
            )
        req = request.Request(url, headers={"User-Agent": "AgentGraph/1.0"})
        try:
            with request.urlopen(req, timeout=self.config.timeout_seconds) as response:
                content_type = response.headers.get("Content-Type", "")
                body = response.read().decode("utf-8", errors="replace")
                status_code = getattr(response, "status", 200)
        except error.HTTPError as exc:
            return ToolResult(
                success=False,
                error=str(exc),
                error_type="http_error",
                metadata={"status_code": exc.code},
            )
        except error.URLError as exc:
            return ToolResult(
                success=False, error=str(exc.reason), error_type="network_error"
            )
        return ToolResult(
            success=True,
            data={
                "url": url,
                "status_code": status_code,
                "content_type": content_type,
                "content": body,
            },
        )

    def web_crawl(self, args: dict[str, Any]) -> ToolResult:
        start_url = str(args.get("url") or args.get("query") or "").strip()
        if not start_url.startswith(("http://", "https://")):
            return ToolResult(
                success=False,
                error="start_url must be an absolute URL",
                error_type="input_error",
            )
        max_pages = int(args.get("max_pages", 3))
        allowed_domains = set(
            args.get("allowed_domains") or [parse.urlparse(start_url).netloc]
        )
        visited: set[str] = set()
        queue: deque[str] = deque([start_url])
        pages: list[dict[str, Any]] = []
        while queue and len(pages) < max_pages:
            url = queue.popleft()
            if url in visited:
                continue
            visited.add(url)
            fetched = self.web_fetch({"url": url})
            if not fetched.success:
                continue
            content = str(fetched.data.get("content", ""))
            parsed = self._extract_html(content)
            pages.append(
                {
                    "url": url,
                    "title": parsed["title"],
                    "excerpt": parsed["text"][:500],
                    "links": parsed["links"][:10],
                }
            )
            for href in parsed["links"]:
                absolute = parse.urljoin(url, href)
                netloc = parse.urlparse(absolute).netloc
                if netloc in allowed_domains and absolute not in visited:
                    queue.append(absolute)
        return ToolResult(
            success=True, data={"pages": pages, "visited": sorted(visited)}
        )

    def web_download(self, args: dict[str, Any]) -> ToolResult:
        candidate_urls = self._resolve_download_urls(args)
        if not candidate_urls:
            return ToolResult(
                success=False,
                error="url must start with http:// or https://",
                error_type="input_error",
            )
        attempted_urls: list[str] = []
        last_error: ToolResult | None = None
        for url in candidate_urls:
            if not url.startswith(("http://", "https://")):
                continue
            attempted_urls.append(url)
            filename = self._resolve_download_filename(args, url)
            if not self._is_allowed_download_extension(filename):
                last_error = ToolResult(
                    success=False,
                    error=f"blocked file extension for download target: {filename}",
                    error_type="extension_blocked",
                    metadata={
                        "url": url,
                        "attempted_urls": list(attempted_urls),
                        "retriable": False,
                    },
                )
                continue

            target_path = self._build_quarantine_target(Path(filename))
            req = request.Request(url, headers={"User-Agent": "AgentGraph/1.0"})
            try:
                with request.urlopen(
                    req, timeout=self.config.timeout_seconds
                ) as response:
                    body = response.read()
                    content_type = response.headers.get("Content-Type", "")
                    status_code = getattr(response, "status", 200)
            except error.HTTPError as exc:
                last_error = ToolResult(
                    success=False,
                    error=str(exc),
                    error_type="http_error",
                    metadata={
                        "status_code": exc.code,
                        "url": url,
                        "attempted_urls": list(attempted_urls),
                        "retriable": exc.code >= 500
                        or exc.code in {408, 409, 425, 429},
                    },
                )
                continue
            except error.URLError as exc:
                last_error = ToolResult(
                    success=False,
                    error=str(exc.reason),
                    error_type="network_error",
                    metadata={
                        "url": url,
                        "attempted_urls": list(attempted_urls),
                        "retriable": True,
                    },
                )
                continue

            size_bytes = len(body)
            max_bytes = 100 * 1024 * 1024
            if size_bytes > max_bytes:
                last_error = ToolResult(
                    success=False,
                    error=f"download exceeds size limit: {size_bytes} bytes",
                    error_type="size_limit_exceeded",
                    metadata={
                        "max_bytes": max_bytes,
                        "url": url,
                        "attempted_urls": list(attempted_urls),
                        "retriable": False,
                    },
                )
                continue

            target_path.parent.mkdir(parents=True, exist_ok=True)
            target_path.write_bytes(body)
            file_hash = sha256(body).hexdigest()
            expected_hash = str(args.get("expected_hash") or "").strip().lower()
            if not expected_hash:
                task = args.get("task") or {}
                metadata = task.get("metadata", {}) if isinstance(task, dict) else {}
                expected_hash = str(metadata.get("expected_hash") or "").strip().lower()
            if expected_hash and expected_hash != file_hash:
                target_path.unlink(missing_ok=True)
                last_error = ToolResult(
                    success=False,
                    error="download hash mismatch",
                    error_type="hash_mismatch",
                    metadata={
                        "expected_hash": expected_hash,
                        "actual_hash": file_hash,
                        "url": url,
                        "attempted_urls": list(attempted_urls),
                        "retriable": False,
                    },
                )
                continue

            return ToolResult(
                success=True,
                data={
                    "url": url,
                    "path": str(target_path),
                    "filename": filename,
                    "quarantine_filename": target_path.name,
                    "bytes_downloaded": size_bytes,
                    "sha256": file_hash,
                    "content_type": content_type,
                    "status_code": status_code,
                    "quarantined": True,
                    "entity_type": "file",
                    "attempted_urls": list(attempted_urls),
                    "results": [
                        {
                            "title": target_path.name,
                            "snippet": _localized_message(
                                args,
                                en=f"Downloaded file from {url} to quarantine at {target_path}.",
                                ru=f"Файл загружен из {url} в карантин по пути {target_path}.",
                            ),
                        }
                    ],
                },
            )

        if last_error is not None:
            return last_error
        return ToolResult(
            success=False,
            error="url must start with http:// or https://",
            error_type="input_error",
        )

    def parse_extract(self, args: dict[str, Any]) -> ToolResult:
        content = str(
            args.get("content") or args.get("html") or args.get("query") or ""
        )
        if "<html" in content.lower() or "<body" in content.lower():
            return ToolResult(success=True, data=self._extract_html(content))
        text = " ".join(content.split())
        return ToolResult(
            success=True,
            data={
                "title": "",
                "text": text,
                "links": re.findall(r"https?://\S+", text),
                "word_count": len(text.split()),
            },
        )

    def citation_verify(self, args: dict[str, Any]) -> ToolResult:
        citations = args.get("citations") or args.get("context_refs") or []
        verified: list[dict[str, Any]] = []
        for citation in citations:
            url = str(citation)
            if not url.startswith(("http://", "https://")):
                verified.append(
                    {"citation": url, "verified": False, "reason": "not_url"}
                )
                continue
            result = self.web_fetch({"url": url})
            verified.append(
                {
                    "citation": url,
                    "verified": result.success,
                    "status_code": (result.data or {}).get("status_code")
                    if result.success
                    else result.metadata.get("status_code"),
                }
            )
        return ToolResult(success=True, data={"citations": verified})

    def fs_list_dir(self, args: dict[str, Any]) -> ToolResult:
        path = self._resolve_read_path(str(args.get("path", ".")))
        if not path.exists() or not path.is_dir():
            return ToolResult(
                success=False,
                error=f"directory not found: {path}",
                error_type="not_found",
            )
        entries = [
            {
                "name": item.name,
                "is_dir": item.is_dir(),
                "size": item.stat().st_size if item.is_file() else None,
            }
            for item in sorted(path.iterdir(), key=lambda item: item.name.lower())
        ]
        return ToolResult(
            success=True,
            data={
                "path": str(path),
                "entries": entries,
                "results": [
                    {
                        "title": str(path),
                        "snippet": _localized_message(
                            args,
                            en=f"Listed directory at {path}.",
                            ru=f"Показано содержимое папки {path}.",
                        ),
                    }
                ],
            },
        )

    def fs_read_file(self, args: dict[str, Any]) -> ToolResult:
        path = self._resolve_read_path(self._resolve_fs_path(args))
        if not path.exists() or not path.is_file():
            return ToolResult(
                success=False, error=f"file not found: {path}", error_type="not_found"
            )
        max_chars = int(args.get("max_chars", 20000))
        content = path.read_text(encoding="utf-8", errors="replace")
        return ToolResult(
            success=True,
            data={
                "path": str(path),
                "content": content[:max_chars],
                "truncated": len(content) > max_chars,
                "results": [
                    {
                        "title": str(path),
                        "snippet": _localized_message(
                            args,
                            en=f"Read file at {path}.",
                            ru=f"Файл прочитан по пути {path}.",
                        ),
                    }
                ],
            },
        )

    def fs_create_dir(self, args: dict[str, Any]) -> ToolResult:
        path = self._resolve_write_path(self._resolve_fs_path(args))
        path.mkdir(parents=True, exist_ok=True)
        return ToolResult(
            success=True,
            data={
                "path": str(path),
                "created": True,
                "entity_type": "directory",
                "results": [
                    {
                        "title": str(path),
                        "snippet": _localized_message(
                            args,
                            en=f"Created directory at {path}.",
                            ru=f"Папка создана по пути {path}.",
                        ),
                    }
                ],
            },
        )

    def fs_write_file(self, args: dict[str, Any]) -> ToolResult:
        path = self._resolve_write_path(self._resolve_fs_path(args))
        content = self._resolve_fs_content(args)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return ToolResult(
            success=True,
            data={
                "path": str(path),
                "bytes_written": len(content.encode("utf-8")),
                "entity_type": "file",
                "results": [
                    {
                        "title": str(path),
                        "snippet": _localized_message(
                            args,
                            en=f"Wrote file at {path}.",
                            ru=f"Файл записан по пути {path}.",
                        ),
                    }
                ],
            },
        )

    def fs_export(self, args: dict[str, Any]) -> ToolResult:
        source_raw = str(args.get("source_path") or "").strip()
        destination_raw = str(args.get("destination_path") or "").strip()
        if not source_raw or not destination_raw:
            return ToolResult(
                success=False,
                error="source_path and destination_path are required",
                error_type="invalid_input",
            )

        source_path = self._resolve_existing_move_source(source_raw)
        if not source_path.exists():
            return ToolResult(
                success=False,
                error=f"source not found: {source_path}",
                error_type="not_found",
            )

        destination_path = self._resolve_export_path(
            destination_raw,
            allowed_paths=args.get("allowed_paths"),
            task=args.get("task"),
        )
        if destination_path.exists():
            if source_path.is_file() and destination_path.is_dir():
                destination_path = destination_path / source_path.name
            else:
                return ToolResult(
                    success=False,
                    error=f"destination already exists: {destination_path}",
                    error_type="already_exists",
                )
        if destination_path.exists():
            return ToolResult(
                success=False,
                error=f"destination already exists: {destination_path}",
                error_type="already_exists",
            )

        destination_path.parent.mkdir(parents=True, exist_ok=True)
        if source_path.is_dir():
            shutil.copytree(source_path, destination_path)
            entity_type = "directory"
        else:
            shutil.copy2(source_path, destination_path)
            entity_type = "file"

        return ToolResult(
            success=True,
            data={
                "source_path": str(source_path),
                "exported_path": str(destination_path),
                "entity_type": entity_type,
                "exported": True,
                "results": [
                    {
                        "title": str(destination_path),
                        "snippet": _localized_message(
                            args,
                            en=f"Exported {entity_type} from {source_path} to {destination_path}.",
                            ru=(
                                f"{'Папка' if entity_type == 'directory' else 'Файл'} "
                                f"экспортирована из песочницы по пути {destination_path}."
                            ),
                        ),
                    }
                ],
            },
        )

    def fs_move(self, args: dict[str, Any]) -> ToolResult:
        source_raw, target_raw = self._resolve_move_paths(args)
        source_path = self._resolve_existing_move_source(source_raw)
        if not source_path.exists():
            return ToolResult(
                success=False,
                error=f"source not found: {source_path}",
                error_type="not_found",
            )

        target_path = self._resolve_write_path(target_raw)
        if target_path.exists():
            return ToolResult(
                success=False,
                error=f"target already exists: {target_path}",
                error_type="already_exists",
            )

        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source_path), str(target_path))
        entity_type = "directory" if target_path.is_dir() else "file"
        return ToolResult(
            success=True,
            data={
                "source_path": str(source_path),
                "target_path": str(target_path),
                "moved": True,
                "entity_type": entity_type,
                "results": [
                    {
                        "title": str(target_path),
                        "snippet": _localized_message(
                            args,
                            en=f"Moved {entity_type} from {source_path} to {target_path}.",
                            ru=f"{'Папка' if entity_type == 'directory' else 'Файл'} перемещена из {source_path} в {target_path}.",
                        ),
                    }
                ],
            },
        )

    def fs_delete(self, args: dict[str, Any]) -> ToolResult:
        source_raw = self._resolve_delete_path(args)
        source_path = self._resolve_existing_move_source(source_raw)
        if not source_path.exists():
            return ToolResult(
                success=False,
                error=f"target not found: {source_path}",
                error_type="not_found",
            )

        quarantine_target = self._build_quarantine_target(source_path)
        quarantine_target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source_path), str(quarantine_target))
        entity_type = "directory" if quarantine_target.is_dir() else "file"
        return ToolResult(
            success=True,
            data={
                "deleted": True,
                "quarantined": True,
                "source_path": str(source_path),
                "quarantine_path": str(quarantine_target),
                "entity_type": entity_type,
                "results": [
                    {
                        "title": str(source_path),
                        "snippet": _localized_message(
                            args,
                            en=f"Deleted {entity_type} at {source_path} by moving it to quarantine.",
                            ru=f"{'Папка' if entity_type == 'directory' else 'Файл'} по пути {source_path} перемещена в карантин.",
                        ),
                    }
                ],
            },
        )

    def package_install(self, args: dict[str, Any]) -> ToolResult:
        return self._package_action(args, action="install")

    def package_update(self, args: dict[str, Any]) -> ToolResult:
        return self._package_action(args, action="update")

    def app_launch(self, args: dict[str, Any]) -> ToolResult:
        app_name = self._resolve_app_name(args)
        executable = self._resolve_app_executable(app_name)
        if executable is None:
            return ToolResult(
                success=False,
                error=f"allowed application not found: {app_name}",
                error_type="not_found",
            )

        arguments = [str(value) for value in args.get("arguments") or []]
        apply = bool(
            args.get("apply")
            or args.get("approved")
            or self._tool_is_approved(args, "app.launch")
        )
        command = [executable, *arguments]
        if not apply:
            return ToolResult(
                success=True,
                data={
                    "app_name": app_name,
                    "executable": executable,
                    "arguments": arguments,
                    "launched": False,
                    "dry_run": True,
                    "command_preview": " ".join(command),
                    "results": [
                        {
                            "title": app_name,
                            "snippet": _localized_message(
                                args,
                                en=f"Prepared application launch for {app_name} using {executable}.",
                                ru=f"Подготовлен запуск приложения {app_name} через {executable}.",
                            ),
                        }
                    ],
                },
            )

        try:
            process = subprocess.Popen(command)
        except OSError as exc:
            return ToolResult(
                success=False,
                error=str(exc),
                error_type="launch_error",
                metadata={"command": command},
            )

        return ToolResult(
            success=True,
            data={
                "app_name": app_name,
                "executable": executable,
                "arguments": arguments,
                "dry_run": False,
                "launched": True,
                "pid": process.pid,
                "command_preview": " ".join(command),
                "results": [
                    {
                        "title": app_name,
                        "snippet": _localized_message(
                            args,
                            en=f"Launched {app_name} using {executable}.",
                            ru=f"Приложение {app_name} запущено через {executable}.",
                        ),
                    }
                ],
            },
        )

    def cache_store(self, args: dict[str, Any]) -> ToolResult:
        key = self._slugify(
            str(
                args.get("key")
                or args.get("task_id")
                or datetime.now(timezone.utc).timestamp()
            )
        )
        path = self.config.cache_dir / f"{key}.json"
        payload = {
            "query": args.get("query"),
            "context_refs": args.get("context_refs", []),
            "shared_context": args.get("shared_context", {}),
            "data": args.get("data") or args.get("content") or {},
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
        return ToolResult(success=True, data={"key": key, "path": str(path)})

    def history_query(self, args: dict[str, Any]) -> ToolResult:
        query = str(args.get("query", "")).lower()
        matches: list[dict[str, Any]] = []
        for candidate in sorted(self.config.cache_dir.glob("*.json")):
            content = candidate.read_text(encoding="utf-8", errors="replace")
            if query and query not in content.lower():
                continue
            matches.append({"path": str(candidate), "excerpt": content[:400]})
        return ToolResult(success=True, data={"matches": matches[:10]})

    def dependency_analyze(self, args: dict[str, Any]) -> ToolResult:
        text = str(args.get("query") or args.get("description") or "")
        parts = [
            part.strip() for part in re.split(r"\band\b|,|;", text) if part.strip()
        ]
        dependencies = []
        for index, part in enumerate(parts):
            dependencies.append(
                {
                    "task": part,
                    "depends_on": [] if index == 0 else [parts[index - 1]],
                }
            )
        return ToolResult(
            success=True,
            data={"dependencies": dependencies, "parallelizable": len(parts) > 1},
        )

    def effort_estimate(self, args: dict[str, Any]) -> ToolResult:
        text = str(args.get("query") or "")
        keywords = {"analyze": 2, "compare": 2, "plan": 2, "research": 3, "migrate": 4}
        score = 1 + len(text.split()) // 20
        for keyword, weight in keywords.items():
            if keyword in text.lower():
                score += weight
        bucket = "low" if score <= 2 else "medium" if score <= 5 else "high"
        return ToolResult(
            success=True, data={"effort_score": score, "effort_bucket": bucket}
        )

    def resource_check(self, args: dict[str, Any]) -> ToolResult:
        cache_files = list(self.config.cache_dir.glob("*"))
        return ToolResult(
            success=True,
            data={
                "workspace_root": str(self.config.workspace_root),
                "cache_file_count": len(cache_files),
                "openai_key_present": bool(os.environ.get("OPENAI_API_KEY")),
                "allowed_read_roots": [
                    str(path) for path in self.config.allowed_read_roots
                ],
                "allowed_write_roots": [
                    str(path) for path in self.config.allowed_write_roots
                ],
            },
        )

    def template_apply(self, args: dict[str, Any]) -> ToolResult:
        template = str(args.get("template", "bullet_summary"))
        data = args.get("data") or args.get("output_candidate") or {}
        if template == "report":
            rendered = (
                f"Summary: {data.get('summary', '')}\n"
                f"Citations: {', '.join(data.get('citations', []))}"
            )
        else:
            rendered = "\n".join(
                [
                    f"- Summary: {data.get('summary', '')}",
                    f"- Citations: {', '.join(data.get('citations', []))}",
                ]
            )
        return ToolResult(
            success=True, data={"template": template, "rendered": rendered}
        )

    def format_validate(self, args: dict[str, Any]) -> ToolResult:
        data = args.get("data") or args.get("output_candidate") or {}
        required_fields = args.get("required_fields") or ["summary", "citations"]
        missing = [field for field in required_fields if field not in data]
        return ToolResult(
            success=True, data={"valid": not missing, "missing_fields": missing}
        )

    def cross_ref_link(self, args: dict[str, Any]) -> ToolResult:
        data = args.get("data") or args.get("output_candidate") or {}
        summary_tokens = set(_tokens(str(data.get("summary", ""))))
        citations = data.get("citations", [])
        links = []
        for citation in citations:
            overlap = sorted(summary_tokens & set(_tokens(str(citation))))
            if overlap:
                links.append({"citation": citation, "overlap": overlap})
        return ToolResult(success=True, data={"links": links})

    def draft_version(self, args: dict[str, Any]) -> ToolResult:
        key = self._slugify(
            str(
                args.get("task_id")
                or args.get("key")
                or datetime.now(timezone.utc).timestamp()
            )
        )
        path = self.config.drafts_dir / f"{key}.md"
        rendered = str(
            args.get("content")
            or json.dumps(
                args.get("output_candidate") or {}, ensure_ascii=False, indent=2
            )
        )
        path.write_text(rendered, encoding="utf-8")
        return ToolResult(success=True, data={"path": str(path), "versioned": True})

    def policy_check(self, args: dict[str, Any]) -> ToolResult:
        output = args.get("output_candidate") or args.get("data") or {}
        required_terms = args.get("required_terms") or []
        min_citations = int(args.get("min_citations", 1))
        errors = []
        summary = str(output.get("summary", ""))
        citations = output.get("citations", [])
        for term in required_terms:
            if str(term).lower() not in summary.lower():
                errors.append(f"missing required term: {term}")
        if len(citations) < min_citations:
            errors.append("insufficient citations")
        return ToolResult(success=True, data={"valid": not errors, "errors": errors})

    def fact_cross_reference(self, args: dict[str, Any]) -> ToolResult:
        output = args.get("output_candidate") or {}
        summary_tokens = Counter(_tokens(str(output.get("summary", ""))))
        citations = [str(item) for item in output.get("citations", [])]
        citation_tokens = Counter(
            token for citation in citations for token in _tokens(citation)
        )
        unsupported = [
            token
            for token in summary_tokens
            if len(token) > 5 and token not in citation_tokens
        ]
        return ToolResult(
            success=True,
            data={"unsupported_terms": unsupported[:10], "supported": not unsupported},
        )

    def self_audit(self, args: dict[str, Any]) -> ToolResult:
        output = args.get("output_candidate") or {}
        issues = []
        if not output:
            issues.append("no output candidate")
        if not output.get("summary"):
            issues.append("summary missing")
        if not output.get("citations"):
            issues.append("citations missing")
        return ToolResult(
            success=True, data={"issues": issues, "issue_count": len(issues)}
        )

    def confidence_calibrate(self, args: dict[str, Any]) -> ToolResult:
        output = args.get("output_candidate") or {}
        confidence = float(output.get("confidence", args.get("confidence", 0.5)))
        citations = len(output.get("citations", []))
        errors = len(args.get("errors", []))
        calibrated = max(0.0, min(1.0, confidence + 0.05 * citations - 0.1 * errors))
        return ToolResult(success=True, data={"confidence": calibrated})

    def vector_update(self, args: dict[str, Any]) -> ToolResult:
        record = {
            "task_id": args.get("task_id"),
            "query": args.get("query"),
            "shared_context": args.get("shared_context", {}),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        with self.config.vector_index_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        return ToolResult(
            success=True, data={"path": str(self.config.vector_index_path)}
        )

    def link_generate(self, args: dict[str, Any]) -> ToolResult:
        content = str(
            args.get("content")
            or json.dumps(args.get("output_candidate") or {}, ensure_ascii=False)
        )
        links = [f"[[{token.title()}]]" for token in _important_tokens(content)]
        return ToolResult(success=True, data={"links": links[:10]})

    def stale_detect(self, args: dict[str, Any]) -> ToolResult:
        threshold_days = int(args.get("threshold_days", 7))
        cutoff = datetime.now(timezone.utc) - timedelta(days=threshold_days)
        stale = []
        for directory in (self.config.cache_dir, self.config.drafts_dir):
            for path in directory.glob("*"):
                modified = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
                if modified < cutoff:
                    stale.append(
                        {"path": str(path), "modified_at": modified.isoformat()}
                    )
        return ToolResult(success=True, data={"stale_items": stale})

    def index_optimize(self, args: dict[str, Any]) -> ToolResult:
        if not self.config.vector_index_path.exists():
            return ToolResult(success=True, data={"optimized": False, "records": 0})
        seen: set[str] = set()
        unique_lines: list[str] = []
        for line in self.config.vector_index_path.read_text(
            encoding="utf-8"
        ).splitlines():
            if line not in seen:
                seen.add(line)
                unique_lines.append(line)
        self.config.vector_index_path.write_text(
            "\n".join(unique_lines) + ("\n" if unique_lines else ""), encoding="utf-8"
        )
        return ToolResult(
            success=True, data={"optimized": True, "records": len(unique_lines)}
        )

    def quarantine_merge(self, args: dict[str, Any]) -> ToolResult:
        merged = []
        for path in self.config.quarantine_dir.glob("*"):
            target = self.config.drafts_dir / path.name
            target.write_bytes(path.read_bytes())
            path.unlink()
            merged.append(str(target))
        return ToolResult(success=True, data={"merged_paths": merged})

    def state_inspect(self, args: dict[str, Any]) -> ToolResult:
        shared_context = args.get("shared_context", {})
        return ToolResult(
            success=True,
            data={
                "task_id": args.get("task_id"),
                "query": args.get("query"),
                "shared_context_keys": sorted(shared_context.keys()),
                "context_ref_count": len(args.get("context_refs", [])),
            },
        )

    def routing_simulate(self, args: dict[str, Any]) -> ToolResult:
        query = str(args.get("query", "")).lower()
        if any(
            keyword in query
            for keyword in (
                "folder",
                "directory",
                "mkdir",
                "file",
                "write file",
                "create file",
                "create folder",
                "папк",
                "каталог",
                "директор",
                "файл",
                "создай",
                "создать",
                "запиши",
                "записать",
                "сохрани",
                "сохранить",
            )
        ):
            recommended = "file_executor"
        elif any(
            keyword in query
            for keyword in ("plan", "steps", "decompose", "шаг", "план")
        ):
            recommended = "planner"
        elif any(
            keyword in query
            for keyword in ("validate", "check", "verify", "проверь", "валидац")
        ):
            recommended = "validator"
        elif any(
            keyword in query for keyword in ("risk", "bias", "critic", "риск", "смещен")
        ):
            recommended = "critic"
        else:
            recommended = "researcher"
        return ToolResult(success=True, data={"recommended_agent": recommended})

    def quota_check(self, args: dict[str, Any]) -> ToolResult:
        return ToolResult(
            success=True,
            data={
                "openai_api_key_present": bool(os.environ.get("OPENAI_API_KEY")),
                "cache_dir_exists": self.config.cache_dir.exists(),
                "draft_dir_exists": self.config.drafts_dir.exists(),
            },
        )

    def conflict_resolve(self, args: dict[str, Any]) -> ToolResult:
        options = args.get("options") or []
        if not options:
            return ToolResult(
                success=True, data={"winner": None, "reason": "no_options"}
            )
        winner = sorted(options, key=lambda item: (-len(str(item)), str(item)))[0]
        return ToolResult(
            success=True, data={"winner": winner, "reason": "longest_option_heuristic"}
        )

    def gap_analyze(self, args: dict[str, Any]) -> ToolResult:
        output = args.get("output_candidate") or {}
        gaps = []
        if not output.get("summary"):
            gaps.append("missing summary")
        if not output.get("citations"):
            gaps.append("missing citations")
        if len(str(output.get("summary", "")).split()) < 20:
            gaps.append("summary may be too short")
        return ToolResult(success=True, data={"gaps": gaps})

    def alternative_generate(self, args: dict[str, Any]) -> ToolResult:
        query = str(args.get("query") or "")
        return ToolResult(
            success=True,
            data={
                "alternatives": [
                    f"Research-first approach for: {query}",
                    f"Planner-led decomposition for: {query}",
                    f"Validator-heavy approach for: {query}",
                ]
            },
        )

    def bias_detect(self, args: dict[str, Any]) -> ToolResult:
        text = str(
            args.get("content")
            or json.dumps(args.get("output_candidate") or {}, ensure_ascii=False)
        )
        flagged = [
            word
            for word in ("always", "never", "obviously", "clearly")
            if word in text.lower()
        ]
        return ToolResult(
            success=True, data={"flagged_terms": flagged, "biased": bool(flagged)}
        )

    def tradeoff_evaluate(self, args: dict[str, Any]) -> ToolResult:
        options = [str(option) for option in (args.get("options") or [])]
        evaluated = [
            {
                "option": option,
                "pros": [f"{option} is simple"],
                "cons": [f"{option} may be incomplete"],
            }
            for option in options
        ]
        return ToolResult(success=True, data={"tradeoffs": evaluated})

    def _resolve_read_path(self, raw_path: str) -> Path:
        return _resolve_path(
            raw_path,
            base=self.config.workspace_root,
            allowed_roots=self.config.allowed_read_roots,
        )

    def _resolve_write_path(self, raw_path: str) -> Path:
        return _resolve_path(
            raw_path,
            base=self.config.workspace_actions_dir,
            allowed_roots=self.config.allowed_write_roots,
        )

    def _resolve_fs_path(self, args: dict[str, Any]) -> str:
        explicit_path = str(args.get("path", "")).strip()
        if explicit_path:
            return explicit_path
        task = args.get("task") or {}
        metadata = task.get("metadata", {}) if isinstance(task, dict) else {}
        metadata_path = str(metadata.get("target_path", "")).strip()
        if metadata_path:
            thread_id = str(args.get("thread_id") or "thread")
            metadata_path = metadata_path.replace("{thread_id}", thread_id)
            return metadata_path
        thread_id = str(args.get("thread_id") or "thread")
        query = str(args.get("query") or "")
        filename_match = re.search(
            r"(?:named|called)\s+([A-Za-z0-9._/-]+)", query, flags=re.IGNORECASE
        )
        inferred_name = (
            filename_match.group(1) if filename_match is not None else "MEMORY"
        )
        return f"workspace/{thread_id}/{inferred_name}"

    def _resolve_fs_content(self, args: dict[str, Any]) -> str:
        explicit_content = str(args.get("content", ""))
        if explicit_content:
            return explicit_content
        task = args.get("task") or {}
        metadata = task.get("metadata", {}) if isinstance(task, dict) else {}
        metadata_content = metadata.get("file_content")
        if metadata_content is not None:
            return str(metadata_content)
        query = str(args.get("query") or "")
        content_match = re.search(
            r"(?:with|containing)\s+['\"](.+?)['\"]",
            query,
            flags=re.IGNORECASE,
        )
        if content_match is not None:
            return content_match.group(1)
        return "Created by AgentGraph file executor."

    def _resolve_move_paths(self, args: dict[str, Any]) -> tuple[str, str]:
        source_explicit = str(args.get("source_path", "")).strip()
        target_explicit = str(args.get("target_path", "")).strip()
        if source_explicit and target_explicit:
            return source_explicit, target_explicit

        task = args.get("task") or {}
        metadata = task.get("metadata", {}) if isinstance(task, dict) else {}
        source_metadata = str(metadata.get("source_path", "")).strip()
        target_metadata = str(metadata.get("target_path", "")).strip()
        if source_metadata and target_metadata:
            thread_id = str(args.get("thread_id") or "thread")
            return (
                source_metadata.replace("{thread_id}", thread_id),
                target_metadata.replace("{thread_id}", thread_id),
            )

        query = str(args.get("query") or "")
        match = re.search(
            r"(?:move|перемести|перенеси)\s+(?:file|folder|directory|файл|папк\w*|каталог\w*|директор\w*)?\s*([A-Za-z0-9._/\-]+)\s+(?:to|в)\s+([A-Za-z0-9._/\-]+)",
            query,
            flags=re.IGNORECASE,
        )
        if match is not None:
            return match.group(1), match.group(2)

        thread_id = str(args.get("thread_id") or "thread")
        return f"workspace/{thread_id}/MEMORY", f"workspace/{thread_id}/ARCHIVE"

    def _resolve_delete_path(self, args: dict[str, Any]) -> str:
        explicit_path = str(args.get("path", "")).strip()
        if explicit_path:
            return explicit_path

        task = args.get("task") or {}
        metadata = task.get("metadata", {}) if isinstance(task, dict) else {}
        metadata_path = str(metadata.get("target_path", "")).strip()
        if metadata_path:
            thread_id = str(args.get("thread_id") or "thread")
            return metadata_path.replace("{thread_id}", thread_id)

        query = str(args.get("query") or "")
        match = re.search(
            r"(?:delete|удали)\s+(?:file|folder|directory|файл|папк\w*|каталог\w*|директор\w*)?\s*([A-Za-z0-9._/\-]+)",
            query,
            flags=re.IGNORECASE,
        )
        if match is not None:
            return match.group(1)

        thread_id = str(args.get("thread_id") or "thread")
        return f"workspace/{thread_id}/MEMORY"

    def _resolve_download_url(self, args: dict[str, Any]) -> str:
        urls = self._resolve_download_urls(args)
        return urls[0] if urls else ""

    def _resolve_download_urls(self, args: dict[str, Any]) -> list[str]:
        candidates: list[str] = []
        seen: set[str] = set()

        def append(candidate: str) -> None:
            normalized = candidate.strip()
            if not normalized or normalized in seen:
                return
            seen.add(normalized)
            candidates.append(normalized)

        explicit_url = str(args.get("url") or "").strip()
        if explicit_url:
            append(explicit_url)

        task = args.get("task") or {}
        metadata = task.get("metadata", {}) if isinstance(task, dict) else {}
        metadata_url = str(metadata.get("url") or "").strip()
        if metadata_url:
            append(metadata_url)

        context_urls = self._candidate_download_urls_from_context(args)
        for context_url in context_urls:
            append(context_url)

        query = str(args.get("query") or "")
        match = re.search(r"https?://[^\s\"']+", query, flags=re.IGNORECASE)
        if match is not None:
            append(match.group(0))
        return candidates

    def _candidate_download_urls_from_context(self, args: dict[str, Any]) -> list[str]:
        shared_context = args.get("shared_context")
        if not isinstance(shared_context, dict):
            return []

        query = str(args.get("query") or "").lower()
        prefer_pdf = "pdf" in query
        urls = self._extract_urls(shared_context)
        if prefer_pdf:
            pdf_urls = [url for url in urls if ".pdf" in url.lower()]
            if pdf_urls:
                return pdf_urls
        return urls

    def _extract_urls(self, value: Any) -> list[str]:
        seen: set[str] = set()
        collected: list[str] = []

        def visit(candidate: Any) -> None:
            if isinstance(candidate, str):
                for match in re.findall(r"https?://[^\s\"'<>]+", candidate):
                    if match not in seen:
                        seen.add(match)
                        collected.append(match)
                return
            if isinstance(candidate, list):
                for item in candidate:
                    visit(item)
                return
            if isinstance(candidate, dict):
                direct_url = str(
                    candidate.get("url")
                    or candidate.get("href")
                    or candidate.get("source_url")
                    or candidate.get("sourceUrl")
                    or ""
                ).strip()
                if direct_url and direct_url not in seen:
                    seen.add(direct_url)
                    collected.append(direct_url)
                for nested in candidate.values():
                    visit(nested)

        visit(value)
        return collected

    def _resolve_download_filename(self, args: dict[str, Any], url: str) -> str:
        explicit_name = str(args.get("target_filename") or "").strip()
        if explicit_name:
            return explicit_name

        task = args.get("task") or {}
        metadata = task.get("metadata", {}) if isinstance(task, dict) else {}
        metadata_name = str(metadata.get("target_filename") or "").strip()
        if metadata_name:
            return metadata_name

        parsed = parse.urlparse(url)
        inferred = Path(parsed.path).name
        return inferred or "download.bin"

    @staticmethod
    def _is_allowed_download_extension(filename: str) -> bool:
        allowed = {".pdf", ".docx", ".xlsx", ".zip", ".txt", ".csv"}
        return Path(filename).suffix.lower() in allowed

    def _package_action(self, args: dict[str, Any], *, action: str) -> ToolResult:
        manager = self._resolve_package_manager(args)
        package_name = self._resolve_package_name(args)
        if not package_name:
            return ToolResult(
                success=False,
                error="package_name is required",
                error_type="input_error",
            )

        if manager not in {"pip", "npm", "brew", "winget", "apt"}:
            return ToolResult(
                success=False,
                error=f"unsupported package manager: {manager}",
                error_type="unsupported_manager",
            )

        source = str(args.get("source") or "official").strip().lower()
        if source not in {"official", "verified"}:
            return ToolResult(
                success=False,
                error=f"untrusted package source: {source}",
                error_type="source_blocked",
            )

        if not self._package_manager_available(manager):
            return ToolResult(
                success=False,
                error=f"package manager not available: {manager}",
                error_type="manager_missing",
            )

        version = str(args.get("version") or args.get("target_version") or "").strip()
        command = self._build_package_command(
            action=action,
            manager=manager,
            package_name=package_name,
            version=version,
        )
        installed_before = self._query_package_state(manager, package_name)
        apply = bool(
            args.get("apply")
            or args.get("approved")
            or self._tool_is_approved(args, f"package.{action}")
        )

        if not apply:
            verb = "install" if action == "install" else "update"
            return ToolResult(
                success=True,
                data={
                    "manager": manager,
                    "package_name": package_name,
                    "version": version,
                    "applied": False,
                    "dry_run": True,
                    "installed_before": installed_before,
                    "command_preview": " ".join(command),
                    "results": [
                        {
                            "title": package_name,
                            "snippet": _localized_message(
                                args,
                                en=f"Prepared package {verb} for {package_name} via {manager}.",
                                ru=f"Подготовлено действие {verb} для пакета {package_name} через {manager}.",
                            ),
                        }
                    ],
                },
            )

        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=max(int(self.config.timeout_seconds), 30),
                check=False,
            )
        except OSError as exc:
            return ToolResult(
                success=False,
                error=str(exc),
                error_type="execution_error",
                metadata={"command": command},
            )

        installed_after = self._query_package_state(manager, package_name)
        success = completed.returncode == 0
        verb = "installed" if action == "install" else "updated"
        return ToolResult(
            success=success,
            data={
                "manager": manager,
                "package_name": package_name,
                "version": version,
                "applied": True,
                "dry_run": False,
                "installed_before": installed_before,
                "installed_after": installed_after,
                "command_preview": " ".join(command),
                "stdout": completed.stdout[-4000:],
                "stderr": completed.stderr[-4000:],
                "results": [
                    {
                        "title": package_name,
                        "snippet": _localized_message(
                            args,
                            en=f"{verb.capitalize()} package {package_name} via {manager}.",
                            ru=f"Пакет {package_name} {('установлен' if action == 'install' else 'обновлён')} через {manager}.",
                        ),
                    }
                ],
            },
            error=None
            if success
            else completed.stderr[-4000:] or completed.stdout[-4000:],
            error_type=None if success else "command_failed",
        )

    def _resolve_package_manager(self, args: dict[str, Any]) -> str:
        explicit = str(args.get("manager") or "").strip().lower()
        if explicit:
            return explicit
        task = args.get("task") or {}
        metadata = task.get("metadata", {}) if isinstance(task, dict) else {}
        metadata_manager = str(metadata.get("manager") or "").strip().lower()
        if metadata_manager:
            return metadata_manager
        query = str(args.get("query") or "").lower()
        for candidate in ("pip", "npm", "brew", "winget", "apt"):
            if candidate in query:
                return candidate
        return "pip"

    def _resolve_package_name(self, args: dict[str, Any]) -> str:
        explicit = str(args.get("package_name") or "").strip()
        if explicit:
            return explicit
        task = args.get("task") or {}
        metadata = task.get("metadata", {}) if isinstance(task, dict) else {}
        metadata_package = str(metadata.get("package_name") or "").strip()
        if metadata_package:
            return metadata_package
        query = str(args.get("query") or "")
        manager = self._resolve_package_manager(args)
        pattern = rf"(?:install|update|установи|обнови)\s+([A-Za-z0-9._-]+)(?:\s+through|\s+via|\s+using|\s+with|\s+через|\s+{manager})?"
        match = re.search(pattern, query, flags=re.IGNORECASE)
        if match is not None:
            return match.group(1)
        return ""

    @staticmethod
    def _package_manager_available(manager: str) -> bool:
        if manager == "pip":
            return True
        return shutil.which(manager) is not None

    @staticmethod
    def _build_package_command(
        *,
        action: str,
        manager: str,
        package_name: str,
        version: str,
    ) -> list[str]:
        pinned = (
            f"{package_name}=={version}"
            if version and manager == "pip"
            else package_name
        )
        if manager == "pip":
            if action == "install":
                return [sys.executable, "-m", "pip", "install", pinned]
            return [sys.executable, "-m", "pip", "install", "--upgrade", pinned]
        if manager == "npm":
            if action == "install":
                return ["npm", "install", package_name]
            return ["npm", "update", package_name]
        if manager == "brew":
            if action == "install":
                return ["brew", "install", package_name]
            return ["brew", "upgrade", package_name]
        if manager == "winget":
            if action == "install":
                return ["winget", "install", "--id", package_name]
            return ["winget", "upgrade", "--id", package_name]
        if action == "install":
            return ["apt", "install", package_name]
        return ["apt", "upgrade", package_name]

    @staticmethod
    def _query_package_state(manager: str, package_name: str) -> dict[str, Any]:
        if manager == "pip":
            try:
                version = importlib_metadata.version(package_name)
            except importlib_metadata.PackageNotFoundError:
                return {"installed": False, "version": None}
            return {"installed": True, "version": version}
        return {"installed": None, "version": None}

    def _resolve_app_name(self, args: dict[str, Any]) -> str:
        explicit = str(args.get("app_name") or "").strip().lower()
        if explicit:
            return explicit
        task = args.get("task") or {}
        metadata = task.get("metadata", {}) if isinstance(task, dict) else {}
        metadata_name = str(metadata.get("app_name") or "").strip().lower()
        if metadata_name:
            return metadata_name
        query = str(args.get("query") or "").lower()
        for candidate in ("browser", "calendar", "mail", "terminal", "text_editor"):
            if candidate in query:
                return candidate
        if "брауз" in query:
            return "browser"
        if "почт" in query:
            return "mail"
        if "календар" in query:
            return "calendar"
        if "терминал" in query:
            return "terminal"
        if "редактор" in query or "блокнот" in query:
            return "text_editor"
        return "browser"

    @staticmethod
    def _resolve_app_executable(app_name: str) -> str | None:
        alias_map = {
            "win32": {
                "browser": ["chrome.exe", "msedge.exe", "firefox.exe"],
                "mail": ["outlook.exe"],
                "calendar": ["outlook.exe"],
                "terminal": ["powershell.exe", "cmd.exe", "wt.exe"],
                "text_editor": ["notepad.exe", "code.exe"],
            },
            "darwin": {
                "browser": ["Google Chrome", "Safari"],
                "mail": ["Mail"],
                "calendar": ["Calendar"],
                "terminal": ["Terminal", "iTerm"],
                "text_editor": ["TextEdit", "Visual Studio Code"],
            },
        }
        aliases = alias_map.get(sys.platform, {}).get(app_name)
        if aliases is None:
            aliases = {
                "browser": ["chromium-browser", "firefox"],
                "mail": ["thunderbird"],
                "calendar": ["gnome-calendar"],
                "terminal": ["gnome-terminal", "xterm", "konsole"],
                "text_editor": ["gedit", "code"],
            }.get(app_name, [])
        for alias in aliases:
            resolved = shutil.which(alias)
            if resolved is not None:
                return resolved
        return None

    def _resolve_existing_move_source(self, raw_path: str) -> Path:
        try:
            candidate = self._resolve_write_path(raw_path)
            if candidate.exists():
                return candidate
        except ValueError:
            pass
        return self._resolve_read_path(raw_path)

    def _resolve_export_path(
        self,
        raw_path: str,
        *,
        allowed_paths: Any = None,
        task: Any = None,
    ) -> Path:
        destination = Path(os.path.expanduser(raw_path))
        if not destination.is_absolute():
            destination = (Path.home() / destination).resolve(strict=False)
        else:
            destination = destination.resolve(strict=False)

        allowed_roots = self._allowed_export_roots(
            allowed_paths=allowed_paths, task=task
        )
        for root in allowed_roots:
            try:
                destination.relative_to(root)
            except ValueError:
                continue
            return destination
        raise ValueError(f"export path outside allowed roots: {raw_path}")

    def _allowed_export_roots(
        self, *, allowed_paths: Any = None, task: Any = None
    ) -> list[Path]:
        roots: list[Path] = []
        if isinstance(allowed_paths, list):
            for value in allowed_paths:
                if not value:
                    continue
                roots.append(Path(os.path.expanduser(str(value))).resolve(strict=False))

        metadata = task.get("metadata", {}) if isinstance(task, dict) else {}
        metadata_allowed = metadata.get("allowed_paths", [])
        if isinstance(metadata_allowed, list):
            for value in metadata_allowed:
                if not value:
                    continue
                roots.append(Path(os.path.expanduser(str(value))).resolve(strict=False))

        home = Path.home()
        roots.extend(
            [
                (home / "Desktop").resolve(strict=False),
                (home / "Documents").resolve(strict=False),
                (home / "Downloads").resolve(strict=False),
            ]
        )
        deduped: list[Path] = []
        seen: set[str] = set()
        for root in roots:
            key = str(root).lower()
            if key in seen:
                continue
            seen.add(key)
            deduped.append(root)
        return deduped

    @staticmethod
    def _tool_is_approved(args: dict[str, Any], tool_ref: str) -> bool:
        shared_context = args.get("shared_context")
        if not isinstance(shared_context, dict):
            return False
        approved_tools = shared_context.get("approved_tools", [])
        if not isinstance(approved_tools, list):
            return False
        return tool_ref in approved_tools

    def _build_quarantine_target(self, source_path: Path) -> Path:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        sanitized = self._slugify(source_path.name or "item")
        target = self.config.quarantine_dir / f"{timestamp}-{sanitized}"
        suffix = 1
        while target.exists():
            target = self.config.quarantine_dir / f"{timestamp}-{sanitized}-{suffix}"
            suffix += 1
        return target

    @staticmethod
    def _extract_html(content: str) -> dict[str, Any]:
        parser = _HTMLSummaryParser()
        parser.feed(content)
        return parser.as_dict()

    @staticmethod
    def _slugify(value: str) -> str:
        normalized = re.sub(r"[^a-zA-Z0-9._-]+", "-", value).strip("-")
        return normalized or "item"


def _resolve_path(raw_path: str, *, base: Path, allowed_roots: list[Path]) -> Path:
    candidate = Path(raw_path)
    resolved = (candidate if candidate.is_absolute() else base / candidate).resolve(
        strict=False
    )
    for root in allowed_roots:
        try:
            resolved.relative_to(root.resolve(strict=False))
        except ValueError:
            continue
        return resolved
    raise ValueError(f"path outside allowed roots: {raw_path}")


def _tokens(text: str) -> list[str]:
    cleaned = "".join(ch.lower() if ch.isalnum() else " " for ch in text)
    return [token for token in cleaned.split() if token]


def _important_tokens(text: str) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for token in _tokens(text):
        if len(token) < 5 or token in seen:
            continue
        seen.add(token)
        result.append(token)
    return result


def _localized_message(args: dict[str, Any], *, en: str, ru: str) -> str:
    locale = str(args.get("locale") or "").strip().lower()
    if not locale:
        task = args.get("task") or {}
        metadata = task.get("metadata", {}) if isinstance(task, dict) else {}
        locale = str(metadata.get("locale") or "").strip().lower()
    return ru if locale.startswith("ru") else en
