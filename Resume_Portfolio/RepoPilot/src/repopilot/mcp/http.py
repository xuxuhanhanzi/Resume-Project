"""Loopback-only Streamable HTTP MCP transport with bounded JSON-RPC."""

from __future__ import annotations

import asyncio
import json
from typing import cast
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from repopilot.core.contracts import JSONValue
from repopilot.mcp.config import MCPServerConfig

_PROTOCOL_VERSION = "2025-11-25"


class StreamableHttpMCPTransport:
    """Support a local HTTP MCP server without cookies, OAuth, or external network access."""

    def __init__(self, config: MCPServerConfig, *, timeout_seconds: float = 30.0) -> None:
        if config.url is None:
            raise ValueError("HTTP transport requires an HTTP MCP configuration")
        if not 1 <= timeout_seconds <= 120:
            raise ValueError("MCP timeout_seconds must be between 1 and 120")
        self.config = config
        self.timeout_seconds = timeout_seconds
        self._session_id: str | None = None

    async def request(self, payload: dict[str, JSONValue]) -> dict[str, JSONValue]:
        return await asyncio.to_thread(self._request_sync, payload)

    def _request_sync(self, payload: dict[str, JSONValue]) -> dict[str, JSONValue]:
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
            "MCP-Protocol-Version": _PROTOCOL_VERSION,
        }
        if self._session_id is not None:
            headers["MCP-Session-Id"] = self._session_id
        request = Request(
            cast(str, self.config.url),
            data=json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:  # noqa: S310
                session_id = response.headers.get("MCP-Session-Id")
                if session_id:
                    if not session_id.isascii() or any(ord(char) < 0x21 for char in session_id):
                        raise RuntimeError("HTTP MCP server returned an invalid session ID")
                    self._session_id = session_id
                content_type = response.headers.get("Content-Type", "").casefold()
                body = response.read(2_000_000).decode("utf-8")
        except HTTPError as error:
            raise RuntimeError(f"loopback HTTP MCP server returned HTTP {error.code}") from error
        except (URLError, TimeoutError) as error:
            raise RuntimeError(f"loopback HTTP MCP server is unavailable: {error}") from error
        if "text/event-stream" in content_type:
            return self._parse_sse_response(body, payload)
        try:
            raw = json.loads(body)
        except json.JSONDecodeError as error:
            raise RuntimeError("HTTP MCP server returned invalid JSON") from error
        if not isinstance(raw, dict):
            raise RuntimeError("HTTP MCP response must be an object")
        return cast(dict[str, JSONValue], raw)

    @staticmethod
    def _parse_sse_response(body: str, request: dict[str, JSONValue]) -> dict[str, JSONValue]:
        data_lines: list[str] = []
        for line in body.splitlines():
            if line.startswith("data:"):
                data_lines.append(line.removeprefix("data:").strip())
                continue
            if line or not data_lines:
                continue
            raw = StreamableHttpMCPTransport._decode_event(data_lines)
            data_lines.clear()
            if raw is not None and raw.get("id") == request.get("id"):
                return raw
        raw = StreamableHttpMCPTransport._decode_event(data_lines)
        if raw is not None and raw.get("id") == request.get("id"):
            return raw
        raise RuntimeError("HTTP MCP SSE response had no matching JSON-RPC result")

    @staticmethod
    def _decode_event(lines: list[str]) -> dict[str, JSONValue] | None:
        if not lines:
            return None
        try:
            raw = json.loads("\n".join(lines))
        except json.JSONDecodeError:
            return None
        return cast(dict[str, JSONValue], raw) if isinstance(raw, dict) else None

    async def close(self) -> None:
        """Forget the local session; no DELETE is sent implicitly."""

        self._session_id = None
