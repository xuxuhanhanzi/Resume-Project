"""Explicitly approved, DNS-pinned HTTPS fetches for documentation research."""

from __future__ import annotations

import asyncio
import http.client
import ipaddress
import socket
import ssl
from urllib.parse import urlsplit

from repopilot.core.contracts import ErrorType, Permission, ToolCall, ToolResult, ToolSpec
from repopilot.tools.base import ToolContext

_MAX_BYTES = 1_000_000
_TEXT_TYPES = ("text/", "application/json", "application/xml", "application/javascript")


class _PinnedHttpsConnection(http.client.HTTPSConnection):
    """Keep one validated DNS answer through TLS connection establishment."""

    def __init__(self, host: str, port: int, address: str, *, timeout: float) -> None:
        context = ssl.create_default_context()
        super().__init__(host, port=port, timeout=timeout, context=context)
        self._address = address
        self._ssl_context = context

    def connect(self) -> None:
        sock = socket.create_connection((self._address, self.port), self.timeout)
        self.sock = self._ssl_context.wrap_socket(sock, server_hostname=self.host)


class WebFetchTool:
    """Fetch one public HTTPS page only after an explicit per-destination approval."""

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            "web_fetch",
            "Fetch a public HTTPS documentation page after showing its destination for approval.",
            {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "minLength": 12, "maxLength": 2048},
                    "allowed_domains": {
                        "type": "array",
                        "maxItems": 16,
                        "items": {"type": "string", "minLength": 1, "maxLength": 253},
                    },
                },
                "required": ["url"],
                "additionalProperties": False,
            },
            permission=Permission.HIGH_RISK,
            timeout_seconds=20.0,
        )

    async def run(self, call: ToolCall, context: ToolContext) -> ToolResult:
        if call.call_id not in context.approved_call_ids:
            return ToolResult(
                call.call_id,
                call.name,
                False,
                error="web fetch destination was not explicitly approved",
                error_type=ErrorType.PERMISSION,
            )
        raw_url = call.arguments.get("url")
        raw_domains = call.arguments.get("allowed_domains", [])
        if (
            not isinstance(raw_url, str)
            or not isinstance(raw_domains, list)
            or not all(isinstance(item, str) for item in raw_domains)
        ):
            return ToolResult(call.call_id, call.name, False, error="invalid web fetch arguments")
        try:
            data = await asyncio.to_thread(
                self._fetch,
                raw_url,
                tuple(item.casefold() for item in raw_domains),
                self.spec.timeout_seconds,
            )
        except (OSError, RuntimeError, ValueError) as error:
            return ToolResult(
                call.call_id,
                call.name,
                False,
                error=str(error),
                error_type=ErrorType.EXECUTION,
                recoverable=True,
            )
        return ToolResult(call.call_id, call.name, True, data)

    @staticmethod
    def _fetch(
        url: str, allowed_domains: tuple[str, ...], timeout_seconds: float
    ) -> dict[str, object]:
        parsed = urlsplit(url)
        if (
            parsed.scheme != "https"
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or parsed.fragment
        ):
            raise ValueError("web_fetch requires an HTTPS URL without credentials or fragments")
        host = parsed.hostname.casefold()
        if allowed_domains and not any(
            host == domain or host.endswith("." + domain) for domain in allowed_domains
        ):
            raise ValueError("URL host is outside allowed_domains")
        if host == "localhost" or host.endswith(".local"):
            raise ValueError("web_fetch does not allow loopback or local hostnames")
        addresses = {
            item[4][0]
            for item in socket.getaddrinfo(host, parsed.port or 443, type=socket.SOCK_STREAM)
        }
        if not addresses:
            raise RuntimeError("URL host did not resolve")
        public_addresses: list[str] = []
        for address in addresses:
            ip = ipaddress.ip_address(address)
            if not ip.is_global:
                raise ValueError("web_fetch rejected a non-public resolved address")
            public_addresses.append(address)
        path = (parsed.path or "/") + (f"?{parsed.query}" if parsed.query else "")
        connection = _PinnedHttpsConnection(
            host, parsed.port or 443, sorted(public_addresses)[0], timeout=timeout_seconds
        )
        try:
            connection.request(
                "GET",
                path,
                headers={"Accept": "text/plain, text/html, application/json, application/xml"},
            )
            response = connection.getresponse()
            if response.status < 200 or response.status >= 300:
                raise RuntimeError(f"web server returned HTTP {response.status}")
            content_type = (
                response.getheader("Content-Type", "text/plain").split(";", 1)[0].casefold()
            )
            if not any(content_type.startswith(prefix) for prefix in _TEXT_TYPES):
                raise ValueError(f"web_fetch rejected non-text response type {content_type}")
            body = response.read(_MAX_BYTES + 1)
        finally:
            connection.close()
        if len(body) > _MAX_BYTES:
            raise ValueError("web_fetch response exceeds 1 MB")
        return {
            "url": url,
            "content_type": content_type,
            "content": body.decode("utf-8", errors="replace"),
        }


def default_web_tools() -> list[WebFetchTool]:
    return [WebFetchTool()]
