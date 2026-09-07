"""Standard newline-delimited JSON-RPC transport for stdio MCP servers."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

from repopilot.core.contracts import JSONValue
from repopilot.mcp.config import MCPServerConfig
from repopilot.security.redaction import redact_text

_MAX_STDERR_BYTES = 8_000


class StdioMCPTransport:
    """Own one tokenized stdio child process and serialize JSON-RPC exchanges."""

    def __init__(
        self, config: MCPServerConfig, *, cwd: Path, timeout_seconds: float = 30.0
    ) -> None:
        if not 1 <= timeout_seconds <= 120:
            raise ValueError("MCP timeout_seconds must be between 1 and 120")
        self.config = config
        self.cwd = cwd.resolve()
        self.timeout_seconds = timeout_seconds
        self._process: asyncio.subprocess.Process | None = None
        self._lock = asyncio.Lock()
        self._stderr_tail = bytearray()
        self._stderr_reader: asyncio.Task[None] | None = None

    async def start(self) -> None:
        """Launch the configured child without a shell after workspace trust has occurred."""
        if self._process is not None and self._process.returncode is None:
            return
        if not self.cwd.is_dir():
            raise ValueError(f"MCP cwd does not exist: {self.cwd}")
        environment = {"PATH": os.environ.get("PATH", ""), "PYTHONIOENCODING": "utf-8"}
        # These variables are required by normal Windows process creation but
        # do not carry provider credentials.  Do not inherit the full parent
        # environment: an MCP child must not receive API keys by accident.
        for name in ("SYSTEMROOT", "WINDIR", "COMSPEC", "TEMP", "TMP"):
            value = os.environ.get(name)
            if value:
                environment[name] = value
        self._process = await asyncio.create_subprocess_exec(
            *self.config.command,
            cwd=str(self.cwd),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=environment,
        )
        self._stderr_tail.clear()
        self._stderr_reader = asyncio.create_task(self._collect_stderr(self._process.stderr))

    async def request(self, payload: dict[str, JSONValue]) -> dict[str, JSONValue]:
        """Send one line-framed JSON-RPC request and await its matching response line."""
        async with self._lock:
            await self.start()
            process = self._process
            if process is None or process.stdin is None or process.stdout is None:
                raise RuntimeError("MCP stdio process did not expose required streams")
            if process.returncode is not None:
                raise RuntimeError(
                    f"MCP server exited with code {process.returncode}{self._stderr_context()}"
                )
            encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
            try:
                process.stdin.write(encoded + b"\n")
                await process.stdin.drain()
            except (BrokenPipeError, ConnectionResetError) as error:
                await process.wait()
                raise RuntimeError(f"MCP server closed stdin{self._stderr_context()}") from error
            line = await asyncio.wait_for(process.stdout.readline(), timeout=self.timeout_seconds)
            if not line:
                exit_code = await process.wait()
                raise RuntimeError(
                    f"MCP server exited with code {exit_code} before responding"
                    + self._stderr_context()
                )
            try:
                raw = json.loads(line.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise RuntimeError(f"MCP server sent invalid JSON-RPC: {error}") from error
            if not isinstance(raw, dict):
                raise RuntimeError("MCP server response must be an object")
            if raw.get("jsonrpc") != "2.0" or raw.get("id") != payload.get("id"):
                raise RuntimeError("MCP server response did not match the JSON-RPC request")
            return raw

    async def close(self) -> None:
        """Terminate only the one stdio child owned by this transport."""
        process = self._process
        self._process = None
        try:
            if process is None or process.returncode is not None:
                return
            process.terminate()
            try:
                await asyncio.wait_for(process.wait(), timeout=3.0)
            except TimeoutError:
                process.kill()
                await process.wait()
        finally:
            reader = self._stderr_reader
            self._stderr_reader = None
            if reader is not None:
                try:
                    await asyncio.wait_for(reader, timeout=1.0)
                except TimeoutError:
                    reader.cancel()
                    await asyncio.gather(reader, return_exceptions=True)

    async def _collect_stderr(self, stream: asyncio.StreamReader | None) -> None:
        if stream is None:
            return
        while chunk := await stream.read(4_096):
            remaining = _MAX_STDERR_BYTES - len(self._stderr_tail)
            if remaining > 0:
                self._stderr_tail.extend(chunk[-remaining:])

    def _stderr_context(self) -> str:
        if not self._stderr_tail:
            return ""
        text = redact_text(bytes(self._stderr_tail).decode("utf-8", errors="replace")).strip()
        return f"; stderr: {text[-2_000:]}" if text else ""
