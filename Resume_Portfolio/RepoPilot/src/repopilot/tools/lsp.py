"""Opt-in stdio Language Server Protocol tools for a trusted workspace."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any, cast

from repopilot.core.contracts import ErrorType, Permission, ToolCall, ToolResult, ToolSpec
from repopilot.security.paths import resolve_workspace_path
from repopilot.tools.base import ToolContext
from repopilot.tools.shell import validate_tokenized_command

_LANGUAGES = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascriptreact",
    ".ts": "typescript",
    ".tsx": "typescriptreact",
    ".go": "go",
    ".rs": "rust",
    ".java": "java",
}
_CONFIG_RELATIVE = Path(".repopilot") / "lsp.json"


def lsp_config_path(workspace: Path) -> Path:
    return workspace.resolve() / _CONFIG_RELATIVE


def load_lsp_servers(workspace: Path) -> dict[str, tuple[str, ...]]:
    """Load a small, tokenized, project-local language-server map without launching it."""

    path = lsp_config_path(workspace)
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid LSP configuration: {error}") from error
    servers = raw.get("servers", {}) if isinstance(raw, dict) else {}
    if not isinstance(servers, dict) or len(servers) > 8:
        raise ValueError("LSP configuration must contain at most eight language servers")
    loaded: dict[str, tuple[str, ...]] = {}
    for language, definition in servers.items():
        if not isinstance(language, str) or not isinstance(definition, dict):
            raise ValueError("LSP servers must map language names to objects")
        command = definition.get("command")
        arguments = definition.get("args", [])
        if (
            not isinstance(command, str)
            or not isinstance(arguments, list)
            or not all(isinstance(item, str) for item in arguments)
        ):
            raise ValueError(f"LSP server {language!r} needs command and string args")
        tokens = (command, *arguments)
        validate_tokenized_command(list(tokens))
        loaded[language] = tokens
    return loaded


def save_lsp_servers(workspace: Path, servers: dict[str, tuple[str, ...]]) -> None:
    """Persist validated LSP commands; launching still needs a high-risk approval."""

    if len(servers) > 8:
        raise ValueError("LSP configuration must contain at most eight language servers")
    for language, command in servers.items():
        if not language or not command:
            raise ValueError("LSP language and command must not be empty")
        validate_tokenized_command(list(command))
    path = lsp_config_path(workspace)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "servers": {
            language: {"command": command[0], "args": list(command[1:])}
            for language, command in sorted(servers.items())
        }
    }
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


class _LspStdioClient:
    """Minimal request/response client; notification payloads are deliberately ignored."""

    def __init__(
        self, command: tuple[str, ...], *, workspace: Path, timeout_seconds: float = 20.0
    ) -> None:
        self.command = command
        self.workspace = workspace.resolve()
        self.timeout_seconds = timeout_seconds
        self.process: asyncio.subprocess.Process | None = None
        self._next_id = 1

    async def start(self) -> None:
        environment = {"PATH": os.environ.get("PATH", ""), "PYTHONIOENCODING": "utf-8"}
        self.process = await asyncio.create_subprocess_exec(
            *self.command,
            cwd=str(self.workspace),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=environment,
        )

    async def initialize_document(self, path: Path) -> None:
        language = _LANGUAGES.get(path.suffix.lower())
        if language is None:
            raise ValueError(f"no LSP language mapping for {path.suffix or 'this file type'}")
        await self.request(
            "initialize",
            {
                "processId": None,
                "rootUri": self.workspace.as_uri(),
                "capabilities": {"textDocument": {"diagnostic": {}}},
            },
        )
        await self.notify("initialized", {})
        await self.notify(
            "textDocument/didOpen",
            {
                "textDocument": {
                    "uri": path.as_uri(),
                    "languageId": language,
                    "version": 1,
                    "text": path.read_text(encoding="utf-8"),
                }
            },
        )

    async def request(self, method: str, params: dict[str, Any]) -> object:
        request_id = self._next_id
        self._next_id += 1
        await self._send({"jsonrpc": "2.0", "id": request_id, "method": method, "params": params})
        while True:
            message = await asyncio.wait_for(self._read(), timeout=self.timeout_seconds)
            if message.get("id") != request_id:
                continue
            if "error" in message:
                raise RuntimeError(f"LSP {method} failed: {message['error']}")
            return message.get("result")

    async def notify(self, method: str, params: dict[str, Any]) -> None:
        await self._send({"jsonrpc": "2.0", "method": method, "params": params})

    async def _send(self, payload: dict[str, Any]) -> None:
        process = self.process
        if process is None or process.stdin is None or process.returncode is not None:
            raise RuntimeError("LSP server is not running")
        encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        process.stdin.write(f"Content-Length: {len(encoded)}\r\n\r\n".encode("ascii") + encoded)
        await process.stdin.drain()

    async def _read(self) -> dict[str, Any]:
        process = self.process
        if process is None or process.stdout is None:
            raise RuntimeError("LSP server has no stdout")
        size: int | None = None
        while True:
            line = await process.stdout.readline()
            if not line:
                raise RuntimeError("LSP server closed stdout")
            if line in {b"\n", b"\r\n"}:
                break
            name, separator, value = line.decode("ascii", errors="replace").partition(":")
            if separator and name.casefold() == "content-length":
                try:
                    size = int(value.strip())
                except ValueError as error:
                    raise RuntimeError("LSP sent an invalid Content-Length") from error
        if size is None or not 0 <= size <= 2_000_000:
            raise RuntimeError("LSP response has an invalid Content-Length")
        try:
            raw = json.loads((await process.stdout.readexactly(size)).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise RuntimeError(f"LSP server sent invalid JSON: {error}") from error
        if not isinstance(raw, dict):
            raise RuntimeError("LSP response must be an object")
        return cast(dict[str, Any], raw)

    async def close(self) -> None:
        process = self.process
        self.process = None
        if process is None or process.returncode is not None:
            return
        process.terminate()
        try:
            await asyncio.wait_for(process.wait(), timeout=2.0)
        except TimeoutError:
            process.kill()
            await process.wait()


class _LspTool:
    method: str
    description: str

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            self.method,
            self.description,
            self._schema(),
            permission=Permission.HIGH_RISK,
            timeout_seconds=25.0,
            read_only=True,
        )

    async def run(self, call: ToolCall, context: ToolContext) -> ToolResult:
        try:
            raw_path = call.arguments.get("path")
            if not isinstance(raw_path, str):
                raise ValueError("path must be a string")
            path = resolve_workspace_path(context.task, raw_path)
            if not path.is_file() or path.stat().st_size > 1_000_000:
                raise ValueError("LSP requires a text file no larger than 1 MB")
            language = _LANGUAGES.get(path.suffix.lower())
            if language is None:
                raise ValueError(f"no LSP language mapping for {path.suffix or 'this file type'}")
            command = load_lsp_servers(context.task.workspace).get(language)
            if command is None:
                raise ValueError(
                    f"no {language} LSP server is configured at {_CONFIG_RELATIVE.as_posix()}"
                )
            client = _LspStdioClient(command, workspace=context.task.workspace)
            await client.start()
            try:
                await client.initialize_document(path)
                data = await self._request(client, path, call.arguments)
            finally:
                await client.close()
            return ToolResult(call.call_id, call.name, True, {"path": raw_path, "result": data})
        except (OSError, RuntimeError, ValueError) as error:
            return ToolResult(
                call.call_id,
                call.name,
                False,
                error=str(error),
                error_type=ErrorType.EXECUTION,
                recoverable=True,
            )

    def _schema(self) -> dict[str, object]:
        raise NotImplementedError

    async def _request(
        self, client: _LspStdioClient, path: Path, arguments: dict[str, object]
    ) -> object:
        raise NotImplementedError


class LspDiagnosticsTool(_LspTool):
    method = "lsp_diagnostics"
    description = "Ask a configured local language server for diagnostics on one file."

    def _schema(self) -> dict[str, object]:
        return {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
            "additionalProperties": False,
        }

    async def _request(
        self, client: _LspStdioClient, path: Path, arguments: dict[str, object]
    ) -> object:
        del arguments
        return await client.request(
            "textDocument/diagnostic", {"textDocument": {"uri": path.as_uri()}}
        )


class LspDefinitionTool(_LspTool):
    method = "lsp_definition"
    description = (
        "Ask a configured local language server for the definition at a zero-based position."
    )

    def _schema(self) -> dict[str, object]:
        return {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "line": {"type": "integer", "minimum": 0},
                "character": {"type": "integer", "minimum": 0},
            },
            "required": ["path", "line", "character"],
            "additionalProperties": False,
        }

    async def _request(
        self, client: _LspStdioClient, path: Path, arguments: dict[str, object]
    ) -> object:
        line = arguments.get("line")
        character = arguments.get("character")
        if not isinstance(line, int) or not isinstance(character, int) or line < 0 or character < 0:
            raise ValueError("line and character must be non-negative integers")
        return await client.request(
            "textDocument/definition",
            {
                "textDocument": {"uri": path.as_uri()},
                "position": {"line": line, "character": character},
            },
        )


def default_lsp_tools() -> list[LspDiagnosticsTool | LspDefinitionTool]:
    return [LspDiagnosticsTool(), LspDefinitionTool()]
