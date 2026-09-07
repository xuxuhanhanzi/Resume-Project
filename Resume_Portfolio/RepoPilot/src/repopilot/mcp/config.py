"""Validated project-local configuration for stdio MCP servers."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

from repopilot.tools.shell import validate_tokenized_command

_SERVER_NAME = re.compile(r"[A-Za-z][A-Za-z0-9_-]{0,63}\Z")
_TOOL_NAME = re.compile(r"[A-Za-z][A-Za-z0-9_-]{0,63}\Z")


@dataclass(frozen=True, slots=True)
class MCPServerConfig:
    """One project-approved stdio or loopback-HTTP server definition."""

    name: str
    command: tuple[str, ...] = ()
    url: str | None = None
    allowed_tools: tuple[str, ...] = ()
    denied_tools: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not _SERVER_NAME.fullmatch(self.name):
            raise ValueError("MCP server names must use letters, digits, underscores, or hyphens")
        if len(self.allowed_tools) > 64 or len(self.denied_tools) > 64:
            raise ValueError("MCP tool policy can contain at most 64 names per list")
        if not all(
            _TOOL_NAME.fullmatch(name) for name in (*self.allowed_tools, *self.denied_tools)
        ):
            raise ValueError("MCP tool policy names must be simple remote tool names")
        if len(set(self.allowed_tools)) != len(self.allowed_tools) or len(
            set(self.denied_tools)
        ) != len(self.denied_tools):
            raise ValueError("MCP tool policy names must not be duplicated")
        overlap = set(self.allowed_tools) & set(self.denied_tools)
        if overlap:
            raise ValueError("MCP tool policy cannot both allow and deny the same tool")
        if self.url is None:
            validate_tokenized_command(list(self.command))
            return
        if self.command:
            raise ValueError("HTTP MCP configuration must not include a command")
        parts = urlsplit(self.url)
        if (
            parts.scheme != "http"
            or parts.hostname not in {"127.0.0.1", "localhost", "::1"}
            or not parts.netloc
            or parts.username is not None
            or parts.password is not None
            or parts.fragment
        ):
            raise ValueError("HTTP MCP URL must use loopback HTTP with no credentials")


def load_project_mcp_config(project_root: Path) -> tuple[MCPServerConfig, ...]:
    """Load up to eight configured stdio servers; configuration never launches them."""
    path = project_mcp_config_path(project_root)
    if not path.exists():
        return ()
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid MCP configuration: {error}") from error
    servers = raw.get("servers", {}) if isinstance(raw, dict) else {}
    if not isinstance(servers, dict) or len(servers) > 8:
        raise ValueError("MCP configuration must contain at most eight named servers")
    result: list[MCPServerConfig] = []
    for name, definition in servers.items():
        if not isinstance(name, str) or not isinstance(definition, dict):
            raise ValueError("MCP server definitions must map names to objects")
        endpoint = definition.get("url")
        executable = definition.get("command")
        arguments = definition.get("args", [])
        allowed_tools = definition.get("allow_tools", [])
        denied_tools = definition.get("deny_tools", [])
        if (
            not isinstance(allowed_tools, list)
            or not all(isinstance(name, str) for name in allowed_tools)
            or not isinstance(denied_tools, list)
            or not all(isinstance(name, str) for name in denied_tools)
        ):
            raise ValueError(f"MCP server {name!r} policy lists must contain strings")
        if isinstance(endpoint, str):
            if set(definition) - {"url", "allow_tools", "deny_tools"}:
                raise ValueError(f"HTTP MCP server {name!r} has unsupported fields")
            result.append(
                MCPServerConfig(
                    name,
                    url=endpoint,
                    allowed_tools=tuple(allowed_tools),
                    denied_tools=tuple(denied_tools),
                )
            )
            continue
        if (
            set(definition) - {"command", "args", "allow_tools", "deny_tools"}
            or not isinstance(executable, str)
            or not isinstance(arguments, list)
            or not all(isinstance(argument, str) for argument in arguments)
        ):
            raise ValueError(f"MCP server {name!r} needs command and string args")
        result.append(
            MCPServerConfig(
                name,
                (executable, *arguments),
                allowed_tools=tuple(allowed_tools),
                denied_tools=tuple(denied_tools),
            )
        )
    return tuple(result)


def project_mcp_config_path(project_root: Path) -> Path:
    """Return the only project-scoped MCP configuration location."""
    return project_root.resolve() / ".repopilot" / "mcp.json"


def save_project_mcp_config(project_root: Path, configs: tuple[MCPServerConfig, ...]) -> None:
    """Atomically write a bounded, tokenized project MCP configuration."""
    if len(configs) > 8:
        raise ValueError("MCP configuration must contain at most eight named servers")
    names = [config.name for config in configs]
    if len(names) != len(set(names)):
        raise ValueError("MCP server names must be unique")
    path = project_mcp_config_path(project_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    servers: dict[str, dict[str, object]] = {}
    for config in sorted(configs, key=lambda item: item.name):
        if config.url is not None:
            servers[config.name] = {"url": config.url}
        else:
            servers[config.name] = {
                "command": config.command[0],
                "args": list(config.command[1:]),
            }
        if config.allowed_tools:
            servers[config.name]["allow_tools"] = list(config.allowed_tools)
        if config.denied_tools:
            servers[config.name]["deny_tools"] = list(config.denied_tools)
    payload: dict[str, object] = {"servers": servers}
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def upsert_project_mcp_config(project_root: Path, config: MCPServerConfig) -> None:
    """Add or replace one named project MCP server without launching it."""
    existing = {item.name: item for item in load_project_mcp_config(project_root)}
    existing[config.name] = config
    save_project_mcp_config(project_root, tuple(existing.values()))


def remove_project_mcp_config(project_root: Path, name: str) -> bool:
    """Remove one named entry while retaining the configuration file itself."""
    if not _SERVER_NAME.fullmatch(name):
        raise ValueError("MCP server names must use letters, digits, underscores, or hyphens")
    existing = {item.name: item for item in load_project_mcp_config(project_root)}
    if name not in existing:
        return False
    del existing[name]
    save_project_mcp_config(project_root, tuple(existing.values()))
    return True
