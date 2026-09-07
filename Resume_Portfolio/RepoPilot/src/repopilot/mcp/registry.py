"""Connect trusted stdio MCP servers while preserving RepoPilot tool governance."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from repopilot.core.contracts import Permission, ToolCall, ToolResult, ToolSpec
from repopilot.mcp.config import MCPServerConfig
from repopilot.mcp.http import StreamableHttpMCPTransport
from repopilot.mcp.protocol import MCPClient, MCPPrompt, MCPPromptMessage, MCPResource
from repopilot.mcp.stdio import StdioMCPTransport
from repopilot.tools.base import Tool, ToolContext

_REMOTE_TOOL_NAME = re.compile(r"[A-Za-z][A-Za-z0-9_-]{0,63}\Z")


class GovernedMCPTool:
    """Prefix a remote tool and classify it as high-risk until separately profiled."""

    def __init__(self, server_name: str, client: MCPClient, source_spec: ToolSpec) -> None:
        if not _REMOTE_TOOL_NAME.fullmatch(source_spec.name):
            raise ValueError(
                "MCP remote tool names must use letters, digits, underscores, or hyphens"
            )
        self.client = client
        self.source_spec = source_spec
        self.server_name = server_name
        self._spec = ToolSpec(
            name=f"mcp__{server_name}__{source_spec.name}",
            description=f"MCP server {server_name}: {source_spec.description}",
            input_schema=source_spec.input_schema,
            permission=Permission.HIGH_RISK,
            read_only=False,
            idempotent=False,
        )

    @property
    def spec(self) -> ToolSpec:
        return self._spec

    async def run(self, call: ToolCall, context: ToolContext) -> ToolResult:
        del context
        forwarded = ToolCall(call.call_id, self.source_spec.name, call.arguments)
        result = await self.client.call_tool(forwarded)
        return ToolResult(
            call.call_id,
            self.spec.name,
            result.ok,
            result.data,
            result.error,
            result.error_type,
            result.recoverable,
            result.side_effect,
            result.cached,
        )


@dataclass(frozen=True, slots=True)
class MCPServerCapabilities:
    """Cached optional MCP discovery results for one trusted stdio server."""

    name: str
    client: MCPClient
    tool_names: tuple[str, ...] = ()
    resources: tuple[MCPResource, ...] = ()
    prompts: tuple[MCPPrompt, ...] = ()


@dataclass(slots=True)
class MCPProjectRegistry:
    """Own configured transport lifetimes and their high-risk tool adapters."""

    transports: tuple[StdioMCPTransport | StreamableHttpMCPTransport, ...]
    tools: tuple[Tool, ...]
    servers: tuple[MCPServerCapabilities, ...] = ()

    @classmethod
    async def connect(
        cls,
        configs: tuple[MCPServerConfig, ...],
        *,
        project_root: Path,
        timeout_seconds: float = 30.0,
    ) -> MCPProjectRegistry:
        if not 1 <= timeout_seconds <= 120:
            raise ValueError("MCP timeout_seconds must be between 1 and 120")
        transports: list[StdioMCPTransport | StreamableHttpMCPTransport] = []
        tools: list[Tool] = []
        servers: list[MCPServerCapabilities] = []
        try:
            for config in configs:
                transport: StdioMCPTransport | StreamableHttpMCPTransport
                if config.url is not None:
                    transport = StreamableHttpMCPTransport(config, timeout_seconds=timeout_seconds)
                else:
                    transport = StdioMCPTransport(
                        config, cwd=project_root, timeout_seconds=timeout_seconds
                    )
                transports.append(transport)
                client = MCPClient(transport)
                await client.initialize()
                remote_specs = await client.list_tools()
                exposed_specs = _apply_tool_policy(config, remote_specs)
                tools.extend(GovernedMCPTool(config.name, client, spec) for spec in exposed_specs)
                resources = await _optional_resources(client)
                prompts = await _optional_prompts(client)
                servers.append(
                    MCPServerCapabilities(
                        config.name,
                        client,
                        tuple(spec.name for spec in exposed_specs),
                        resources,
                        prompts,
                    )
                )
        except Exception:
            for transport in transports:
                await transport.close()
            raise
        return cls(tuple(transports), tuple(tools), tuple(servers))

    async def read_resource(self, server_name: str, uri: str) -> tuple[MCPResource, ...]:
        """Read a named resource from a server started for this trusted session."""
        return await self._server(server_name).client.read_resource(uri)

    async def get_prompt(
        self, server_name: str, prompt_name: str, *, arguments: dict[str, str] | None = None
    ) -> tuple[MCPPromptMessage, ...]:
        """Fetch a named MCP prompt without injecting it into model context."""
        return await self._server(server_name).client.get_prompt(prompt_name, arguments=arguments)

    def _server(self, name: str) -> MCPServerCapabilities:
        matches = [server for server in self.servers if server.name == name]
        if len(matches) != 1:
            raise ValueError(f"MCP server is not active: {name}")
        return matches[0]

    async def close(self) -> None:
        for transport in self.transports:
            await transport.close()


def _apply_tool_policy(
    config: MCPServerConfig, specs: tuple[ToolSpec, ...]
) -> tuple[ToolSpec, ...]:
    """Apply a server-local availability policy without reducing high-risk governance."""

    allowed = set(config.allowed_tools)
    denied = set(config.denied_tools)
    return tuple(
        spec for spec in specs if (not allowed or spec.name in allowed) and spec.name not in denied
    )


async def _optional_resources(client: MCPClient) -> tuple[MCPResource, ...]:
    try:
        return await client.list_resources()
    except RuntimeError:
        return ()


async def _optional_prompts(client: MCPClient) -> tuple[MCPPrompt, ...]:
    try:
        return await client.list_prompts()
    except RuntimeError:
        return ()
