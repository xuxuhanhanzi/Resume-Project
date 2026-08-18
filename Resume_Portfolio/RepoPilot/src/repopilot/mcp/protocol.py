"""A minimal MCP-shaped JSON-RPC boundary for learning and in-process tests.

This intentionally implements only ``initialize``, ``tools/list``, and ``tools/call``.
It demonstrates discovery and remote tool adaptation without claiming full wire-level
compatibility with every MCP transport or protocol revision.
"""

from __future__ import annotations

from typing import Protocol, cast

from repopilot.core.contracts import JSONValue, Permission, ToolCall, ToolResult, ToolSpec
from repopilot.tools.base import ToolContext, ToolRegistry


class MCPServer:
    """Serve a ToolRegistry through a tiny JSON-RPC method surface."""

    def __init__(
        self, registry: ToolRegistry, context: ToolContext, *, name: str = "repopilot"
    ) -> None:
        self.registry = registry
        self.context = context
        self.name = name

    async def handle(self, request: dict[str, JSONValue]) -> dict[str, JSONValue]:
        request_id = request.get("id")
        method = request.get("method")
        params = request.get("params", {})
        try:
            if method == "initialize":
                result: JSONValue = {
                    "serverInfo": {"name": self.name, "version": "0.1.0"},
                    "capabilities": {"tools": {"listChanged": False}},
                    "protocolVersion": (
                        params.get("protocolVersion", "educational-subset")
                        if isinstance(params, dict)
                        else "educational-subset"
                    ),
                }
            elif method == "tools/list":
                result = {"tools": [self._tool_description(spec) for spec in self.registry.specs()]}
            elif method == "tools/call":
                result = await self._call_tool(params)
            else:
                return self._error(request_id, -32601, f"unknown method: {method}")
        except (KeyError, TypeError, ValueError) as error:
            return self._error(request_id, -32602, str(error))
        return {"jsonrpc": "2.0", "id": request_id, "result": result}

    @staticmethod
    def _tool_description(spec: ToolSpec) -> dict[str, JSONValue]:
        return {
            "name": spec.name,
            "description": spec.description,
            "inputSchema": spec.input_schema,
            "x-repopilot": {
                "permission": spec.permission.value,
                "readOnly": spec.read_only,
                "idempotent": spec.idempotent,
                "timeoutSeconds": spec.timeout_seconds,
            },
        }

    async def _call_tool(self, params: JSONValue) -> JSONValue:
        if not isinstance(params, dict):
            raise ValueError("tools/call params must be an object")
        name = str(params["name"])
        arguments = params.get("arguments", {})
        if not isinstance(arguments, dict):
            raise ValueError("tools/call arguments must be an object")
        call_id = str(params.get("callId", "mcp-call"))
        tool = self.registry.get(name)
        if tool is None:
            raise ValueError(f"unknown tool: {name}")
        result = await tool.run(ToolCall(call_id, name, arguments), self.context)
        return {
            "content": [{"type": "text", "text": str(result.to_dict())}],
            "toolResult": result.to_dict(),
        }

    @staticmethod
    def _error(request_id: JSONValue, code: int, message: str) -> dict[str, JSONValue]:
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {"code": code, "message": message},
        }


class MCPTransport(Protocol):
    """Transport boundary so stdio/HTTP can replace the in-process demo later."""

    async def request(self, payload: dict[str, JSONValue]) -> dict[str, JSONValue]:
        """Round-trip one JSON-RPC request."""
        ...


class InProcessMCPTransport:
    """Zero-network transport used to demonstrate protocol behavior deterministically."""

    def __init__(self, server: MCPServer) -> None:
        self.server = server

    async def request(self, payload: dict[str, JSONValue]) -> dict[str, JSONValue]:
        return await self.server.handle(payload)


class MCPClient:
    """Discover and call remote tools through an MCP transport."""

    def __init__(self, transport: MCPTransport) -> None:
        self.transport = transport
        self._next_id = 1

    async def initialize(self) -> dict[str, JSONValue]:
        return await self._request("initialize", {"protocolVersion": "educational-subset"})

    async def list_tools(self) -> tuple[ToolSpec, ...]:
        result = await self._request("tools/list", {})
        raw_tools = result.get("tools", [])
        if not isinstance(raw_tools, list):
            raise RuntimeError("MCP tools/list result is malformed")
        specs: list[ToolSpec] = []
        for raw in raw_tools:
            if not isinstance(raw, dict):
                continue
            metadata = raw.get("x-repopilot", {})
            if not isinstance(metadata, dict):
                metadata = {}
            schema = raw.get("inputSchema", {})
            if not isinstance(schema, dict):
                raise RuntimeError("MCP tool inputSchema must be an object")
            specs.append(
                ToolSpec(
                    name=str(raw["name"]),
                    description=str(raw.get("description", "")),
                    input_schema=schema,
                    permission=Permission(str(metadata.get("permission", Permission.READ.value))),
                    timeout_seconds=float(metadata.get("timeoutSeconds", 10.0)),
                    read_only=bool(metadata.get("readOnly", True)),
                    idempotent=bool(metadata.get("idempotent", True)),
                )
            )
        return tuple(specs)

    async def call_tool(self, call: ToolCall) -> ToolResult:
        result = await self._request(
            "tools/call",
            {"name": call.name, "arguments": call.arguments, "callId": call.call_id},
        )
        raw_result = result.get("toolResult")
        if not isinstance(raw_result, dict):
            raise RuntimeError("MCP tools/call result is malformed")
        return ToolResult.from_dict(cast(dict[str, JSONValue], raw_result))

    async def _request(self, method: str, params: dict[str, JSONValue]) -> dict[str, JSONValue]:
        request_id = self._next_id
        self._next_id += 1
        response = await self.transport.request(
            {"jsonrpc": "2.0", "id": request_id, "method": method, "params": params}
        )
        if "error" in response:
            raise RuntimeError(f"MCP request failed: {response['error']}")
        result = response.get("result")
        if not isinstance(result, dict):
            raise RuntimeError("MCP response result must be an object")
        return result


class MCPRemoteTool:
    """Adapt one discovered MCP capability back into RepoPilot's Tool protocol."""

    def __init__(self, client: MCPClient, spec: ToolSpec) -> None:
        self.client = client
        self._spec = spec

    @property
    def spec(self) -> ToolSpec:
        return self._spec

    async def run(self, call: ToolCall, context: ToolContext) -> ToolResult:
        del context
        return await self.client.call_tool(call)
