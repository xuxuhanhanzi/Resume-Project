"""A bounded MCP JSON-RPC boundary for tools, resources, and prompts.

The stdio transport intentionally remains local and explicit: this module supports
the common discovery surface without pretending to implement OAuth or every MCP
transport revision.  Unknown optional capabilities degrade gracefully in the
project registry so older tool-only servers remain usable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, cast

from repopilot.core.contracts import JSONValue, Permission, ToolCall, ToolResult, ToolSpec
from repopilot.tools.base import ToolContext, ToolRegistry


@dataclass(frozen=True, slots=True)
class MCPResource:
    """A small text resource exposed by an MCP server."""

    uri: str
    name: str
    text: str = ""
    description: str = ""
    mime_type: str = "text/plain"

    def __post_init__(self) -> None:
        if not self.uri.strip() or not self.name.strip():
            raise ValueError("MCP resources require non-empty uri and name")
        if len(self.text) > 200_000:
            raise ValueError("MCP resource text must not exceed 200000 characters")


@dataclass(frozen=True, slots=True)
class MCPPrompt:
    """A named prompt template exposed by an MCP server."""

    name: str
    template: str
    description: str = ""
    arguments: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.name.strip() or not all(argument.strip() for argument in self.arguments):
            raise ValueError("MCP prompt names and arguments must be non-empty")
        if len(self.template) > 32_000:
            raise ValueError("MCP prompt templates must not exceed 32000 characters")


@dataclass(frozen=True, slots=True)
class MCPPromptMessage:
    """One text message returned by ``prompts/get``."""

    role: str
    text: str


class MCPServer:
    """Serve a ToolRegistry through a tiny JSON-RPC method surface."""

    def __init__(
        self,
        registry: ToolRegistry,
        context: ToolContext,
        *,
        name: str = "repopilot",
        resources: tuple[MCPResource, ...] = (),
        prompts: tuple[MCPPrompt, ...] = (),
    ) -> None:
        self.registry = registry
        self.context = context
        self.name = name
        self.resources = {resource.uri: resource for resource in resources}
        self.prompts = {prompt.name: prompt for prompt in prompts}
        if len(self.resources) != len(resources) or len(self.prompts) != len(prompts):
            raise ValueError("MCP resource URIs and prompt names must be unique")

    async def handle(self, request: dict[str, JSONValue]) -> dict[str, JSONValue]:
        request_id = request.get("id")
        method = request.get("method")
        params = request.get("params", {})
        try:
            if method == "initialize":
                result: JSONValue = {
                    "serverInfo": {"name": self.name, "version": "0.1.0"},
                    "capabilities": {
                        "tools": {"listChanged": False},
                        "resources": {"listChanged": False, "subscribe": False},
                        "prompts": {"listChanged": False},
                    },
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
            elif method == "resources/list":
                result = {
                    "resources": [
                        self._resource_description(resource) for resource in self.resources.values()
                    ]
                }
            elif method == "resources/read":
                result = self._read_resource(params)
            elif method == "prompts/list":
                result = {
                    "prompts": [
                        self._prompt_description(prompt) for prompt in self.prompts.values()
                    ]
                }
            elif method == "prompts/get":
                result = self._get_prompt(params)
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

    @staticmethod
    def _resource_description(resource: MCPResource) -> dict[str, JSONValue]:
        return {
            "uri": resource.uri,
            "name": resource.name,
            "description": resource.description,
            "mimeType": resource.mime_type,
        }

    @staticmethod
    def _prompt_description(prompt: MCPPrompt) -> dict[str, JSONValue]:
        return {
            "name": prompt.name,
            "description": prompt.description,
            "arguments": [{"name": argument, "required": False} for argument in prompt.arguments],
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

    def _read_resource(self, params: JSONValue) -> JSONValue:
        if not isinstance(params, dict) or not isinstance(params.get("uri"), str):
            raise ValueError("resources/read params need a string uri")
        resource = self.resources.get(params["uri"])
        if resource is None:
            raise ValueError(f"unknown resource: {params['uri']}")
        return {
            "contents": [
                {
                    "uri": resource.uri,
                    "mimeType": resource.mime_type,
                    "text": resource.text,
                }
            ]
        }

    def _get_prompt(self, params: JSONValue) -> JSONValue:
        if not isinstance(params, dict) or not isinstance(params.get("name"), str):
            raise ValueError("prompts/get params need a string name")
        prompt = self.prompts.get(params["name"])
        if prompt is None:
            raise ValueError(f"unknown prompt: {params['name']}")
        arguments = params.get("arguments", {})
        if not isinstance(arguments, dict) or not all(
            isinstance(key, str) and isinstance(value, str) for key, value in arguments.items()
        ):
            raise ValueError("prompts/get arguments must map strings to strings")
        text = prompt.template
        for name in prompt.arguments:
            text = text.replace("{{" + name + "}}", str(arguments.get(name, "")))
        return {
            "description": prompt.description,
            "messages": [{"role": "user", "content": {"type": "text", "text": text}}],
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
        return await self._request("initialize", {"protocolVersion": "2025-11-25"})

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
        if isinstance(raw_result, dict):
            return ToolResult.from_dict(cast(dict[str, JSONValue], raw_result))
        # Standard MCP servers return content blocks rather than RepoPilot's
        # educational ToolResult wrapper. Preserve them as structured evidence.
        content = result.get("content")
        if not isinstance(content, list):
            raise RuntimeError("MCP tools/call result is malformed")
        return ToolResult(
            call.call_id,
            call.name,
            not bool(result.get("isError", False)),
            {"content": content},
            error="remote MCP tool reported an error" if result.get("isError") else None,
        )

    async def list_resources(self) -> tuple[MCPResource, ...]:
        result = await self._request("resources/list", {})
        raw_resources = result.get("resources", [])
        if not isinstance(raw_resources, list):
            raise RuntimeError("MCP resources/list result is malformed")
        resources: list[MCPResource] = []
        for raw in raw_resources:
            if not isinstance(raw, dict):
                continue
            uri = raw.get("uri")
            name = raw.get("name")
            if not isinstance(uri, str) or not isinstance(name, str):
                raise RuntimeError("MCP resource needs string uri and name")
            resources.append(
                MCPResource(
                    uri=uri,
                    name=name,
                    description=str(raw.get("description", "")),
                    mime_type=str(raw.get("mimeType", "text/plain")),
                )
            )
        return tuple(resources)

    async def read_resource(self, uri: str) -> tuple[MCPResource, ...]:
        result = await self._request("resources/read", {"uri": uri})
        raw_contents = result.get("contents", [])
        if not isinstance(raw_contents, list):
            raise RuntimeError("MCP resources/read result is malformed")
        contents: list[MCPResource] = []
        for raw in raw_contents:
            if not isinstance(raw, dict):
                continue
            resource_uri = raw.get("uri")
            text = raw.get("text")
            if not isinstance(resource_uri, str) or not isinstance(text, str):
                raise RuntimeError("MCP resource content needs string uri and text")
            contents.append(
                MCPResource(
                    uri=resource_uri,
                    name=resource_uri,
                    text=text,
                    mime_type=str(raw.get("mimeType", "text/plain")),
                )
            )
        return tuple(contents)

    async def list_prompts(self) -> tuple[MCPPrompt, ...]:
        result = await self._request("prompts/list", {})
        raw_prompts = result.get("prompts", [])
        if not isinstance(raw_prompts, list):
            raise RuntimeError("MCP prompts/list result is malformed")
        prompts: list[MCPPrompt] = []
        for raw in raw_prompts:
            if not isinstance(raw, dict) or not isinstance(raw.get("name"), str):
                raise RuntimeError("MCP prompt needs a string name")
            raw_arguments = raw.get("arguments", [])
            if not isinstance(raw_arguments, list):
                raise RuntimeError("MCP prompt arguments must be a list")
            arguments = tuple(
                item["name"]
                for item in raw_arguments
                if isinstance(item, dict) and isinstance(item.get("name"), str)
            )
            if len(arguments) != len(raw_arguments):
                raise RuntimeError("MCP prompt arguments need string names")
            prompts.append(
                MCPPrompt(
                    name=raw["name"],
                    template="",
                    description=str(raw.get("description", "")),
                    arguments=arguments,
                )
            )
        return tuple(prompts)

    async def get_prompt(
        self, name: str, *, arguments: dict[str, str] | None = None
    ) -> tuple[MCPPromptMessage, ...]:
        result = await self._request("prompts/get", {"name": name, "arguments": arguments or {}})
        raw_messages = result.get("messages", [])
        if not isinstance(raw_messages, list):
            raise RuntimeError("MCP prompts/get result is malformed")
        messages: list[MCPPromptMessage] = []
        for raw in raw_messages:
            if not isinstance(raw, dict) or not isinstance(raw.get("role"), str):
                raise RuntimeError("MCP prompt message needs a string role")
            content = raw.get("content")
            if not isinstance(content, dict) or content.get("type") != "text":
                raise RuntimeError("MCP prompt message must contain text content")
            text = content.get("text")
            if not isinstance(text, str):
                raise RuntimeError("MCP prompt text must be a string")
            messages.append(MCPPromptMessage(raw["role"], text))
        return tuple(messages)

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
