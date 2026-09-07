"""MCP teaching subset plus governed stdio extension support."""

from repopilot.mcp.audit import MCPProbeAuditStore, MCPProbeRecord, capability_drift
from repopilot.mcp.config import (
    MCPServerConfig,
    load_project_mcp_config,
    project_mcp_config_path,
    remove_project_mcp_config,
    upsert_project_mcp_config,
)
from repopilot.mcp.http import StreamableHttpMCPTransport
from repopilot.mcp.protocol import (
    InProcessMCPTransport,
    MCPClient,
    MCPPrompt,
    MCPPromptMessage,
    MCPRemoteTool,
    MCPResource,
    MCPServer,
)
from repopilot.mcp.registry import GovernedMCPTool, MCPProjectRegistry, MCPServerCapabilities
from repopilot.mcp.stdio import StdioMCPTransport

__all__ = [
    "GovernedMCPTool",
    "MCPProbeAuditStore",
    "MCPProbeRecord",
    "MCPProjectRegistry",
    "MCPPrompt",
    "MCPPromptMessage",
    "MCPResource",
    "MCPServerCapabilities",
    "MCPServerConfig",
    "InProcessMCPTransport",
    "MCPClient",
    "MCPRemoteTool",
    "MCPServer",
    "StdioMCPTransport",
    "StreamableHttpMCPTransport",
    "load_project_mcp_config",
    "capability_drift",
    "project_mcp_config_path",
    "remove_project_mcp_config",
    "upsert_project_mcp_config",
]
