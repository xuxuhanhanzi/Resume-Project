from __future__ import annotations

import asyncio
import json
import shutil
import sys
from pathlib import Path

import pytest

from repopilot import cli
from repopilot.core.contracts import Permission, ToolCall, ToolSpec
from repopilot.mcp.audit import MCPProbeAuditStore, capability_drift
from repopilot.mcp.config import (
    MCPServerConfig,
    load_project_mcp_config,
    remove_project_mcp_config,
    upsert_project_mcp_config,
)
from repopilot.mcp.protocol import InProcessMCPTransport, MCPClient, MCPServer
from repopilot.mcp.registry import (
    GovernedMCPTool,
    MCPProjectRegistry,
    MCPServerCapabilities,
    _apply_tool_policy,
)
from repopilot.mcp.stdio import StdioMCPTransport
from repopilot.runtime.runner import LocalTrustedRunner
from repopilot.tools.base import ToolContext, ToolRegistry
from repopilot.tools.coding import ListFilesTool
from repopilot.workspace.contracts import InteractiveTask


def test_project_mcp_config_is_tokenized_and_bounded(tmp_path: Path) -> None:
    config_dir = tmp_path / ".repopilot"
    config_dir.mkdir()
    (config_dir / "mcp.json").write_text(
        json.dumps({"servers": {"demo": {"command": "python", "args": ["-m", "demo"]}}}),
        encoding="utf-8",
    )
    configs = load_project_mcp_config(tmp_path)
    assert configs == (MCPServerConfig("demo", ("python", "-m", "demo")),)


def test_project_mcp_config_can_be_upserted_and_removed_without_deleting_file(
    tmp_path: Path,
) -> None:
    upsert_project_mcp_config(tmp_path, MCPServerConfig("demo", ("python", "-m", "demo")))
    upsert_project_mcp_config(tmp_path, MCPServerConfig("demo", ("python", "-m", "replacement")))

    assert load_project_mcp_config(tmp_path) == (
        MCPServerConfig("demo", ("python", "-m", "replacement")),
    )
    assert remove_project_mcp_config(tmp_path, "demo")
    assert load_project_mcp_config(tmp_path) == ()
    assert (tmp_path / ".repopilot" / "mcp.json").is_file()


def test_mcp_config_persists_per_server_tool_policy(tmp_path: Path) -> None:
    upsert_project_mcp_config(
        tmp_path,
        MCPServerConfig(
            "demo",
            ("python", "-m", "demo"),
            allowed_tools=("read",),
            denied_tools=("write",),
        ),
    )

    assert load_project_mcp_config(tmp_path) == (
        MCPServerConfig(
            "demo",
            ("python", "-m", "demo"),
            allowed_tools=("read",),
            denied_tools=("write",),
        ),
    )


def test_mcp_server_tool_policy_filters_availability_but_not_permission_class() -> None:
    specs = (
        ToolSpec("read", "", {}),
        ToolSpec("write", "", {}),
        ToolSpec("other", "", {}),
    )
    config = MCPServerConfig(
        "demo", ("python", "-m", "demo"), allowed_tools=("read", "other"), denied_tools=("write",)
    )

    assert [item.name for item in _apply_tool_policy(config, specs)] == ["read", "other"]


def test_mcp_cli_requires_explicit_trust_for_config_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    args = cli.build_parser().parse_args(["mcp", "add", "demo", "python", "-m", "demo"])

    with pytest.raises(ValueError, match="require --trust"):
        asyncio.run(cli._mcp_command(args))


def test_mcp_doctor_does_not_start_a_server_and_flags_missing_launcher(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    upsert_project_mcp_config(tmp_path, MCPServerConfig("missing", ("not-a-real-launcher",)))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(shutil, "which", lambda _name: None)
    args = cli.build_parser().parse_args(["mcp", "doctor"])

    assert asyncio.run(cli._mcp_command(args)) == 1  # noqa: SLF001
    output = capsys.readouterr().out
    assert "not started" in output
    assert "MCP tools remain disabled" in output


def test_mcp_probe_starts_only_the_selected_server_after_explicit_trust(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    upsert_project_mcp_config(tmp_path, MCPServerConfig("demo", ("python", "-m", "demo")))

    class _Registry:
        def __init__(self) -> None:
            self.closed = False
            self.servers = (
                MCPServerCapabilities("demo", None, ("read",), (), ()),  # type: ignore[arg-type]
            )

        async def close(self) -> None:
            self.closed = True

    registry = _Registry()

    async def connect(
        configs: tuple[MCPServerConfig, ...], *, project_root: Path, timeout_seconds: float
    ) -> _Registry:
        assert configs[0].name == "demo"
        assert project_root == tmp_path
        assert timeout_seconds == 30.0
        return registry

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(MCPProjectRegistry, "connect", connect)
    args = cli.build_parser().parse_args(
        ["--trust", "--session-root", str(tmp_path / "state"), "mcp", "probe", "demo"]
    )

    assert asyncio.run(cli._mcp_command(args)) == 0  # noqa: SLF001
    assert "MCP probe: demo" in capsys.readouterr().out
    assert registry.closed


def test_mcp_probe_audit_is_local_redacted_and_reports_capability_drift(tmp_path: Path) -> None:
    config = MCPServerConfig("demo", ("python", "-m", "demo"), allowed_tools=("read",))
    store = MCPProbeAuditStore(tmp_path / "state")
    first = store.record(
        project_root=tmp_path,
        config=config,
        timeout_seconds=2.0,
        tools=("read",),
        resources=1,
        prompts=("guide",),
    )
    assert first.ok
    assert capability_drift(first, ("read", "write")) == {"added": ["write"], "removed": []}
    failed = store.record(
        project_root=tmp_path,
        config=config,
        timeout_seconds=2.0,
        error="api_key=definitely-not-kept",
    )

    history = store.history(project_root=tmp_path)
    assert [record.ok for record in history] == [False, True]
    assert failed.error == "api_key=[REDACTED]"
    assert store.latest_success(project_root=tmp_path, server="demo") == first


def test_mcp_policy_command_updates_only_one_server_without_starting_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    upsert_project_mcp_config(tmp_path, MCPServerConfig("demo", ("python", "-m", "demo")))
    monkeypatch.chdir(tmp_path)
    args = cli.build_parser().parse_args(
        [
            "--trust",
            "--session-root",
            str(tmp_path / "state"),
            "mcp",
            "policy",
            "demo",
            "--allow-remote-tool",
            "read",
            "--deny-remote-tool",
            "write",
        ]
    )

    assert asyncio.run(cli._mcp_command(args)) == 0  # noqa: SLF001
    config = load_project_mcp_config(tmp_path)[0]
    assert config.allowed_tools == ("read",)
    assert config.denied_tools == ("write",)
    assert "Updated local tool policy" in capsys.readouterr().out


def test_governed_mcp_tool_prefixes_name_and_forces_high_risk(tmp_path: Path) -> None:
    task = InteractiveTask("session", tmp_path, "List files")
    context = ToolContext(task, LocalTrustedRunner(trusted=True))
    server = MCPServer(ToolRegistry([ListFilesTool()]), context)
    client = MCPClient(InProcessMCPTransport(server))

    async def exercise() -> None:
        await client.initialize()
        remote_spec = (await client.list_tools())[0]
        tool = GovernedMCPTool("demo", client, remote_spec)
        assert tool.spec.name == "mcp__demo__list_files"
        assert tool.spec.permission is Permission.HIGH_RISK
        result = await tool.run(ToolCall("remote", tool.spec.name, {"path": "."}), context)
        assert result.ok
        assert result.tool_name == tool.spec.name

    asyncio.run(exercise())


def test_stdio_transport_exchanges_line_framed_jsonrpc(tmp_path: Path) -> None:
    script = (
        "import json, sys; "
        "[print(json.dumps({'jsonrpc':'2.0','id':(request:=json.loads(line))['id'],"
        "'result':{'echo':request['method']}}), flush=True) for line in sys.stdin]"
    )

    async def exercise() -> None:
        transport = StdioMCPTransport(
            MCPServerConfig("echo", (sys.executable, "-u", "-c", script)), cwd=tmp_path
        )
        response = await transport.request({"jsonrpc": "2.0", "id": 7, "method": "ping"})
        assert response["result"] == {"echo": "ping"}
        await transport.close()

    asyncio.run(exercise())


def test_stdio_transport_drains_noisy_stderr_and_redacts_failure_context(tmp_path: Path) -> None:
    noisy = "import sys; sys.stderr.write('x' * 12000); sys.stderr.flush(); sys.exit(7)"

    async def exercise() -> None:
        transport = StdioMCPTransport(
            MCPServerConfig("noisy", (sys.executable, "-u", "-c", noisy)), cwd=tmp_path
        )
        with pytest.raises(RuntimeError, match="exited with code 7"):
            await transport.request({"jsonrpc": "2.0", "id": 1, "method": "ping"})
        await transport.close()

    asyncio.run(exercise())


def test_interactive_mcp_registry_closes_when_session_setup_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class RecordingRegistry:
        tools: tuple[object, ...] = ()

        def __init__(self) -> None:
            self.closed = False

        async def close(self) -> None:
            self.closed = True

    registry = RecordingRegistry()

    async def connect(*_args: object, **_kwargs: object) -> RecordingRegistry:
        return registry

    async def fail_session(*_args: object, **_kwargs: object) -> int:
        raise RuntimeError("session setup failed")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(MCPProjectRegistry, "connect", connect)
    monkeypatch.setattr(cli, "_interactive_session", fail_session)
    args = cli.build_parser().parse_args(
        ["--mcp", "--trust", "--session-root", str(tmp_path / "state")]
    )

    with pytest.raises(RuntimeError, match="session setup failed"):
        asyncio.run(cli._interactive_command(args))
    assert registry.closed
