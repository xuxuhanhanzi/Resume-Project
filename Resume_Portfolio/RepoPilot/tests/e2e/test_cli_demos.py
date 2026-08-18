from __future__ import annotations

import json
from pathlib import Path

import pytest

from repopilot.cli import entrypoint


def test_scripted_cli_demo(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert entrypoint(["demo", "scripted", "--artifacts", str(tmp_path)]) == 0
    output = json.loads(capsys.readouterr().out)
    list_result = json.loads(output["messages"][1]["content"])
    assert list_result["data"]["files"] == ["calculator.py"]
    checkpoints = list(tmp_path.glob("demo_*/checkpoint.json"))
    assert checkpoints


def test_mcp_permission_and_recovery_demos(tmp_path: Path) -> None:
    assert entrypoint(["demo", "mcp", "--artifacts", str(tmp_path)]) == 0
    assert entrypoint(["demo", "permission", "--artifacts", str(tmp_path)]) == 0
    assert entrypoint(["demo", "recovery", "--artifacts", str(tmp_path)]) == 0
