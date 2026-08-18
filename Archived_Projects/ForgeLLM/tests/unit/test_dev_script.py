"""Tests for the cross-platform development command wrapper."""

import subprocess
import sys
from pathlib import Path


def test_dev_script_help() -> None:
    repo_root = Path(__file__).parents[2]
    completed = subprocess.run(
        [sys.executable, str(repo_root / "scripts" / "dev.py"), "--help"],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert completed.returncode == 0
    assert "Run ForgeLLM development checks" in completed.stdout


def test_dev_script_rejects_unknown_command() -> None:
    repo_root = Path(__file__).parents[2]
    completed = subprocess.run(
        [sys.executable, str(repo_root / "scripts" / "dev.py"), "unknown"],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert completed.returncode == 2
    assert "invalid choice" in completed.stderr
