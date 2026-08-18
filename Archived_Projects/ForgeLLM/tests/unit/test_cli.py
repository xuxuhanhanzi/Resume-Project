"""Unit tests for the ForgeLLM CLI."""

import json

import pytest

from forgellm import __version__
from forgellm.cli import main


def test_no_arguments_prints_help(capsys: pytest.CaptureFixture[str]) -> None:
    assert main([]) == 0
    assert "usage: forgellm" in capsys.readouterr().out


def test_doctor_returns_non_sensitive_runtime_summary(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["doctor"]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["forgellm"] == __version__
    assert set(payload) == {"forgellm", "platform", "python"}


def test_unknown_command_fails() -> None:
    with pytest.raises(SystemExit) as error:
        main(["unknown-command"])

    assert error.value.code == 2
