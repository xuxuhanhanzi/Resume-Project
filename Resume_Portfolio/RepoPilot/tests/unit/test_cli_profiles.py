from __future__ import annotations

import argparse
import ast
import asyncio
import json
import shutil
import sys
from pathlib import Path
from typing import cast

import pytest

import repopilot.cli as cli
from repopilot.configuration import (
    ModelProfile,
    UserConfiguration,
    UserConfigurationStore,
)
from repopilot.core.contracts import (
    Message,
    ModelRequest,
    ModelResponse,
    ModelResponseDiagnostics,
    ToolCall,
)
from repopilot.providers.base import ModelProviderError
from repopilot.providers.deepseek import DeepSeekProvider
from repopilot.providers.scripted import ScriptedProvider
from repopilot.runtime.policy import PermissionMode, StaticApprovalHandler
from repopilot.runtime.runner import LocalTrustedRunner
from repopilot.security.redaction import redact_source_text
from repopilot.session.runtime import SessionRuntime
from repopilot.session.store import SessionStore
from repopilot.tools.coding import default_coding_tools


class _FakeCredentials:
    def __init__(self, values: dict[str, str] | None = None) -> None:
        self.values = values or {}

    def get(self, provider: str) -> str | None:
        return self.values.get(provider)

    def set(self, provider: str, secret: str) -> None:
        self.values[provider] = secret

    def delete(self, provider: str) -> bool:
        return self.values.pop(provider, None) is not None


def test_source_redaction_preserves_python_expressions_while_masking_literal_credentials() -> None:
    source = (
        "secret = os.environ.get('DEEPSEEK_API_KEY')\n"
        "api_key = 'literal-sensitive-value'\n"
        "headers = {'Authorization': f'Bearer {secret}'}\n"
    )

    redacted = redact_source_text(source)

    ast.parse(redacted)
    assert "secret = os.environ.get" in redacted
    assert "literal-sensitive-value" not in redacted
    assert "api_key = '[REDACTED]'" in redacted


def test_profile_selection_prefers_in_memory_credential_without_persisting_it() -> None:
    configuration = UserConfiguration(
        "deepseek-main",
        (ModelProfile("deepseek-main", "deepseek", "deepseek-v4-flash"),),
    )
    args = cli.build_parser().parse_args(["--profile", "deepseek-main"])

    selection = cli._resolve_provider_selection(  # noqa: SLF001
        args, configuration=configuration, credentials=_FakeCredentials({"deepseek": "secret"})
    )
    provider = cli._provider_from_selection(selection)  # noqa: SLF001

    assert selection.profile == "deepseek-main"
    assert isinstance(provider, DeepSeekProvider)
    assert provider.deepseek_config.api_key == "secret"


def test_config_and_project_commands_persist_an_openable_shortcut(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    session_root = tmp_path / "state"
    project = tmp_path / "project"
    project.mkdir()
    parser = cli.build_parser()
    profile_args = parser.parse_args(
        [
            "--session-root",
            str(session_root),
            "config",
            "set-profile",
            "deepseek-main",
            "deepseek",
            "deepseek-v4-flash",
            "--default",
        ]
    )
    assert asyncio.run(cli._config_command(profile_args)) == 0  # noqa: SLF001
    project_args = parser.parse_args(
        [
            "--session-root",
            str(session_root),
            "project",
            "add",
            "repopilot",
            str(project),
            "--profile",
            "deepseek-main",
        ]
    )
    assert asyncio.run(cli._project_command(project_args)) == 0  # noqa: SLF001
    observed: dict[str, object] = {}

    async def fake_interactive(args: argparse.Namespace) -> int:
        observed["cwd"] = Path.cwd()
        observed["profile"] = args.profile
        return 0

    monkeypatch.setattr(cli, "_interactive_command", fake_interactive)
    open_args = parser.parse_args(["--session-root", str(session_root), "open", "repopilot"])

    assert asyncio.run(cli._open_project_shortcut(open_args, "repopilot")) == 0  # noqa: SLF001
    assert observed == {"cwd": project.resolve(), "profile": "deepseek-main"}
    assert cli._rewrite_project_alias(["repopilot"]) == ["open", "repopilot"]  # noqa: SLF001


def test_auth_login_uses_a_hidden_prompt_and_never_echoes_the_secret(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    credentials = _FakeCredentials()
    monkeypatch.setattr(cli, "CredentialStore", lambda: credentials)
    monkeypatch.setattr("repopilot.cli.getpass.getpass", lambda _prompt: "secret")
    args = cli.build_parser().parse_args(["auth", "login", "qwen"])

    assert asyncio.run(cli._auth_command(args)) == 0  # noqa: SLF001
    assert credentials.values == {"qwen": "secret"}
    assert "secret" not in capsys.readouterr().out


def test_auth_test_uses_the_saved_credential_without_printing_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    credentials = _FakeCredentials({"deepseek": "secret"})

    class _Provider:
        async def complete(self, _request: object) -> ModelResponse:
            return ModelResponse(content="OK", model="deepseek-v4-flash")

    monkeypatch.setattr(cli, "CredentialStore", lambda: credentials)
    monkeypatch.setattr(cli, "_provider_from_selection", lambda _selection: _Provider())
    args = cli.build_parser().parse_args(
        ["--session-root", str(tmp_path / "state"), "auth", "test", "deepseek"]
    )

    assert asyncio.run(cli._auth_command(args)) == 0  # noqa: SLF001
    output = capsys.readouterr().out
    assert "verified" in output
    assert "secret" not in output


def test_auth_probe_uses_only_a_fixed_public_marker_and_reports_stream_diagnostics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    credentials = _FakeCredentials({"deepseek": "secret"})

    class _StreamingProvider:
        async def complete(self, _request: ModelRequest) -> ModelResponse:
            raise AssertionError("the public probe must use streaming")

        async def complete_stream(
            self, request: ModelRequest, *, on_text_delta: object
        ) -> ModelResponse:
            assert request.tools == ()
            assert request.reasoning_mode == "disabled"
            assert request.messages[-1].content == "Reply with exactly: REPOPILOT_STREAM_OK"
            assert callable(on_text_delta)
            update = on_text_delta("REPOPILOT_STREAM_OK")
            assert hasattr(update, "__await__")
            await update
            return ModelResponse(
                model="deepseek-v4-flash",
                diagnostics=ModelResponseDiagnostics(
                    transport="sse",
                    http_status=200,
                    choice_count=2,
                    finish_reason="stop",
                    stream_chunks=2,
                    visible_text_deltas=1,
                    visible_text_characters=20,
                    stream_done_received=True,
                ),
            )

    monkeypatch.setattr(cli, "CredentialStore", lambda: credentials)
    monkeypatch.setattr(cli, "_provider_from_selection", lambda _selection: _StreamingProvider())
    args = cli.build_parser().parse_args(
        [
            "--session-root",
            str(tmp_path / "state"),
            "--output-format",
            "jsonl",
            "auth",
            "probe",
            "deepseek",
        ]
    )

    assert asyncio.run(cli._auth_command(args)) == 0  # noqa: SLF001
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "passed"
    assert payload["marker_matched"] is True
    assert payload["streamed_delta_count"] == 1
    assert payload["diagnostics"]["finish_reason"] == "stop"
    assert "secret" not in json.dumps(payload)


def test_shell_init_uses_a_non_conflicting_windows_launcher(
    capsys: pytest.CaptureFixture[str],
) -> None:
    args = cli.build_parser().parse_args(["shell-init", "powershell"])

    assert asyncio.run(cli._shell_init_command(args)) == 0  # noqa: SLF001
    assert "function repopilot" in capsys.readouterr().out


def test_shell_doctor_reports_a_path_collision_without_changing_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    other_launcher = tmp_path / "anaconda" / "Scripts" / "repopilot.exe"
    other_launcher.parent.mkdir(parents=True)
    other_launcher.write_text("placeholder", encoding="utf-8")
    monkeypatch.setattr(shutil, "which", lambda _name: str(other_launcher))
    args = cli.build_parser().parse_args(["shell-doctor", "powershell"])

    assert asyncio.run(cli._shell_doctor_command(args)) == 0  # noqa: SLF001
    output = capsys.readouterr().out
    assert "collision detected" in output
    assert "Invoke-Expression" in output
    assert "No PATH" in output
    assert other_launcher.is_file()


def test_shell_doctor_supports_a_machine_readable_session_only_recovery(
    capsys: pytest.CaptureFixture[str],
) -> None:
    args = cli.build_parser().parse_args(["--output-format", "jsonl", "shell-doctor", "powershell"])

    assert asyncio.run(cli._shell_doctor_command(args)) == 0  # noqa: SLF001

    payload = json.loads(capsys.readouterr().out)
    assert payload["type"] == "shell_doctor"
    assert "shell-init powershell" in payload["session_only_command"]
    assert "No PATH" in payload["note"]


def test_doctor_fix_plan_is_advisory_and_does_not_expose_credentials(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    launcher = tmp_path / "venv" / "Scripts" / "repopilot.exe"
    launcher.parent.mkdir(parents=True)
    launcher.write_text("placeholder", encoding="utf-8")
    conflicting = tmp_path / "other" / "Scripts" / "repopilot.exe"
    conflicting.parent.mkdir(parents=True)
    conflicting.write_text("placeholder", encoding="utf-8")
    monkeypatch.setattr(sys, "executable", str(launcher.parent / "python.exe"))
    monkeypatch.setattr(sys, "argv", [str(launcher)])
    monkeypatch.setattr(shutil, "which", lambda _name: str(conflicting))
    monkeypatch.setattr(cli, "CredentialStore", lambda: _FakeCredentials({"deepseek": "secret"}))
    args = cli.build_parser().parse_args(
        [
            "--session-root",
            str(tmp_path / "state"),
            "doctor",
            "--project",
            str(project),
            "--fix-plan",
        ]
    )

    assert asyncio.run(cli._doctor_command(args)) == 0  # noqa: SLF001
    output = capsys.readouterr().out
    assert "Fix plan (not executed):" in output
    assert "launcher function" in output
    assert "secret" not in output


def test_doctor_and_eval_summary_stay_offline_and_redact_credentials(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    session_root = tmp_path / "state"
    configuration = UserConfigurationStore(session_root)
    configuration.set_profile(
        UserConfiguration(),
        ModelProfile("deepseek-main", "deepseek", "deepseek-v4-flash"),
        make_default=True,
    )
    monkeypatch.setattr(cli, "CredentialStore", lambda: _FakeCredentials({"deepseek": "secret"}))
    parser = cli.build_parser()
    doctor_args = parser.parse_args(
        ["--session-root", str(session_root), "doctor", "--project", str(project)]
    )

    assert asyncio.run(cli._doctor_command(doctor_args)) == 0  # noqa: SLF001
    doctor_output = capsys.readouterr().out
    assert "deepseek=available" in doctor_output
    assert "secret" not in doctor_output

    records = tmp_path / "records.jsonl"
    records.write_text(
        '{"task_id":"one","status":"completed","hidden_tests_passed":true,'
        '"iterations":1,"tool_calls":2,"input_tokens":3,"output_tokens":4,'
        '"wall_seconds":0.5,"changed_files":1}\n',
        encoding="utf-8",
    )
    eval_args = parser.parse_args(["eval", "summary", str(records)])
    assert asyncio.run(cli._eval_command(eval_args)) == 0  # noqa: SLF001
    assert '"resolve_rate": 1.0' in capsys.readouterr().out
    assert cli._rewrite_project_alias(["doctor"]) == ["doctor"]  # noqa: SLF001


def test_doctor_reports_an_installation_path_collision_without_touching_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    path_launcher = tmp_path / "other" / "repopilot.exe"
    path_launcher.parent.mkdir()
    path_launcher.write_text("placeholder", encoding="utf-8")
    monkeypatch.setattr(shutil, "which", lambda _name: str(path_launcher))
    args = cli.build_parser().parse_args(["doctor", "--project", str(project)])

    assert asyncio.run(cli._doctor_command(args)) == 0  # noqa: SLF001
    output = capsys.readouterr().out
    assert "PATH command:" in output
    assert "Action:" in output
    assert path_launcher.is_file()


def test_doctor_normalizes_a_windows_console_launcher_without_exe_suffix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    launcher = tmp_path / "Scripts" / "repopilot.exe"
    launcher.parent.mkdir()
    launcher.write_text("placeholder", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", [str(launcher.with_suffix(""))])
    monkeypatch.setattr(sys, "executable", str(launcher.parent / "python.exe"))
    monkeypatch.setattr(shutil, "which", lambda _name: str(launcher))

    installation = cli._doctor_installation_status()  # noqa: SLF001

    assert installation["launcher"] == str(launcher.resolve())
    warnings = cast(list[str], installation["warnings"])
    assert "the launcher does not belong to the current Python environment" not in warnings


def test_doctor_prefers_the_venv_launcher_for_python_module_invocation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    launcher = tmp_path / "Scripts" / "repopilot.exe"
    launcher.parent.mkdir()
    launcher.write_text("placeholder", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", [str(tmp_path / "src" / "repopilot" / "__main__.py")])
    monkeypatch.setattr(sys, "executable", str(launcher.parent / "python.exe"))
    monkeypatch.setattr(shutil, "which", lambda _name: str(launcher))

    installation = cli._doctor_installation_status()  # noqa: SLF001

    assert installation["launcher"] == str(launcher.resolve())


def test_readonly_acceptance_package_transmits_only_named_non_secret_files(tmp_path: Path) -> None:
    project = tmp_path / "project"
    source = project / "src" / "module.py"
    source.parent.mkdir(parents=True)
    source.write_text("VALUE = 1\n", encoding="utf-8")
    secret = project / ".env"
    secret.write_text("TOKEN=must-not-send\n", encoding="utf-8")

    files, message = cli._readonly_acceptance_package(  # noqa: SLF001
        project,
        includes=("src/module.py",),
        prompt="Review this module.",
    )

    assert files[0]["path"] == "src/module.py"
    assert "VALUE = 1" in message
    assert "must-not-send" not in message
    with pytest.raises(ValueError, match="protected"):
        cli._readonly_acceptance_package(  # noqa: SLF001
            project,
            includes=(".env",),
            prompt="Review this module.",
        )


def test_readonly_acceptance_prefers_streamed_user_facing_text() -> None:
    class _StreamingProvider:
        async def complete(self, _request: ModelRequest) -> ModelResponse:
            raise AssertionError("streaming provider should not use complete")

        async def complete_stream(
            self, _request: ModelRequest, *, on_text_delta: object
        ) -> ModelResponse:
            assert callable(on_text_delta)
            result = on_text_delta("streamed assessment")
            assert hasattr(result, "__await__")
            await result
            return ModelResponse(content="")

    response, answer = asyncio.run(
        cli._complete_readonly_acceptance(  # noqa: SLF001
            _StreamingProvider(), ModelRequest((Message("user", "review"),), ())
        )
    )

    assert response.content == ""
    assert answer == "streamed assessment"


def test_readonly_acceptance_records_an_empty_final_response_locally(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    project = tmp_path / "project"
    source = project / "src" / "module.py"
    source.parent.mkdir(parents=True)
    source.write_text("UNIQUE_SOURCE_SENTINEL = 1\n", encoding="utf-8")
    artifacts = tmp_path / "artifacts"
    monkeypatch.chdir(project)
    monkeypatch.setattr(cli, "CredentialStore", lambda: _FakeCredentials({"deepseek": "secret"}))

    class _EmptyProvider:
        async def complete(self, _request: ModelRequest) -> ModelResponse:
            return ModelResponse(content="")

    monkeypatch.setattr(cli, "_provider_from_selection", lambda _selection: _EmptyProvider())
    args = cli.build_parser().parse_args(
        [
            "--provider",
            "deepseek",
            "--trust",
            "acceptance",
            "--send-project-files",
            "--include",
            "src/module.py",
            "--artifacts",
            str(artifacts),
        ]
    )

    with pytest.raises(ModelProviderError, match="no final user-facing assessment"):
        asyncio.run(cli._acceptance_command(args))  # noqa: SLF001

    receipts = list((artifacts / "acceptance").glob("*.receipt.json"))
    intents = list((artifacts / "acceptance").glob("*.intent.json"))
    assert len(receipts) == 1
    assert len(intents) == 1
    receipt_text = receipts[0].read_text(encoding="utf-8")
    receipt = json.loads(receipt_text)
    intent_text = intents[0].read_text(encoding="utf-8")
    intent = json.loads(intent_text)
    assert intent["kind"] == "readonly_deepseek_acceptance_intent"
    assert intent["reasoning_mode"] == "disabled"
    assert receipt["outcome"] == "inconclusive"
    assert receipt["reason"] == "provider returned no final user-facing assessment"
    assert receipt["assessment"] is None
    assert receipt["files"][0]["path"] == "src/module.py"
    assert "UNIQUE_SOURCE_SENTINEL" not in receipt_text
    assert "UNIQUE_SOURCE_SENTINEL" not in intent_text

    verify_args = cli.build_parser().parse_args(
        ["--output-format", "jsonl", "acceptance", "verify", str(receipts[0])]
    )
    assert asyncio.run(cli._acceptance_command(verify_args)) == 0  # noqa: SLF001
    verified = json.loads(capsys.readouterr().out)
    assert verified["type"] == "acceptance_receipt"
    assert verified["valid"] is True
    assert verified["assessment_characters"] == 0


def test_acceptance_receipt_verification_detects_a_tampered_assessment(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    receipt_path = tmp_path / "receipt.json"
    selection = cli._ProviderSelection(  # noqa: SLF001
        provider="deepseek",
        model="deepseek-v4-flash",
        api_key_env="DEEPSEEK_API_KEY",
        base_url="http://127.0.0.1:11434",
    )
    receipt = cli._acceptance_final_receipt(  # noqa: SLF001
        run_id="audit_test",
        project_root=tmp_path,
        selection=selection,
        files=[{"path": "src/module.py", "bytes": 12, "sha256": "a" * 64}],
        request_sha256="b" * 64,
        intent_sha256="c" * 64,
        outcome="completed",
        reason=None,
        response=ModelResponse(content="SAFE ASSESSMENT", model="deepseek-v4-flash"),
        answer="SAFE ASSESSMENT",
    )
    cli._write_acceptance_receipt(receipt_path, receipt)  # noqa: SLF001

    assert cli._acceptance_receipt_command("verify", receipt_path, "jsonl") == 0  # noqa: SLF001
    verified = json.loads(capsys.readouterr().out)
    assert verified["valid"] is True
    assert "SAFE ASSESSMENT" not in json.dumps(verified)

    tampered = json.loads(receipt_path.read_text(encoding="utf-8"))
    tampered["assessment"] = "TAMPERED"
    receipt_path.write_text(json.dumps(tampered), encoding="utf-8")

    assert cli._acceptance_receipt_command("verify", receipt_path, "jsonl") == 1  # noqa: SLF001
    assert json.loads(capsys.readouterr().out)["checks"]["assessment_hash"] is False


def test_eval_diagnose_swe_classifies_existing_artifacts_offline(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    predictions = tmp_path / "predictions.jsonl"
    predictions.write_text(
        "\n".join(
            (
                '{"instance_id":"empty","model_patch":""}',
                '{"instance_id":"apply","model_patch":"diff --git a/a b/a"}',
                '{"instance_id":"regression","model_patch":"diff --git a/b b/b"}',
                '{"instance_id":"failed","model_patch":"diff --git a/c b/c"}',
                '{"instance_id":"ok","model_patch":"diff --git a/d b/d"}',
            )
        )
        + "\n",
        encoding="utf-8",
    )
    results = tmp_path / "resolved_results.json"
    apply_failure = (
        '{"instance_id":"apply","resolved":false,"fail_to_pass_pass":false,'
        '"pass_to_pass_pass":false,"ftp_output_tail":"error: patch failed: module.py"},'
    )
    result_payload = (
        '{"report":['
        '{"instance_id":"empty","resolved":false,"fail_to_pass_pass":false,"pass_to_pass_pass":false},'
        + apply_failure
        + '{"instance_id":"regression","resolved":false,"fail_to_pass_pass":true,'
        '"pass_to_pass_pass":false},'
        + '{"instance_id":"failed","resolved":false,"fail_to_pass_pass":false,'
        '"pass_to_pass_pass":true},'
        + '{"instance_id":"ok","resolved":true,"fail_to_pass_pass":true,'
        '"pass_to_pass_pass":true}]}'
    )
    results.write_text(result_payload, encoding="utf-8")
    args = cli.build_parser().parse_args(["eval", "diagnose-swe", str(predictions), str(results)])

    assert asyncio.run(cli._eval_command(args)) == 0  # noqa: SLF001
    output = capsys.readouterr().out
    assert '"empty_patch": 1' in output
    assert '"patch_apply_failed": 1' in output
    assert '"regression": 1' in output
    assert '"fail_to_pass_failed": 1' in output
    assert '"resolved": 1' in output


def test_unknown_slash_command_is_suggested_instead_of_sent_to_the_model() -> None:
    assert cli._slash_command_suggestion("/statu") == "/status"  # noqa: SLF001
    assert cli._slash_command_suggestion("/definitely-not-a-command") is None  # noqa: SLF001


def test_model_command_switches_the_active_session_and_persists_the_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    session_root = tmp_path / "state"
    project = tmp_path / "project"
    project.mkdir()
    configuration_store = UserConfigurationStore(session_root)
    configuration = configuration_store.set_profile(
        UserConfiguration(),
        ModelProfile("deepseek-main", "deepseek", "deepseek-v4-flash"),
    )
    runtime = SessionRuntime(
        provider=ScriptedProvider([]),
        tools=[],
        runner=LocalTrustedRunner(trusted=True),
        store=SessionStore(session_root),
        model="qwen2.5:7b",
        permission_mode=PermissionMode.MANUAL,
        approval=StaticApprovalHandler(False),
    )

    class _Terminal:
        def __init__(self) -> None:
            self.responses = iter(("/model deepseek-main", "/exit"))

        async def read_prompt_async(self, _prompt: str = "> ") -> str:
            return next(self.responses)

        def status(self, _message: str) -> None:
            return None

    monkeypatch.chdir(project)
    monkeypatch.setattr(cli, "ConsoleTerminal", _Terminal)
    args = argparse.Namespace(
        resume_session=None,
        continue_session=False,
        fork=False,
        print_prompt=None,
        output_format="text",
        verbose=False,
        preflight_tests=False,
    )
    selection = cli._ProviderSelection(  # noqa: SLF001
        provider="local",
        model="qwen2.5:7b",
        api_key_env=None,
        base_url="http://127.0.0.1:11434",
    )

    assert (
        asyncio.run(
            cli._interactive_session(  # noqa: SLF001
                args,
                project_root=project,
                store=SessionStore(session_root),
                runtime=runtime,
                mcp_registry=None,
                preflight_evidence=None,
                user_config_store=configuration_store,
                user_configuration=configuration,
                credentials=_FakeCredentials({"deepseek": "secret"}),
                selection=selection,
            )
        )
        == 0
    )
    metadata = SessionStore(session_root).latest(project)
    assert metadata is not None
    assert metadata.model == "deepseek-v4-flash"
    assert isinstance(runtime.kernel.provider, DeepSeekProvider)
    assert configuration_store.load().default_profile == "deepseek-main"


def test_export_command_requires_confirmation_and_creates_a_local_session_export(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    session_root = tmp_path / "state"
    project = tmp_path / "project"
    project.mkdir()
    configuration_store = UserConfigurationStore(session_root)
    runtime = SessionRuntime(
        provider=ScriptedProvider([]),
        tools=[],
        runner=LocalTrustedRunner(trusted=True),
        store=SessionStore(session_root),
        model="qwen2.5:7b",
        permission_mode=PermissionMode.MANUAL,
        approval=StaticApprovalHandler(False),
    )

    class _Terminal:
        def __init__(self) -> None:
            self.responses = iter(("/export review.md", "y", "/exit"))

        async def read_prompt_async(self, _prompt: str = "> ") -> str:
            return next(self.responses)

        def status(self, _message: str) -> None:
            return None

    monkeypatch.chdir(project)
    monkeypatch.setattr(cli, "ConsoleTerminal", _Terminal)
    args = argparse.Namespace(
        resume_session=None,
        continue_session=False,
        fork=False,
        print_prompt=None,
        output_format="text",
        verbose=False,
        preflight_tests=False,
    )
    selection = cli._ProviderSelection(  # noqa: SLF001
        provider="local",
        model="qwen2.5:7b",
        api_key_env=None,
        base_url="http://127.0.0.1:11434",
    )

    assert (
        asyncio.run(
            cli._interactive_session(  # noqa: SLF001
                args,
                project_root=project,
                store=SessionStore(session_root),
                runtime=runtime,
                mcp_registry=None,
                preflight_evidence=None,
                user_config_store=configuration_store,
                user_configuration=UserConfiguration(),
                credentials=_FakeCredentials(),
                selection=selection,
            )
        )
        == 0
    )
    metadata = SessionStore(session_root).latest(project)
    assert metadata is not None
    session_dir = SessionStore(session_root).session_dir(project, metadata.session_id)
    target = session_dir / "exports/review.md"
    assert target.is_file()
    assert "Conversation exported to:" in capsys.readouterr().out


def test_status_command_renders_local_session_health(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    session_root = tmp_path / "state"
    project = tmp_path / "project"
    project.mkdir()
    runtime = SessionRuntime(
        provider=ScriptedProvider([]),
        tools=[],
        runner=LocalTrustedRunner(trusted=True),
        store=SessionStore(session_root),
        model="qwen2.5:7b",
        permission_mode=PermissionMode.MANUAL,
        approval=StaticApprovalHandler(False),
    )

    class _Terminal:
        def __init__(self) -> None:
            self.responses = iter(("/status", "/exit"))

        async def read_prompt_async(self, _prompt: str = "> ") -> str:
            return next(self.responses)

        def status(self, _message: str) -> None:
            return None

    monkeypatch.chdir(project)
    monkeypatch.setattr(cli, "ConsoleTerminal", _Terminal)
    args = argparse.Namespace(
        resume_session=None,
        continue_session=False,
        fork=False,
        print_prompt=None,
        output_format="text",
        verbose=False,
        preflight_tests=False,
    )
    selection = cli._ProviderSelection(  # noqa: SLF001
        provider="local",
        model="qwen2.5:7b",
        api_key_env=None,
        base_url="http://127.0.0.1:11434",
    )

    assert (
        asyncio.run(
            cli._interactive_session(  # noqa: SLF001
                args,
                project_root=project,
                store=SessionStore(session_root),
                runtime=runtime,
                mcp_registry=None,
                preflight_evidence=None,
                user_config_store=UserConfigurationStore(session_root),
                user_configuration=UserConfiguration(),
                credentials=_FakeCredentials(),
                selection=selection,
            )
        )
        == 0
    )
    output = capsys.readouterr().out
    assert "Verification: not run" in output
    assert "Background tasks: none" in output
    assert "MCP: not started: 0 configured" in output


def test_workflow_command_renders_only_local_next_step(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    session_root = tmp_path / "state"
    project = tmp_path / "project"
    project.mkdir()
    runtime = SessionRuntime(
        provider=ScriptedProvider([]),
        tools=[],
        runner=LocalTrustedRunner(trusted=True),
        store=SessionStore(session_root),
        model="qwen2.5:7b",
        permission_mode=PermissionMode.MANUAL,
        approval=StaticApprovalHandler(False),
    )

    class _Terminal:
        def __init__(self) -> None:
            self.responses = iter(("/workflow", "/exit"))

        async def read_prompt_async(self, _prompt: str = "> ") -> str:
            return next(self.responses)

        def status(self, _message: str) -> None:
            return None

    monkeypatch.chdir(project)
    monkeypatch.setattr(cli, "ConsoleTerminal", _Terminal)
    args = argparse.Namespace(
        resume_session=None,
        continue_session=False,
        fork=False,
        print_prompt=None,
        output_format="text",
        verbose=False,
        preflight_tests=False,
    )
    selection = cli._ProviderSelection(  # noqa: SLF001
        provider="local",
        model="qwen2.5:7b",
        api_key_env=None,
        base_url="http://127.0.0.1:11434",
    )

    assert (
        asyncio.run(
            cli._interactive_session(  # noqa: SLF001
                args,
                project_root=project,
                store=SessionStore(session_root),
                runtime=runtime,
                mcp_registry=None,
                preflight_evidence=None,
                user_config_store=UserConfigurationStore(session_root),
                user_configuration=UserConfiguration(),
                credentials=_FakeCredentials(),
                selection=selection,
            )
        )
        == 0
    )
    output = capsys.readouterr().out
    assert "Workflow (local only" in output
    assert "Next safe action:" in output
    assert "permission policy" in output


def test_changes_summary_is_local_and_reports_the_latest_revision(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    target = project / "module.py"
    target.write_text("VALUE = 1\n", encoding="utf-8")
    runtime = SessionRuntime(
        provider=ScriptedProvider(
            [
                ModelResponse(
                    tool_calls=(
                        ToolCall(
                            "patch",
                            "apply_patch",
                            {"path": "module.py", "old_text": "VALUE = 1", "new_text": "VALUE = 2"},
                        ),
                    )
                ),
                ModelResponse(content="Updated."),
            ]
        ),
        tools=default_coding_tools(),
        runner=LocalTrustedRunner(trusted=True),
        store=SessionStore(tmp_path / "sessions"),
        model="scripted",
        approval=StaticApprovalHandler(True),
    )

    result = asyncio.run(runtime.run_turn(runtime.start(project), "Update the module."))
    summary = cli._render_changes_summary(runtime, result.metadata)  # noqa: SLF001

    assert "local only" in summary
    assert "module.py: +1 -1" in summary
    assert "/diff" in summary
