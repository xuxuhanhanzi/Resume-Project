from __future__ import annotations

from pathlib import Path


def test_windows_setup_script_is_project_local_and_non_destructive() -> None:
    script = (Path(__file__).resolve().parents[2] / "scripts" / "setup_windows.ps1").read_text(
        encoding="utf-8"
    )

    assert "Scripts\\python.exe" in script
    assert 'pip", "install", "-e' in script
    assert "shell-init powershell" in script
    assert "Remove-Item" not in script
    assert "rmdir" not in script.casefold()
    assert "Set-Content $PROFILE" not in script
    assert "Add-Content $PROFILE" not in script
    assert "either -WithDev or -WithService" in script


def test_windows_project_launcher_is_pinned_to_the_project_venv() -> None:
    script = (Path(__file__).resolve().parents[2] / "scripts" / "repopilot.ps1").read_text(
        encoding="utf-8"
    )

    assert ".venv\\Scripts\\repopilot.exe" in script
    assert "ValueFromRemainingArguments" in script
    assert "Set-Location -LiteralPath $repository" in script
    assert "Remove-Item" not in script
    assert "Set-Content $PROFILE" not in script


def test_windows_acceptance_script_stays_offline_and_does_not_change_shell_or_credentials() -> None:
    script = (Path(__file__).resolve().parents[2] / "scripts" / "acceptance_windows.ps1").read_text(
        encoding="utf-8"
    )

    assert "shell-init powershell" in script
    assert "mcp doctor" in script
    assert "Credential Manager" not in script
    assert "auth login" not in script
    assert "Set-Content $PROFILE" not in script
    assert "Remove-Item" not in script
    assert "isolated child PowerShell" in script
    assert "collision-proof project PowerShell launcher" in script
    assert "No cloud model or API key was used" in script


def test_windows_release_script_is_local_and_non_destructive() -> None:
    script = (
        Path(__file__).resolve().parents[2] / "scripts" / "release_check_windows.ps1"
    ).read_text(encoding="utf-8")

    assert "pip check" in script
    assert "eval coding-dev plan" in script
    assert "acceptance_windows.ps1" in script
    assert "Remove-Item" not in script
    assert "Set-Content $PROFILE" not in script
    assert "auth login" not in script


def test_windows_package_smoke_is_opt_in_and_retains_evidence() -> None:
    root = Path(__file__).resolve().parents[2]
    release_script = (root / "scripts" / "release_check_windows.ps1").read_text(encoding="utf-8")
    smoke_script = (root / "scripts" / "package_smoke_windows.ps1").read_text(encoding="utf-8")

    assert "[switch]$PackageSmoke" in release_script
    assert "package_smoke_windows.ps1" in release_script
    assert "pip wheel" in smoke_script
    assert "--no-build-isolation" in smoke_script
    assert "--no-index --find-links" in smoke_script
    assert "artifacts\\package_smoke" in smoke_script
    assert "Remove-Item" not in smoke_script
    assert "Set-Content $PROFILE" not in smoke_script


def test_windows_release_baseline_and_docker_gate_are_explicit_and_fail_closed() -> None:
    root = Path(__file__).resolve().parents[2]
    release_script = (root / "scripts" / "release_check_windows.ps1").read_text(encoding="utf-8")
    baseline_script = (root / "scripts" / "release_baseline_windows.ps1").read_text(
        encoding="utf-8"
    )
    docker_script = (root / "scripts" / "docker_security_gate_windows.ps1").read_text(
        encoding="utf-8"
    )

    assert "[switch]$Baseline" in release_script
    assert "[switch]$DockerSecurity" in release_script
    assert "[switch]$ReleaseScope" in release_script
    assert "[switch]$CIPreflight" in release_script
    assert "release_baseline_windows.ps1" in release_script
    assert "docker_security_gate_windows.ps1" in release_script
    assert "release_scope_windows.ps1" in release_script
    assert "ci_preflight_windows.ps1" in release_script
    assert "status --porcelain=v1 --untracked-files=normal" in baseline_script
    assert "diff --check" in baseline_script
    assert "git commit" not in baseline_script
    assert "Remove-Item" not in baseline_script
    assert "Docker daemon is unavailable" in docker_script
    assert "No host-execution fallback" in docker_script
    assert "Remove-Item" not in docker_script


def test_windows_release_scope_ci_preflight_and_acceptance_scripts_are_bounded() -> None:
    root = Path(__file__).resolve().parents[2]
    scope_script = (root / "scripts" / "release_scope_windows.ps1").read_text(encoding="utf-8")
    preflight_script = (root / "scripts" / "ci_preflight_windows.ps1").read_text(encoding="utf-8")
    provider_script = (root / "scripts" / "interactive_acceptance_windows.ps1").read_text(
        encoding="utf-8"
    )
    transcript_script = (root / "scripts" / "interactive_transcript_windows.ps1").read_text(
        encoding="utf-8"
    )

    assert "status --porcelain=v1 --untracked-files=normal" in scope_script
    assert "review_then_include" in scope_script
    assert "git commit" not in scope_script
    assert "Remove-Item" not in scope_script
    assert "remote get-url origin" in preflight_script
    assert "Docker daemon is unavailable" in preflight_script
    assert "git push" not in preflight_script
    assert "REPOPILOT_STREAM_OK" in provider_script
    assert "Do not read, analyze, or modify project files" in provider_script
    assert "test_session_interruption_recovery.py" in provider_script
    assert "REPOPILOT_STREAM_OK" not in transcript_script
    assert "--provider local" in transcript_script
    for script in (preflight_script, provider_script, transcript_script):
        assert "Remove-Item" not in script
        assert "Set-Content $PROFILE" not in script


def test_package_smoke_receipt_and_verifier_are_offline_and_integrity_bound() -> None:
    root = Path(__file__).resolve().parents[2]
    smoke_script = (root / "scripts" / "package_smoke_windows.ps1").read_text(encoding="utf-8")
    verifier = (root / "scripts" / "verify_package_smoke_receipt_windows.ps1").read_text(
        encoding="utf-8"
    )

    assert "package-smoke.receipt.json" in smoke_script
    assert "Get-FileHash" in smoke_script
    assert "wheel_sha256" in smoke_script
    assert "Get-FileHash" in verifier
    assert "SHA-256 does not match" in verifier
    assert "pip install" not in verifier
    assert "Remove-Item" not in verifier
