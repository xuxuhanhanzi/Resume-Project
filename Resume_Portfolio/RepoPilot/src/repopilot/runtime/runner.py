"""Trusted-local and fail-closed Docker command runners."""

from __future__ import annotations

import asyncio
import os
import shutil
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True, slots=True)
class ExecutionResult:
    """Bounded command result returned to a tool or verifier."""

    command: tuple[str, ...]
    exit_code: int
    stdout: str
    stderr: str
    timed_out: bool = False


class CommandRunner(Protocol):
    """Execution boundary used by tests and setup commands."""

    async def run(
        self, command: Sequence[str], *, cwd: Path, timeout_seconds: float
    ) -> ExecutionResult:
        """Execute a tokenized command without a shell."""
        ...


def _bounded_output(value: str, max_output_bytes: int) -> str:
    encoded = value.encode("utf-8", errors="replace")
    if len(encoded) <= max_output_bytes:
        return value
    suffix = b"\n...[output truncated by RepoPilot]"
    return (encoded[: max_output_bytes - len(suffix)] + suffix).decode("utf-8", errors="replace")


class LocalTrustedRunner:
    """Run only project-owned fixtures; never pretend to sandbox untrusted code."""

    def __init__(self, *, trusted: bool, max_output_bytes: int = 64_000) -> None:
        self.trusted = trusted
        self.max_output_bytes = max_output_bytes

    async def run(
        self, command: Sequence[str], *, cwd: Path, timeout_seconds: float
    ) -> ExecutionResult:
        if not self.trusted:
            raise PermissionError("untrusted execution requires DockerSandboxRunner")
        if not command:
            raise ValueError("command must not be empty")
        return await asyncio.to_thread(self._run_sync, tuple(command), cwd, timeout_seconds)

    def _run_sync(
        self, command: tuple[str, ...], cwd: Path, timeout_seconds: float
    ) -> ExecutionResult:
        safe_env = {
            "PATH": os.environ.get("PATH", ""),
            "PYTHONIOENCODING": "utf-8",
            "PYTHONDONTWRITEBYTECODE": "1",
        }
        try:
            result = subprocess.run(
                command,
                cwd=cwd,
                env=safe_env,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout_seconds,
                check=False,
                shell=False,
            )
        except subprocess.TimeoutExpired as error:
            return ExecutionResult(
                command,
                -1,
                _bounded_output(str(error.stdout or ""), self.max_output_bytes),
                _bounded_output(str(error.stderr or ""), self.max_output_bytes),
                timed_out=True,
            )
        return ExecutionResult(
            command,
            result.returncode,
            _bounded_output(result.stdout, self.max_output_bytes),
            _bounded_output(result.stderr, self.max_output_bytes),
        )


@dataclass(frozen=True, slots=True)
class DockerSandboxConfig:
    """Minimal resource and isolation policy for an untrusted task container."""

    image: str
    memory: str = "1g"
    cpus: float = 1.0
    pids_limit: int = 64
    user: str = "65534:65534"
    max_output_bytes: int = 64_000

    def __post_init__(self) -> None:
        if not self.image or self.cpus <= 0 or self.pids_limit <= 0:
            raise ValueError("sandbox image and positive resources are required")


class DockerSandboxRunner:
    """Execute in a networkless, non-root, read-only-root Docker container."""

    def __init__(self, config: DockerSandboxConfig) -> None:
        self.config = config

    def build_command(
        self, command: Sequence[str], *, cwd: Path, timeout_seconds: float
    ) -> tuple[str, ...]:
        if not command:
            raise ValueError("command must not be empty")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        workspace = str(cwd.resolve(strict=True))
        return (
            "docker",
            "run",
            "--rm",
            "--network",
            "none",
            "--read-only",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "--pids-limit",
            str(self.config.pids_limit),
            "--memory",
            self.config.memory,
            "--cpus",
            str(self.config.cpus),
            "--user",
            self.config.user,
            "--tmpfs",
            "/tmp:rw,noexec,nosuid,size=64m",
            "--mount",
            f"type=bind,source={workspace},target=/workspace",
            "--workdir",
            "/workspace",
            self.config.image,
            *tuple(command),
        )

    async def run(
        self, command: Sequence[str], *, cwd: Path, timeout_seconds: float
    ) -> ExecutionResult:
        if shutil.which("docker") is None:
            raise RuntimeError("Docker CLI is unavailable; refusing untrusted execution")
        docker_command = self.build_command(command, cwd=cwd, timeout_seconds=timeout_seconds)
        trusted_runner = LocalTrustedRunner(
            trusted=True, max_output_bytes=self.config.max_output_bytes
        )
        return await trusted_runner.run(
            docker_command, cwd=cwd, timeout_seconds=timeout_seconds + 5.0
        )
