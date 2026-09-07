"""Trusted-local and fail-closed Docker command runners."""

from __future__ import annotations

import asyncio
import os
import shutil
import signal
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from repopilot.runtime.cancellation import CancellationToken, OperationCancelledError


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
        self,
        command: Sequence[str],
        *,
        cwd: Path,
        timeout_seconds: float,
        cancellation: CancellationToken | None = None,
    ) -> ExecutionResult:
        """Execute a tokenized command without a shell."""
        ...


def _bounded_output(value: str, max_output_bytes: int) -> str:
    if max_output_bytes <= 0:
        raise ValueError("max_output_bytes must be positive")
    encoded = value.encode("utf-8", errors="replace")
    if len(encoded) <= max_output_bytes:
        return value
    suffix = b"\n...[output truncated by RepoPilot]"
    if max_output_bytes <= len(suffix):
        return suffix[:max_output_bytes].decode("ascii")
    prefix = encoded[: max_output_bytes - len(suffix)]
    # Do not create a U+FFFD replacement character merely because the byte
    # limit happens to fall in the middle of a UTF-8 code point.
    while prefix:
        try:
            rendered_prefix = prefix.decode("utf-8")
            break
        except UnicodeDecodeError as error:
            prefix = prefix[: error.start]
    else:
        rendered_prefix = ""
    return rendered_prefix + suffix.decode("ascii")


class LocalTrustedRunner:
    """Run only project-owned fixtures; never pretend to sandbox untrusted code."""

    def __init__(self, *, trusted: bool, max_output_bytes: int = 64_000) -> None:
        if max_output_bytes <= 0:
            raise ValueError("max_output_bytes must be positive")
        self.trusted = trusted
        self.max_output_bytes = max_output_bytes

    async def run(
        self,
        command: Sequence[str],
        *,
        cwd: Path,
        timeout_seconds: float,
        cancellation: CancellationToken | None = None,
    ) -> ExecutionResult:
        if not self.trusted:
            raise PermissionError("untrusted execution requires DockerSandboxRunner")
        if not command:
            raise ValueError("command must not be empty")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if cancellation is not None:
            cancellation.raise_if_cancelled()
        safe_env = {
            "PATH": os.environ.get("PATH", ""),
            "PYTHONIOENCODING": "utf-8",
            "PYTHONDONTWRITEBYTECODE": "1",
            # A venv can otherwise discover pytest entry-point plugins installed
            # by the Python that originally created it (Anaconda on this host).
            # Project-declared plugins and explicit ``-p`` plugins still work.
            "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
            "PYTHONNOUSERSITE": "1",
        }
        # Windows system DLLs and Winsock initialisation require these inherited
        # operating-system values. They do not reintroduce Python package paths.
        for name in (
            "COMSPEC",
            "HOMEDRIVE",
            "HOMEPATH",
            "PATHEXT",
            "SYSTEMROOT",
            "TEMP",
            "TMP",
            "USERPROFILE",
            "WINDIR",
        ):
            if value := os.environ.get(name):
                safe_env[name] = value
        creationflags = 0
        if os.name == "nt":
            # Keep an invoked test or shell command outside the caller's console
            # control group.  A Ctrl+C broadcast from a host shell must cancel
            # the agent turn, not inject KeyboardInterrupt into the child test
            # process (which used to make a healthy pytest run look failed).
            import subprocess

            creationflags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS
        if os.name == "nt":
            process = await asyncio.create_subprocess_exec(
                *tuple(command),
                cwd=str(cwd),
                env=safe_env,
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                creationflags=creationflags,
            )
        else:
            # The child owns a fresh session/process group, allowing the
            # cancellation path to terminate descendants as well as the shell
            # or test runner initially launched by RepoPilot.
            process = await asyncio.create_subprocess_exec(
                *tuple(command),
                cwd=str(cwd),
                env=safe_env,
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                start_new_session=True,
            )
        communicate = asyncio.create_task(process.communicate())
        timeout = asyncio.create_task(asyncio.sleep(timeout_seconds))
        cancelled = asyncio.create_task(cancellation.wait()) if cancellation is not None else None
        waiters = {communicate, timeout}
        if cancelled is not None:
            waiters.add(cancelled)
        try:
            done, _ = await asyncio.wait(waiters, return_when=asyncio.FIRST_COMPLETED)
            if cancelled is not None and cancelled in done:
                await self._stop_process(process)
                await communicate
                assert cancellation is not None
                raise OperationCancelledError(cancellation.reason)
            if communicate in done:
                stdout, stderr = await communicate
                return ExecutionResult(
                    tuple(command),
                    process.returncode if process.returncode is not None else -1,
                    _bounded_output(_decode_output(stdout), self.max_output_bytes),
                    _bounded_output(_decode_output(stderr), self.max_output_bytes),
                )
            await self._stop_process(process)
            stdout, stderr = await communicate
            return ExecutionResult(
                tuple(command),
                -1,
                _bounded_output(_decode_output(stdout), self.max_output_bytes),
                _bounded_output(_decode_output(stderr), self.max_output_bytes),
                timed_out=True,
            )
        except asyncio.CancelledError:
            await self._stop_process(process)
            await communicate
            raise
        finally:
            for waiter in (timeout, cancelled):
                if waiter is not None and not waiter.done():
                    waiter.cancel()
            await asyncio.gather(
                *(waiter for waiter in (timeout, cancelled) if waiter is not None),
                return_exceptions=True,
            )

    @staticmethod
    async def _stop_process(process: asyncio.subprocess.Process) -> None:
        """Stop the launched process and its owned descendant tree when possible."""
        if process.returncode is not None:
            return
        if os.name == "nt" and process.pid is not None:
            terminator = await asyncio.create_subprocess_exec(
                "taskkill",
                "/PID",
                str(process.pid),
                "/T",
                "/F",
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            try:
                await asyncio.wait_for(terminator.wait(), timeout=3.0)
            except TimeoutError:
                terminator.kill()
                await terminator.wait()
            try:
                await asyncio.wait_for(process.wait(), timeout=3.0)
                return
            except TimeoutError:
                pass
        elif os.name != "nt" and process.pid is not None:
            if LocalTrustedRunner._signal_owned_process_group(process.pid, signal.SIGTERM):
                try:
                    await asyncio.wait_for(process.wait(), timeout=3.0)
                    return
                except TimeoutError:
                    LocalTrustedRunner._signal_owned_process_group(
                        process.pid, getattr(signal, "SIGKILL", signal.SIGTERM)
                    )
                    await process.wait()
                    return
        try:
            process.terminate()
        except ProcessLookupError:
            return
        try:
            await asyncio.wait_for(process.wait(), timeout=3.0)
        except TimeoutError:
            try:
                process.kill()
            except ProcessLookupError:
                return
            await process.wait()

    @staticmethod
    def _signal_owned_process_group(pid: int, signal_number: int) -> bool:
        """Signal the POSIX session started by this runner, when the OS supports it."""
        kill_process_group = getattr(os, "killpg", None)
        if kill_process_group is None:
            return False
        try:
            kill_process_group(pid, signal_number)
        except (ProcessLookupError, PermissionError):
            return False
        return True


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
        if not self.image or self.cpus <= 0 or self.pids_limit <= 0 or self.max_output_bytes <= 0:
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
            f"type=bind,source={workspace},target=/workspace,readonly",
            "--workdir",
            "/workspace",
            self.config.image,
            *tuple(command),
        )

    async def run(
        self,
        command: Sequence[str],
        *,
        cwd: Path,
        timeout_seconds: float,
        cancellation: CancellationToken | None = None,
    ) -> ExecutionResult:
        if shutil.which("docker") is None:
            raise RuntimeError("Docker CLI is unavailable; refusing untrusted execution")
        docker_command = self.build_command(command, cwd=cwd, timeout_seconds=timeout_seconds)
        trusted_runner = LocalTrustedRunner(
            trusted=True, max_output_bytes=self.config.max_output_bytes
        )
        return await trusted_runner.run(
            docker_command,
            cwd=cwd,
            timeout_seconds=timeout_seconds + 5.0,
            cancellation=cancellation,
        )


def _decode_output(value: bytes | None) -> str:
    # Match ``subprocess.run(text=True)`` from the previous runner so transcripts
    # remain stable across Windows and POSIX hosts.
    return (
        (value or b"").decode("utf-8", errors="replace").replace("\r\n", "\n").replace("\r", "\n")
    )
