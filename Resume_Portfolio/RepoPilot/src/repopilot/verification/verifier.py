"""Outcome-first graders; model prose is never completion evidence."""

from __future__ import annotations

from repopilot.core.contracts import VerificationResult
from repopilot.runtime.runner import CommandRunner
from repopilot.task import EvaluatorTaskSpec, PublicTaskSpec
from repopilot.tools.coding import capture_text_snapshot


class DeterministicVerifier:
    """Check changed-file policy and execute the immutable visible test command."""

    def __init__(self, runner: CommandRunner, baseline: dict[str, str]) -> None:
        self.runner = runner
        self.baseline = baseline

    async def verify(self, task: PublicTaskSpec) -> VerificationResult:
        current = capture_text_snapshot(task.workspace, include_paths=task.allowed_paths)
        changed = sorted(
            path
            for path in set(self.baseline) | set(current)
            if self.baseline.get(path) != current.get(path)
        )
        if not changed:
            return VerificationResult(False, "no repository file changed", recoverable=True)
        if len(changed) > task.max_changed_files:
            return VerificationResult(
                False,
                "changed-file budget exceeded",
                details={"changed_files": changed},
            )
        if not task.test_command:
            return VerificationResult(
                False, "task has no immutable test command", recoverable=False
            )
        try:
            execution = await self.runner.run(
                task.test_command,
                cwd=task.workspace,
                timeout_seconds=min(task.budget.max_wall_seconds, 120.0),
            )
        except (OSError, RuntimeError, PermissionError, ValueError) as error:
            return VerificationResult(False, f"verifier could not execute: {error}")
        passed = execution.exit_code == 0 and not execution.timed_out
        summary = "visible verification passed" if passed else "visible verification failed"
        details: dict[str, object] = {
            "changed_files": changed,
            "changed_file_count": len(changed),
            "exit_code": execution.exit_code,
            "stdout": execution.stdout,
            "stderr": execution.stderr,
            "timed_out": execution.timed_out,
        }
        if not passed and execution.stderr:
            stderr_lines = execution.stderr.strip().splitlines()
            exc_lines = [
                line
                for line in stderr_lines
                if "Error" in line or "Exception" in line or "FAILED" in line
            ]
            details["exception_type"] = exc_lines[-1].strip() if exc_lines else "unknown"
            details["shortest_traceback"] = "\n".join(stderr_lines[-8:])
        return VerificationResult(
            passed,
            summary,
            recoverable=not execution.timed_out,
            details=details,
        )


class HiddenTestGrader:
    """Run evaluator-only test IDs without adding them to agent state or context."""

    def __init__(self, runner: CommandRunner) -> None:
        self.runner = runner

    async def grade(self, spec: EvaluatorTaskSpec) -> VerificationResult:
        if not spec.hidden_tests:
            return VerificationResult(True, "no hidden tests configured")
        if not spec.public.test_command:
            return VerificationResult(False, "public task has no test command")
        command = (*spec.public.test_command, *spec.hidden_tests)
        execution = await self.runner.run(
            command,
            cwd=spec.public.workspace,
            timeout_seconds=min(spec.public.budget.max_wall_seconds, 120.0),
        )
        passed = execution.exit_code == 0 and not execution.timed_out
        return VerificationResult(
            passed,
            "hidden verification passed" if passed else "hidden verification failed",
            details={
                "exit_code": execution.exit_code,
                "stdout": execution.stdout,
                "stderr": execution.stderr,
            },
        )
