"""Small environment implementations shared by benchmark smoke runs."""

from __future__ import annotations

from repopilot.benchmarks.contracts import EnvironmentHandle, PreparedTask


class LocalPreparedEnvironment:
    """Expose an adapter-prepared directory without running untrusted code."""

    async def start(self, prepared: PreparedTask) -> EnvironmentHandle:
        if not prepared.work_dir.is_dir():
            raise ValueError(f"prepared work directory does not exist: {prepared.work_dir}")
        return EnvironmentHandle(
            handle_id=f"local:{prepared.task.task_id}",
            work_dir=prepared.work_dir,
            metadata={"network_policy": prepared.task.network_policy.value},
        )

    async def reset(self, handle: EnvironmentHandle) -> None:
        del handle

    async def close(self, handle: EnvironmentHandle) -> None:
        del handle
