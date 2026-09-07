from __future__ import annotations

import asyncio
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from repopilot.supervisor import LocalSupervisor, SupervisorJob, SupervisorStore


async def _wait_for_status(
    store: SupervisorStore, job_id: str, expected: set[str], *, attempts: int = 100
) -> str:
    for _ in range(attempts):
        status = store.resolve(job_id).status
        if status in expected:
            return status
        await asyncio.sleep(0.03)
    raise AssertionError(f"job did not reach one of {sorted(expected)}")


def test_supervisor_runs_an_explicit_job_and_keeps_a_redacted_local_log(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    store = SupervisorStore(tmp_path / "state")
    job = store.submit(
        project_root=project,
        cwd=project,
        command=(sys.executable, "-c", "print('supervisor complete')"),
    )

    async def scenario() -> None:
        stop_event = asyncio.Event()
        service = asyncio.create_task(
            LocalSupervisor(store, project_root=project).serve(
                poll_seconds=0.05, stop_event=stop_event
            )
        )
        assert await _wait_for_status(store, job.job_id, {"succeeded"}) == "succeeded"
        stop_event.set()
        await asyncio.wait_for(service, timeout=2.0)

    asyncio.run(scenario())
    completed = store.resolve(job.job_id, project_root=project)
    assert completed.status == "succeeded"
    assert "supervisor complete" in store.read_log(completed)


def test_supervisor_requires_explicit_retry_after_a_restart_interruption(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    store = SupervisorStore(tmp_path / "state")
    job = store.submit(
        project_root=project,
        cwd=project,
        command=(sys.executable, "-c", "print('never replay automatically')"),
    )
    store._write(replace(job, status="running", process_id=4321))  # noqa: SLF001

    interrupted = store.mark_interrupted(project_root=project)

    assert [item.job_id for item in interrupted] == [job.job_id]
    assert store.resolve(job.job_id).status == "interrupted"
    retried = store.retry(store.resolve(job.job_id))
    assert retried.status == "queued"
    assert retried.retry_of == job.job_id


def test_supervisor_rejects_a_command_that_would_persist_a_credential(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()

    with pytest.raises(ValueError, match="credential"):
        SupervisorStore(tmp_path / "state").submit(
            project_root=project,
            cwd=project,
            command=(sys.executable, "api_key=not-for-a-job"),
        )


def test_supervisor_enforces_a_project_queue_limit_and_reads_legacy_receipts(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    store = SupervisorStore(tmp_path / "state", max_queued_jobs=1)
    first = store.submit(
        project_root=project,
        cwd=project,
        command=(sys.executable, "-c", "print('one')"),
        label="first test",
        timeout_seconds=12.5,
    )

    assert first.label == "first test"
    assert first.timeout_seconds == 12.5
    with pytest.raises(ValueError, match="queue is full"):
        store.submit(
            project_root=project,
            cwd=project,
            command=(sys.executable, "-c", "print('two')"),
        )
    legacy = first.to_dict()
    for field in ("label", "timeout_seconds", "failure_reason"):
        del legacy[field]
    restored = SupervisorJob.from_dict(legacy)
    assert restored.label == ""
    assert restored.timeout_seconds == 900.0


def test_supervisor_marks_a_timed_out_job_failed_with_a_diagnosis(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    store = SupervisorStore(tmp_path / "state")
    job = store.submit(
        project_root=project,
        cwd=project,
        command=(sys.executable, "-c", "import time; time.sleep(5)"),
        label="timeout test",
        timeout_seconds=0.1,
    )

    async def scenario() -> None:
        stop_event = asyncio.Event()
        service = asyncio.create_task(
            LocalSupervisor(store, project_root=project).serve(
                poll_seconds=0.05, stop_event=stop_event
            )
        )
        assert await _wait_for_status(store, job.job_id, {"failed"}, attempts=200) == "failed"
        stop_event.set()
        await asyncio.wait_for(service, timeout=2.0)

    asyncio.run(scenario())
    completed = store.resolve(job.job_id, project_root=project)
    assert completed.failure_reason == "job timed out after 0.1 seconds"
    assert completed.retryable


def test_supervisor_records_a_launch_failure_instead_of_leaving_a_job_queued(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    store = SupervisorStore(tmp_path / "state")
    job = store.submit(
        project_root=project,
        cwd=project,
        command=("definitely-not-a-repopilot-executable",),
    )

    async def scenario() -> None:
        stop_event = asyncio.Event()
        service = asyncio.create_task(
            LocalSupervisor(store, project_root=project).serve(
                poll_seconds=0.05, stop_event=stop_event
            )
        )
        assert await _wait_for_status(store, job.job_id, {"failed"}) == "failed"
        stop_event.set()
        await asyncio.wait_for(service, timeout=2.0)

    asyncio.run(scenario())
    failed = store.resolve(job.job_id, project_root=project)
    assert failed.failure_reason is not None
    assert failed.failure_reason.startswith("could not start job:")
