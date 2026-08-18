from __future__ import annotations

import asyncio
from functools import partial
from pathlib import Path
from typing import Any, cast

import pytest
import yaml

from repopilot.retrieval.bm25 import BM25CodeIndex
from repopilot.runtime.runner import LocalTrustedRunner
from repopilot.security.paths import task_path_is_visible
from repopilot.task import PublicTaskSpec


@pytest.mark.integration
def test_ten_case_benchmark_is_frozen_localizable_and_initially_failing() -> None:
    project = Path(__file__).resolve().parents[2]
    manifest_path = project / "evaluation" / "micro_benchmark.yaml"
    raw = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    assert isinstance(raw, dict)
    tasks = cast(dict[str, Any], raw)["tasks"]
    assert isinstance(tasks, list)
    assert len(tasks) == 10
    task_ids: set[str] = set()
    runner = LocalTrustedRunner(trusted=True)

    for raw_case in tasks:
        assert isinstance(raw_case, dict)
        case = cast(dict[str, Any], raw_case)
        task_path = manifest_path.parent / str(case["path"])
        spec = PublicTaskSpec.load(task_path)
        task_ids.add(spec.task_id)
        execution = asyncio.run(
            runner.run(spec.test_command, cwd=spec.workspace, timeout_seconds=10)
        )
        assert execution.exit_code != 0, f"{spec.task_id} no longer starts fail-to-pass"
        index = BM25CodeIndex.build(spec.workspace, path_filter=partial(task_path_is_visible, spec))
        hits = index.search(spec.problem_statement)
        assert hits and hits[0].path == "module.py"

    assert len(task_ids) == 10
