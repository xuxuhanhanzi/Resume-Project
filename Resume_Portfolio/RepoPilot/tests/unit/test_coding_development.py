from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

import repopilot.cli as cli
from repopilot.core.contracts import ModelResponse, ToolCall
from repopilot.evaluation.coding_development import (
    CodingDevelopmentManifest,
    coding_release_holdout_plan,
    compare_development_runs,
    development_case_receipt,
    development_outcomes_sha256,
    development_record,
    development_reliability_dashboard,
    development_run_receipt,
    inspect_development_run,
    materialize_coding_development_cases,
    verify_development_run,
    write_development_receipt,
    write_development_records,
)
from repopilot.providers.scripted import ScriptedProvider


def test_coding_development_plan_is_public_synthetic_and_offline(
    capsys: pytest.CaptureFixture[str],
) -> None:
    args = cli.build_parser().parse_args(["eval", "coding-dev", "plan"])

    assert asyncio.run(cli._eval_command(args)) == 0  # noqa: SLF001

    output = capsys.readouterr().out
    assert "repopilot-synthetic-dev-v1" in output
    assert "capability benchmark" in output
    assert "hidden" in output.casefold()


def test_release_holdout_plan_exposes_only_metadata_and_no_evaluator_payload(
    capsys: pytest.CaptureFixture[str],
) -> None:
    plan = coding_release_holdout_plan()
    args = cli.build_parser().parse_args(["eval", "coding-dev", "holdout-plan"])

    assert plan["suite_id"] == "repopilot-swebench-live-release-holdout-v1"
    assert plan["task_count"] == 5
    assert "gold patches" in str(plan["visibility"])
    assert "task_ids" not in plan
    assert asyncio.run(cli._eval_command(args)) == 0  # noqa: SLF001
    assert "external_immutable_holdout" in capsys.readouterr().out


def test_coding_development_materializes_new_fixture_workspaces_without_overwrite(
    tmp_path: Path,
) -> None:
    manifest = CodingDevelopmentManifest.load()
    artifacts = tmp_path / "artifacts"

    tasks = materialize_coding_development_cases(
        manifest, artifacts=artifacts, run_id="first-run", max_cases=2
    )

    assert [task.task_id for task in tasks] == ["clamp-upper-bound", "normalize-whitespace"]
    assert (tasks[0].workspace / "src/bounds.py").is_file()
    assert (tasks[0].workspace / "tests/test_bounds.py").is_file()
    with pytest.raises(ValueError, match="already exists"):
        materialize_coding_development_cases(
            manifest, artifacts=artifacts, run_id="first-run", max_cases=1
        )


def test_coding_development_runtime_produces_a_redacted_verified_funnel_record(
    tmp_path: Path,
) -> None:
    manifest = CodingDevelopmentManifest.load()
    task = materialize_coding_development_cases(
        manifest, artifacts=tmp_path / "artifacts", run_id="scripted-run", max_cases=1
    )[0]
    provider = ScriptedProvider(
        [
            ModelResponse(
                tool_calls=(
                    ToolCall(
                        "read-bounds",
                        "read_file",
                        {"path": "src/bounds.py", "start_line": 1, "end_line": 20},
                    ),
                )
            ),
            ModelResponse(
                tool_calls=(
                    ToolCall(
                        "fix-bounds",
                        "apply_patch",
                        {
                            "path": "src/bounds.py",
                            "old_text": "return max(lower, min(value, lower))",
                            "new_text": "return max(lower, min(value, upper))",
                        },
                    ),
                )
            ),
            ModelResponse(content="Fixed the upper bound."),
        ]
    )
    runtime = cli._coding_development_runtime(  # noqa: SLF001
        provider=provider, artifacts=task.workspace.parent / "runtime"
    )

    state = asyncio.run(runtime.run(task, run_id="agent", resume=False))
    record = development_record(task.task_id, state)
    records_path = task.workspace.parent / "outcomes.redacted.jsonl"
    summary = write_development_records(records_path, [record])

    assert state.status.value == "completed"
    assert record == {
        "task_id": "clamp-upper-bound",
        "localized": True,
        "patch": "applied",
        "patch_applied": True,
        "verification_passed": True,
    }
    assert summary.categories == {"verified": 1}
    assert "return max" not in records_path.read_text(encoding="utf-8")


def test_coding_development_runtime_finalizes_after_a_patched_fixture_test_passes(
    tmp_path: Path,
) -> None:
    manifest = CodingDevelopmentManifest.load()
    task = materialize_coding_development_cases(
        manifest, artifacts=tmp_path / "artifacts", run_id="auto-finalize", max_cases=1
    )[0]
    provider = ScriptedProvider(
        [
            ModelResponse(
                tool_calls=(
                    ToolCall(
                        "fix-bounds",
                        "apply_patch",
                        {
                            "path": "src/bounds.py",
                            "old_text": "return max(lower, min(value, lower))",
                            "new_text": "return max(lower, min(value, upper))",
                        },
                    ),
                )
            ),
            ModelResponse(tool_calls=(ToolCall("test-bounds", "run_tests", {}),)),
            ModelResponse(tool_calls=(ToolCall("unneeded-diff", "git_diff", {}),)),
        ]
    )
    runtime = cli._coding_development_runtime(  # noqa: SLF001
        provider=provider, artifacts=task.workspace.parent / "runtime"
    )

    state = asyncio.run(runtime.run(task, run_id="agent", resume=False))

    assert state.status.value == "completed"
    assert state.iteration == 2
    assert state.tool_calls == 2
    assert "verified" in state.final_answer


def test_coding_development_run_requires_explicit_cloud_and_fixture_edit_consent() -> None:
    args = cli.build_parser().parse_args(["eval", "coding-dev", "run"])
    manifest = CodingDevelopmentManifest.load()

    with pytest.raises(ValueError, match="explicit --provider"):
        asyncio.run(cli._run_coding_development(args, manifest))  # noqa: SLF001

    args = cli.build_parser().parse_args(["--provider", "deepseek", "eval", "coding-dev", "run"])
    with pytest.raises(ValueError, match="requires --trust"):
        asyncio.run(cli._run_coding_development(args, manifest))  # noqa: SLF001


def test_coding_development_receipt_and_report_are_local_redacted_and_evidence_led(
    tmp_path: Path,
) -> None:
    manifest = CodingDevelopmentManifest.load()
    run_directory = tmp_path / "artifacts" / "coding_development" / "runs" / "receipt-run"
    run_directory.mkdir(parents=True)
    records_path = run_directory / "outcomes.redacted.jsonl"
    write_development_records(
        records_path,
        [
            {
                "task_id": "clamp-upper-bound",
                "localized": True,
                "patch": "applied",
                "patch_applied": True,
                "verification_passed": False,
            }
        ],
    )
    receipt = development_run_receipt(
        manifest,
        run_id="receipt-run",
        provider="deepseek",
        model="model-name",
        selected_case_ids=("clamp-upper-bound",),
        cases=[
            development_case_receipt(
                "clamp-upper-bound",
                None,
                elapsed_seconds=1.2345,
                runtime_error=RuntimeError("api_key=secret\nconnection stopped"),
            )
        ],
        elapsed_seconds=2.0,
    )
    write_development_receipt(run_directory / "run.receipt.json", receipt)

    report = inspect_development_run(run_directory)

    summary = report["summary"]
    actions = report["next_actions"]
    assert isinstance(summary, dict)
    assert isinstance(actions, list)
    assert summary["categories"] == {"verification_failed": 1}
    assert all(isinstance(action, str) for action in actions)
    assert any("verification_failed" in action for action in actions)
    stored = (run_directory / "run.receipt.json").read_text(encoding="utf-8")
    assert "secret" not in stored
    assert "prompts" in stored


def test_coding_development_report_cli_does_not_construct_a_provider(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    manifest = CodingDevelopmentManifest.load()
    run_directory = tmp_path / "run"
    run_directory.mkdir()
    write_development_records(
        run_directory / "outcomes.redacted.jsonl",
        [
            {
                "task_id": "clamp-upper-bound",
                "localized": False,
                "patch": "",
                "patch_applied": False,
                "verification_passed": False,
            }
        ],
    )
    write_development_receipt(
        run_directory / "run.receipt.json",
        development_run_receipt(
            manifest,
            run_id="report-run",
            provider="deepseek",
            model="model-name",
            selected_case_ids=("clamp-upper-bound",),
            cases=[development_case_receipt("clamp-upper-bound", None, elapsed_seconds=0.0)],
            elapsed_seconds=0.0,
        ),
    )
    args = cli.build_parser().parse_args(["eval", "coding-dev", "report", str(run_directory)])

    assert asyncio.run(cli._eval_command(args)) == 0  # noqa: SLF001
    output = capsys.readouterr().out
    assert "localization_failed" in output
    assert "next_actions" in output


def test_coding_development_compare_marks_legacy_profile_change_directional(
    tmp_path: Path,
) -> None:
    manifest = CodingDevelopmentManifest.load()
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    baseline.mkdir()
    candidate.mkdir()
    records = {
        "baseline": {
            "task_id": "clamp-upper-bound",
            "localized": True,
            "patch": "applied",
            "patch_applied": True,
            "verification_passed": False,
        },
        "candidate": {
            "task_id": "clamp-upper-bound",
            "localized": True,
            "patch": "applied",
            "patch_applied": True,
            "verification_passed": True,
        },
    }
    for label, directory in (("baseline", baseline), ("candidate", candidate)):
        write_development_records(directory / "outcomes.redacted.jsonl", [records[label]])
        profile = None
        if label == "candidate":
            profile = {
                "repopilot_version": "1.13.0",
                "auto_finalize_after_successful_test": True,
                "task_budget": {
                    "max_iterations": 8,
                    "max_tool_calls": 20,
                    "max_total_tokens": 12000,
                    "max_wall_seconds": 180.0,
                },
                "tool_surface": ["read_file", "apply_patch", "run_tests"],
            }
        write_development_receipt(
            directory / "run.receipt.json",
            development_run_receipt(
                manifest,
                run_id=label,
                provider="deepseek",
                model="deepseek-v4-flash",
                selected_case_ids=("clamp-upper-bound",),
                cases=[development_case_receipt("clamp-upper-bound", None, elapsed_seconds=0.0)],
                elapsed_seconds=0.0,
                execution_profile=profile,
            ),
        )

    comparison = compare_development_runs(baseline, candidate)

    comparability = comparison["comparability"]
    delta = comparison["funnel_delta"]
    assert isinstance(comparability, dict)
    assert isinstance(delta, dict)
    assert comparability["classification"] == "directional_only"
    assert comparability["fully_controlled"] is False
    assert delta["verified"] == 1
    assert comparison["case_changes"] == [
        {
            "task_id": "clamp-upper-bound",
            "baseline_category": "verification_failed",
            "candidate_category": "verified",
        }
    ]


def test_coding_development_rejects_receipt_with_unsupported_field(tmp_path: Path) -> None:
    manifest = CodingDevelopmentManifest.load()
    receipt = development_run_receipt(
        manifest,
        run_id="strict-receipt",
        provider="deepseek",
        model="deepseek-v4-flash",
        selected_case_ids=("clamp-upper-bound",),
        cases=[development_case_receipt("clamp-upper-bound", None, elapsed_seconds=0.0)],
        elapsed_seconds=0.0,
    )
    receipt["untrusted_raw_trace"] = "must not become report output"

    with pytest.raises(ValueError, match="unsupported fields"):
        write_development_receipt(tmp_path / "run.receipt.json", receipt)


def test_coding_development_receipt_binds_and_detects_outcome_tampering(tmp_path: Path) -> None:
    manifest = CodingDevelopmentManifest.load()
    run_directory = tmp_path / "bound-run"
    run_directory.mkdir()
    outcomes = run_directory / "outcomes.redacted.jsonl"
    write_development_records(
        outcomes,
        [
            {
                "task_id": "clamp-upper-bound",
                "localized": True,
                "patch": "applied",
                "patch_applied": True,
                "verification_passed": True,
            }
        ],
    )
    write_development_receipt(
        run_directory / "run.receipt.json",
        development_run_receipt(
            manifest,
            run_id="bound-run",
            provider="deepseek",
            model="deepseek-v4-flash",
            selected_case_ids=("clamp-upper-bound",),
            cases=[development_case_receipt("clamp-upper-bound", None, elapsed_seconds=0.0)],
            elapsed_seconds=0.0,
            outcomes_sha256=development_outcomes_sha256(outcomes),
        ),
    )

    verification = verify_development_run(run_directory)

    integrity = verification["integrity"]
    assert isinstance(integrity, dict)
    assert integrity["classification"] == "bound_verified"
    assert integrity["verified"] is True

    outcomes.write_text(
        '{"task_id":"clamp-upper-bound","localized":true,"patch":"applied",'
        '"patch_applied":true,"verification_passed":false}\n',
        encoding="utf-8",
    )
    tampered = verify_development_run(run_directory)
    tampered_integrity = tampered["integrity"]
    assert isinstance(tampered_integrity, dict)
    assert tampered_integrity["classification"] == "bound_mismatch"
    with pytest.raises(ValueError, match="outcomes_sha256"):
        inspect_development_run(run_directory)


def test_coding_development_dashboard_is_local_and_never_calls_public_fixtures_a_holdout(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    manifest = CodingDevelopmentManifest.load()
    run_directory = tmp_path / "dashboard-run"
    run_directory.mkdir()
    outcomes = run_directory / "outcomes.redacted.jsonl"
    write_development_records(
        outcomes,
        [
            {
                "task_id": "clamp-upper-bound",
                "localized": True,
                "patch": "applied",
                "patch_applied": True,
                "verification_passed": True,
            }
        ],
    )
    write_development_receipt(
        run_directory / "run.receipt.json",
        development_run_receipt(
            manifest,
            run_id="dashboard-run",
            provider="deepseek",
            model="deepseek-v4-flash",
            selected_case_ids=("clamp-upper-bound",),
            cases=[development_case_receipt("clamp-upper-bound", None, elapsed_seconds=0.0)],
            elapsed_seconds=0.0,
            outcomes_sha256=development_outcomes_sha256(outcomes),
        ),
    )

    dashboard = development_reliability_dashboard((run_directory,))
    args = cli.build_parser().parse_args(["eval", "coding-dev", "dashboard", str(run_directory)])

    assert dashboard["aggregate"] == {
        "run_count": 1,
        "case_count": 1,
        "verified_case_count": 1,
        "receipts_without_verified_outcome_binding": 0,
    }
    boundary = dashboard["evidence_boundary"]
    assert isinstance(boundary, dict)
    assert boundary["release_holdout_required"] is True
    assert boundary["capability_score_claimed"] is False
    assert asyncio.run(cli._eval_command(args)) == 0  # noqa: SLF001
    assert "release_holdout_required" in capsys.readouterr().out
