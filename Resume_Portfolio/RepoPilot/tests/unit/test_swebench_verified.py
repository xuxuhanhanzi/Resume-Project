from __future__ import annotations

from pathlib import Path

import pytest

from repopilot.evaluation.swebench_verified import (
    build_swebench_verified_manifest,
    extract_public_records,
    load_public_records,
    public_dataset_sha256,
    write_public_records_once,
)


def _row(instance_id: str, repo: str, *, patch: str = "gold") -> dict[str, object]:
    return {
        "instance_id": instance_id,
        "repo": repo,
        "base_commit": f"base-{instance_id}",
        "problem_statement": f"Fix {instance_id}",
        "version": "1.0",
        "image": f"swebench/{instance_id}:latest",
        "patch": patch,
        "test_patch": "hidden test patch",
        "FAIL_TO_PASS": '["hidden_test"]',
        "PASS_TO_PASS": '["regression_test"]',
        "eval_script": "secret evaluator script",
    }


def test_public_projection_excludes_evaluator_material_and_roundtrips(tmp_path: Path) -> None:
    records = extract_public_records([_row("repo_a-1", "owner/a"), _row("repo_b-1", "owner/b")])

    assert all("patch" not in record for record in records)
    assert all("FAIL_TO_PASS" not in record for record in records)
    output = tmp_path / "public.jsonl"
    write_public_records_once(output, records)

    assert load_public_records(output) == records
    assert len(public_dataset_sha256(records)) == 64
    with pytest.raises(ValueError, match="refusing to overwrite"):
        write_public_records_once(output, records)


def test_public_loader_rejects_evaluator_field(tmp_path: Path) -> None:
    output = tmp_path / "leaked.jsonl"
    output.write_text(
        '{"instance_id":"x","repo":"owner/x","base_commit":"base","problem_statement":"fix",'
        '"version":"1","image":"image","patch":"gold"}\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="evaluator-only"):
        load_public_records(output)


def test_manifest_assigns_a_repository_to_exactly_one_split() -> None:
    records = extract_public_records(
        [
            _row("repo_a-1", "owner/a"),
            _row("repo_a-2", "owner/a"),
            _row("repo_b-1", "owner/b"),
            _row("repo_c-1", "owner/c"),
            _row("repo_d-1", "owner/d"),
        ]
    )

    manifest = build_swebench_verified_manifest(
        records,
        dataset_version="SWE-bench/SWE-bench_Verified@test-revision",
        source_sha256="a" * 64,
    )
    splits_by_id = {
        instance_id: split
        for split, instance_ids in manifest.assignments.items()
        for instance_id in instance_ids
    }

    assert splits_by_id["repo_a-1"] == splits_by_id["repo_a-2"]
    assert set(manifest.assignments) == {"development", "validation", "final_holdout"}
