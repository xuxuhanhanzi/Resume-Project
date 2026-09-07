from __future__ import annotations

from pathlib import Path

import pytest

from repopilot.evidence.protocol import (
    DatasetInstance,
    ExperimentReceipt,
    build_grouped_split_manifest,
)
from repopilot.evidence.statistics import paired_bootstrap_mean_difference, wilson_interval


def test_grouped_manifest_is_stable_and_never_splits_connected_group(tmp_path: Path) -> None:
    instances = [
        DatasetInstance("q1", ("s1", "s2"), "a"),
        DatasetInstance("q2", ("s2", "s3"), "a"),
        DatasetInstance("q3", ("s4",), "b"),
        DatasetInstance("q4", ("s5",), "b"),
        DatasetInstance("q5", ("s6",), "a"),
    ]
    manifest = build_grouped_split_manifest(
        dataset_name="toy",
        dataset_version="v1",
        dataset_sha256="a" * 64,
        instances=instances,
    )
    repeated = build_grouped_split_manifest(
        dataset_name="toy",
        dataset_version="v1",
        dataset_sha256="a" * 64,
        instances=instances,
    )

    assert manifest.to_dict() == repeated.to_dict()
    containing_q1 = next(name for name, ids in manifest.assignments.items() if "q1" in ids)
    assert "q2" in manifest.assignments[containing_q1]
    path = tmp_path / "manifest.json"
    manifest.write_once(path)
    with pytest.raises(ValueError, match="overwrite"):
        manifest.write_once(path)


def test_grouped_manifest_allocates_large_atomic_groups_before_small_seed_groups() -> None:
    """A repository-level split should not collapse development to a tiny repo."""

    sizes = (231, 75, 44, 34, 32, 22, 22, 19, 10, 8, 2, 1)
    instances = [
        DatasetInstance(f"repo-{repo}-{index}", (f"repo:{repo}",), f"repo-{repo}")
        for repo, size in enumerate(sizes)
        for index in range(size)
    ]

    manifest = build_grouped_split_manifest(
        dataset_name="toy-repositories",
        dataset_version="v1",
        dataset_sha256="a" * 64,
        instances=instances,
    )

    assert len(manifest.assignments["development"]) >= 75
    assert len(manifest.assignments["validation"]) >= 150
    assert len(manifest.assignments["final_holdout"]) >= 150
    assert manifest.split_algorithm.endswith("_v4")


def test_receipt_and_confidence_intervals_are_deterministic(tmp_path: Path) -> None:
    interval = wilson_interval(3, 10)
    difference = paired_bootstrap_mean_difference([0, 0, 1], [1, 0, 1], repetitions=200, seed=3)
    receipt = ExperimentReceipt(
        schema_version=1,
        run_id="r1-a",
        variant="bm25",
        manifest_sha256="b" * 64,
        source_commit="abc123",
        environment={"python": "3.12"},
        fixed_controls={"temperature": 0.0, "budget": 4096},
        raw_output_sha256=None,
    )

    assert interval.estimate == 0.3
    assert 0 < interval.lower < interval.upper < 1
    assert difference.estimate == pytest.approx(1 / 3)
    receipt.write_once(tmp_path / "receipt.json")
