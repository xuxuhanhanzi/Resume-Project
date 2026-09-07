"""Evidence protocol primitives shared by the R1--R4 experiments.

The module intentionally has no provider, Docker, or dataset dependency.  It
creates stable split manifests and receipts before any expensive run starts,
and it refuses to overwrite an artifact that already represents evidence.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

SplitName = Literal["development", "validation", "final_holdout"]
_SPLITS: tuple[SplitName, ...] = ("development", "validation", "final_holdout")
_SPLIT_WEIGHTS: dict[SplitName, float] = {
    "development": 0.20,
    "validation": 0.40,
    "final_holdout": 0.40,
}


@dataclass(frozen=True, slots=True)
class DatasetInstance:
    """One public evaluation instance and its atomic leakage group(s)."""

    instance_id: str
    group_ids: tuple[str, ...]
    stratum: str

    def __post_init__(self) -> None:
        if not self.instance_id.strip():
            raise ValueError("instance_id must not be empty")
        if not self.group_ids or any(not group.strip() for group in self.group_ids):
            raise ValueError("every instance needs at least one non-empty group ID")
        if not self.stratum.strip():
            raise ValueError("stratum must not be empty")


@dataclass(frozen=True, slots=True)
class FrozenManifest:
    """A content-addressed, immutable assignment of public IDs to splits."""

    schema_version: int
    dataset_name: str
    dataset_version: str
    dataset_sha256: str
    split_algorithm: str
    assignments: dict[str, tuple[str, ...]]
    strata_counts: dict[str, dict[str, int]]
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("unsupported manifest schema version")
        if not self.dataset_name.strip() or not self.dataset_version.strip():
            raise ValueError("dataset name and version are required")
        if len(self.dataset_sha256) != 64 or any(
            char not in "0123456789abcdef" for char in self.dataset_sha256.casefold()
        ):
            raise ValueError("dataset_sha256 must be a SHA-256 hex digest")
        if set(self.assignments) != set(_SPLITS):
            raise ValueError("manifest must contain exactly development, validation, final_holdout")
        all_ids = [item for values in self.assignments.values() for item in values]
        if not all_ids or len(all_ids) != len(set(all_ids)):
            raise ValueError("manifest assignments must be non-empty and disjoint")
        if any(tuple(sorted(values)) != values for values in self.assignments.values()):
            raise ValueError("manifest assignments must be sorted")
        if set(self.strata_counts) != set(_SPLITS):
            raise ValueError("manifest must record per-split strata counts")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "dataset_name": self.dataset_name,
            "dataset_version": self.dataset_version,
            "dataset_sha256": self.dataset_sha256,
            "split_algorithm": self.split_algorithm,
            "assignments": {name: list(self.assignments[name]) for name in _SPLITS},
            "strata_counts": {
                name: dict(sorted(self.strata_counts[name].items())) for name in _SPLITS
            },
            "notes": list(self.notes),
        }

    @property
    def sha256(self) -> str:
        return canonical_sha256(self.to_dict())

    def write_once(self, path: Path) -> None:
        """Persist the canonical manifest once; replacement is never implicit."""

        if path.exists():
            raise ValueError(f"refusing to overwrite frozen manifest: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )


@dataclass(frozen=True, slots=True)
class ExperimentReceipt:
    """Comparable, payload-free evidence binding for one experiment run."""

    schema_version: int
    run_id: str
    variant: str
    manifest_sha256: str
    source_commit: str
    environment: dict[str, str]
    fixed_controls: dict[str, Any]
    raw_output_sha256: str | None
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("unsupported receipt schema version")
        for name, value in (
            ("run_id", self.run_id),
            ("variant", self.variant),
            ("source_commit", self.source_commit),
        ):
            if not value.strip() or len(value) > 160:
                raise ValueError(f"{name} must be a bounded non-empty string")
        if not _is_sha256(self.manifest_sha256):
            raise ValueError("manifest_sha256 must be a SHA-256 hex digest")
        if self.raw_output_sha256 is not None and not _is_sha256(self.raw_output_sha256):
            raise ValueError("raw_output_sha256 must be a SHA-256 hex digest")
        if not self.environment or not self.fixed_controls:
            raise ValueError("receipt must record environment and fixed controls")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "variant": self.variant,
            "manifest_sha256": self.manifest_sha256,
            "source_commit": self.source_commit,
            "environment": dict(sorted(self.environment.items())),
            "fixed_controls": self.fixed_controls,
            "raw_output_sha256": self.raw_output_sha256,
            "notes": list(self.notes),
        }

    def write_once(self, path: Path) -> None:
        if path.exists():
            raise ValueError(f"refusing to overwrite experiment receipt: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )


def canonical_sha256(value: object) -> str:
    """Hash JSON-compatible content independent of platform formatting."""

    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(char in "0123456789abcdef" for char in value.casefold())


def sha256_file(path: Path) -> str:
    """Return a file digest without loading a dataset into a second copy of RAM."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1_048_576), b""):
            digest.update(block)
    return digest.hexdigest()


def build_grouped_split_manifest(
    *,
    dataset_name: str,
    dataset_version: str,
    dataset_sha256: str,
    instances: list[DatasetInstance],
    notes: tuple[str, ...] = (),
) -> FrozenManifest:
    """Split connected leakage groups deterministically while balancing strata.

    Each connected component of shared group IDs is assigned as a unit.  A
    deterministic greedy objective balances both total count and each
    question-type stratum against the pre-registered 20/40/40 allocation.
    This makes grouping and split decisions reproducible without looking at
    model outputs.
    """

    if not instances:
        raise ValueError("instances must not be empty")
    if len({item.instance_id for item in instances}) != len(instances):
        raise ValueError("duplicate instance_id in split input")
    components = _connected_components(instances)
    totals = {item.stratum: 0 for item in instances}
    for item in instances:
        totals[item.stratum] += 1
    components_by_stratum: dict[str, set[int]] = {}
    for component_index, component in enumerate(components):
        for stratum in {member.stratum for member in component}:
            components_by_stratum.setdefault(stratum, set()).add(component_index)
    # A stratum contained in exactly one atomic leakage component cannot be
    # proportionally represented in all three splits.  Penalizing its
    # unavoidable missing development/validation/final allocations makes the
    # greedy objective choose a 40% split forever.  Retain stratum balancing
    # where a stratum actually has independently assignable components, and
    # use global count balance for atomic strata such as one repository group.
    splittable_strata = {
        stratum
        for stratum, component_ids in components_by_stratum.items()
        if len(component_ids) > 1
    }
    assigned: dict[SplitName, list[str]] = {name: [] for name in _SPLITS}
    assigned_strata: dict[SplitName, dict[str, int]] = {name: {} for name in _SPLITS}
    assigned_total: dict[SplitName, int] = {name: 0 for name in _SPLITS}

    # Allocate larger atomic groups first.  The earlier hash-only seeded
    # ordering could put three arbitrary groups into distinct splits before
    # considering their size; on repository-grouped SWE-bench this produced an
    # 8/500 development partition despite a feasible near-20% allocation.
    # The digest remains a deterministic tie breaker, not an outcome-driven
    # choice.  Every component is then scored by the same global target cost.
    ordered_components = sorted(
        components,
        key=lambda members: (
            -len(members),
            hashlib.sha256(
                (
                    dataset_version
                    + "\0"
                    + "\0".join(sorted(member.instance_id for member in members))
                ).encode("utf-8")
            ).hexdigest(),
        ),
    )
    for component in ordered_components:
        stratum_counts: dict[str, int] = {}
        for member in component:
            stratum_counts[member.stratum] = stratum_counts.get(member.stratum, 0) + 1
        selected = min(
            _SPLITS,
            key=lambda split: _assignment_cost(
                split,
                component_count=len(component),
                component_strata=stratum_counts,
                assigned_total=assigned_total,
                assigned_strata=assigned_strata,
                total_instances=len(instances),
                total_strata=totals,
                splittable_strata=splittable_strata,
            ),
        )
        assigned[selected].extend(member.instance_id for member in component)
        assigned_total[selected] += len(component)
        for stratum, count in stratum_counts.items():
            assigned_strata[selected][stratum] = assigned_strata[selected].get(stratum, 0) + count

    if any(not assigned[name] for name in _SPLITS):
        component_sizes = sorted((len(component) for component in components), reverse=True)
        raise ValueError(
            "group isolation cannot produce non-empty development, validation, and final_holdout "
            f"splits; connected component sizes start {component_sizes[:5]}"
        )
    return FrozenManifest(
        schema_version=1,
        dataset_name=dataset_name,
        dataset_version=dataset_version,
        dataset_sha256=dataset_sha256,
        split_algorithm="grouped_sha256_size_first_feasible_strata_greedy_v4",
        assignments={name: tuple(sorted(assigned[name])) for name in _SPLITS},
        strata_counts={name: dict(sorted(assigned_strata[name].items())) for name in _SPLITS},
        notes=notes,
    )


def _assignment_cost(
    split: SplitName,
    *,
    component_count: int,
    component_strata: dict[str, int],
    assigned_total: dict[SplitName, int],
    assigned_strata: dict[SplitName, dict[str, int]],
    total_instances: int,
    total_strata: dict[str, int],
    splittable_strata: set[str],
) -> tuple[float, float, str]:
    """Score the complete allocation after assigning a candidate component."""

    count_error = 0.0
    stratum_error = 0.0
    fill_ratio = 0.0
    for candidate_split in _SPLITS:
        total_target = total_instances * _SPLIT_WEIGHTS[candidate_split]
        future_total = assigned_total[candidate_split] + (
            component_count if candidate_split == split else 0
        )
        count_error += abs(future_total - total_target) / max(total_target, 1.0)
        fill_ratio += future_total / max(total_target, 1.0)
        for stratum in sorted(splittable_strata):
            total = total_strata[stratum]
            future_stratum = assigned_strata[candidate_split].get(stratum, 0) + (
                component_strata.get(stratum, 0) if candidate_split == split else 0
            )
            target = total * _SPLIT_WEIGHTS[candidate_split]
            stratum_error += abs(future_stratum - target) / max(target, 1.0)
    return (round(stratum_error + count_error, 12), round(fill_ratio, 12), split)


def _connected_components(instances: list[DatasetInstance]) -> list[list[DatasetInstance]]:
    """Union records that share any session/repository leakage identifier."""

    parent = {item.instance_id: item.instance_id for item in instances}

    def find(value: str) -> str:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(left: str, right: str) -> None:
        root_left, root_right = find(left), find(right)
        if root_left != root_right:
            parent[max(root_left, root_right)] = min(root_left, root_right)

    first_by_group: dict[str, str] = {}
    for item in instances:
        for group_id in sorted(set(item.group_ids)):
            if group_id in first_by_group:
                union(item.instance_id, first_by_group[group_id])
            else:
                first_by_group[group_id] = item.instance_id
    grouped: dict[str, list[DatasetInstance]] = {}
    for item in instances:
        grouped.setdefault(find(item.instance_id), []).append(item)
    return [sorted(group, key=lambda item: item.instance_id) for group in grouped.values()]
