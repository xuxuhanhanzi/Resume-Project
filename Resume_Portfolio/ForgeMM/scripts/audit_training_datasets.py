from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from forgemm.swift_plugin import score_swift_batch  # noqa: E402


def _iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid_json:{path}:{line_number}:{exc}") from exc
            if not isinstance(payload, dict):
                raise ValueError(f"row_not_object:{path}:{line_number}")
            yield payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def audit(sft_path: Path, grpo_path: Path, output: Path, project_root: Path) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing_to_overwrite:{output}")
    started = time.perf_counter()
    grpo_rows = list(_iter_jsonl(grpo_path))
    by_group: dict[str, dict[str, Any]] = {}
    missing_images: list[str] = []
    for row in grpo_rows:
        group_key = str(row["group_key"])
        if group_key in by_group:
            raise ValueError(f"duplicate_grpo_group:{group_key}")
        by_group[group_key] = row
        for image in row["images"]:
            path = Path(str(image))
            resolved = path if path.is_absolute() else project_root / path
            if not resolved.is_file():
                missing_images.append(str(image))

    sft_count = 0
    samples: set[str] = set()
    templates_by_group: dict[str, set[int]] = {}
    reward_failures: list[dict[str, Any]] = []
    for row in _iter_jsonl(sft_path):
        sft_count += 1
        sample_id = str(row["sample_id"])
        if sample_id in samples:
            raise ValueError(f"duplicate_sft_sample:{sample_id}")
        samples.add(sample_id)
        group_key = str(row["group_key"])
        if group_key not in by_group:
            raise ValueError(f"sft_group_missing_from_grpo:{group_key}")
        templates_by_group.setdefault(group_key, set()).add(int(row["template_id"]))
        messages = row["messages"]
        if len(messages) != 2 or messages[-1].get("role") != "assistant":
            raise ValueError(f"invalid_sft_messages:{sample_id}")
        gold = by_group[group_key]
        bundles = score_swift_batch([str(messages[-1]["content"])], **gold)
        bundle = bundles[0]
        if (bundle.task, bundle.evidence, bundle.operation) != (1.0, 1.0, 1.0):
            reward_failures.append(
                {
                    "sample_id": sample_id,
                    "task": bundle.task,
                    "evidence": bundle.evidence,
                    "operation": bundle.operation,
                    "parse_error": bundle.parse_error,
                }
            )

    invalid_template_groups = sorted(
        group for group, templates in templates_by_group.items() if templates != {0, 1}
    )
    report = {
        "schema_version": 1,
        "experiment_scope": "local_full_training_dataset_and_reward_replay",
        "sft_rows": sft_count,
        "grpo_rows": len(grpo_rows),
        "unique_sft_sample_ids": len(samples),
        "unique_grpo_groups": len(by_group),
        "groups_with_exactly_templates_0_and_1": len(by_group) - len(invalid_template_groups),
        "invalid_template_groups": invalid_template_groups[:20],
        "missing_image_references": len(missing_images),
        "missing_image_examples": missing_images[:20],
        "gold_reward_replays": sft_count,
        "gold_reward_failures": len(reward_failures),
        "gold_reward_failure_examples": reward_failures[:20],
        "all_checks_passed": (
            sft_count == 2 * len(grpo_rows)
            and len(samples) == sft_count
            and not invalid_template_groups
            and not missing_images
            and not reward_failures
        ),
        "wall_seconds": time.perf_counter() - started,
        "sft_sha256": _sha256(sft_path),
        "grpo_sha256": _sha256(grpo_path),
        "claim_boundary": (
            "validates local data transport, parsing and deterministic rewards only; "
            "it is not a model-quality or ms-swift training result"
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit every generated SFT and GRPO row")
    parser.add_argument("--sft", required=True)
    parser.add_argument("--grpo", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    arguments = parser.parse_args()
    report = audit(
        Path(arguments.sft),
        Path(arguments.grpo),
        Path(arguments.output),
        Path(arguments.project_root).resolve(),
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["all_checks_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
