"""Freeze public-only SWE-bench Verified inputs and the R2 split manifest.

Run this in the isolated evaluator environment.  It reads the complete
Hugging Face dataset only during offline preparation, writes an allowlisted
public JSONL, then writes an immutable repository-grouped manifest.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from datasets import load_dataset

from repopilot.evaluation.swebench_verified import (
    DATASET_ID,
    DATASET_SPLIT,
    build_swebench_verified_manifest,
    extract_public_records,
    public_dataset_sha256,
    source_dataset_sha256,
    write_public_records_once,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf-cache", type=Path, required=True)
    parser.add_argument("--dataset-revision", required=True)
    parser.add_argument("--public-output", type=Path, required=True)
    parser.add_argument("--manifest-output", type=Path, required=True)
    arguments = parser.parse_args()
    if arguments.manifest_output.exists():
        raise ValueError(f"refusing to overwrite frozen manifest: {arguments.manifest_output}")
    dataset = load_dataset(
        DATASET_ID,
        split=DATASET_SPLIT,
        cache_dir=str(arguments.hf_cache),
        revision=arguments.dataset_revision,
    )
    source_rows: list[dict[str, Any]] = [dict(row) for row in dataset]
    public_records = extract_public_records(source_rows)
    source_sha = source_dataset_sha256(source_rows)
    manifest = build_swebench_verified_manifest(
        public_records,
        dataset_version=f"{DATASET_ID}@{arguments.dataset_revision}",
        source_sha256=source_sha,
    )
    write_public_records_once(arguments.public_output, public_records)
    manifest.write_once(arguments.manifest_output)
    print(
        json.dumps(
            {
                "dataset": DATASET_ID,
                "dataset_revision": arguments.dataset_revision,
                "source_sha256": source_sha,
                "public_sha256": public_dataset_sha256(public_records),
                "public_records": len(public_records),
                "manifest_sha256": manifest.sha256,
                "splits": {name: len(values) for name, values in manifest.assignments.items()},
                "public_output": str(arguments.public_output),
                "manifest_output": str(arguments.manifest_output),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
