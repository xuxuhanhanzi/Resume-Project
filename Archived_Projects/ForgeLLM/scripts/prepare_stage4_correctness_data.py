"""Create the frozen 96-record project-original Stage 4 correctness dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Sequence
from pathlib import Path

from forgellm.post_training.correctness_data import (
    build_correctness_records,
    split_correctness_records,
)
from forgellm.post_training.schema import dataset_fingerprint, write_instruction_jsonl


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def main(argv: Sequence[str] | None = None) -> int:
    """Write split JSONL files and one deterministic Manifest."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    output_dir: Path = args.output_dir
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists():
        raise FileExistsError(f"Stage 4 correctness data already exists: {manifest_path}")
    splits = split_correctness_records(build_correctness_records())
    split_reports: dict[str, object] = {}
    for name, records in splits.items():
        path = output_dir / f"{name}.jsonl"
        write_instruction_jsonl(path, records)
        split_reports[name] = {
            "path": str(path),
            "records": len(records),
            "sha256": _sha256(path),
            "dataset_fingerprint": dataset_fingerprint(records),
        }
    manifest = {
        "schema_version": "forgellm-stage4-correctness-manifest-v1",
        "dataset": "sft_correctness_v1",
        "source": "ForgeLLM project-original deterministic generator",
        "license": "project-original",
        "split_policy": "ordered 64/16/16 over generator indices 0..95",
        "splits": split_reports,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(manifest_path.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
