"""Create the frozen Stage 5 verifier-backed preference dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import cast

from forgellm.alignment.preference_data import (
    GENERATOR_VERSION,
    VERIFIER_VERSION,
    adversarial_responses,
    build_preference_splits,
    verify_response,
)
from forgellm.alignment.schema import assert_split_disjoint, write_preference_jsonl
from forgellm.structured_logging import JsonValue


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create deterministic 384/64/64 verifier-backed Stage 5 preference pairs."
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output_dir = cast(Path, args.output_dir)
    splits = build_preference_splits()
    assert_split_disjoint(splits)
    output_dir.mkdir(parents=True, exist_ok=True)
    files: dict[str, JsonValue] = {}
    for split, records in splits.items():
        path = output_dir / f"{split}.jsonl"
        write_preference_jsonl(path, records)
        files[split] = {
            "path": path.as_posix(),
            "records": len(records),
            "sha256": _sha256(path),
        }
    test_records = splits["test"]
    attacks = 0
    attacks_rejected = 0
    for record in test_records:
        for response in adversarial_responses(record).values():
            attacks += 1
            attacks_rejected += int(verify_response(record, response)["total"] == 0.0)
    manifest: dict[str, JsonValue] = {
        "schema_version": "forgellm-stage5-preference-data-v1",
        "generator_version": GENERATOR_VERSION,
        "verifier_version": VERIFIER_VERSION,
        "split_policy": "SHA-256 sort of task semantics before rejected response construction",
        "files": files,
        "adversarial_audit": {
            "attacks": attacks,
            "rejected": attacks_rejected,
            "rejection_rate": attacks_rejected / attacks,
        },
        "license": "ForgeLLM project-original educational data",
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
