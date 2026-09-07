#!/usr/bin/env python3
"""Download and verify the pinned public benchmark sources used by RepoPilot."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download

FRAMES_REVISION = "58d9fb6330f3ab1316d1eca12e5e8ef23dcc22ef"
DABENCH_REVISION = "b455d578e30fee513abd79936cbdf7a6de026cb5"
SWE_REVISION = "a637bd46829f3132e12938c8a0ca93173a977b8e"
FRAMES_SHA256 = "4255093c93b595b5b04c7c8dde290b48ec87d72ca0fb0b760d9dd02740d669ff"
DABENCH_QUESTIONS_SHA256 = "430306c5813bf4f2b0e38f1b6ccee9693c36d5920b3322bd7c8ae46f475ce443"
DABENCH_LABELS_SHA256 = "83b8fb8133c1794fcb6b42435e9cce124d22f68b71e2eedb37c973e40b5a18c2"
SWE_SHA256 = "7ee0a75c41bfc954fd441b67ce738fc5c1cbae00721c4e30e7db4d893057c9ab"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def copy_verified(source: Path, target: Path, expected: str | None = None) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        shutil.copy2(source, target)
    if expected and sha256(target) != expected:
        raise RuntimeError(f"hash mismatch: {target}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--profile", choices=("smoke", "full"), default="smoke")
    args = parser.parse_args()
    project = args.project_root.expanduser().resolve()
    data = project / "evaluation" / "benchmarks" / "data"

    frames = Path(
        hf_hub_download(
            "google/frames-benchmark",
            "test.tsv",
            repo_type="dataset",
            revision=FRAMES_REVISION,
        )
    )
    copy_verified(frames, data / "frames" / FRAMES_REVISION / "test.tsv", FRAMES_SHA256)

    dabench_snapshot = Path(
        snapshot_download(
            "infiagent/DABench",
            repo_type="dataset",
            revision=DABENCH_REVISION,
            allow_patterns=["da-dev-questions.jsonl", "da-dev-labels.jsonl", "da-dev-tables/*.csv"],
        )
    )
    dabench = data / "dabench" / DABENCH_REVISION
    copy_verified(
        dabench_snapshot / "da-dev-questions.jsonl",
        dabench / "da-dev-questions.jsonl",
        DABENCH_QUESTIONS_SHA256,
    )
    copy_verified(
        dabench_snapshot / "da-dev-labels.jsonl",
        dabench / "da-dev-labels.jsonl",
        DABENCH_LABELS_SHA256,
    )
    for table in (dabench_snapshot / "da-dev-tables").glob("*.csv"):
        copy_verified(table, dabench / table.name)

    manifest = json.loads(
        (project / "evaluation/benchmarks/manifests/dabench_validation35_v2.json").read_text(
            encoding="utf-8"
        )
    )
    for name, expected in manifest["tables"].items():
        if sha256(dabench / name) != expected:
            raise RuntimeError(f"DABench hash mismatch: {name}")

    sources: dict[str, object] = {
        "frames": {"repo": "google/frames-benchmark", "revision": FRAMES_REVISION},
        "dabench": {"repo": "infiagent/DABench", "revision": DABENCH_REVISION},
    }
    if args.profile == "full":
        swe = Path(
            hf_hub_download(
                "SWE-bench-Live/SWE-bench-Live",
                "data/lite-00000-of-00001.parquet",
                repo_type="dataset",
                revision=SWE_REVISION,
            )
        )
        copy_verified(swe, data / "swebench_live" / SWE_REVISION / "lite.parquet", SWE_SHA256)
        sources["swebench_live"] = {
            "repo": "SWE-bench-Live/SWE-bench-Live",
            "revision": SWE_REVISION,
        }

    receipt = {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "profile": args.profile,
        "sources": sources,
    }
    receipt_path = data / "download-receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(receipt_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
