"""Freeze a small Apache-2.0 SmolTalk constraints subset for Stage 4."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Literal, cast

from datasets import Dataset, load_dataset  # type: ignore[import-untyped]

from forgellm.post_training.schema import (
    InstructionDataError,
    InstructionRecord,
    Message,
    dataset_fingerprint,
    write_instruction_jsonl,
)

DATASET_ID = "HuggingFaceTB/smoltalk"
DATASET_CONFIG = "smol-constraints"
DATASET_REVISION = "5feaf2fd3ffca7c237fc38d1861bc30365d48ffa"
MAX_CHARACTERS = 1600


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _convert_row(row: Mapping[str, object], *, split: str, index: int) -> InstructionRecord:
    raw_messages = row.get("messages")
    if not isinstance(raw_messages, list):
        raise InstructionDataError("SmolTalk row has no messages list")
    messages: list[Message] = []
    for item in raw_messages:
        if not isinstance(item, dict):
            raise InstructionDataError("SmolTalk message must be an object")
        role = item.get("role")
        content = item.get("content")
        if role not in ("system", "user", "assistant") or not isinstance(content, str):
            raise InstructionDataError("SmolTalk message has invalid role/content")
        messages.append(Message(cast(Literal["system", "user", "assistant"], role), content))
    total_characters = sum(len(message.content) for message in messages)
    if total_characters > MAX_CHARACTERS:
        raise InstructionDataError("conversation exceeds Stage 4 character cap")
    canonical = json.dumps(
        [message.as_dict() for message in messages],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    content_hash = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return InstructionRecord(
        record_id=f"smol-constraints-{split}-{content_hash[:16]}",
        messages=tuple(messages),
        source=f"{DATASET_ID}/{DATASET_CONFIG}@{DATASET_REVISION}",
        license="apache-2.0",
        metadata={"upstream_index": index, "upstream_split": split},
    )


def _accepted_rows(dataset: Dataset, *, split: str) -> tuple[list[InstructionRecord], Counter[str]]:
    accepted: dict[str, InstructionRecord] = {}
    rejected: Counter[str] = Counter()
    for index, row in enumerate(dataset):
        try:
            record = _convert_row(cast(Mapping[str, object], row), split=split, index=index)
        except InstructionDataError as error:
            rejected[str(error)] += 1
            continue
        accepted.setdefault(record.content_fingerprint(), record)
    return [accepted[key] for key in sorted(accepted)], rejected


def main(argv: Sequence[str] | None = None) -> int:
    """Download through Datasets cache, validate, hash-sort and freeze three splits."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    output_dir: Path = args.output_dir
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists():
        raise FileExistsError(f"Stage 4 public data already exists: {manifest_path}")
    upstream_train = cast(
        Dataset,
        load_dataset(
            DATASET_ID,
            DATASET_CONFIG,
            split="train",
            revision=DATASET_REVISION,
        ),
    )
    upstream_test = cast(
        Dataset,
        load_dataset(
            DATASET_ID,
            DATASET_CONFIG,
            split="test",
            revision=DATASET_REVISION,
        ),
    )
    train_candidates, train_rejected = _accepted_rows(upstream_train, split="train")
    test_candidates, test_rejected = _accepted_rows(upstream_test, split="test")
    if len(train_candidates) < 2304 or len(test_candidates) < 256:
        raise RuntimeError("not enough valid SmolTalk records after frozen filtering")
    splits = {
        "train": tuple(train_candidates[:2048]),
        "validation": tuple(train_candidates[2048:2304]),
        "test": tuple(test_candidates[:256]),
    }
    fingerprints: set[str] = set()
    split_reports: dict[str, object] = {}
    for name, records in splits.items():
        contents = {record.content_fingerprint() for record in records}
        if fingerprints & contents:
            raise RuntimeError("content hash leakage across Stage 4 splits")
        fingerprints.update(contents)
        path = output_dir / f"{name}.jsonl"
        write_instruction_jsonl(path, records)
        split_reports[name] = {
            "records": len(records),
            "path": str(path),
            "sha256": _sha256(path),
            "dataset_fingerprint": dataset_fingerprint(records),
        }
    manifest = {
        "schema_version": "forgellm-stage4-public-data-manifest-v1",
        "dataset_id": DATASET_ID,
        "dataset_config": DATASET_CONFIG,
        "dataset_revision": DATASET_REVISION,
        "license": "apache-2.0 for the SmolTalk new smol-constraints subset",
        "selection": {
            "maximum_characters": MAX_CHARACTERS,
            "policy": "validate, exact-deduplicate, sort by content SHA-256, take fixed prefix",
        },
        "upstream_rows": {"train": len(upstream_train), "test": len(upstream_test)},
        "rejected": {"train": dict(train_rejected), "test": dict(test_rejected)},
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
