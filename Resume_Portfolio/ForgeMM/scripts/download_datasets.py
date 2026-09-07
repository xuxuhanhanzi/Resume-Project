#!/usr/bin/env python3
"""Materialize pinned ChartQA and ChartQAPro files into the ignored data roots."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import hf_hub_download

CHARTQA_REVISION = "af8b6f5c08c95085271561c2a3f9d15f2b5a9031"
CHARTQA_PRO_REVISION = "e27c2874825874d6767d2bbc538ed4f0dc2c64c2"
EXPECTED = {
    "train/train_human.json": "4eaaa03e406dbbbe43caff925dbec9930b87fadba960f1c2ab30a5def287c384",
    "train/train_augmented.json": (
        "77342d8527d2a194a6011ce3f4389ea32d9c50cfe60b10c74dbe1cd2e49724da"
    ),
    "val/val_human.json": "e297ce5b38b4ab79bd0e22a93eb19a65cb89f9bd07ce886cb78f2e333fb0ba14",
    "val/val_augmented.json": "cafc675781fba27282b78cab40eb49b77a5bf890ef1e89294bb67c994bdfde47",
    "test/test_human.json": "5fba61b508a96b85cd5906bb14adfc1ba552bdcf88ac568ed6e2f79dcf0460b7",
    "test/test_augmented.json": "9df43a15e93ecb8496c3e5dc23f6a733cfeccac838ab41126ab56576b9305de1",
}
CHARTQA_PRO_SHA256 = "6209a9a6f7307b761e70ff9e708cb7505e0327d7eb932aa26152b8240f633da5"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_extract(archive_path: Path, destination: Path) -> None:
    marker = destination / ".complete"
    if marker.exists():
        return
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.infolist():
            target = (destination / member.filename).resolve()
            if root not in target.parents and target != root:
                raise ValueError(f"archive member escapes destination: {member.filename}")
        archive.extractall(destination)
    marker.write_text(sha256(archive_path) + "\n", encoding="utf-8")


def copy_tree_without_overwrite(source: Path, target: Path) -> None:
    for path in source.rglob("*"):
        relative = path.relative_to(source)
        destination = target / relative
        if path.is_dir():
            destination.mkdir(parents=True, exist_ok=True)
        elif not destination.exists():
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    args = parser.parse_args()
    project = args.project_root.expanduser().resolve()
    cache = args.cache_root.expanduser().resolve()
    cache.mkdir(parents=True, exist_ok=True)

    archive = Path(
        hf_hub_download(
            repo_id="ahmed-masry/ChartQA",
            filename="ChartQA Dataset.zip",
            repo_type="dataset",
            revision=CHARTQA_REVISION,
            local_dir=cache / "downloads" / "ChartQA",
        )
    )
    expanded = cache / "expanded" / "ChartQA"
    safe_extract(archive, expanded)
    candidates = [
        path
        for path in expanded.rglob("train")
        if (path.parent / "val").is_dir() and (path.parent / "test").is_dir()
    ]
    if len(candidates) != 1:
        raise RuntimeError(f"could not identify one ChartQA split root: {candidates}")
    source_root = candidates[0].parent
    chartqa_root = project / "datasets" / "ChartQA"
    for split in ("train", "val", "test"):
        copy_tree_without_overwrite(source_root / split, chartqa_root / split)

    for relative, expected in EXPECTED.items():
        actual = sha256(chartqa_root / relative)
        if actual != expected:
            raise RuntimeError(f"ChartQA hash mismatch for {relative}: {actual}")

    pro_source = Path(
        hf_hub_download(
            repo_id="ahmed-masry/ChartQAPro",
            filename="data/test-00000-of-00001.parquet",
            repo_type="dataset",
            revision=CHARTQA_PRO_REVISION,
            local_dir=cache / "downloads" / "ChartQAPro",
        )
    )
    pro_target = project / "datasets" / "ChartQAPro" / "chartqapro_test.parquet"
    pro_target.parent.mkdir(parents=True, exist_ok=True)
    if not pro_target.exists():
        shutil.copy2(pro_source, pro_target)
    if sha256(pro_target) != CHARTQA_PRO_SHA256:
        raise RuntimeError("ChartQAPro hash mismatch")

    receipt = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "chartqa": {
            "repo": "ahmed-masry/ChartQA",
            "revision": CHARTQA_REVISION,
            "verified": EXPECTED,
        },
        "chartqapro": {
            "repo": "ahmed-masry/ChartQAPro",
            "revision": CHARTQA_PRO_REVISION,
            "sha256": CHARTQA_PRO_SHA256,
        },
    }
    path = cache / "download-receipt.json"
    path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
