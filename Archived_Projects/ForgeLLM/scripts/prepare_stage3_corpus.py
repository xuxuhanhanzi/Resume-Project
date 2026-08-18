"""Convert the frozen TinyStories text into document-split audited JSONL."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Sequence
from pathlib import Path

from forgellm.data.config import load_data_config
from forgellm.data.pipeline import run_data_pipeline

EXPECTED_SHA256 = "94e431816c4cce81ff71e4408ff8d3bda9a42e8d2663986697c3954288cb38b4"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare(raw_path: Path, source_jsonl: Path, output_dir: Path, config_path: Path) -> Path:
    """Verify upstream bytes, extract stories, then invoke the existing data pipeline."""
    actual_hash = _sha256(raw_path)
    if actual_hash != EXPECTED_SHA256:
        raise RuntimeError(f"raw TinyStories SHA-256 mismatch: {actual_hash}")
    if source_jsonl.exists():
        raise FileExistsError(f"refusing to overwrite source JSONL: {source_jsonl}")
    source_jsonl.parent.mkdir(parents=True, exist_ok=True)
    text = raw_path.read_text(encoding="utf-8")
    stories = [story.strip() for story in text.split("<|endoftext|>") if story.strip()]
    if len(stories) < 1000:
        raise RuntimeError("unexpectedly few TinyStories documents were parsed")
    with source_jsonl.open("x", encoding="utf-8", newline="\n") as stream:
        for index, story in enumerate(stories):
            record = {"id": f"tinystories-stage3-{index:06d}", "text": story}
            stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    result = run_data_pipeline(
        load_data_config(config_path),
        source_jsonl,
        output_dir,
        source_name="TinyStories 19.4 MB bounded teaching subset",
        source_license="CDLA-Sharing-1.0",
    )
    provenance = {
        "claim_boundary": (
            "The upstream validation file is re-split for method education; results are not "
            "TinyStories benchmark validation results."
        ),
        "documents_parsed": len(stories),
        "upstream_file": raw_path.name,
        "upstream_sha256": actual_hash,
        "upstream_url": (
            "https://huggingface.co/datasets/roneneldan/TinyStories/blob/main/TinyStories-valid.txt"
        ),
    }
    (output_dir / "upstream_provenance.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result.manifest_path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=Path("data/raw/tinystories_valid.txt"))
    parser.add_argument(
        "--source-jsonl",
        type=Path,
        default=Path("data/raw/tinystories_stage3_source.jsonl"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/stage3_tinystories"),
    )
    parser.add_argument("--config", type=Path, default=Path("configs/data/stage3_tinystories.toml"))
    args = parser.parse_args(argv)
    print(prepare(args.raw, args.source_jsonl, args.output_dir, args.config).resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
