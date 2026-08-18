"""Fetch the frozen 19.4 MB TinyStories teaching-corpus source."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import urllib.request
import uuid
from collections.abc import Sequence
from pathlib import Path

SOURCE_URL = (
    "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/"
    "TinyStories-valid.txt?download=true"
)
EXPECTED_SHA256 = "94e431816c4cce81ff71e4408ff8d3bda9a42e8d2663986697c3954288cb38b4"
MAXIMUM_BYTES = 25 * 1024 * 1024


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch(output: Path) -> Path:
    """Download to a temporary file, verify bytes, then atomically publish it."""
    received = 0
    if output.exists():
        if sha256_file(output) != EXPECTED_SHA256:
            raise RuntimeError(f"existing corpus has an unexpected SHA-256: {output}")
        received = output.stat().st_size
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_name(f".{output.name}.{uuid.uuid4().hex}.tmp")
        request = urllib.request.Request(SOURCE_URL, headers={"User-Agent": "ForgeLLM-Stage3/1.0"})
        digest = hashlib.sha256()
        try:
            with (
                urllib.request.urlopen(request, timeout=60) as response,
                temporary.open("xb") as target,
            ):
                while chunk := response.read(1024 * 1024):
                    received += len(chunk)
                    if received > MAXIMUM_BYTES:
                        raise RuntimeError("download exceeds the documented 25 MiB safety cap")
                    digest.update(chunk)
                    target.write(chunk)
                target.flush()
                os.fsync(target.fileno())
            if digest.hexdigest() != EXPECTED_SHA256:
                raise RuntimeError(
                    f"download SHA-256 mismatch: expected {EXPECTED_SHA256}, "
                    f"got {digest.hexdigest()}"
                )
            os.replace(temporary, output)
        except Exception:
            if temporary.is_file():
                temporary.unlink()
            raise
    manifest = {
        "bytes": received,
        "license": "CDLA-Sharing-1.0",
        "name": "TinyStories official validation file used as a bounded teaching subset",
        "sha256": EXPECTED_SHA256,
        "source_url": SOURCE_URL,
        "warning": (
            "This upstream validation file is re-split locally and is not used for "
            "benchmark claims."
        ),
    }
    output.with_suffix(".source.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/raw/tinystories_valid.txt"),
    )
    args = parser.parse_args(argv)
    print(fetch(args.output).resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
