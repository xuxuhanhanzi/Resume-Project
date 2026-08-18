from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else ".").resolve()
    manifest_path = root / "CLOUD_BUNDLE_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    failures = []
    for item in manifest["files"]:
        path = root / item["path"]
        if not path.is_file():
            failures.append({"path": item["path"], "error": "missing"})
            continue
        if path.stat().st_size != item["bytes"]:
            failures.append({"path": item["path"], "error": "size"})
            continue
        if sha256(path) != item["sha256"]:
            failures.append({"path": item["path"], "error": "sha256"})
    print(
        json.dumps(
            {
                "project": manifest["project"],
                "version": manifest["version"],
                "files": len(manifest["files"]),
                "failures": failures,
                "passed": not failures,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
