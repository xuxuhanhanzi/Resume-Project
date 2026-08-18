from __future__ import annotations

import hashlib
import json
import subprocess
import zipfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

PORTFOLIO = Path(__file__).resolve().parents[1]
OUTPUT = Path(__file__).resolve().parent
STORE_SUFFIXES = {".png", ".jpg", ".jpeg", ".parquet", ".zip", ".gz", ".pkl"}


@dataclass(frozen=True)
class Bundle:
    project: str
    version: str
    source: Path
    readme: Path
    zip_name: str
    extra_roots: tuple[Path, ...] = ()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tracked_files(bundle: Bundle) -> list[Path]:
    output = subprocess.check_output(
        ["git", "ls-tree", "-r", "--name-only", bundle.version],
        cwd=bundle.source,
        text=True,
        encoding="utf-8",
    )
    return [
        bundle.source / line
        for line in output.splitlines()
        if line and (bundle.source / line).is_file()
    ]


def extra_files(bundle: Bundle) -> list[Path]:
    files = []
    for root in bundle.extra_roots:
        if not root.is_dir():
            raise FileNotFoundError(root)
        files.extend(path for path in root.rglob("*") if path.is_file())
    return files


def relative_path(bundle: Bundle, path: Path) -> str:
    return path.relative_to(bundle.source).as_posix()


def build(bundle: Bundle) -> dict[str, object]:
    target = OUTPUT / bundle.zip_name
    checksum_path = target.with_suffix(target.suffix + ".sha256")
    if target.exists() or checksum_path.exists():
        raise FileExistsError(f"refusing to overwrite existing package: {target}")

    files_by_relative = {
        relative_path(bundle, path): path
        for path in [*tracked_files(bundle), *extra_files(bundle)]
    }
    files_by_relative["CLOUD_README.md"] = bundle.readme
    files_by_relative["scripts/verify_cloud_bundle.py"] = (
        OUTPUT / "verify_cloud_bundle.py"
    )

    records = []
    for relative, path in sorted(files_by_relative.items()):
        records.append(
            {"path": relative, "bytes": path.stat().st_size, "sha256": sha256(path)}
        )
    manifest = {
        "schema_version": 1,
        "project": bundle.project,
        "version": bundle.version,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "source_commit": subprocess.check_output(
            ["git", "rev-list", "-n", "1", bundle.version],
            cwd=bundle.source,
            text=True,
            encoding="utf-8",
        ).strip(),
        "files": records,
        "total_uncompressed_bytes": sum(item["bytes"] for item in records),
    }
    manifest_bytes = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )

    with zipfile.ZipFile(target, "x", allowZip64=True) as archive:
        prefix = f"{bundle.project}/"
        for relative, path in sorted(files_by_relative.items()):
            compression = (
                zipfile.ZIP_STORED
                if path.suffix.lower() in STORE_SUFFIXES
                else zipfile.ZIP_DEFLATED
            )
            archive.write(
                path, prefix + relative, compress_type=compression, compresslevel=6
            )
        archive.writestr(
            prefix + "CLOUD_BUNDLE_MANIFEST.json",
            manifest_bytes,
            compress_type=zipfile.ZIP_DEFLATED,
            compresslevel=6,
        )

    bad_member = None
    with zipfile.ZipFile(target, "r", allowZip64=True) as archive:
        bad_member = archive.testzip()
        names = set(archive.namelist())
        if prefix + "CLOUD_BUNDLE_MANIFEST.json" not in names:
            raise RuntimeError("bundle manifest missing from zip")
    if bad_member is not None:
        raise RuntimeError(f"zip CRC verification failed: {bad_member}")

    digest = sha256(target)
    checksum_path.write_text(
        f"{digest}  {target.name}\n", encoding="utf-8", newline="\n"
    )
    return {
        "zip": str(target),
        "sha256_file": str(checksum_path),
        "sha256": digest,
        "compressed_bytes": target.stat().st_size,
        "uncompressed_bytes": manifest["total_uncompressed_bytes"],
        "payload_files": len(records),
        "zip_crc_passed": True,
    }


def main() -> int:
    bundles = (
        Bundle(
            project="DriveVLA-Guard",
            version="v1.0.2",
            source=PORTFOLIO / "DriveVLA-Guard",
            readme=OUTPUT / "DRIVEVLA_CLOUD_README.md",
            zip_name="DriveVLA-Guard_cloud_v1.0.2.zip",
        ),
        Bundle(
            project="ForgeMM",
            version="v0.2.3",
            source=PORTFOLIO / "ForgeMM",
            readme=OUTPUT / "FORGEMM_CLOUD_README.md",
            zip_name="ForgeMM_cloud_v0.2.3_with_data.zip",
            extra_roots=(
                PORTFOLIO / "ForgeMM" / "datasets" / "ChartQA",
                PORTFOLIO / "ForgeMM" / "datasets" / "ChartQAPro",
                PORTFOLIO
                / "ForgeMM"
                / "artifacts"
                / "runs"
                / "stage03_evidence_store_v7",
                PORTFOLIO
                / "ForgeMM"
                / "artifacts"
                / "runs"
                / "stage04_training_data_v2",
            ),
        ),
    )
    reports = [build(bundle) for bundle in bundles]
    print(json.dumps(reports, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
