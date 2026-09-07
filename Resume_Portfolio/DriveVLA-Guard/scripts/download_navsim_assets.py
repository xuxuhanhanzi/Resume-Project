#!/usr/bin/env python3
"""Download and safely extract the pinned public NAVSIM test assets."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tarfile
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import snapshot_download

OPENSCENE_REPO = "OpenDriveLab/OpenScene"
OPENSCENE_REVISION = "a76f840b65e972bc45e56c2adced897498e9a026"
MAP_URL = "https://motional-nuplan.s3-ap-northeast-1.amazonaws.com/public/nuplan-v1.1/nuplan-maps-v1.1.zip"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_destination(root: Path, member: str) -> Path:
    destination = (root / member).resolve()
    if root.resolve() not in destination.parents and destination != root.resolve():
        raise ValueError(f"archive member escapes destination: {member}")
    return destination


def extract_tar(path: Path, destination: Path) -> None:
    marker = destination / ".extracted" / f"{path.name}.json"
    if marker.exists():
        return
    with tarfile.open(path) as archive:
        members = archive.getmembers()
        for member in members:
            safe_destination(destination, member.name)
            if member.issym() or member.islnk():
                raise ValueError(f"links are not accepted in downloaded archive: {member.name}")
            if not member.isdir() and not member.isfile():
                raise ValueError(f"special archive member is not accepted: {member.name}")
        archive.extractall(destination, members=members)
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(json.dumps({"archive": str(path), "sha256": sha256(path)}, indent=2) + "\n")


def extract_zip(path: Path, destination: Path) -> None:
    marker = destination / ".extracted" / f"{path.name}.json"
    if marker.exists():
        return
    with zipfile.ZipFile(path) as archive:
        for member in archive.infolist():
            safe_destination(destination, member.filename)
        archive.extractall(destination)
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(json.dumps({"archive": str(path), "sha256": sha256(path)}, indent=2) + "\n")


def link_once(link: Path, target: Path) -> None:
    if not target.is_dir():
        raise FileNotFoundError(f"expected extracted directory: {target}")
    if link.exists() or link.is_symlink():
        if link.resolve() != target.resolve():
            raise FileExistsError(f"refusing to replace existing path: {link}")
        return
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(target, target_is_directory=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.asset_root.expanduser().resolve()
    archives = root / "downloads"
    data = root / "dataset" / "nuplan"
    archives.mkdir(parents=True, exist_ok=True)
    data.mkdir(parents=True, exist_ok=True)

    snapshot = Path(
        snapshot_download(
            repo_id=OPENSCENE_REPO,
            repo_type="dataset",
            revision=OPENSCENE_REVISION,
            allow_patterns=[
                "openscene-v1.1/openscene_metadata_test.tgz",
                "openscene-v1.1/openscene_sensor_test_camera/*.tgz",
            ],
            local_dir=archives / "OpenScene",
        )
    )
    open_scene_archives = sorted(snapshot.rglob("*.tgz"))
    if len(open_scene_archives) < 33:
        raise RuntimeError(f"expected metadata plus 32 sensor archives, found {len(open_scene_archives)}")
    for archive in open_scene_archives:
        print(f"extract {archive.name}", flush=True)
        extract_tar(archive, data)

    map_archive = archives / "nuplan-maps-v1.1.zip"
    if not map_archive.exists():
        print(f"download {MAP_URL}", flush=True)
        with urllib.request.urlopen(MAP_URL) as response, map_archive.open("wb") as output:
            shutil.copyfileobj(response, output, length=8 * 1024 * 1024)
    extract_zip(map_archive, data)

    link_once(data / "test_navsim_logs", data / "openscene-v1.1" / "meta_datas")
    link_once(data / "navsim_logs" / "test", data / "openscene-v1.1" / "meta_datas")
    link_once(data / "sensor_blobs" / "test", data / "openscene-v1.1" / "sensor_blobs")
    link_once(data / "maps", data / "nuplan-maps-v1.0")
    receipt = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": f"https://huggingface.co/datasets/{OPENSCENE_REPO}",
        "revision": OPENSCENE_REVISION,
        "map_url": MAP_URL,
        "archive_count": len(open_scene_archives) + 1,
        "layout": {
            "metadata": str((data / "test_navsim_logs").resolve()),
            "navsim_metadata": str((data / "navsim_logs" / "test").resolve()),
            "sensor": str((data / "sensor_blobs" / "test").resolve()),
            "maps": str((data / "maps").resolve()),
        },
    }
    receipt_path = root / "download-receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(receipt_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
