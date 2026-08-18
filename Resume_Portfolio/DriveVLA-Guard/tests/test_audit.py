import json
import zipfile
from pathlib import Path

from drivevla_guard import audit


def test_audit_autovla_accepts_verified_codeload_archive(tmp_path) -> None:
    commit = "a" * 40
    root = tmp_path / "AutoVLA"
    required = [
        "LICENSE",
        "README.md",
        "models/autovla.py",
        "models/action_tokenizer.py",
        "codebook_cache/agent_vocab.pkl",
        "navsim/navsim/agents/autovla_agent.py",
    ]
    for name in required:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixture\n", encoding="utf-8")
    archive = tmp_path / "AutoVLA.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr(f"AutoVLA-{commit}/README.md", "fixture\n")
    manifest = {
        "schema_version": "autovla-codeload-v1",
        "upstream": "https://github.com/ucla-mobility/AutoVLA",
        "commit": commit,
        "archive": "../AutoVLA.zip",
        "archive_sha256": audit.sha256_file(archive),
    }
    (root / ".autovla-source.json").write_text(json.dumps(manifest), encoding="utf-8")

    result = audit.audit_autovla(root)

    assert result["ready"]
    assert result["commit"] == commit
    assert result["source_kind"] == "github_codeload_archive"
    assert result["source_archive"]["sha256"] == manifest["archive_sha256"]


def _make_paths(root: Path) -> dict[str, str]:
    directories = ["autovla_root", "qwen_model", "sensor_data", "metric_cache"]
    files = ["checkpoint", "model_config", "json_data"]
    paths: dict[str, str] = {}
    for name in directories:
        path = root / name
        path.mkdir()
        paths[name] = str(path)
    for name in files:
        path = root / name
        path.write_text("fixture\n", encoding="utf-8")
        paths[name] = str(path)
    return paths


def test_official_preflight_requires_gpu_capacity_and_commit(tmp_path, monkeypatch) -> None:
    paths = _make_paths(tmp_path)
    monkeypatch.setattr(
        audit,
        "_module_snapshot",
        lambda names: {name: {"available": True, "version": "test"} for name in names},
    )
    monkeypatch.setattr(
        audit,
        "_gpu_snapshot",
        lambda: {
            "available": True,
            "devices": [{"name": "test", "memory_total_mib": 48 * 1024, "driver_version": "test"}],
            "raw": "test, 49152 MiB, test",
        },
    )
    monkeypatch.setattr(
        audit,
        "audit_autovla",
        lambda root: {"ready": True, "commit": "expected", "root": str(root)},
    )

    result = audit.official_preflight(
        paths,
        min_vram_gb=24,
        expected_upstream_commit="expected",
    )

    assert result["ready"]
    assert result["gpu"]["capacity_ok"]
    assert not result["blockers"]


def test_official_preflight_rejects_wrong_path_kind_and_low_vram(tmp_path, monkeypatch) -> None:
    paths = _make_paths(tmp_path)
    paths["checkpoint"] = paths["qwen_model"]
    monkeypatch.setattr(
        audit,
        "_module_snapshot",
        lambda names: {name: {"available": True, "version": "test"} for name in names},
    )
    monkeypatch.setattr(
        audit,
        "_gpu_snapshot",
        lambda: {
            "available": True,
            "devices": [{"name": "test", "memory_total_mib": 8192.0, "driver_version": "test"}],
            "raw": "test, 8192 MiB, test",
        },
    )
    monkeypatch.setattr(
        audit,
        "audit_autovla",
        lambda root: {"ready": True, "commit": "wrong", "root": str(root)},
    )

    result = audit.official_preflight(paths, expected_upstream_commit="expected")

    assert not result["ready"]
    assert "invalid_path:checkpoint" in result["blockers"]
    assert any(item.startswith("insufficient_vram:") for item in result["blockers"])
    assert any(item.startswith("upstream_commit_mismatch:") for item in result["blockers"])
