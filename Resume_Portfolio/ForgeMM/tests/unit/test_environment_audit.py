from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType


def _load_script() -> ModuleType:
    path = Path(__file__).resolve().parents[2] / "scripts" / "audit_environment.py"
    spec = importlib.util.spec_from_file_location("audit_environment", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_environment_audit_verifies_frozen_dataset_identities() -> None:
    project = Path(__file__).resolve().parents[2]
    module = _load_script()
    report = module.build_report(project)

    assert report["schema_version"] == "forgemm-environment-audit-v1"
    assert report["gates"]["python_supported"]
    status = report["gates"]["dataset_identity_status"]
    assert status in {"verified", "unavailable"}
    if report["gates"]["dataset_payload_present"]:
        assert report["gates"]["dataset_identity_passed"]
        assert report["datasets"]["chartqa"]["passed"]
        assert report["datasets"]["chartqapro"]["passed"]
    else:
        assert not report["gates"]["dataset_identity_passed"]
    assert report["ms_swift_candidate"]["version"] == "4.2.2"
