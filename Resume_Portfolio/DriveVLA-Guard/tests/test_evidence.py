from drivevla_guard.evidence import evaluate_frozen_evidence, verify_evidence


def test_frozen_synthetic_evidence_is_semantically_valid(project_root) -> None:
    report = evaluate_frozen_evidence(project_root)
    assert report["ready"], report["errors"]
    assert report["checks"]["manifest"]["scenes"] == 12
    assert all(
        report["checks"][variant]["summary_recomputed_ok"] for variant in ("b0", "b1", "e1", "e2", "e3", "e4")
    )


def test_frozen_synthetic_hash_snapshot(project_root) -> None:
    snapshot = project_root / "artifacts/evidence_manifest.json"
    assert snapshot.is_file()
    report = verify_evidence(project_root, snapshot)
    assert report["ready"], report["errors"]
