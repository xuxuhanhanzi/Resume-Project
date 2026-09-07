from __future__ import annotations

from pathlib import Path

from forgemm.controlled.dataset import make_controlled_record
from forgemm.controlled.protocol import (
    ControlledRecord,
    render_controlled_completion,
    verify_controlled_completion,
)


def _record(tmp_path: Path) -> ControlledRecord:
    root = tmp_path / "project"
    target = root / "datasets" / "controlled"
    target.mkdir(parents=True)
    return make_controlled_record(
        split="test",
        index=0,
        root_seed=7,
        dataset_root=target,
        project_root=root,
    )


def test_controlled_gold_completion_requires_value_operation_and_visual_bbox(
    tmp_path: Path,
) -> None:
    record = _record(tmp_path)
    completion = render_controlled_completion(record)

    verdict = verify_controlled_completion(record, completion)

    assert verdict.full_pass
    assert verdict.visual_iou_min == 1.0


def test_controlled_verifier_rejects_value_correct_but_wrong_visual_region(tmp_path: Path) -> None:
    record = _record(tmp_path)
    completion = render_controlled_completion(record)
    bbox = next(
        item.bbox for item in record.evidence if item.source_id == record.gold_evidence_ids[0]
    )
    rendered_bbox = ",".join(str(value) for value in bbox)
    wrong_region = completion.replace(f"bbox=[{rendered_bbox}]", "bbox=[0,0,1,1]", 1)

    verdict = verify_controlled_completion(record, wrong_region)

    assert verdict.answer_ok
    assert verdict.evidence_ok is False
    assert verdict.full_pass is False
    assert "VISUAL_REGION_MISMATCH" in verdict.errors
