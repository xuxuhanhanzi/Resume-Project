import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from forgemm.controlled.dataset import make_controlled_record
from forgemm.controlled.protocol import render_controlled_completion
from forgemm.swift_plugin import (
    ForgeMMControlledEvidenceReward,
    ForgeMMControlledOperationReward,
    ForgeMMControlledTaskReward,
    ForgeMMEvidenceReward,
    ForgeMMOperationReward,
    ForgeMMTaskReward,
    _install_trl_transformers_451_compat,
    orms,
)

COMPLETION = """<evidence>
e1=cell(row="2015", column="Value", value=38)
e2=cell(row="2016", column="Value", value=43)
</evidence>
<operation>
difference(ref=e2, ref=e1)
</operation>
<answer>
5
</answer>"""
GOLD = json.dumps(
    [
        {"evidence_id": "a", "row": "2015", "column": "Value", "value": "38"},
        {"evidence_id": "b", "row": "2016", "column": "Value", "value": "43"},
    ]
)


def test_swift_reward_classes_consume_extra_dataset_columns() -> None:
    completions = [COMPLETION, "invalid"]
    answers = ["5", "5"]
    evidence = [GOLD, GOLD]
    assert ForgeMMTaskReward()(completions, answers, evidence) == [1.0, 0.0]
    assert ForgeMMEvidenceReward()(completions, answers, evidence) == [1.0, 0.0]
    assert ForgeMMOperationReward()(completions, answers, evidence) == [1.0, 0.0]


def test_swift_reward_names_are_registered() -> None:
    assert {
        "forgemm_task",
        "forgemm_evidence",
        "forgemm_operation",
        "forgemm_controlled_task",
        "forgemm_controlled_evidence",
        "forgemm_controlled_operation",
    } <= set(orms)


def test_controlled_rewards_require_generator_backed_visual_bbox(tmp_path: Path) -> None:
    project = tmp_path / "project"
    dataset = project / "datasets" / "controlled"
    dataset.mkdir(parents=True)
    record = make_controlled_record(
        split="train",
        index=0,
        root_seed=19,
        dataset_root=dataset,
        project_root=project,
    )
    gold = json.dumps(record.to_dict())
    completion = render_controlled_completion(record)
    box = next(
        item.bbox for item in record.evidence if item.source_id == record.gold_evidence_ids[0]
    )
    wrong_box = completion.replace(
        f"bbox=[{','.join(str(value) for value in box)}]", "bbox=[0,0,1,1]", 1
    )

    assert ForgeMMControlledTaskReward()([completion], gold) == [1.0]
    assert ForgeMMControlledEvidenceReward()([completion], gold) == [1.0]
    assert ForgeMMControlledOperationReward()([completion], gold) == [1.0]
    assert ForgeMMControlledEvidenceReward()([wrong_box], gold) == [0.0]


def test_transformers_451_compat_exposes_trl_optional_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeTrainer:
        args: Any
        current_gradient_accumulation_steps: int

    fake_transformers = SimpleNamespace(Trainer=FakeTrainer)
    monkeypatch.setenv("FORGEMM_TRL_TRANSFORMERS_451", "1")
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    _install_trl_transformers_451_compat()

    assert fake_transformers.is_trackio_available() is False
    trainer = FakeTrainer()
    trainer.args = SimpleNamespace(gradient_accumulation_steps=3)
    assert trainer.current_gradient_accumulation_steps == 3
