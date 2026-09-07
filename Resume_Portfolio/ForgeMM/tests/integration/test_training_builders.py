import json

from forgemm.data.schemas import EvidenceCell, EvidenceRecord, Operation, OperationArgument
from forgemm.data.training_builders import (
    build_answer_sft_rows,
    build_grpo_rows,
    build_sft_rows,
    build_strict_eval_rows,
    render_structured_answer,
)
from forgemm.reasoning.parser import parse_prediction


def _record() -> EvidenceRecord:
    return EvidenceRecord(
        record_id="train:human:0",
        dataset="ChartQA",
        split="train",
        source="human",
        image_path="chart.png",
        image_sha256="abc",
        question="What is the difference?",
        reference_answer="5",
        cells=(
            EvidenceCell("a", "2015", "Value", "38"),
            EvidenceCell("b", "2016", "Value", "43"),
        ),
        gold_evidence_ids=("a", "b"),
        gold_operation=Operation(
            "difference",
            (OperationArgument("ref", "b", True), OperationArgument("ref", "a", True)),
        ),
        evidence_mask=True,
        operation_mask=True,
        conflict_status="exact",
    )


def test_sft_builder_creates_two_traceable_views_with_parseable_target() -> None:
    rows = build_sft_rows([_record()])
    assert len(rows) == 2
    assert {row["group_key"] for row in rows} == {"train:human:0"}
    assert len({row["sample_id"] for row in rows}) == 2
    prediction = parse_prediction(rows[0]["messages"][1]["content"])
    assert prediction.answer == "5"


def test_grpo_builder_keeps_reward_columns_outside_messages() -> None:
    row = build_grpo_rows([_record()])[0]
    assert len(row["messages"]) == 1
    assert row["reference_answer"] == "5"
    assert len(json.loads(row["gold_evidence"])) == 2
    assert row["evidence_mask"] and row["operation_mask"]


def test_answer_control_and_strict_eval_use_one_frozen_record_population() -> None:
    answer_rows = build_answer_sft_rows([_record()])
    strict_rows = build_strict_eval_rows([_record()])

    assert len(answer_rows) == 2
    assert {row["group_key"] for row in answer_rows} == {"train:human:0"}
    assert {row["messages"][-1]["content"] for row in answer_rows} == {"5"}
    assert len(strict_rows) == 1
    assert strict_rows[0]["sample_id"] == "train:human:0"
    assert parse_prediction(strict_rows[0]["messages"][-1]["content"]).answer == "5"


def test_structured_render_preserves_operation_reference_order() -> None:
    rendered = render_structured_answer(_record())
    assert "difference(ref=e2, ref=e1)" in rendered
