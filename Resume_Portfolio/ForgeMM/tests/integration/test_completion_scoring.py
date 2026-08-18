from forgemm.data.schemas import EvidenceCell
from forgemm.rewards.scoring import score_completion


def test_completion_to_three_rewards() -> None:
    completion = """<evidence>
e1=cell(row="2015", column="Value", value=38)
e2=cell(row="2016", column="Value", value=43)
</evidence>
<operation>
difference(ref=e2, ref=e1)
</operation>
<answer>
5
</answer>"""
    gold = (
        EvidenceCell("gold-a", "2015", "Value", "38.0"),
        EvidenceCell("gold-b", "2016", "Value", "43"),
    )

    reward = score_completion(completion, "5", gold)

    assert reward.task == 1.0
    assert reward.evidence == 1.0
    assert reward.operation == 1.0
    assert reward.parse_error is None


def test_parse_failure_zeros_all_rewards() -> None:
    reward = score_completion("unstructured answer", "5", ())

    assert (reward.task, reward.evidence, reward.operation) == (0.0, 0.0, 0.0)
    assert reward.parse_error == "invalid_protocol"


def test_masks_zero_inapplicable_constraint_channels() -> None:
    completion = """<evidence>
e1=cell(row="2015", column="Value", value=5)
</evidence>
<operation>
lookup(ref=e1)
</operation>
<answer>
5
</answer>"""

    reward = score_completion(completion, "5", (), evidence_mask=False, operation_mask=False)

    assert reward.task == 1.0
    assert reward.evidence == 0.0
    assert reward.operation == 0.0
