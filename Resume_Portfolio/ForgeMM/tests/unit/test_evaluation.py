import numpy as np
import pytest

from forgemm.data.schemas import EvidenceCell
from forgemm.evaluation.metrics import evaluate_completions
from forgemm.evaluation.statistics import paired_bootstrap_delta, paired_mcnemar

VALID = """<evidence>
e1=cell(row="2015", column="Value", value=5)
</evidence>
<operation>
lookup(ref=e1)
</operation>
<answer>
5
</answer>"""


def test_faithful_metrics_separate_correct_from_consistent() -> None:
    gold: list[tuple[EvidenceCell, ...]] = [(EvidenceCell("gold", "2015", "Value", "5"),)] * 2
    result = evaluate_completions(
        [VALID, "invalid"],
        ["5", "5"],
        gold,
        [True, True],
        [True, True],
        [False, True],
    )
    assert result.samples == 2
    assert result.format_compliance == 0.5
    assert result.faithful_correct_rate == 0.5
    assert result.truncation_rate == 0.5


def test_paired_bootstrap_is_deterministic() -> None:
    baseline = np.array([0.0, 0.0, 1.0, 1.0])
    candidate = np.array([1.0, 0.0, 1.0, 1.0])
    first = paired_bootstrap_delta(baseline, candidate, samples=1000, seed=42)
    second = paired_bootstrap_delta(baseline, candidate, samples=1000, seed=42)
    assert first == second
    assert first["mean_delta"] == pytest.approx(0.25)


def test_exact_mcnemar_counts_direction() -> None:
    result = paired_mcnemar(
        [True, False, False, True],
        [False, True, True, True],
    )
    assert result.baseline_only_correct == 1
    assert result.candidate_only_correct == 2
    assert 0 <= result.exact_p_value <= 1
