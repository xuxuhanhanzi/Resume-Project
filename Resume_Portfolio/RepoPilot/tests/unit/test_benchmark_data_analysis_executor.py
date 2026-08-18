from repopilot.benchmarks.contracts import BenchmarkDomain, BenchmarkTask
from repopilot.benchmarks.executors import DockerDataAnalysisExecutor


def test_extract_python_accepts_fenced_and_plain_code() -> None:
    assert DockerDataAnalysisExecutor.extract_python("```python\nprint(1)\n```") == "print(1)"
    assert DockerDataAnalysisExecutor.extract_python("print(2)") == "print(2)"


def test_diagnostic_hint_classifies_nan_failure() -> None:
    hint = DockerDataAnalysisExecutor.diagnostic_hint("ValueError: Input X contains NaN")
    assert "missing-value" in hint
    assert "LinearRegression" in hint


def test_semantic_policy_rejects_wrong_required_estimator() -> None:
    task = BenchmarkTask(
        "fixture",
        "revision",
        "task-1",
        BenchmarkDomain.DATA_ANALYSIS,
        "Apply the linear regression algorithm.",
        "docker",
    )
    violation = DockerDataAnalysisExecutor.semantic_violation(
        task, "from sklearn.linear_model import LogisticRegression"
    )
    assert "LinearRegression" in violation
