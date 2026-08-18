from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from forgemm.trainers.chart_fgrpo import compose_chart_fgrpo_advantage  # noqa: E402
from forgemm.trainers.dual_state import DualConfig, DualState  # noqa: E402

RewardBatch = tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.int_],
    NDArray[np.bool_],
    NDArray[np.bool_],
]


def _required_float(value: float | int | None, name: str) -> float:
    if value is None:
        raise ValueError(f"missing_numeric_update:{name}")
    return float(value)


def _make_batches(seed: int, steps: int, groups: int, group_size: int) -> list[RewardBatch]:
    rng = np.random.default_rng(seed)
    group_ids = np.repeat(np.arange(groups), group_size)
    batches: list[RewardBatch] = []
    for step in range(steps):
        progress = step / max(steps - 1, 1)
        latent = rng.normal(0.0, 0.9, size=len(group_ids))
        task = np.clip(0.62 + 0.10 * progress + 0.16 * latent, 0.0, 1.0)
        evidence = np.clip(0.70 + 0.17 * progress + 0.18 * latent, 0.0, 1.0)
        operation = np.clip(0.76 + 0.16 * progress + 0.14 * latent, 0.0, 1.0)
        evidence_mask = rng.random(len(group_ids)) > 0.08
        operation_mask = rng.random(len(group_ids)) > 0.12
        batches.append(
            (
                task.astype(np.float64),
                evidence.astype(np.float64),
                operation.astype(np.float64),
                group_ids.astype(np.int_),
                evidence_mask.astype(np.bool_),
                operation_mask.astype(np.bool_),
            )
        )
    return batches


def _run_method(
    method: str,
    batches: list[RewardBatch],
    state: DualState | None = None,
) -> tuple[dict[str, Any], DualState | None]:
    advantage_stds = []
    nonzero_rates = []
    lambda_trace = []
    constraint_means = []
    for task, evidence, operation, group_ids, evidence_mask, operation_mask in batches:
        if method == "e3_task_only":
            lambda_evidence = 0.0
            lambda_operation = 0.0
        elif method == "e4_fixed_multi_reward":
            lambda_evidence = 1.0
            lambda_operation = 1.0
        elif method == "e5_dynamic_chart_fgrpo" and state is not None:
            lambda_evidence = state.lambda_evidence
            lambda_operation = state.lambda_operation
        else:
            raise ValueError(f"unsupported_method_or_state:{method}")
        composed = compose_chart_fgrpo_advantage(
            task,
            evidence,
            operation,
            group_ids,
            evidence_mask,
            operation_mask,
            lambda_evidence=lambda_evidence,
            lambda_operation=lambda_operation,
        )
        total = composed["combined"]
        advantage_stds.append(float(total.std()))
        nonzero_rates.append(float(np.mean(np.abs(total) > 1e-12)))
        if method == "e5_dynamic_chart_fgrpo" and state is not None:
            update = state.update(evidence, operation, evidence_mask, operation_mask)
            lambda_trace.append(
                [
                    _required_float(update["lambda_evidence"], "lambda_evidence"),
                    _required_float(update["lambda_operation"], "lambda_operation"),
                ]
            )
            constraint_means.append(
                [
                    _required_float(update["evidence_mean"], "evidence_mean"),
                    _required_float(update["operation_mean"], "operation_mean"),
                ]
            )
    result: dict[str, Any] = {
        "steps": len(batches),
        "mean_advantage_std": float(np.mean(advantage_stds)),
        "mean_nonzero_advantage_rate": float(np.mean(nonzero_rates)),
        "all_advantages_finite": bool(np.isfinite(advantage_stds).all()),
    }
    if state is not None:
        result.update(
            {
                "final_lambda_evidence": state.lambda_evidence,
                "final_lambda_operation": state.lambda_operation,
                "max_lambda_evidence": max(item[0] for item in lambda_trace),
                "max_lambda_operation": max(item[1] for item in lambda_trace),
                "first_constraint_means": constraint_means[0],
                "last_constraint_means": constraint_means[-1],
                "lambda_trace_sha256": hashlib.sha256(
                    json.dumps(lambda_trace, separators=(",", ":")).encode()
                ).hexdigest(),
            }
        )
    return result, state


def run_benchmark(output: Path, steps: int, groups: int, group_size: int) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing_to_overwrite:{output}")
    started = time.perf_counter()
    seeds: dict[str, Any] = {}
    for seed in (17, 42, 2026):
        batches = _make_batches(seed, steps, groups, group_size)
        e3, _ = _run_method("e3_task_only", batches)
        e4, _ = _run_method("e4_fixed_multi_reward", batches)
        full_state = DualState(DualConfig())
        e5, final_full_state = _run_method("e5_dynamic_chart_fgrpo", batches, full_state)

        midpoint = len(batches) // 2
        resumed_state = DualState(DualConfig())
        _run_method("e5_dynamic_chart_fgrpo", batches[:midpoint], resumed_state)
        with tempfile.TemporaryDirectory(prefix="forgemm-dual-") as temp_dir:
            checkpoint = Path(temp_dir) / "dual_state.json"
            resumed_state.save(checkpoint)
            resumed_state = DualState.load(checkpoint)
        resumed, final_resumed_state = _run_method(
            "e5_dynamic_chart_fgrpo", batches[midpoint:], resumed_state
        )
        exact_resume = bool(
            final_full_state is not None
            and final_resumed_state is not None
            and final_full_state.to_dict() == final_resumed_state.to_dict()
            and e5["lambda_trace_sha256"] != resumed.get("lambda_trace_sha256")
        )
        seeds[str(seed)] = {
            "e3_task_only": e3,
            "e4_fixed_multi_reward": e4,
            "e5_dynamic_chart_fgrpo": e5,
            "checkpoint_resume_final_state_exact": exact_resume,
        }

    report = {
        "schema_version": 1,
        "experiment_scope": "cpu_fixed_tensor_algorithm_diagnostic_not_model_training",
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "numpy": np.__version__,
        },
        "configuration": {
            "steps": steps,
            "groups_per_step": groups,
            "group_size": group_size,
            "dual": DualState(DualConfig()).to_dict()["config"],
        },
        "seeds": seeds,
        "all_checks_passed": all(
            row["checkpoint_resume_final_state_exact"]
            and row["e3_task_only"]["all_advantages_finite"]
            and row["e4_fixed_multi_reward"]["all_advantages_finite"]
            and row["e5_dynamic_chart_fgrpo"]["all_advantages_finite"]
            for row in seeds.values()
        ),
        "wall_seconds": time.perf_counter() - started,
        "claim_boundary": (
            "verifies advantage composition, constraint response and exact dual-state resume "
            "on deterministic CPU tensors; no model-quality claim is permitted"
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark Chart-FGRPO math on CPU tensors")
    parser.add_argument("--output", required=True)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--groups", type=int, default=32)
    parser.add_argument("--group-size", type=int, default=4)
    arguments = parser.parse_args()
    if min(arguments.steps, arguments.groups, arguments.group_size) < 1:
        parser.error("steps, groups and group-size must be positive")
    report = run_benchmark(
        Path(arguments.output), arguments.steps, arguments.groups, arguments.group_size
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["all_checks_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
