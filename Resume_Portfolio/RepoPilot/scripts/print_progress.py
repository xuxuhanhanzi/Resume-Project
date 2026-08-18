"""Summarise every frozen benchmark run into one auditable progress table.

Reads `artifacts/benchmarks/<run-id>/results.json` for each registered experiment and
prints a per-domain summary. Domains are always reported separately: RepoPilot never
aggregates a cross-domain score, because the three benchmarks measure different things
and their primary metrics are not commensurable.

Usage:
    python scripts/print_progress.py
    python scripts/print_progress.py --json      # machine-readable output
"""

from __future__ import annotations

import argparse
import json
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict

PROJECT = Path(__file__).resolve().parents[1]
ARTIFACTS = PROJECT / "artifacts" / "benchmarks"
TRAINING = PROJECT / "artifacts" / "training"


@dataclass(frozen=True, slots=True)
class Experiment:
    """One experiment configuration, possibly averaged over repeated runs."""

    label: str
    run_ids: tuple[str, ...]
    note: str = ""


FRAMES_EXPERIMENTS: tuple[Experiment, ...] = (
    Experiment(
        "B1 BM25 (baseline)",
        (
            "20260807_frames_agent_qwen2_5_7b_smoke10_v1",
            "20260807_frames_p5_b1_agent_qwen2_5_7b_repeat2",
            "20260807_frames_p5_b1_agent_qwen2_5_7b_repeat3",
        ),
    ),
    Experiment("R1 Dense", ("20260808_p8a_r1_dense_run1",)),
    Experiment(
        "R2 Hybrid (BM25+Dense)",
        (
            "20260808_p8a_r2_hybrid_run1",
            "20260808_p8a_r2_hybrid_run2",
            "20260808_p8a_r2_hybrid_run3",
        ),
        note="high variance: 20/30/10",
    ),
    Experiment("R3 Hybrid + Rerank", ("20260808_p8a_r3_hybrid_rerank_run1",)),
    Experiment("R4 Hybrid + QueryRewrite", ("20260808_p8a_r4_hybrid_qrewrite_run1",)),
    Experiment("A2 No-retrieval (ablation)", ("20260808_p9_a2_no_retrieval_run1",)),
)

DABENCH_EXPERIMENTS: tuple[Experiment, ...] = (
    Experiment(
        "A1 smoke10 no-feedback",
        (
            "20260807_dabench_p6_a1_no_feedback_qwen2_5_7b_quick10",
            "20260807_dabench_p6_a1_no_feedback_qwen2_5_7b_repeat2",
            "20260807_dabench_p6_a1_no_feedback_qwen2_5_7b_repeat3",
        ),
        note="10 tasks only - not extrapolatable",
    ),
    Experiment(
        "B1 smoke10 feedback",
        (
            "20260807_dabench_p5_b1_agent_qwen2_5_7b_quick10",
            "20260807_dabench_p5_b1_agent_qwen2_5_7b_repeat2",
            "20260807_dabench_p5_b1_agent_qwen2_5_7b_repeat3",
        ),
        note="10 tasks only - overestimates by ~19pp",
    ),
    Experiment(
        "val35 base",
        ("20260809_p8c_dabench_val35_run3",),
        note="PRIMARY METRIC (35 tasks)",
    ),
    Experiment("val35 D1 guardrail", ("20260809_p8c_dabench_val35_run4_d1",)),
    Experiment("val35 D2 guardrail", ("20260809_p8c_dabench_val35_run5_d2",)),
)

SWE_EXPERIMENTS: tuple[Experiment, ...] = (
    Experiment(
        "Agent smoke3 (official eval)",
        ("20260807_swebench_live_agent_official_smoke3",),
        note="3 tasks, dev-contaminated - dev smoke only",
    ),
)


class SummaryRow(TypedDict):
    """One row of the per-domain progress summary table."""

    label: str
    runs: int
    expected_runs: int
    per_run: list[float]
    mean: float | None
    note: str


def _load_accuracy(run_id: str) -> float | None:
    path = ARTIFACTS / run_id / "results.json"
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and "accuracy" in payload:
        return float(payload["accuracy"])
    records = payload if isinstance(payload, list) else payload.get("records", [])
    if not records:
        return None
    correct = sum(1 for r in records if float(r.get("primary_metric", 0.0)) >= 1.0)
    return correct / len(records)


def _empty_answer_count(run_id: str) -> int | None:
    path = ARTIFACTS / run_id / "results.json"
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload if isinstance(payload, list) else payload.get("records", [])
    if not records:
        return None
    return sum(1 for r in records if not str(r.get("prediction", "")).strip())


def summarise(experiments: Sequence[Experiment]) -> list[SummaryRow]:
    rows: list[SummaryRow] = []
    for exp in experiments:
        accuracies = [a for a in (_load_accuracy(r) for r in exp.run_ids) if a is not None]
        rows.append(
            {
                "label": exp.label,
                "runs": len(accuracies),
                "expected_runs": len(exp.run_ids),
                "per_run": accuracies,
                "mean": sum(accuracies) / len(accuracies) if accuracies else None,
                "note": exp.note,
            }
        )
    return rows


def _print_domain(title: str, metric: str, rows: Sequence[SummaryRow]) -> None:
    print()
    print("-" * 78)
    print(f"{title}   [primary metric: {metric}]")
    print("-" * 78)
    for row in rows:
        label = row["label"]
        if row["mean"] is None:
            print(f"  {label:<28s} (no data)")
            continue
        per_run = " / ".join(f"{a:.0%}" for a in row["per_run"])
        mean = float(row["mean"])
        note = f"  # {row['note']}" if row["note"] else ""
        print(f"  {label:<28s} {per_run:<22s} mean={mean:6.1%}{note}")


def _print_training() -> None:
    print()
    print("-" * 78)
    print("P10 fine-tuning artifacts")
    print("-" * 78)
    for name, path in (
        ("SFT adapter", TRAINING / "p10_qlora" / "adapter" / "adapter_model.safetensors"),
        ("DPO adapter", TRAINING / "p10_dpo" / "adapter" / "adapter_model.safetensors"),
    ):
        if path.exists():
            print(f"  {name:<28s} {path.stat().st_size / 1e6:.1f} MB")
        else:
            print(f"  {name:<28s} (not trained)")


def _git_commit_count() -> int:
    result = subprocess.run(  # noqa: S603
        ["git", "rev-list", "--count", "HEAD"],  # noqa: S607
        capture_output=True,
        text=True,
        cwd=PROJECT,
        check=False,
    )
    return int(result.stdout.strip() or 0)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    args = parser.parse_args()

    domains = {
        "frames": summarise(FRAMES_EXPERIMENTS),
        "dabench": summarise(DABENCH_EXPERIMENTS),
        "swebench_live": summarise(SWE_EXPERIMENTS),
    }

    if args.json:
        print(json.dumps({"commits": _git_commit_count(), "domains": domains}, indent=2))
        return 0

    print("=" * 78)
    print(f"RepoPilot experiment status   ({_git_commit_count()} commits)")
    print("=" * 78)
    _print_domain("FRAMES (knowledge research)", "answer accuracy", domains["frames"])
    _print_domain("InfiAgent-DABench (data analysis)", "question accuracy", domains["dabench"])
    _print_domain(
        "SWE-bench-Live (software engineering)", "resolved rate", domains["swebench_live"]
    )

    base_empty = _empty_answer_count("20260809_p8c_dabench_val35_run3")
    d2_empty = _empty_answer_count("20260809_p8c_dabench_val35_run5_d2")
    if base_empty is not None and d2_empty is not None:
        print()
        print(f"  DABench empty-answer rate: base {base_empty}/35 -> D2 {d2_empty}/35")

    _print_training()
    print()
    print("Domains are reported separately by design; no cross-domain score is computed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
