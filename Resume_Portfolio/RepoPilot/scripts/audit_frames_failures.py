"""FRAMES per-question failure audit (read-only, no algorithm change)."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import TypedDict

PROJECT = Path(__file__).resolve().parents[1]
DATA = PROJECT / "evaluation" / "benchmarks" / "data" / "frames"
REVISION = "58d9fb6330f3ab1316d1eca12e5e8ef23dcc22ef"
TSV = DATA / REVISION / "test.tsv"

RUN_DIRS = [
    PROJECT / "artifacts" / "benchmarks" / "20260807_frames_agent_qwen2_5_7b_smoke10_v1",
    PROJECT / "artifacts" / "benchmarks" / "20260807_frames_p5_b1_agent_qwen2_5_7b_repeat2",
    PROJECT / "artifacts" / "benchmarks" / "20260807_frames_p5_b1_agent_qwen2_5_7b_repeat3",
]


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", "", text.lower())).strip()


class RunEntry(TypedDict, total=False):
    """One run's outcome for a single FRAMES task."""

    run: str
    correct: bool | None
    pred: str
    ftype: str


class AuditEntry(TypedDict):
    """Aggregated audit info for one FRAMES task."""

    gold: str
    runs: list[RunEntry]


def load_gold() -> dict[str, str]:
    gold: dict[str, str] = {}
    with open(TSV, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for i, row in enumerate(reader):
            gold[f"frames-test-{i:04d}"] = row["Answer"]
    return gold


def extract_final(text: str) -> str:
    m = re.search(r"FINAL:\s*(.+?)(?:\n|$)", text, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return text.strip()[:200]


def audit() -> None:
    gold = load_gold()
    print(f"Loaded {len(gold)} gold answers\n")
    run_names = ["Run1", "Run2", "Run3"]

    header = (
        f"{'Task':<20} {'Gold':<20} {'Run':<5} {'Correct':<8} {'Prediction':<40} {'FailureType'}"
    )
    print(header)
    print("-" * len(header))

    summary: dict[str, AuditEntry] = {}

    for tid in sorted(gold):
        g = gold[tid]
        g_norm = normalize(g)
        g_keywords = set(g_norm.split()) if g_norm else set()
        summary[tid] = {"gold": g, "runs": []}

        for run_name, run_dir in zip(run_names, RUN_DIRS, strict=False):
            task_dir = run_dir / tid
            if not task_dir.exists():
                summary[tid]["runs"].append({"run": run_name, "correct": None, "pred": "MISSING"})
                continue

            reviewer_path = task_dir / "reviewer_response.json"
            model_path = task_dir / "model_response.json"
            ctx_path = task_dir / "retrieval_context.txt"

            pred = ""
            if reviewer_path.exists():
                pred = extract_final(
                    json.loads(reviewer_path.read_text(encoding="utf-8")).get("content", "")
                )
            elif model_path.exists():
                pred = extract_final(
                    json.loads(model_path.read_text(encoding="utf-8")).get("content", "")
                )

            pred_norm = normalize(pred)
            correct = pred_norm == g_norm if pred_norm else False

            ctx_has_gold = False
            if ctx_path.exists():
                ctx = normalize(ctx_path.read_text(encoding="utf-8"))
                if g_norm and g_norm in ctx or g_keywords and g_keywords.issubset(set(ctx.split())):
                    ctx_has_gold = True

            if correct:
                ftype = "correct"
            elif not pred_norm:
                ftype = "extraction_failure"
            elif ctx_has_gold:
                ftype = "reasoning_failure"
            else:
                ftype = "retrieval_failure"

            summary[tid]["runs"].append(
                {"run": run_name, "correct": correct, "pred": pred[:38], "ftype": ftype}
            )
            print(f"{tid:<20} {g:<20} {run_name:<5} {str(correct):<8} {pred[:38]:<40} {ftype}")

    print("\n=== Failure type summary ===")
    ftypes: dict[str, int] = {}
    for _tid, info in summary.items():
        for r in info["runs"]:
            ft = r.get("ftype", "unknown")
            ftypes[ft] = ftypes.get(ft, 0) + 1
    for ft, count in sorted(ftypes.items(), key=lambda x: -x[1]):
        total = len(summary) * 3
        print(f"  {ft}: {count} ({count}/{total} = {count / total * 100:.0f}%)")

    print("\n=== Per-question correctness (3 runs) ===")
    for tid, info in summary.items():
        corrects = [r["correct"] for r in info["runs"] if r["correct"] is not None]
        n = sum(corrects)
        print(f"  {tid}: {n}/3 correct, gold={info['gold']}")

    out = PROJECT / "artifacts" / "benchmarks" / "20260808_p7_frames_failure_audit.json"
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nDetailed audit saved to {out}")


if __name__ == "__main__":
    audit()
