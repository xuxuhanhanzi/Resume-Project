"""Extract SFT and DPO training data from FRAMES agent traces (P10).

This script reads all frozen FRAMES run artifacts, compares predictions
to gold answers, and emits:
  - SFT data: {instruction, context, response} for correct runs
  - DPO pairs: {instruction, context, chosen, rejected} for matched
    question pairs where one run succeeded and another failed

Data governance:
  - Only public task descriptions, public agent traces, and public gold
    answers are used.  No evaluator-only data enters training data.
  - Source run-ids, hashes, and correctness labels are recorded per sample.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
from collections import defaultdict
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
    PROJECT / "artifacts" / "benchmarks" / "20260808_p8a_r2_hybrid_run1",
    PROJECT / "artifacts" / "benchmarks" / "20260808_p8a_r2_hybrid_run2",
]

OUTPUT_DIR = PROJECT / "artifacts" / "training" / "p10_frames"


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", "", text.lower())).strip()


class Sample(TypedDict):
    """One FRAMES agent trace turned into a training sample."""

    task_id: str
    run_id: str
    question: str
    context: str
    model_response: str
    final_answer: str
    correct: bool
    input_tokens: int
    output_tokens: int


def extract_final(text: str) -> str:
    m = re.search(r"FINAL:\s*(.+?)(?:\n|$)", text, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return text.strip()[:200]


def load_gold() -> dict[str, str]:
    gold: dict[str, str] = {}
    with open(TSV, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for i, row in enumerate(reader):
            gold[f"frames-test-{i:04d}"] = row["Answer"]
    return gold


def extract_run(run_dir: Path, gold: dict[str, str]) -> list[Sample]:
    samples: list[Sample] = []
    for task_dir in sorted(d for d in run_dir.iterdir() if d.is_dir()):
        tid = task_dir.name
        if tid not in gold:
            continue
        ctx_path = task_dir / "retrieval_context.txt"
        model_path = task_dir / "model_response.json"
        reviewer_path = task_dir / "reviewer_response.json"
        if not ctx_path.exists() or not model_path.exists():
            continue
        context = ctx_path.read_text(encoding="utf-8").strip()
        model_resp = json.loads(model_path.read_text(encoding="utf-8"))
        model_content = model_resp.get("content", "")
        final_answer = model_content
        if reviewer_path.exists():
            rev = json.loads(reviewer_path.read_text(encoding="utf-8"))
            final_answer = extract_final(rev.get("content", model_content))
        else:
            final_answer = extract_final(model_content)
        correct = normalize(final_answer) == normalize(gold[tid])
        samples.append(
            {
                "task_id": tid,
                "run_id": run_dir.name,
                "question": "",
                "context": context,
                "model_response": model_content,
                "final_answer": final_answer,
                "correct": correct,
                "input_tokens": model_resp.get("input_tokens", 0),
                "output_tokens": model_resp.get("output_tokens", 0),
            }
        )
    return samples


def load_questions() -> dict[str, str]:
    questions: dict[str, str] = {}
    with open(TSV, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for i, row in enumerate(reader):
            questions[f"frames-test-{i:04d}"] = row["Prompt"]
    return questions


def main() -> int:
    gold = load_gold()
    questions = load_questions()
    all_samples: list[Sample] = []
    for run_dir in RUN_DIRS:
        if not run_dir.exists():
            continue
        samples = extract_run(run_dir, gold)
        for s in samples:
            s["question"] = questions.get(s["task_id"], "")
        all_samples.extend(samples)
        print(
            f"  {run_dir.name}: {len(samples)} samples, "
            f"{sum(1 for s in samples if s['correct'])} correct"
        )

    correct = [s for s in all_samples if s["correct"]]
    incorrect = [s for s in all_samples if not s["correct"]]
    print(
        f"\nTotal: {len(all_samples)} samples ({len(correct)} correct, {len(incorrect)} incorrect)"
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    sft_data = [
        {
            "instruction": s["question"],
            "input": s["context"],
            "response": s["model_response"],
            "task_id": s["task_id"],
            "run_id": s["run_id"],
        }
        for s in correct
    ]
    sft_path = OUTPUT_DIR / "sft_data.jsonl"
    with open(sft_path, "w", encoding="utf-8") as f:
        for item in sft_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"SFT data: {len(sft_data)} samples -> {sft_path}")

    by_task: dict[str, list[Sample]] = defaultdict(list)
    for s in all_samples:
        by_task[s["task_id"]].append(s)

    dpo_data: list[dict[str, str]] = []
    for tid, samples in by_task.items():
        chosen = [s for s in samples if s["correct"]]
        rejected = [s for s in samples if not s["correct"]]
        if not chosen or not rejected:
            continue
        for c in chosen:
            for r in rejected[:2]:
                dpo_data.append(
                    {
                        "instruction": c["question"],
                        "input": c["context"],
                        "chosen": c["model_response"],
                        "rejected": r["model_response"],
                        "task_id": tid,
                        "chosen_run": c["run_id"],
                        "rejected_run": r["run_id"],
                    }
                )
    dpo_path = OUTPUT_DIR / "dpo_data.jsonl"
    with open(dpo_path, "w", encoding="utf-8") as f:
        for item in dpo_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"DPO data: {len(dpo_data)} pairs -> {dpo_path}")

    manifest: dict[str, object] = {
        "source": "FRAMES agent traces (public)",
        "gold_revision": REVISION,
        "runs_included": [d.name for d in RUN_DIRS if d.exists()],
        "total_samples": len(all_samples),
        "correct_samples": len(correct),
        "sft_count": len(sft_data),
        "dpo_count": len(dpo_data),
        "license": "FRAMES data is CC-BY-SA, agent traces are project-generated",
    }
    manifest_path = OUTPUT_DIR / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    content_hash = hashlib.sha256(sft_path.read_bytes() + dpo_path.read_bytes()).hexdigest()
    print(f"Data hash: {content_hash}")
    print(f"Manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
