"""Build docs/evidence/artifact_index.json from authoritative benchmark runs.

Registers, for each citable / reference run: the command context (frozen model
digest, temperature, seed, network policy), the manifest + its SHA-256, the
results path + its SHA-256, and the recomputed primary metric. Small summaries
are inlined; large raw artifacts stay in artifacts/ and are referenced by hash.

Usage:
    python scripts/build_artifact_index.py
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts" / "benchmarks"
EVID = ROOT / "docs" / "evidence"

MODEL = {
    "name": "qwen2.5:7b",
    "digest": "845dbda0ea48ed749caafd9e6037047aa19acfcfd82e704d7ca97d631a0b697e",
    "temperature": 0,
    "seed": 7,
    "network_policy": "deny",
}

DABENCH_MANIFEST = "evaluation/benchmarks/manifests/dabench_validation35_v2.json"


def sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _results_path(run_id: str) -> Path | None:
    """Resolve the authoritative results file for a run (supports both names)."""
    for name in ("results.json", "resolved_results.json"):
        rp = ART / run_id / name
        if rp.exists():
            return rp
    return None


def accuracy(run_id: str) -> tuple[int, int]:
    rp = _results_path(run_id)
    if rp is None:
        return 0, 0
    top = json.loads(rp.read_text(encoding="utf-8"))
    # SWE-bench-Live style: top-level resolved / tasks
    if "resolved" in top:
        tasks = top.get("tasks")
        tot = len(tasks) if isinstance(tasks, list) else tasks
        return int(top["resolved"]), int(tot) if tot is not None else 0
    recs = top.get("records", [])
    if not recs:
        return 0, 0
    ok = sum(
        1
        for r in recs
        if r.get("task_success") is True or float(r.get("primary_metric", 0.0)) >= 1.0
    )
    return ok, len(recs)


def register(
    run_id: str,
    domain: str,
    primary_metric: str,
    citable: bool,
    note: str,
    commit: str,
    manifest: str | None = None,
) -> dict[str, object]:
    ok, tot = accuracy(run_id)
    rp = _results_path(run_id)
    mani_path = ROOT / manifest if manifest else None
    return {
        "id": run_id,
        "domain": domain,
        "primary_metric": primary_metric,
        "score": ok,
        "total": tot,
        "accuracy": round(ok / tot, 4) if tot else None,
        "citable": citable,
        "note": note,
        "commit": commit,
        "manifest": manifest,
        "manifest_sha256": sha256(mani_path) if mani_path else None,
        "results_path": f"artifacts/benchmarks/{run_id}/{rp.name}" if rp else None,
        "results_sha256": sha256(rp) if rp else None,
    }


def main() -> int:
    EVID.mkdir(parents=True, exist_ok=True)
    # R2 frozen DABench baselines: 3 repeats x {base, guardrail, D3} over the
    # frozen val35 manifest. base repeats are the citable metric components;
    # guardrail / D3 are variant configs (guardrail ~ +3pp, D3 ~ -11pp vs base
    # mean — see docs/STATUS.md). Supersedes the earlier single-run run3/run4_d1/
    # run5_d2 / core5_d3 entries.
    R2_COMMIT = "63b47c8"
    r2_runs = []
    for cfg, citable in (
        ("base", True),
        ("guardrail", False),
        ("d3", False),
    ):
        for rep in (1, 2, 3):
            rid = f"r2_20260809_{cfg}_rep{rep}"
            if cfg == "d3" and rep == 3:
                # original r2_20260809_d3_rep3 crashed mid-way (partial, 10/35);
                # preserved per project rules, re-run produced this clean id.
                rid = "r2_20260809_d3_rep3b"
            r2_runs.append(
                register(
                    rid,
                    "dabench",
                    "question_accuracy",
                    citable,
                    f"R2 frozen baseline — {cfg} config repeat {rep}",
                    R2_COMMIT,
                    DABENCH_MANIFEST,
                )
            )
    runs = r2_runs + [
        register(
            "r6_20260809_frames60b_rep1",
            "frames",
            "answer_accuracy",
            True,
            "R6 citable FRAMES — 60-task oracle-document repeat 1",
            "6be3187",
            "evaluation/benchmarks/corpora/frames/58d9fb6330f3ab1316d1eca12e5e8ef23dcc22ef/smoke60/manifest.json",
        ),
        register(
            "r6_20260809_frames60b_rep2",
            "frames",
            "answer_accuracy",
            True,
            "R6 citable FRAMES — 60-task oracle-document repeat 2",
            "6be3187",
            "evaluation/benchmarks/corpora/frames/58d9fb6330f3ab1316d1eca12e5e8ef23dcc22ef/smoke60/manifest.json",
        ),
        register(
            "r6_20260809_frames60b_rep3",
            "frames",
            "answer_accuracy",
            True,
            "R6 citable FRAMES — 60-task oracle-document repeat 3",
            "6be3187",
            "evaluation/benchmarks/corpora/frames/58d9fb6330f3ab1316d1eca12e5e8ef23dcc22ef/smoke60/manifest.json",
        ),
        register(
            "20260808_p8a_r2_hybrid_run1",
            "frames",
            "answer_accuracy",
            False,
            "R2 hybrid repeat 1 (reference only)",
            "unknown",
        ),
        register(
            "20260808_p8a_r2_hybrid_run2",
            "frames",
            "answer_accuracy",
            False,
            "R2 hybrid repeat 2 (reference only)",
            "unknown",
        ),
        register(
            "20260808_p8a_r2_hybrid_run3",
            "frames",
            "answer_accuracy",
            False,
            "R2 hybrid repeat 3 (reference only)",
            "unknown",
        ),
        register(
            "20260807_swebench_live_agent_official_smoke3",
            "swebench_live",
            "resolved_rate",
            False,
            "dev smoke only (contaminated tasks)",
            "unknown",
        ),
        register(
            "r5_20260809_fresh_combined5",
            "swebench_live",
            "resolved_rate",
            True,
            "R5 clean holdout — 5 fresh (uncontaminated) SWE-bench-Live tasks, qwen2.5:7b",
            "2d56acb",
        ),
        register(
            "r4_1p5b_base_rep1",
            "dabench",
            "question_accuracy",
            False,
            "R4 1.5B ablation — Qwen2.5-1.5B base (no adapter) on DABench val35",
            "2d56acb",
            DABENCH_MANIFEST,
        ),
        register(
            "r4_1p5b_adapter_rep2",
            "dabench",
            "question_accuracy",
            False,
            "R4 1.5B ablation — 1.5B + P10 SFT+DPO adapter (transformers serve)",
            "2d56acb",
            DABENCH_MANIFEST,
        ),
    ]
    index = {"generated": "2026-08-10", "model": MODEL, "runs": runs}
    out = EVID / "artifact_index.json"
    out.write_text(json.dumps(index, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {out}")
    for r in runs:
        print(
            f"  {r['id']:<42} {r['score']}/{r['total']} "
            f"citable={r['citable']} sha={str(r['results_sha256'])[:12]}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
