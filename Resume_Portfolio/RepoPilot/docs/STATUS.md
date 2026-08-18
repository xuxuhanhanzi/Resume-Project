# RepoPilot — Current Status (authoritative, single source of truth)

> This file is the canonical description of the project's current state. Where it
> conflicts with any other doc (README, project_context_summary, agent card, old
> summaries), this file wins. Last updated: 2026-08-14.

## What RepoPilot is

A local-first, feedback-driven, governable agent runtime plus a three-domain
evaluation harness covering **knowledge research / data analysis / software
engineering**. Each domain reports its own official primary metric; **no
cross-domain composite score is ever computed.**

## Completion status

The historical P7–P12 delivery stages are committed. This does **not** mean the
final release definition of done is closed: service/load validation, security hardening
and a release/tag remain open.

| Stage | Scope | Status |
|-------|-------|--------|
| P7 | Hand-off validation & state freeze | ✅ |
| P8A | FRAMES retrieval ablation | ✅ |
| P8B | SWE-bench-Live failure-driven improvement | ✅ |
| P8C | DABench scale-up + prompt guardrails (D1/D2) | ✅ |
| P9 | Mechanism ablation (A2 no-retrieval) | ✅ |
| P10 | QLoRA fine-tuning (SFT + DPO) | ✅ weights + 1.5B downstream negative ablation |
| P11 | Multi-agent harness | ✅ graph + routing + authenticated HTTP/SSE + Runtime/Verifier load + real Qwen 1/4/8 capacity |
| P12 | Evaluation protocol & delivery | ✅ evidence, clean-clone record and local release/tag delivered |

## Quality gates (verified 2026-08-14)

Run with the project venv (`.venv`, Python 3.12.3):

| Gate | Command | Result |
|------|---------|--------|
| Format | `ruff format --check .` | ✅ 141 files formatted |
| Lint | `ruff check .` | ✅ all checks passed |
| Type | `mypy` (strict, src+tests+scripts) | ✅ 113 source files, 0 errors |
| Test | `pytest -q` | ✅ 90 passed, 1 live-Docker test skipped by default |
| Live safety | `REPOPILOT_RUN_DOCKER_SECURITY=1 ... dev.py safety` | ✅ 6 passed |

`python scripts/dev.py check` completed successfully in the current workspace;
the executable harness, routing, service and HTTP/SSE tests pass. The Docker test is isolated in a
separate CI job because it requires the fixed local image; it passed in the recorded live run.
`dev.py` uses `--no-incremental` and
`-p no:cacheprovider`, so the gate does not depend on mypy or pytest caches.

## Three-domain primary metrics

| Domain | Dataset | Scale | Primary metric | Citable? |
|--------|---------|-------|----------------|----------|
| Knowledge research | Google FRAMES | **60** questions | **18.3% (11/60)** | ✅ citable (R6, oracle-document, 3 repeats) |
| Data analysis | InfiAgent-DABench | **35** questions × 3 | **73.3% mean** (25/35, 26/35, 26/35) | ✅ citable (hash-frozen, repeated) |
| Software engineering | SWE-bench-Live | 5 fresh (R5) | **0.0% (0/5)** | ✅ citable (R5, clean holdout) |

**Honesty notes (do not inflate):**
- FRAMES 18.3% is now a **citable** number: 60 questions (oracle-document protocol,
  qwen2.5:7b), 3 deterministic repeats all 11/60 = 0.183. Supersedes the old 10-question
  ~20% reference figure (which spread 20/30/10 across repeats and was directional only).
- DABench's repeat-validated baseline mean is 73.3% (25/35, 26/35, 26/35).
  The old 71.4% is the first frozen repeat, and the earlier 90% 10-task smoke
  overestimated the expanded result and must not be cited as a dataset-level score.
- SWE-bench-Live R5 uses 5 **fresh, uncontaminated** tasks (linkding / torchtune /
  weasyprint / fast-f1 / haystack) selected from the lite parquet; resolved-rate is
  computed self-contained by `scripts/eval_swebench_resolved.py` (model_patch +
  gold test_patch applied, FAIL_TO_PASS run inside the official eval image).
  **Result: 0/5 = 0.0% resolved** (all 5 failed; 4 empty patches + 1 insufficient partial).
  This is the citable SWE number. The old smoke3 0/3 is a dev-smoke signal only, never a
  resolved-rate claim.
- P10 produced real SFT+DPO adapter weights (Qwen2.5-1.5B) and a matching-architecture
  downstream ablation. The base scored 9/35; the adapter scored 0/35. This is a
  negative, single-repeat diagnostic result and cannot alter the 7B main metric.

## Stable-failure core set (5 DABench questions)

Heterogeneous, not a uniform capability wall:
- 0028 (hard), 0056 (easy), 0062 (medium): **parsing/robustness crashes** —
  fixable by prompt guardrails (D3 implemented, see below).
- 0055 (easy), 0109 (hard): **genuine reasoning errors** — need model-capability
  work, not prompt wording.

D3 (additive `parse_robustness_guardrail`, default off) was implemented and
validated on the 5 core questions: it **removed the crashes but recovered 0/5
scores**, confirming core finding #3 (fixing a crash ≠ raising accuracy). This is
a documented negative result, not a silent failure.

## Key findings (P7–P12)

1. Small samples overestimate (10-task DABench 90% vs expanded repeated baseline ≈73%).
2. A stable total can mask high per-question churn (28.6% of questions flipped
   between runs; 20 steady-correct / 10 flipped / 5 steady-wrong).
3. Fixing a crash does not raise accuracy (D1/D2/D3 all 0 net gain on score).
4. Over-specified methods suppress model adaptation (D1 regressed 3 questions D2
   recovered by giving the method choice back).
5. Under oracle retrieval, FRAMES gains are bounded — evidence synthesis, not
   retrieval, is the bottleneck (P9 A2).

## Remediation plan status (see `PROJECT_COMPLETION_AUDIT_AND_REMEDIATION_PLAN_2026-08-09.md`)

- **R0 — credible quality gates:** ✅ done (format/lint/type/test all green;
  `dev.py check` green except sandbox-only test-guard caveat above).
- **R1 — transferable evidence index:** ✅ done (`docs/evidence/artifact_index.json` +
  `verify_artifact_index.py`, **19/19 runs consistent**, commit `ab4205d`, re-based
  2026-08-09 after R2; extended through R5/R6).
- **R2 — re-establish frozen baselines:** ✅ **done** (9 full 35q runs via
  `scripts/run_r2_dabench_repeats.sh`, Docker 29.6.1, Ollama qwen2.5:7b; re-registered
  in the evidence index, verify 19/19). Frozen DABench baselines (3 repeats each):
  - **base**: 25/35 (0.714), 26/35 (0.743), 26/35 (0.743) → mean **0.733**
  - **guardrail** (single merged `dtype_guardrail` = old D1/D2): 26/35, 26/35, 26/35
    → stable **0.743** (≈ +1pp over base)
  - **D3** (`--parse-robustness`): 22/35 (0.629), 22/35 (0.629), 23/35 (0.657) →
    mean **≈0.638** (≈ **−10pp** vs base — confirms guardrail prevents crashes but
    does NOT raise reasoning accuracy; core finding #3)
  - Note: original `r2_20260809_d3_rep3` crashed at 10/35 (partial, preserved per
    project rules); re-run `r2_20260809_d3_rep3b` (23/35) is the registered 3rd D3 repeat.
  - Citable component = the 3 base repeats (mean 0.733 ≈ 25.7/35). This is now the
    honest, repeat-validated DABench number, superseding the old single-run 71.4%.
- **R3 — metric honesty pass:** ✅ done (README, project_review, project_final_summary,
  stage_01_06_summary audited; inflated 90% and false "no training" claims corrected).
- **R4 — adapter downstream eval:** ✅ **done (1.5B-only ablation)** — the P10
  adapters were trained on **Qwen/Qwen2.5-1.5B** (`adapter_config.json`), so they
  CANNOT merge into / improve the **qwen2.5:7b** (7B) DABench baseline (73.3% mean).
  R4 therefore evaluates a *separate 1.5B config* (1.5B base vs 1.5B+SFT+DPO adapter
  on DABench val35), NOT an improvement of the 7B baseline.
  - `scripts/merge_adapter_and_prep_eval.py --apply-dpo` merges SFT+DPO LoRA adapters
    into Qwen2.5-1.5B (torch 2.5.1+cu121 / transformers 5.14.1 / peft 0.20.0; committed `3676707`).
  - Ollama 0.32.6's `--experimental` safetensors import forces the MLX runner
    (Apple-only) on Windows → inference fails. Workaround: `scripts/serve_hf_model.py`
    serves the merged HF model via a minimal OpenAI-compatible `/v1/chat/completions`
    endpoint using transformers.
  - **Results (1 repeat each, qwen2.5:1.5b / temp 0 / seed 7 / network deny):**
    - 1.5B base: **9/35 = 25.7%** (digest `65ec06548149...`, run `r4_1p5b_base_rep1`)
    - 1.5B + SFT+DPO adapter: **0/35 = 0.0%** (run `r4_1p5b_adapter_rep2`,
      served via `serve_hf_model.py` on port 8080)
  - These are reference-only (citable=False); they do NOT raise the 7B 73.3% mean.
    They document a severe regression from the current P10 recipe on the matching 1.5B architecture.
- **R5 — SWE clean holdout:** ✅ **done** — 5 fresh (uncontaminated) SWE-bench-Live tasks
  (linkding-984 / weasyprint-2387 / fast-f1-699 / haystack-8609 / torchtune-1806) selected from
  the lite parquet (seed 7), each cloned @ base_commit and run inside its official eval image
  (`starryzhang/sweb.eval.x86_64.*`, `__`→`_1776_`). `allowed_paths: ["."]` so the agent may edit
  any source file exactly like standard SWE-bench evaluation — no evaluator data enters the agent.
  Harness: `scripts/prepare_swebench_fresh_holdout.py` (prep), `scripts/run_swebench_live_fresh_holdout.py`
  (agent run, hardened with `try/except/finally` so `predictions.jsonl` is always emitted even on
  timeout/error, `core.longpaths=true` for Windows long paths), `scripts/eval_swebench_resolved.py`
  (self-contained resolved-rate: model_patch + gold test_patch applied, FAIL_TO_PASS/PASS_TO_PASS run
  inside the image, test-id chunking to avoid `WinError 206`).
  **Result: resolved_rate = 0/5 = 0.0%** (qwen2.5:7b, temp 0, seed 7, network deny). All 5 tasks
  failed: 4 produced empty patches (agent exhausted iterations without a valid `apply_patch`), 1
  (weasyprint) produced a 620-char partial patch on `docs/changelog.rst` (insufficient to pass tests).
  Registered as `r5_20260809_fresh_combined5` in the evidence index (citable, verify 19/19). This
  supersedes the old contaminated smoke3 0/3 (dev-smoke only, never a resolved-rate claim).
  - **Correction to prior STATUS:** previously marked BLOCKED by `network=deny`. On re-check
    the sandbox **host network is reachable** (huggingface/pypi 200, Docker pull works), and
    `network=deny` is only the policy applied to the *scored agent* inside its sandbox — it
    does not block us from fetching data/images and localizing them for evaluation.
- **R6 — FRAMES scale-up:** ✅ **done** — `scripts/run_frames_oracle_smoke.py --count` cap
  raised to 200 and corpus path derived from `--count`; `materialize_frames_corpus.py` now
  `.strip()`s each wiki link and percent-encodes non-ASCII URLs so the frozen corpus manifest
  matches `FramesAdapter` asset URIs exactly (fixed a `KeyError` on e.g. `Calton,_Glasgow`).
  Result: **60 questions, oracle-document protocol, qwen2.5:7b, 3 repeats → 11/60 = 0.183 each
  (deterministic at temperature 0)**. Registered as `r6_20260809_frames60b_rep{1,2,3}` in the
  evidence index (citable, verify 19/19). FRAMES is now a citable metric.

## Evidence

Authoritative run artifacts, commands, commit hashes, manifest/model/image
digests and result paths are registered in `docs/evidence/artifact_index.json`
(verified by `scripts/verify_artifact_index.py`). Large artifacts live in
`artifacts/` and are referenced by path + SHA-256, not inlined.

## Executable multi-agent milestone (2026-08-11)

`MultiAgentHarness.run` now executes a real Planner → AgentRuntime → deterministic
Verifier → read-only Reviewer graph. The executor remains the only node with tools
or write permission. A separate coordinator checkpoint records node/cycle/message
state; the executor retains its own checkpoint and idempotent tool journal. Tests
cover a real bug-fix task, process cancellation followed by reviewer-stage resume,
explicit cancellation, and durable timeout failure. Fixed/rule/model/hybrid routing now
selects complete runtime bindings; model failures cascade to an attributable rule/default
decision, and the target/reason/confidence/cascade are durable. This closes the previous
"data structures only" and "routing prototype only" defects. The service now exposes authenticated
JSON submit/status/cancel and resumable SSE events, with request/rate bounds and task-path allowlisting.
Localhost black-box testing covers 200/401/202, event replay and an attributable model-unavailable
failure. A separate real-runtime load run executed 120 independent file-fix + subprocess-verifier
tasks at each 1/4/8 concurrency (360/360 completed): throughput 10.08/33.26/34.21 task/s and
queue-inclusive P95 11.35/3.45/3.42s. It uses ScriptedProvider, so those numbers remain scheduling
evidence only. A separate fixed-digest Qwen2.5-7B/Q4_K_M run completed 24 requests at each 1/4/8
concurrency (72/72 non-empty): throughput 5.52/16.80/18.62 req/s and queue-inclusive P95
4.16/1.38/1.24s. That closes the local model-service capacity gate but remains a fixed short-request
benchmark, not end-to-end coding-agent throughput.

## Remaining external gates

- Choose a remote repository and publish the already-created local release/tag.
- Keep FRAMES 18.3%, DABench 73.3% mean and SWE 0/5 separate; do not publish a composite score.
