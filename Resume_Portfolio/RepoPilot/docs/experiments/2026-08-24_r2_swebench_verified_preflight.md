# R2 SWE-bench Verified：E 盘预检与可复核冒烟

## Status

Partial engineering evidence only — 2026-08-24.  This record is not an R2
ablation and reports no resolved-rate, error-acceptance-rate, or cost-improvement
claim.

## Frozen inputs and split

- Dataset: `SWE-bench/SWE-bench_Verified`, source revision
  `78f471bf655a3137b2e8a75af1501690ec009ec3`.
- Full source digest: `84385d3374a0c37b692a72ee57509fba15e5cce896671944e1348d62a4a8f4de`.
- Agent-visible public JSONL SHA-256:
  `5994ce5f8a9b4e51e3ca4ebdc3aa73fb7a87feebea885d87f57f41e24564f0b0`.
  It permits only issue metadata (`instance_id`, repository, base commit,
  problem statement, version, image); evaluator-only patches, test identifiers,
  test patches and evaluation scripts are rejected by the loader.
- The first manifest (`swebench_verified_r2_20260824_v1.json`, file SHA-256
  `9ace2cca908914eca17477e397080f493a4c1cafdcf58314b83e46ad255419c9`) had
  an unsound group allocation of 8/265/227.  It is retained for audit and was
  never used for a comparison.
- The active manifest is
  `evaluation/benchmarks/manifests/swebench_verified_r2_20260824_v2.json`
  (canonical SHA-256
  `0aedbbc3863615a34ed7f7cb7dcef3b117a54a11ca181c704d5a40a05a1cd57b`,
  file SHA-256
  `cd94ae459d1aba33fd1d06fdadad313be08c4a6d9205daaa7def4dba01a2bf10`).
  Its deterministic repository-grouped allocation is development 107,
  validation 162, and final holdout 231.  Exact 20/40/40 counts are impossible
  while keeping the 231-instance Django repository indivisible; the manifest
  records the size-first feasible-strata greedy algorithm and its group boundary.

## E-drive and official evaluator preflight

- At preflight, E: had 148.20 GiB free of 255.97 GiB and contains the external
  source checkout, public data, generated workspaces, and run outputs.
- Docker Desktop's image VHDX remains on C:, so E: capacity does not remove the
  Docker-layer capacity requirement.  Existing user images were not pruned.
- Official evaluator checkout: `E:\RepoPilotEval\SWE-bench`, commit
  `7a21e05772954cc81471ae19d56f436cecf43c54`.
- The unmodified Windows evaluator failed its one-instance `--gold` smoke only
  because generated `eval.sh` had CRLF line endings.  ADR 0006 records the
  two-write LF-only portability patch.  After it, the official evaluator resolved
  the same gold `sympy__sympy-20590` task `1/1` in
  `E:\RepoPilotEval\runs\r2_20260824_official_gold_smoke_lf`.

This proves that the evaluator/container path works.  It is not an agent score:
the evaluator supplied the gold patch, and this instance is excluded from agent
generation.

## Agent workspace and direct-control smoke

The workspace preparer exports `/testbed` from the public instance image, resets
to the public `base_commit`, sets workspace-local `core.autocrlf=false` and
`core.filemode=false`, and requires a clean status before generation.  It never
loads evaluator-only fields.  The direct-control workspace was
`E:\RepoPilotEval\workspaces\r2_smoke_scripted2\scikit-learn__scikit-learn-10297`
at base commit `b90661d6a46aa3619d3eec94d5281f5888add501`; its image digest is
`swebench/sweb.eval.x86_64.scikit-learn_1776_scikit-learn-10297@sha256:58b80e9100f9d107291b5ea37c76583ed4851d8e18ec07aef197f1803a290986`.

`r2_20260824_direct_react_smoke_1` ran one development instance with local
`qwen2.5:7b` revision
`845dbda0ea48ed749caafd9e6037047aa19acfcfd82e704d7ca97d631a0b697e`, fixed
temperature 0, an identical eight-tool surface, a 12,000-token ledger, and a
600-second cap.  The agent used three tools, produced no patch, and stopped after
the harness detected repeated verification failure without changed files.  Its
raw generation SHA-256 is
`0b9d130ac1c56604c7a6a14971154ac65a9b8e8bc65cd6f7cfc71438e0c4be1a`; the
predictions SHA-256 is
`8b4cde1d71537f285864e9b666d0b74a17dad62c8db4b773b75ff3e2eb0eadd3`.

The run validates public-data isolation, workspace preparation, local model/tool
traces, shared-budget accounting, and generation receipts.  It deliberately
does **not** invoke official resolution and cannot support an R2 ability claim.

`r2_20260824_pev_risk_smoke_retry1` then exercised the same task through the
actual planner and executor nodes of `pev_risk_gated`.  The harness completed
one executor cycle and recorded seven model calls.  The final call made the
shared ledger reach 13,814 tokens, over its 12,000-token cap, so the raw result
is explicitly marked `token_budget_exceeded=true` and non-comparable.  The
executor did not pass deterministic verification; consequently the risk rule
did not call the reviewer.  Its raw generation SHA-256 is
`a4b150e75f1e73b7bfdfd2ea28564ed3f593c0219d5448e1b36d9aa3724e3be9`.

The first PEV attempt is retained separately at
`E:\RepoPilotEval\runs\r2_20260824_pev_risk_smoke_1`.  It produced only
partial role artifacts because an object-valued `expected_replacements` argument
raised a tool-layer `TypeError`.  The tool now returns a structured validation
error for non-integer replacement counts; the regression test passed before the
retry.  The failed first attempt is neither a score nor a receipt.

### Strict pre-call budget gate

The original ledger could observe an overrun only after a model response was
returned.  It was replaced with a shared pre-call reservation: before each
planner, executor, or reviewer call, the runner reserves the UTF-8 byte upper
bound of the complete provider-neutral request plus 512 framing tokens and the
capped output allowance.  A call that cannot be fully reserved is not sent.
Reported provider usage remains the cost metric; a provider that exceeds its
reservation is explicitly non-comparable.

`r2_20260824_pev_strict_ledger_smoke_1` used 16,000 reserved tokens.  It sent
two calls, reserved 12,718 tokens, used 2,655 reported tokens, and rejected its
third call before transmission; `token_budget_exceeded` is false.  Raw SHA-256:
`6c0af0e2a72a9e49f680d5e1d9ef5b9230c4ac6c050e980a3bd33008a5dadefd`.
The reserve is intentionally conservative and this smoke is not a model score.

The same clean public-base workspace was then used only while its Git status
remained clean to calibrate 32k, 64k, and 96k reservations.  At 32k the third
response was blocked; at 64k the seventh request was blocked after six model
responses; at 96k the planner plus six executor responses completed normally
with `reserved_tokens=73967`, `total_tokens=14970`, and no reservation
violation.  The 96k value is therefore frozen for development in
`configs/evals/r2_swebench_verified_development.yaml`; it is shared by all six
matrix rows and all three requested seeds.  This choice fixes the runtime
allocation, while actual reported token usage remains the cost metric.

The 96k calibration raw SHA-256 is
`4d824e6da1e397678205a987481a147892d9dd966b4be06eea8ce3d67146d371`.

### Authorized scratch reuse

On 2026-08-24, the user authorized baseline restoration only for experiment-
created workspaces below `E:\RepoPilotEval\workspaces\r2_scratch\`.  The
runner now writes predictions, the generation trace, the generation receipt and
a separate reset receipt before it may reset a workspace.  The reset validates
the adjacent immutable workspace receipt (task ID, manifest SHA-256, public
base commit, and resolved workspace path), then runs exactly `git reset --hard
<public-base-commit>`.  It never calls `git clean` or deletes files; any
remaining untracked content is a failed lifecycle check.

`r2_20260824_direct_react_receipt_reset_smoke_1` exercised that lifecycle on
one development task, `scikit-learn__scikit-learn-10297`, seed 7.  It made seven
model calls, reported 17,233 tokens, reserved 87,872 of 96,000 tokens, and
rejected the next request before transmission.  It emitted no patch and is a
failed generation (`token_budget_exceeded=false`), not an official evaluation.
Its raw generation SHA-256 is
`2a8925dbf5bf20771e0a5af810a9198bfc6629f7e391fdc2a44a37a21e9854e7`.
The reset receipt records one successful event, no errors, and confirms the
post-reset clean HEAD is `b90661d6a46aa3619d3eec94d5281f5888add501`.

## Remaining formal gate

R2-A/B/C still require all paired configurations and repetitions on the frozen
development and validation partitions, configuration selection without viewing
the final results, then one final-holdout run per selected configuration with the
official evaluator.  A bounded smoke must not substitute for those runs.
