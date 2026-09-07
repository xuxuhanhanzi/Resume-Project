# Coding development-funnel protocol

This protocol is a **development-only** diagnostic for improving RepoPilot's
repair loop. It is deliberately separate from the frozen five-task
SWE-bench-Live holdout recorded in `docs/experiments/2026-08-14_swe_negative_stop_decision.md`.
No task, hidden test, evaluator output, or model patch from that holdout may be
copied into this development set.

## Record format

For each independently selected local fixture, record exactly one JSONL object:

```json
{"task_id":"local-fixture-001","localized":true,"patch":"<redacted patch>","patch_applied":true,"verification_passed":false}
```

Run `repopilot eval coding-funnel <records.jsonl>`. The command is offline: it
does not invoke a provider, execute a patch, run tests, or print the supplied
patch text. It reports five mutually exclusive categories:

1. `localization_failed`
2. `empty_patch`
3. `patch_apply_failed`
4. `verification_failed`
5. `verified`

The stage ordering is intentionally strict: a non-empty patch requires a
recorded localization result; application requires a non-empty patch; and a
passing verification requires application. This catches falsely optimistic
reports before they are compared.

## Evaluation discipline

- Build the development set from small, versioned, public or project-owned
  fixtures that are not part of a scored holdout.
- Freeze fixture revisions and the verification command before comparing model
  prompts or runtime changes.
- Treat the current `80%` non-empty-and-applicable-patch goal as an acceptance
  threshold for a future real development run, **not** as a result achieved by
  this repository. Do not infer it from sample data or from SWE results.
- Keep raw patches and test logs in a local evidence directory with the normal
  secret-review process; only the redacted JSONL belongs in a report.

## Bundled synthetic starter suite (P12)

`repopilot eval coding-dev plan` reads the bundled `repopilot-synthetic-dev-v1`
manifest only. It contains four small, public Python micro-fixes with visible
tests. It has no hidden tests and shares no task, patch, evaluator result or
implementation detail with the frozen SWE-bench-Live holdout.

The runner is deliberately opt-in because it can spend cloud API quota. It
creates a new artifact workspace for every fixture and never writes to the
opened project. It exposes only bounded file discovery/read/search, exact
`apply_patch`, the immutable fixture test command and a baseline diff; generic
shell, web, MCP, reviewer and subagent tools are absent.

```powershell
# Offline: inspect fixture IDs and paths only.
repopilot eval coding-dev plan

# Explicit paid/provider action. This is a development diagnostic, not a benchmark.
repopilot --provider deepseek --trust eval coding-dev run --allow-fixture-edits `
  --run-id dev_20260823 --max-cases 4
```

The command requires all three explicit choices: a cloud `--provider`,
`--trust`, and `--allow-fixture-edits`. It stores redacted funnel records under
`artifacts/coding_development/runs/<run-id>/outcomes.redacted.jsonl` and a separate
`run.receipt.json`. The receipt contains only the suite digest, selected case IDs,
provider/model names, an explicit runtime/budget profile, bounded per-case counters, statuses and redacted failure reasons;
it excludes prompts, patches, tool payloads, test logs and credentials. Raw local runtime
traces remain beside each synthetic fixture.

Use the following **offline** command before choosing P17 work:

```powershell
repopilot eval coding-dev report .\artifacts\coding_development\runs\dev_20260823
```

It validates that the receipt and funnel agree, then maps observed failures to a small
diagnostic backlog. It does not call a provider or execute a command. Make at most one
bounded runtime/prompt change per selected failure mode, add a scripted-provider regression
test, and repeat the same suite digest, selected case IDs, provider and model. A valid run may
have zero verified cases—the output is evidence for iteration, never a capability claim.

## Receipt comparison and input boundary (P28–P29)

Use the following offline command to compare two already-completed synthetic runs:

```powershell
repopilot eval coding-dev compare .\artifacts\coding_development\runs\baseline `
  .\artifacts\coding_development\runs\candidate
```

The report includes per-case category changes and category-count deltas. It is marked
`controlled` only when the suite ID/digest, selected case IDs, provider, model and explicit
`execution_profile` match exactly. Otherwise it is `directional_only`; that means an observed
change can guide the next local regression test, but cannot support a comparative capability
claim. The P21-to-P23 result is deliberately directional because P21 predates the explicit
execution-profile field.

Both `report` and `compare` accept only a bounded schema-v1 receipt: no unknown top-level or
profile fields, no raw prompt/patch/tool/test payloads, bounded strings and counters, and exact
receipt-to-JSONL task ID alignment. Invalid or tampered receipts are rejected rather than echoed.

## Outcome integrity and reproducibility (P31–P32)

New runs additionally bind the exact bytes of `outcomes.redacted.jsonl` with an
`outcomes_sha256` field and record the fixed fixture verification command in the execution
profile. Check the local evidence without contacting a provider:

```powershell
repopilot eval coding-dev verify-receipt .\artifacts\coding_development\runs\dev_20260823
```

`bound_verified` means the current redacted JSONL matches its receipt. `bound_mismatch` means it
does not, and `report`/`compare` reject it. Receipts produced before this addition remain readable
as `legacy_unbound`; they are not silently upgraded or treated as reproducibly bound evidence.
