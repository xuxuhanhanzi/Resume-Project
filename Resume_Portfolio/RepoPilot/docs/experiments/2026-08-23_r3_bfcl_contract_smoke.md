# R3 BFCL v4 contract-validity smoke

Status: local development smoke complete for B0 and B1.  This is explicitly
not an official BFCL AST/executable score.

## Frozen inputs and scope boundary

- Upstream source: `external/BFCL` at
  `6ea57973c7a6097fd7c5915698c54c17c5b1b6c8`.
- Dataset manifest:
  `evaluation/retrieval/bfcl/manifests/r3_bfcl_v4_20260823.json`, canonical manifest digest
  `34d746de52029dc1da52c15793dec0a0b9d482feec2873db33cc1dacedc4076f`.
- Input categories and counts: simple Python 400, parallel 200, irrelevance
  240, multi-turn base 200.  The frozen 20/40/40 split has 208/416/416 IDs.
- Development run: 168 single-turn IDs (80 simple, 40 parallel, 48
  irrelevance).  The 40 development `multi_turn_base` IDs were not run.

`multi_turn_base` requires BFCL's state engine.  The upstream evaluator cannot
be imported in this repository's minimal environment because importing its
AST checker transitively requires every cloud-provider SDK; installing and
substituting only portions of it would not create an official score.  The
omission is recorded, not counted as failure or success.

The R3-A stage-scoped configuration is also intentionally not forced onto
these arbitrary APIs.  BFCL single-turn prompts do not define a task stage;
deriving a tool sequence from the benchmark's gold path would leak evaluation
information.  Stage-scoped contracts are covered by the RepoPilot unit and
kernel integration tests instead.

## B0/B1 local smoke results

Both variants used local `qwen2.5:1.5b`, temperature 0, a 128-token generation
cap, and 60-second request cap.  B0 provided function descriptions in a strict
text prompt and parsed JSON after generation.  B1 sent the same descriptions
through Ollama's native `tools` field.  In both cases the returned calls were
checked by the same normalised RepoPilot typed contract before any execution.

| Variant | Contract-valid call rate | Irrelevance correct | p50 / p95 latency | Recoverable rejection codes |
|---|---:|---:|---:|---|
| B0 loose text JSON | 0.925676 | 0.416667 | 484.29 / 1101.06 ms | `schema_invalid`: 4; `unknown_tool`: 7 |
| B1 typed native tools | 0.815385 | 0.791667 | 433.54 / 929.03 ms | `schema_invalid`: 24 |

“Contract-valid call rate” is calculated over emitted calls after excluding
correct irrelevance abstentions; “irrelevance correct” is the separate rate of
emitting no call for those 48 prompts.  Thus the results show a trade-off in
this small local model and must not be summarized as “typed contracts improve
BFCL.”  Raw public-prompt responses and receipts are in
`artifacts/evaluations/r3/`.

## What remains necessary for the original R3 claim

An official result requires a compatible isolated BFCL evaluation environment,
the upstream AST/state engines, a valid pre-registered stage task for A, and
validation/final-holdout runs.  Until then, do not report AST accuracy,
executable accuracy, recovery-rate improvements, or B0/B1/A superiority.
