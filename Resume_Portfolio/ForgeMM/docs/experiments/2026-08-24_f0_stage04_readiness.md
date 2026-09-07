# Gate-F0 / Stage 4 readiness — 2026-08-24

## Inputs

- Project: ForgeMM, local working tree at the recorded run time.
- Dataset identity audit:
  `artifacts/runs/2026-08-24_f0/environment.json`.
- Full payload audit:
  `artifacts/runs/2026-08-24_f0/dataset_audit.json`.
- Strict evaluation construction: ChartQA human validation and test only.

## Confirmed facts

- All six frozen ChartQA JSON identities and the ChartQAPro parquet identity
  match their manifests.
- ChartQA contains 16 cross-split duplicate image hashes. The EvidenceStore
  records the affected samples as ineligible rather than silently admitting
  them to strict evidence/operation scoring.
- The full ChartQAPro audit found 1,948 rows, 2,709 question slots, 2,706
  non-empty valid questions, and no corrupt image payload.
- Local hardware is an RTX 4070 Laptop GPU with 8,188 MiB. The local
  environment intentionally lacks the fixed ms-swift/torch training stack;
  it is a development and audit environment, not a substitute for the
  preregistered RTX 4090 training environment.
- Existing Stage 1 AutoDL evidence remains the only verified 4090 smoke result.
  No E0–E5 quality result is claimed here.

## Strict-label result

The conservative table/annotation agreement requirement produced the following
question-level counts after labeler `rules-1.4.0` expanded deterministic global
reduction rules:

| Split | Strict records | Required formal minimum | Result |
|---|---:|---:|---|
| Train | 1,763 | 1,500 | pass |
| ChartQA val (human) | 106 | 200 | stop |
| ChartQA test (human) | 123 | 500 | stop |

The first v1.3.0 construction yielded 95 val and 108 test records; the counts
above are the current labeler output. The 2026-08-24 smoke artifacts preserve
the earlier construction and the reproducible data-gate failure record. Neither
version satisfies the preregistered val/test minimums.

## Engineering actions completed

1. Added answer-only control-data and single-view strict-evaluation builders.
2. Corrected answer-only E0/E1 evaluation to extract the answer from a
   structured reference target instead of assigning zero score solely because
   the control lacks XML blocks.
3. Added an E2 non-inferiority/format gate before any GRPO run.
4. Added a Stage 4 data gate before cloud training. It fails closed when the
   strict train/val/test record counts are below their frozen thresholds.
5. Added idempotent cloud preparation, SFT baseline, and top-level driver
   scripts. They save exit codes and do not overwrite completed artifacts.

## Decision

`stop` for formal E0–E5/A1–A2 training under the current preregistration. The
block is **data sufficiency**, not a model or code failure. Running the matrix
with 106/123 strict examples could produce exploratory diagnostics, but it may
not be described as a supported ForgeMM full-pass improvement.

## Required authorization for the next experiment

Choose one and register a new data/evaluation version before training:

1. add an independently auditable evidence-labelled evaluation set sufficient
   to meet the 200-val/500-test thresholds; or
2. define a separate, clearly named answer-only/executable-consistency study
   whose claims do not call its table-derived signals visual evidence validity.

After the data gate advances, run the prepared cloud driver on the fixed 4090
environment:

```bash
cd /root/autodl-tmp/ForgeMM
FORGEMM_ROOT=$PWD FORGEMM_ENV=/root/autodl-tmp/envs/forgemm \
  FORGEMM_MODEL=/root/autodl-tmp/models/Qwen2.5-VL-3B-Instruct \
  bash scripts/run_cloud_stage04.sh
```

The driver itself does not provision a GPU, upload data, or create a cloud
instance; those are external account actions and remain outside this run.
