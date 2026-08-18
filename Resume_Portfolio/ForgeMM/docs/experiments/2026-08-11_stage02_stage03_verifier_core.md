# Stage 2–3: loader and verifier core

- Date: 2026-08-11
- Status: completed
- Objective: establish a typed, deterministic CPU path from ChartQA/ChartQAPro records to
  structured parsing, safe execution, and decomposed rewards.
- Primary variable: introduction of the ForgeMM-owned data/verifier implementation.
- Frozen inputs: the dataset payloads and hashes recorded by the Stage 0 environment audit.
- Exclusions: no model inference, training, test-set tuning, or inferred gold operation labels.

## Acceptance criteria

1. ChartQA train/val/test records are lazily enumerable with stable source identifiers and
   validated image/annotation/table paths.
2. ChartQAPro is exposed only as `test`; metadata enumeration does not materialize image bytes.
3. EvidenceStore uses a versioned JSONL schema and records conflicts/exclusion reasons.
4. The parser requires exactly one evidence, operation, and answer block and never uses `eval`.
5. The executor permits only registered operations and returns stable failure codes.
6. Answer, evidence, and operation rewards are independently testable and remain in `[0, 1]`.
7. Unit/integration tests cover normal, boundary, and malformed inputs.

## Commands

```powershell
..\ForgeLLM\.venv\Scripts\python.exe scripts\dev.py check
```

## Evidence and conclusion

Full audit artifact: `artifacts/runs/stage02_data_audit/report.json`.

- ChartQA question counts: train human/augmented 7,398/20,901; val 960/960; test 1,250/1,250.
- ChartQA unique charts: train 18,317; val 1,056; test 1,509.
- Image integrity: five corrupt train images, zero corrupt val/test images.
- Split isolation: 16 image hashes occur across more than one split.
- Strict table/annotation exact charts: train 2,074; val 195; test 228. Other charts are retained
  as value conflict, shape conflict, unverifiable layout, or parse error rather than silently trusted.
- ChartQAPro: 1,948 parquet rows flatten to 2,709 slots; 2,706 are valid and three have empty
  answers. All images decode. It remains exposed only as test.

The initial chart-level stores were superseded after review showed that chart agreement is not the
same as question-level evidence. The authoritative v7 artifacts are:

- `artifacts/runs/stage03_evidence_store_v7/train_human.jsonl`: 580 labelled / 7,398 records;
- `artifacts/runs/stage03_evidence_store_v7/train_augmented.jsonl`: 1,183 labelled / 20,901 records;
- schema `1.3.0`, builder `chartqa-store-1.1.0`, labeler `rules-1.3.0`;
  combined strict labels: 1,763.

All labelled records were replayed through the whitelist executor with zero failures. Human
operation distribution: lookup 213, argmax 111, sum 67, difference 61, argmin 36, average 68,
ratio 20, count 4. Augmented: lookup 1,120, argmax 51, argmin 9, average 5, sum 1.

Two rejected intermediate artifacts are intentionally retained. v4 fixed a factorial candidate
enumeration bug; v5 was rejected because it reused ChartQA's 5% final-metric tolerance for gold
operations and consequently admitted visibly wrong reasoning such as `2014 -> 2019` and
`154 -> 161`. v6 and v7 separate final accuracy from labelling: lookup values must be exact, while an
operation result must equal the reference after rounding to the reference's displayed precision.

The duplicate audit found 1 exact duplicate in train-human and 160 in train-augmented; v7 disables
all 161 with `duplicate_qa_record`. ChartQAPro has 319 duplicates among its 2,706 valid records.
Future ChartQAPro evaluation must report both record-level and unique-QA metrics.

The result passes the 1,500-prompt GRPO minimum but not the 3,000-example Structured SFT minimum.
No training-quality or model-improvement claim is made from these data alone.
