# LongMemEval R4 adapter

This directory contains the offline, source-preserving part of R4. The
downloaded data directory is intentionally gitignored; its only accepted
dataset file is longmemeval_s_cleaned.json.

## Pinned sources

- LongMemEval source checkout: 9e0b455f4ef0e2ab8f2e582289761153549043fc
- Dataset URI:
  https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_s_cleaned.json
- Downloaded file SHA-256:
  d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442
- Count verified locally: 500 records

## Manifest command

Run this before any reader, embedding, or card-extraction experiment:

~~~powershell
.\.venv\Scripts\python.exe -m repopilot.evaluation.longmemeval --data evaluation\memory\longmemeval\data\longmemeval_s_cleaned.json --manifest evaluation\memory\longmemeval\manifests\longmemeval_s_cleaned_v1.json --dataset-version huggingface-longmemeval-cleaned@2026-08-23
~~~

The adapter treats every shared haystack_session_id as a leakage edge, as
specified in the experiment plan. On the current official cleaned file this
creates one 500-question connected component because filler sessions are
reused. The command therefore fails before writing a nominal 20/40/40
manifest. The attempted diagnostic is retained locally at
artifacts/dataset_receipts/longmemeval_s_group_leakage_diagnostic_2026-08-23.json.

Do not remove that guard or use the one-component result as a final holdout.
A revised grouping rule needs an explicit experimental-plan amendment before
R4 can proceed.

## Implemented offline boundaries

- repopilot.evaluation.longmemeval validates the official schema and
  calculates session Recall@K and MRR without an LLM judge.
- repopilot.memory.provenance provides the fixed TPM card schema,
  source-turn validation, and content-addressed extraction cache.
- repopilot.evidence creates immutable dataset/run identities and Wilson or
  paired-bootstrap intervals.
