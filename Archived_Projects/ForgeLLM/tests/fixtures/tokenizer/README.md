# Tokenizer correctness fixtures

- Source: original text written for ForgeLLM tests.
- License marker: `project-test-fixture`.
- Purpose: deterministic unit, integration, CLI, round-trip, and metric checks.
- Non-purpose: this is not a representative natural-language corpus and must not support claims about production tokenizer quality.
- Split rule: `train.jsonl` learns merges; `validation.jsonl` and `test.jsonl` are evaluation-only.
- Method lab: `method_lab_evaluation.jsonl` adds explicit English, Chinese, code, number, whitespace, Emoji, and mixed subsets; it is evaluation-only and is not used to tune vocabulary size.
- Schema: each JSONL record contains exactly `id`, `subset`, and `text`.
