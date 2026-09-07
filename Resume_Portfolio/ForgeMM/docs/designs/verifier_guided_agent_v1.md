# Verifier-guided Stateful Agent v1

## Scope and status

This is the first implementation slice for the verifier-guided, bounded Agent
described in the current ForgeMM experiment plan. It is **evaluation and state
infrastructure**, not a claim of an Agent-RL result. It does not alter the
existing Structured SFT, ms-swift, GRPO, or Chart-FGRPO paths.

Implemented modules:

- `forgemm.agent.state`: immutable episode state, typed evidence ledger,
  fixed step/tool/repair budgets, canonical JSON serialization, SHA-256 state
  identity, and immutable snapshot/fork support;
- `forgemm.agent.verifier`: independent component verdict for answer,
  evidence, operation, format, and budget validity;
- unit and integration tests covering replay equivalence, snapshot isolation,
  budget guards, fabricated evidence, and no-gold-leak failure labels.

## State contract

An episode has a stable `episode_id`, `question_id`, image hash, processor
version, verifier version, and fixed `AgentBudget`. The supported action kinds
are `OBSERVE`, `EXTRACT`, `CALCULATE`, `PROPOSE`, `REPAIR`, and `FINISH`.

`EXTRACT` actions must carry deterministic input/output metadata and create one
new `EvidenceItem`. Each item is bound to a known EvidenceStore cell through
`source_evidence_id`; a source cell cannot be extracted twice. A final
`Proposal` must cite existing ledger IDs, include a whitelisted executable
operation, and include a non-empty answer.

`SnapshotStore.save()` serializes the state canonically and records its digest.
`fork()` reconstructs the stored bytes and sets a parent link for one repair
branch; it never mutates or overwrites the parent snapshot. Replaying the same
transition records must yield the same state SHA-256.

## Verifier behavior

The verifier receives the `EvidenceRecord` only on the evaluation side. It
returns category labels such as `ANSWER_MISMATCH`, `EVIDENCE_MISMATCH`,
`INVALID_OPERATION`, and `TOOL_BUDGET_EXCEEDED`; labels intentionally contain
no reference answer, cell value, crop, or source location.

For evidence-masked records, `evidence_ok` is `None` rather than a successful
score. Otherwise a full pass requires all of the following:

1. a completed, structurally valid trajectory;
2. an answer matching the reference answer;
3. cited ledger facts matching their bound source cells and the frozen gold
   evidence set when one is present;
4. an executable proposal operation whose result matches the proposed answer;
5. no step, tool, or repair budget violation.

The format result and budget result are deliberately separate so an episode can
be well formed yet still fail for excessive cost.

## Compatibility and next gate

The existing XML Structured SFT protocol remains unchanged. This state contract
is the internal representation for a later JSON/action decoder adapter, not a
second incompatible training format. A rollout integration may only be added
after it maps model actions to `AgentAction`/`EvidenceItem` deterministically
and persists the exact snapshot payloads, tool transcripts, policy version,
and verifier version.

Before using `VerificationVerdict.full_pass` as an RL reward, add fixtures for
the real OCR/crop and calculator adapters and run the frozen adversarial set
against the exact deployment tool versions. The present tests prove local state
and verifier invariants only; they do not validate visual extraction quality or
report a model improvement.

## Quality gate

```powershell
python -B -m pytest -p no:cacheprovider
python -B -m ruff check --no-cache .
python -B -m mypy --no-incremental
```
