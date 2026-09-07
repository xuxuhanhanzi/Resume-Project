# RepoPilot v1.15.0

## P31–P32 — bound diagnostic evidence

New synthetic coding-development receipts store the SHA-256 of their exact
`outcomes.redacted.jsonl` bytes and record the fixture verification command together with the
existing runtime version, budget and tool-surface profile. `repopilot eval coding-dev
verify-receipt <run-directory>` is offline and distinguishes `bound_verified`, `bound_mismatch`
and `legacy_unbound`. A bound mismatch is rejected by report and compare; the existing P21/P23
receipts intentionally remain historical, unmodified evidence rather than being rewritten.

## P33–P36 — reliable local operation

- Cancellation still creates a durable checkpoint; interactive output now explicitly tells the
  user that `/status` can be inspected before continuing with a new prompt.
- Startup and `/model` copy is DeepSeek-first for the present configuration. Qwen was not
  configured, selected or tested.
- `shell-doctor` now supports JSONL for automation and, when started via `python -m repopilot`,
  correctly emits the active venv `repopilot.exe` session-only PowerShell command.
- No global PATH, PowerShell profile, credentials, Anaconda installation or stale pip metadata was
  automatically changed.

## P37 — explicit-scope DeepSeek acceptance command

`repopilot --provider deepseek --trust acceptance --send-project-files --include <relative-path>`
creates one no-tool, no-edit model request from only the named project-relative source/test files.
It accepts one to four files (at most 16 KB each and 48 KB total), excludes `.git`, `.repopilot`,
`.venv`, `artifacts`, cache directories and common secret names, and writes a redacted local
receipt containing only file metadata/digests, outcome metadata and, when one exists, the
assessment.

This release does **not** claim a successful project-content cloud acceptance result. Two separately
authorized attempts — first non-streaming, then one streaming retry — sent only the following exact
files. DeepSeek returned no final user-facing assessment in either attempt:

1. `src/repopilot/runtime/cancellation.py`
2. `src/repopilot/runtime/runner.py`
3. `tests/unit/test_runtime_cancellation.py`

The provider endpoint and credential were independently verified without source content. P37 uses the
same streaming response path as normal interactive sessions, and its local tests pass. Future
inconclusive responses now produce a local redacted audit receipt (metadata, hashes and outcome only,
not source text) before the command fails. No third transmission will occur without a new, separate
user decision. The corresponding local cancellation and session-reliability tests are part of the
release gate.
