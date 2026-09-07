# RepoPilot v1.16.0

## P38 — content-free DeepSeek streaming diagnosis

`repopilot auth probe deepseek` performs one fixed, public marker request over SSE. It never reads a
project file or session and does not accept a user prompt. Its output contains only safe transport
facts: HTTP status, completion reason, stream/choice counters, visible-text counters and whether the
SSE `DONE` marker arrived. RepoPilot now preserves the same content-free diagnostics on every model
response and adds DeepSeek's documented `thinking: {"type": "disabled"}` option only for the bounded
public probe and read-only acceptance path.

The live P38 probe completed on 2026-08-23 with HTTP 200, `finish_reason=stop`, seven visible SSE
deltas and the expected fixed marker. It transmitted no project source.

## P39–P40 — reproducible read-only acceptance

Every future `acceptance --send-project-files` command writes an intent receipt before constructing a
provider request. A final receipt is then written for successful, inconclusive and transport-failed
attempts. `acceptance show <receipt>` and `acceptance verify <receipt>` are fully offline: they report
metadata, hashes and integrity checks, never source text or the model assessment itself.

With the user-authorized P40 retry, exactly these same three files were sent to DeepSeek and no other
project content was exposed:

1. `src/repopilot/runtime/cancellation.py`
2. `src/repopilot/runtime/runner.py`
3. `tests/unit/test_runtime_cancellation.py`

The model completed normally over SSE and the final receipt plus preflight intent passed local
integrity verification. The external review is advisory only: it did not run tools, edit files, or
automatically create a code change. The recommended local cancellation/reliability tests passed
(`4 passed`).

## P41–P43 — transparent local operation and regression gates

- `/status`, `/workflow` and `/trace` now state the active provider/model, stream mode and approval
  behavior. `doctor --fix-plan` creates a recovery plan but never executes it.
- `scripts/repopilot.ps1` is a collision-proof project launcher pinned to `.venv\Scripts\repopilot.exe`.
  It does not modify PATH, `$PROFILE`, credentials or installed packages. `setup_windows.ps1` prints
  the launcher for users who prefer it to session-local shell initialization.
- Provider empty-response/stream diagnostics, acceptance intent/final receipt integrity and tamper
  detection, cancellation recovery, the project PowerShell launcher and existing safety boundaries are
  covered by the local release gate. Qwen remains unconfigured and was not called.
