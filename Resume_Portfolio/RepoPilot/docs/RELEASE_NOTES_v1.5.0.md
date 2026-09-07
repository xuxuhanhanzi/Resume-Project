# RepoPilot v1.5.0

## P0: Durable revisions and quality gate

- Every returned interactive turn now appends a bounded, text-only workspace revision to the
  local session. Revisions include the durable post-turn checkpoint, context summary, and plan,
  so they remain inspectable after restarting RepoPilot.
- `/diff [turn]` renders a stored unified diff without asking a model or executing a command.
  `/rewind list` shows revision availability; `/rewind <turn> [all|code|session]` previews and
  requires an explicit terminal confirmation before any restoration.
- Rewind verifies that every affected file still matches the last RepoPilot-recorded content. A
  manual edit is reported as a conflict rather than overwritten. Revisions whose inventory was
  incomplete are not traversed. RepoPilot never automatically deletes a file during rewind; a
  requested historical state requiring deletion is reported for manual cleanup.
- The append-only transcript is deliberately retained after a rewind. The live checkpoint may be
  restored, but the audit trail continues to show the original turns and the explicit rewind event.
- Full Ruff and mypy now pass for all source and test files.
