# RepoPilot v1.4.0

## Verification Engine 2.0

- Project verification is now a typed plan: `test`, `lint`, `typecheck`, `build`, and explicit
  `custom` checks. Discovery does not execute any process.
- `/verify list` presents available project checks, while `/verify all`, `/verify test`,
  `/verify lint`, `/verify typecheck`, and `/verify build` run only the selected checks through
  the existing approval boundary. `/verify <argv>` remains available for an explicit custom check.
- Completed reports are appended to the session's redacted `verifications.jsonl`; `/verify last`
  reads the most recent report without re-running anything.
- Python discovery recognises `pytest` plus configured Ruff and mypy sections. Node projects
  recognise conventional `test`, `lint`, `typecheck`, and `build` scripts. Existing Cargo, Go,
  and Maven test discovery remains available.

This release keeps verification user-initiated: model turns cannot silently start a verification
command merely because a plan is available.
