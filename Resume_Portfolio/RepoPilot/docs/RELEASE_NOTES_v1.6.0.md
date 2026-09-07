# RepoPilot v1.6.0

## P1: Auditable review loop

- `/review` runs a user-confirmed, read-only quality review over the current uncommitted
  `git diff --no-ext-diff --unified=3`. The reviewer receives no tools and its structured result
  is appended to the durable session transcript.
- `/security-review` uses the same explicit diff boundary but a security-focused reviewer prompt.
  It requests only concrete newly introduced vulnerabilities with an exploit path, evidence,
  severity, and confidence; style and theoretical hardening findings are excluded.
- Both paths emit durable `review_started` and `review_completed` events, preserve the current
  workspace unchanged, and skip the model call when Git reports an empty diff.
- `git_working_diff` is now a model-visible, read-only Git observation. It disables Git external
  diff execution explicitly and uses the existing tokenized runner with bounded output.
