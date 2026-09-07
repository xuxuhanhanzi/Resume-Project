# RepoPilot v1.8.0

## P3: Safe local session export

- Added Claude-Code-style `/export [name.md]` for the current interactive session. It first shows a
  no-write preview with message count, byte count, and final local path; creating the file requires
  a separate `y` confirmation.
- Exported Markdown/TXT files are written only below the current session's RepoPilot state directory,
  not into the repository. Filenames cannot contain paths, are limited to Markdown or text extensions,
  and existing files are never overwritten.
- The export renders human conversation messages, omits tool payloads and opaque provider reasoning,
  and applies credential redaction a second time before writing.
- A durable `session_exported` audit event records filename, message count, and byte count without
  recording exported content. Tests cover redaction, path rejection, overwrite refusal, and the CLI's
  explicit confirmation flow.
