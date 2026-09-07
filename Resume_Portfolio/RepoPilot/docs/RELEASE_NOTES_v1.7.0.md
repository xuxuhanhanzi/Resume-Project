# RepoPilot v1.7.0

## P2: Terminal interaction baseline

- Interactive TTY sessions now use Prompt Toolkit for Windows-compatible line editing, persistent
  project-local input history, Tab completion for slash commands and bounded workspace-relative
  path completion.
- Input history is stored under the existing local RepoPilot project state, never in the repository.
  Every entry passes through the project credential redactor before it is written.
- Completion examines only one requested directory, caps candidates at 100, and refuses any path
  that resolves outside the trusted workspace. Non-interactive use, injected readers, and missing
  optional terminal support retain the original portable ``input()`` fallback.
- Model text is sanitised before both streaming and final terminal rendering, preventing model
  content from injecting terminal escape sequences.
