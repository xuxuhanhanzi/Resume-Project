# RepoPilot v1.1.0

This release completes the P0–P4 CLI hardening pass for the Claude-Code-style learning agent.

- P0: preserves DeepSeek opaque reasoning data across native tool continuations, adds bounded rate-limit retry, and states the configured provider/model truthfully in context.
- P1: adds provider-aware session recovery, `/sessions`, `/resume`, `/clear`, `/rename`, `/history`, explicit project facts, and layered user/repository/subdirectory instructions.
- P2: adds bounded MCP resource/prompt discovery for trusted stdio servers, user/project skill layering, and inspectable declarative hooks with audited lifecycle events.
- P3: bounds parallel read-only execution, contains one read failure without dropping sibling results, and lists workspace-contained Git worktrees without exposing or deleting external worktrees.
- P4: adds an offline `doctor` command and strict public evaluation-record JSONL summary command. Neither command sends a model request or prints credentials.
- Follow-up UX hardening: interactive text sessions now stream model text by default; `--verbose` adds diagnostics and `--no-stream` restores buffered final-answer rendering.

Known boundaries remain intentional: HTTP/OAuth MCP transports are not implemented; MCP servers are stdio-only and must be explicitly enabled with `--mcp` in a trusted workspace. Project hooks remain declarative and cannot execute code.
