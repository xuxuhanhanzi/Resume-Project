# RepoPilot v2 Compatibility Contract

## Preserved v1 entry points

The following behavior is a regression boundary for v2 development:

- `repopilot run --task TASK --model MODEL`
- offline scripted, permission, recovery, and MCP demos
- `PublicTaskSpec` / `EvaluatorTaskSpec` validation
- deterministic verifier and hidden-test separation
- checkpoint and completed-tool journal behavior for benchmark tasks
- three-domain benchmark adapters and evaluation reports

The v1 `AgentRuntime` name remains available as a compatibility facade while
the implementation moves behind `TaskRuntime` and `AgentKernel`.

## Deliberate non-compatibility

Interactive sessions do not accept a `PublicTaskSpec`. They use a real project
workspace and a persistent session record instead. A user-selected
verification strategy replaces the v1 rule that every final answer must have
changed a file and passed an immutable task command.

## Regression gates

Every v2 stage must run:

```powershell
python scripts/dev.py format-check
python scripts/dev.py lint
python scripts/dev.py typecheck
python scripts/dev.py test
python -m repopilot demo scripted
python -m repopilot demo permission
python -m repopilot demo recovery
python -m repopilot demo mcp
```

The live Docker security gate remains opt-in and must be recorded separately.
