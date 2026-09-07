# RepoPilot v1.14.0

## P28 — offline comparable-run report

`repopilot eval coding-dev compare <baseline> <candidate>` compares two completed public
synthetic coding-development artifact directories without constructing a provider, executing a
test or reading a raw runtime trace. It reports the per-case funnel category changes and category
deltas together with the exact comparability facts.

The recorded P21 → P23 comparison has identical suite digest, selected cases, provider and model,
and shows `parse-boolean-token` moving from `verification_failed` to `verified` (3/4 → 4/4).
It is intentionally reported as `directional_only`, rather than a controlled capability result:
the older P21 receipt does not contain the later execution-profile field. This remains a small
public diagnostic, not a benchmark, holdout result or general coding-agent claim.

## P29 — strict receipt boundary

Offline report and compare commands now accept only bounded schema-v1 synthetic receipts. Their
top-level keys, per-case fields and execution profile are allowlisted; counters, strings, hashes,
tool names and task IDs are type/size checked; and receipt case IDs must match the redacted funnel
JSONL in order. Unknown or malformed fields cause a local validation error and are not rendered.
The receipt continues to exclude prompts, patches, tool payloads, test logs and credentials.

## P30 — regression and local release verification

The release adds unit coverage for legacy-profile comparison classification and unsupported receipt
fields. It remains DeepSeek-only for the current phase: Qwen was not configured, selected or
tested. No real RepoPilot project file was sent to DeepSeek or to another provider.

Run the no-cloud validation from the project root:

```powershell
.\scripts\release_check_windows.ps1
```

It does not modify global PATH, `$PROFILE`, Credential Manager, Anaconda metadata, or provider
credentials. Real-project cloud acceptance remains separately consent-gated by the exact project
file scope and provider.
