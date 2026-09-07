# P40 external advisory triage

This document records the local engineering disposition of the authorized P40
read-only assessment. It is not evidence that an external model executed or
verified a change. Every accepted item below was reproduced or established by
local code inspection and regression tests.

| Advisory | Local disposition | Result |
|---|---|---|
| Cancellation can lose to a simultaneously completed child process | Confirmed | Cancellation now wins when both waiters complete in the same scheduler turn. |
| `asyncio.Event` is safe to set from a UI/service thread | Confirmed unsafe | `CancellationToken` now records cancellation with a thread-safe primitive and wakes its single async loop via `call_soon_threadsafe`. |
| POSIX child processes can leave descendants | Confirmed | Local commands now create a POSIX session and cancellation signals the owned process group. |
| Byte truncation can split UTF-8 | Confirmed | Output truncation backs up to a code-point boundary and has a Unicode regression test. |
| Docker workspace bind mount is read-only | Found false | The prior bind mount was implicitly read/write. It now declares `readonly`, and the live safety probe expects writes to fail. |
| Docker bind mount has a symlink-escape vulnerability | Not accepted as stated | RepoPilot path validation already resolves and rejects workspace symlink escapes. The real verified mount issue was the missing read-only flag above; no additional symlink claim is made without a reproducer. |

The referenced platform behavior is documented by Python's asyncio
synchronization documentation (asyncio primitives are not thread-safe) and
Docker's bind-mount documentation (bind mounts are read/write by default unless
made read-only). No source content is sent by this triage process.
