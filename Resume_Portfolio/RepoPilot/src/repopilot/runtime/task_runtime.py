"""Task-runtime migration seam.

The v1 lifecycle remains available through ``AgentRuntime`` while callers move
to this explicit task-oriented name. Both use the shared ``AgentKernel``.
"""

from __future__ import annotations

from repopilot.core.loop import AgentRuntime


class TaskRuntime(AgentRuntime):
    """The benchmark and batch-task runtime during the v1-to-v2 migration."""

    pass
