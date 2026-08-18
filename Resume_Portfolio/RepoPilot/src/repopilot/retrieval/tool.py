"""Agentic lexical retrieval exposed as a regular tool."""

from __future__ import annotations

from repopilot.core.contracts import JSONValue, ToolCall, ToolResult, ToolSpec
from repopilot.retrieval.bm25 import BM25CodeIndex
from repopilot.security.paths import task_path_is_visible
from repopilot.tools.base import ToolContext


class RetrieveCodeTool:
    """Let the agent decide when and how to retrieve ranked repository context."""

    @property
    def spec(self) -> ToolSpec:
        schema: dict[str, JSONValue] = {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "limit": {"type": "integer", "minimum": 1, "maximum": 20},
            },
            "required": ["query"],
            "additionalProperties": False,
        }
        return ToolSpec("retrieve_code", "Rank repository files with BM25.", schema)

    async def run(self, call: ToolCall, context: ToolContext) -> ToolResult:
        query = str(call.arguments.get("query", ""))
        limit = int(call.arguments.get("limit", 5))
        if not query.strip() or not 1 <= limit <= 20:
            return ToolResult(
                call.call_id,
                call.name,
                False,
                error="query must be non-empty and limit must be between 1 and 20",
                recoverable=True,
            )
        index = BM25CodeIndex.build(
            context.task.workspace,
            path_filter=lambda path: task_path_is_visible(context.task, path),
        )
        hits = index.search(query, limit=limit)
        return ToolResult(
            call.call_id,
            call.name,
            True,
            {
                "hits": [
                    {"path": hit.path, "score": round(hit.score, 6), "excerpt": hit.excerpt}
                    for hit in hits
                ]
            },
        )
