"""Human-readable, local-only session export with conservative redaction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from repopilot.core.contracts import Message
from repopilot.security.redaction import redact_text
from repopilot.session.models import SessionMetadata

_MAX_EXPORT_BYTES = 2_000_000


@dataclass(frozen=True, slots=True)
class ExportPreview:
    """A complete, no-write description of one session export."""

    target: Path
    content: str
    message_count: int

    @property
    def byte_count(self) -> int:
        return len(self.content.encode("utf-8"))


def preview_session_export(
    metadata: SessionMetadata,
    messages: list[Message],
    *,
    export_directory: Path,
    filename: str | None = None,
) -> ExportPreview:
    """Render an export without creating a file or overwriting an existing one."""
    target = _export_target(metadata, export_directory, filename)
    if target.exists():
        raise ValueError(f"export already exists: {target.name}; choose another filename")
    content = _render_markdown(metadata, messages)
    if len(content.encode("utf-8")) > _MAX_EXPORT_BYTES:
        raise ValueError(f"export exceeds the {_MAX_EXPORT_BYTES:,}-byte safety limit")
    return ExportPreview(target, content, len(messages))


def write_session_export(preview: ExportPreview) -> Path:
    """Create one new export file without replacing an existing user file."""
    preview.target.parent.mkdir(parents=True, exist_ok=True)
    try:
        with preview.target.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(preview.content)
    except FileExistsError as error:
        raise ValueError(f"export already exists: {preview.target.name}") from error
    return preview.target


def _export_target(metadata: SessionMetadata, export_directory: Path, filename: str | None) -> Path:
    raw_name = filename.strip() if filename is not None else ""
    name = raw_name or f"conversation-{metadata.session_id[:8]}.md"
    if (
        len(name) > 120
        or name in {".", ".."}
        or name.startswith(".")
        or "/" in name
        or "\\" in name
        or any(ord(character) < 32 for character in name)
    ):
        raise ValueError("export filename must be a plain filename under 120 characters")
    suffix = Path(name).suffix.casefold()
    if not suffix:
        name += ".md"
    elif suffix not in {".md", ".txt"}:
        raise ValueError("export filename must end in .md or .txt")
    root = export_directory.resolve()
    target = (root / name).resolve(strict=False)
    try:
        target.relative_to(root)
    except ValueError as error:
        raise ValueError("export path must remain in the local session export directory") from error
    return target


def _render_markdown(metadata: SessionMetadata, messages: list[Message]) -> str:
    """Render human messages while omitting hidden reasoning and tool payloads."""
    provider = redact_text(metadata.provider or "unknown")
    model = redact_text(metadata.model)
    lines = [
        "# RepoPilot conversation",
        "",
        f"- Session: `{metadata.session_id}`",
        f"- Project: `{redact_text(str(metadata.project_root))}`",
        f"- Model: `{provider}/{model}`",
        "- Export: local-only; common credential forms are redacted again on export.",
        "",
    ]
    for message in messages:
        if message.role == "system":
            continue
        if message.role == "user":
            heading = "## You"
        elif message.role == "assistant":
            heading = "## RepoPilot"
        else:
            tool_name = redact_text(message.name or "tool")
            lines.extend((f"## Tool: `{tool_name}`", "", "Tool payload omitted from export.", ""))
            continue
        content = redact_text(message.content).strip() or "_No text content._"
        lines.extend((heading, "", content, ""))
        if message.role == "assistant" and message.tool_calls:
            tool_names = ", ".join(f"`{redact_text(call.name)}`" for call in message.tool_calls)
            lines.extend((f"Requested tools: {tool_names}", ""))
    return "\n".join(lines).rstrip() + "\n"
