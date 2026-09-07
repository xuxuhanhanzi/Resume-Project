"""Bounded readers for notebooks, PDFs, and image metadata without broad file access."""

from __future__ import annotations

import json
import struct
from pathlib import Path

from repopilot.core.contracts import ErrorType, ToolCall, ToolResult, ToolSpec
from repopilot.security.paths import resolve_workspace_path
from repopilot.tools.base import ToolContext


def _path(call: ToolCall, context: ToolContext, suffix: str, *, max_bytes: int) -> Path:
    raw = call.arguments.get("path")
    if not isinstance(raw, str):
        raise ValueError("path must be a string")
    path = resolve_workspace_path(context.task, raw)
    if not path.is_file() or path.suffix.casefold() != suffix or path.stat().st_size > max_bytes:
        raise ValueError(
            f"reader requires a {suffix} file no larger than {max_bytes // 1_000_000} MB"
        )
    return path


class ReadNotebookTool:
    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            "read_notebook",
            "Read bounded Jupyter notebook cell sources without executing cells.",
            {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "max_cells": {"type": "integer", "minimum": 1, "maximum": 100},
                },
                "required": ["path"],
                "additionalProperties": False,
            },
        )

    async def run(self, call: ToolCall, context: ToolContext) -> ToolResult:
        try:
            path = _path(call, context, ".ipynb", max_bytes=2_000_000)
            raw = json.loads(path.read_text(encoding="utf-8"))
            cells = raw.get("cells", []) if isinstance(raw, dict) else []
            if not isinstance(cells, list):
                raise ValueError("notebook cells must be a list")
            max_cells = call.arguments.get("max_cells", 30)
            if not isinstance(max_cells, int) or not 1 <= max_cells <= 100:
                raise ValueError("max_cells must be 1-100")
            result = []
            for index, cell in enumerate(cells[:max_cells]):
                if not isinstance(cell, dict):
                    continue
                source = cell.get("source", [])
                text = (
                    "".join(source)
                    if isinstance(source, list) and all(isinstance(item, str) for item in source)
                    else str(source)
                )
                result.append(
                    {
                        "index": index,
                        "type": str(cell.get("cell_type", "")),
                        "source": text[:12_000],
                    }
                )
            return ToolResult(
                call.call_id,
                call.name,
                True,
                {"path": path.name, "cells": result, "total_cells": len(cells)},
            )
        except (OSError, UnicodeDecodeError, ValueError, json.JSONDecodeError) as error:
            return ToolResult(
                call.call_id, call.name, False, error=str(error), error_type=ErrorType.VALIDATION
            )


class ReadPdfTool:
    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            "read_pdf",
            "Extract text from up to 20 pages of a PDF using the optional local pypdf dependency.",
            {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "start_page": {"type": "integer", "minimum": 1},
                    "end_page": {"type": "integer", "minimum": 1, "maximum": 20},
                },
                "required": ["path"],
                "additionalProperties": False,
            },
        )

    async def run(self, call: ToolCall, context: ToolContext) -> ToolResult:
        try:
            path = _path(call, context, ".pdf", max_bytes=20_000_000)
            try:
                from pypdf import PdfReader  # type: ignore[import-not-found]
            except ImportError as error:
                raise ValueError("read_pdf requires optional dependency pypdf") from error
            reader = PdfReader(str(path))
            start = call.arguments.get("start_page", 1)
            end = call.arguments.get("end_page", min(len(reader.pages), 20))
            if (
                not isinstance(start, int)
                or not isinstance(end, int)
                or not 1 <= start <= end <= min(len(reader.pages), 20)
            ):
                raise ValueError("PDF page range must be within the first 20 pages")
            pages = [
                reader.pages[index - 1].extract_text() or "" for index in range(start, end + 1)
            ]
            return ToolResult(
                call.call_id,
                call.name,
                True,
                {
                    "path": path.name,
                    "pages": {
                        str(start + index): text[:20_000] for index, text in enumerate(pages)
                    },
                },
            )
        except (OSError, ValueError) as error:
            return ToolResult(
                call.call_id, call.name, False, error=str(error), error_type=ErrorType.VALIDATION
            )


class ReadImageMetadataTool:
    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            "read_image_metadata",
            "Read PNG or JPEG dimensions and MIME type; image pixels are not sent "
            "to a text-only model.",
            {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
                "additionalProperties": False,
            },
        )

    async def run(self, call: ToolCall, context: ToolContext) -> ToolResult:
        try:
            raw = call.arguments.get("path")
            if not isinstance(raw, str):
                raise ValueError("path must be a string")
            path = resolve_workspace_path(context.task, raw)
            if not path.is_file() or path.stat().st_size > 10_000_000:
                raise ValueError("image metadata requires a file no larger than 10 MB")
            with path.open("rb") as handle:
                header = handle.read(32)
            if header.startswith(b"\x89PNG\r\n\x1a\n") and len(header) >= 24:
                width, height = struct.unpack(">II", header[16:24])
                mime = "image/png"
            elif header.startswith(b"\xff\xd8"):
                width, height = _jpeg_dimensions(path)
                mime = "image/jpeg"
            else:
                raise ValueError("only PNG and JPEG metadata are supported")
            return ToolResult(
                call.call_id,
                call.name,
                True,
                {"path": raw, "mime_type": mime, "width": width, "height": height},
            )
        except (OSError, ValueError, struct.error) as error:
            return ToolResult(
                call.call_id, call.name, False, error=str(error), error_type=ErrorType.VALIDATION
            )


def _jpeg_dimensions(path: Path) -> tuple[int, int]:
    with path.open("rb") as handle:
        if handle.read(2) != b"\xff\xd8":
            raise ValueError("invalid JPEG header")
        while True:
            marker = handle.read(1)
            while marker == b"\xff":
                marker = handle.read(1)
            if not marker:
                break
            marker_id = marker[0]
            length_raw = handle.read(2)
            if len(length_raw) != 2:
                break
            length = struct.unpack(">H", length_raw)[0]
            if 0xC0 <= marker_id <= 0xC3:
                payload = handle.read(5)
                if len(payload) != 5:
                    break
                height, width = struct.unpack(">HH", payload[1:5])
                return width, height
            handle.seek(max(length - 2, 0), 1)
    raise ValueError("could not find JPEG dimensions")


def default_rich_read_tools() -> list[ReadNotebookTool | ReadPdfTool | ReadImageMetadataTool]:
    return [ReadNotebookTool(), ReadPdfTool(), ReadImageMetadataTool()]
