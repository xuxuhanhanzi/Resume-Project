"""Typed schemas shared by the ForgeMM data and verifier layers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

SCHEMA_VERSION = "1.3.0"
ChartQASplit = Literal["train", "val", "test"]
ChartQASource = Literal["human", "augmented"]


@dataclass(frozen=True)
class ChartQARecord:
    """One official ChartQA question with traceable local assets."""

    record_id: str
    split: ChartQASplit
    source: ChartQASource
    index: int
    image_name: str
    question: str
    answer: str
    image_path: Path
    annotation_path: Path
    table_path: Path


@dataclass(frozen=True)
class ChartQAProRecord:
    """One ChartQAPro question without eagerly materializing image bytes."""

    record_id: str
    split: Literal["test"]
    row_index: int
    question_index: int
    question: str
    answer: str
    question_type: str
    years: tuple[str, ...]
    paragraph: str
    valid: bool = True
    exclusion_reason: str | None = None


@dataclass(frozen=True)
class EvidenceCell:
    """A chart table cell addressable by a structured prediction."""

    evidence_id: str
    row: str
    column: str
    value: str


@dataclass(frozen=True)
class OperationArgument:
    """A named operation operand; duplicate names are intentionally supported."""

    name: str
    value: str
    is_reference: bool


@dataclass(frozen=True)
class Operation:
    """A parsed whitelist operation."""

    name: str
    arguments: tuple[OperationArgument, ...]


@dataclass(frozen=True)
class EvidenceRecord:
    """Versioned, auditable record used to build training and evaluation data."""

    record_id: str
    dataset: str
    split: str
    source: str
    image_path: str
    image_sha256: str
    question: str
    reference_answer: str
    cells: tuple[EvidenceCell, ...]
    gold_evidence_ids: tuple[str, ...] = ()
    gold_operation: Operation | None = None
    evidence_mask: bool = True
    operation_mask: bool = False
    conflict_status: str = "unchecked"
    exclusion_reason: str | None = None
    labeler_version: str = "none"
    builder_version: str = "table-1.0.0"
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation."""

        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> EvidenceRecord:
        """Load and validate a serialized record."""

        version = payload.get("schema_version")
        if version != SCHEMA_VERSION:
            raise ValueError(f"unsupported_schema_version:{version}")
        raw_cells = payload.get("cells")
        if not isinstance(raw_cells, list):
            raise ValueError("invalid_cells")
        cells = tuple(EvidenceCell(**cell) for cell in raw_cells)
        raw_operation = payload.get("gold_operation")
        operation = None
        if raw_operation is not None:
            arguments = tuple(OperationArgument(**arg) for arg in raw_operation["arguments"])
            operation = Operation(name=raw_operation["name"], arguments=arguments)
        return cls(
            record_id=str(payload["record_id"]),
            dataset=str(payload["dataset"]),
            split=str(payload["split"]),
            source=str(payload["source"]),
            image_path=str(payload["image_path"]),
            image_sha256=str(payload["image_sha256"]),
            question=str(payload["question"]),
            reference_answer=str(payload["reference_answer"]),
            cells=cells,
            gold_evidence_ids=tuple(str(item) for item in payload.get("gold_evidence_ids", [])),
            gold_operation=operation,
            evidence_mask=bool(payload.get("evidence_mask", True)),
            operation_mask=bool(payload.get("operation_mask", False)),
            conflict_status=str(payload.get("conflict_status", "unchecked")),
            exclusion_reason=(
                None
                if payload.get("exclusion_reason") is None
                else str(payload["exclusion_reason"])
            ),
            labeler_version=str(payload.get("labeler_version", "none")),
            builder_version=str(payload.get("builder_version", "table-1.0.0")),
            schema_version=str(version),
        )
