"""Dataset adapters and versioned evidence records."""

from typing import TYPE_CHECKING, Any

from forgemm.data.schemas import (
    ChartQAProRecord,
    ChartQARecord,
    EvidenceCell,
    EvidenceRecord,
    Operation,
    OperationArgument,
)

if TYPE_CHECKING:
    from forgemm.data.chartqa import ChartQALoader
    from forgemm.data.chartqapro import ChartQAProLoader
    from forgemm.data.evidence_store import EvidenceStore


def __getattr__(name: str) -> Any:
    """Load adapters lazily so schema imports cannot create a circular import."""
    if name == "ChartQALoader":
        from forgemm.data.chartqa import ChartQALoader

        return ChartQALoader
    if name == "ChartQAProLoader":
        from forgemm.data.chartqapro import ChartQAProLoader

        return ChartQAProLoader
    if name == "EvidenceStore":
        from forgemm.data.evidence_store import EvidenceStore

        return EvidenceStore
    raise AttributeError(name)


__all__ = [
    "ChartQALoader",
    "ChartQAProLoader",
    "ChartQAProRecord",
    "ChartQARecord",
    "EvidenceCell",
    "EvidenceRecord",
    "EvidenceStore",
    "Operation",
    "OperationArgument",
]
