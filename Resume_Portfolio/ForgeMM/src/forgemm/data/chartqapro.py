"""Metadata-first ChartQAPro loader with a frozen test-only boundary."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from forgemm.data.schemas import ChartQAProRecord

_METADATA_COLUMNS = ["Question", "Answer", "Question Type", "Year", "Paragraph"]


class ChartQAProLoader:
    """Enumerate ChartQAPro questions without reading its image column."""

    def __init__(self, parquet_path: Path | str) -> None:
        self.parquet_path = Path(parquet_path)
        if not self.parquet_path.is_file():
            raise FileNotFoundError(f"missing_chartqapro:{self.parquet_path}")

    def iter_records(self, split: str = "test") -> Iterator[ChartQAProRecord]:
        """Yield flattened questions. ChartQAPro is never exposed as a training split."""

        if split != "test":
            raise ValueError("chartqapro_test_only")
        parquet = pq.ParquetFile(self.parquet_path)
        row_index = 0
        for batch in parquet.iter_batches(columns=_METADATA_COLUMNS):
            for row in batch.to_pylist():
                yield from self._records_for_row(row, row_index)
                row_index += 1

    @staticmethod
    def _records_for_row(row: dict[str, Any], row_index: int) -> Iterator[ChartQAProRecord]:
        questions = tuple(
            "" if value is None else str(value) for value in row.get("Question") or ()
        )
        answers = tuple("" if value is None else str(value) for value in row.get("Answer") or ())
        years = tuple(str(value) for value in row.get("Year") or ())
        for question_index in range(max(len(questions), len(answers))):
            question = questions[question_index] if question_index < len(questions) else ""
            answer = answers[question_index] if question_index < len(answers) else ""
            reason: str | None = None
            if len(questions) != len(answers):
                reason = "question_answer_length_mismatch"
            elif not question:
                reason = "empty_question"
            elif not answer:
                reason = "empty_answer"
            yield ChartQAProRecord(
                record_id=f"chartqapro:test:{row_index}:{question_index}",
                split="test",
                row_index=row_index,
                question_index=question_index,
                question=question,
                answer=answer,
                question_type=str(row.get("Question Type") or ""),
                years=years,
                paragraph=str(row.get("Paragraph") or ""),
                valid=reason is None,
                exclusion_reason=reason,
            )
