"""Lazy, traceable ChartQA loader."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import cast

from forgemm.data.schemas import ChartQARecord, ChartQASource, ChartQASplit

_SPLITS = {"train", "val", "test"}
_SOURCES = {"human", "augmented"}


class ChartQALoader:
    """Enumerate official ChartQA JSON records and validate their linked assets."""

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)

    def iter_records(self, split: str, source: str = "human") -> Iterator[ChartQARecord]:
        """Yield records from one frozen split/source pair."""

        if split not in _SPLITS:
            raise ValueError(f"invalid_split:{split}")
        if source not in _SOURCES:
            raise ValueError(f"invalid_source:{source}")
        typed_split = cast(ChartQASplit, split)
        typed_source = cast(ChartQASource, source)
        split_dir = self.root / split
        qa_path = split_dir / f"{split}_{source}.json"
        if not qa_path.is_file():
            raise FileNotFoundError(f"missing_qa_file:{qa_path}")
        payload = json.loads(qa_path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError(f"invalid_qa_payload:{qa_path}")
        for index, item in enumerate(payload):
            if not isinstance(item, dict):
                raise ValueError(f"invalid_qa_record:{split}:{source}:{index}")
            image_name = str(item.get("imgname", ""))
            question = str(item.get("query", ""))
            answer = str(item.get("label", ""))
            if not image_name or not question or not answer:
                raise ValueError(f"missing_qa_field:{split}:{source}:{index}")
            stem = Path(image_name).stem
            image_path = split_dir / "png" / image_name
            annotation_path = split_dir / "annotations" / f"{stem}.json"
            table_path = split_dir / "tables" / f"{stem}.csv"
            for kind, path in (
                ("image", image_path),
                ("annotation", annotation_path),
                ("table", table_path),
            ):
                if not path.is_file():
                    raise FileNotFoundError(f"missing_{kind}:{split}:{source}:{index}:{path}")
            yield ChartQARecord(
                record_id=f"chartqa:{split}:{source}:{index}:{stem}",
                split=typed_split,
                source=typed_source,
                index=index,
                image_name=image_name,
                question=question,
                answer=answer,
                image_path=image_path,
                annotation_path=annotation_path,
                table_path=table_path,
            )
