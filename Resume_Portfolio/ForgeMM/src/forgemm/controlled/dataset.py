"""Pillow-only deterministic generator and audit for ForgeMM-Controlled v1-Lite."""

from __future__ import annotations

import hashlib
import io
import json
import random
from collections.abc import Iterable
from dataclasses import replace
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from forgemm.controlled.protocol import (
    ControlledRecord,
    VisualEvidence,
    render_controlled_completion,
    verify_controlled_completion,
)
from forgemm.data.schemas import EvidenceCell, Operation, OperationArgument
from forgemm.reasoning.executor import execute

DATASET_VERSION = "forgemm-controlled-v1-lite"
GENERATOR_VERSION = "forgemm-controlled-pillow-1.0.0"
IMAGE_SIZE = (560, 420)
DEFAULT_COUNTS = {"train": 5000, "val": 600, "test": 1000}
_CATEGORIES = (
    "Aster",
    "Birch",
    "Cedar",
    "Dahlia",
    "Elm",
    "Fir",
    "Ginkgo",
    "Hazel",
    "Iris",
    "Juniper",
    "Kite",
    "Linden",
    "Maple",
    "Nile",
    "Orchid",
    "Pine",
    "Quartz",
    "Reed",
    "Sage",
    "Tulip",
)
_PALETTES = (
    ("#1d4ed8", "#60a5fa", "#0f766e", "#f59e0b", "#dc2626"),
    ("#7c3aed", "#a78bfa", "#db2777", "#fb7185", "#0f766e"),
    ("#0369a1", "#0284c7", "#65a30d", "#ca8a04", "#c2410c"),
)
_OPERATIONS = ("lookup", "difference", "sum", "average", "argmax", "argmin", "compare")


def create_controlled_dataset(
    output_root: str | Path,
    *,
    project_root: str | Path,
    counts: dict[str, int] | None = None,
    root_seed: int = 20260824,
) -> dict[str, Any]:
    """Generate immutable images, gold manifests, and model-facing data views.

    The output directory must not already contain files.  This fail-closed rule
    prevents a later invocation from silently replacing a frozen test set.
    """

    output = Path(output_root).resolve()
    project = Path(project_root).resolve()
    selected_counts = dict(DEFAULT_COUNTS if counts is None else counts)
    _validate_counts(selected_counts)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"controlled_dataset_exists:{output}")
    output.mkdir(parents=True, exist_ok=True)
    manifests = output / "manifests"
    images = output / "images"
    views = output / "views"
    manifests.mkdir()
    images.mkdir()
    views.mkdir()

    records_by_split: dict[str, list[ControlledRecord]] = {}
    for split in ("train", "val", "test"):
        split_dir = images / split
        split_dir.mkdir()
        records = []
        for index in range(selected_counts[split]):
            record = make_controlled_record(
                split=split,
                index=index,
                root_seed=root_seed,
                dataset_root=output,
                project_root=project,
            )
            image = render_chart(record)
            image_target = output / "images" / split / f"{index:06d}.png"
            image_target.parent.mkdir(parents=True, exist_ok=True)
            image_target.write_bytes(image)
            records.append(record)
        _write_jsonl(manifests / f"{split}_oracle.jsonl", (item.to_dict() for item in records))
        records_by_split[split] = records

    _write_jsonl(views / "sft_train.jsonl", (_sft_row(item) for item in records_by_split["train"]))
    _write_jsonl(
        views / "answer_sft_train.jsonl",
        (_answer_sft_row(item) for item in records_by_split["train"]),
    )
    _write_jsonl(
        views / "grpo_train.jsonl", (_grpo_row(item) for item in records_by_split["train"])
    )
    for split in ("val", "test"):
        _write_jsonl(
            views / f"{split}_prompts.jsonl",
            (_prompt_row(item) for item in records_by_split[split]),
        )

    summary = {
        "dataset_version": DATASET_VERSION,
        "generator_version": GENERATOR_VERSION,
        "root_seed": root_seed,
        "counts": selected_counts,
        "images": sum(selected_counts.values()),
        "project_relative_output": _relative_to_project(output, project),
        "files": {
            path.relative_to(output).as_posix(): _identity(path)
            for path in sorted(manifests.glob("*.jsonl")) + sorted(views.glob("*.jsonl"))
        },
        "claim_boundary": (
            "controlled synthetic charts have generator-backed visual regions; "
            "ChartQA remains a separate natural-chart answer-generalization evaluation"
        ),
    }
    _write_json(output / "dataset_summary.json", summary)
    return summary


def make_controlled_record(
    *,
    split: str,
    index: int,
    root_seed: int,
    dataset_root: Path,
    project_root: Path,
) -> ControlledRecord:
    """Create one record without writing any data to disk."""

    seed = _sample_seed(root_seed, split, index)
    rng = random.Random(seed)
    categories = tuple(rng.sample(_CATEGORIES, 5))
    values = tuple(rng.sample(range(12, 96), len(categories)))
    chart_type = "bar" if rng.randrange(2) == 0 else "line"
    palette_index = rng.randrange(len(_PALETTES))
    operation_name = _OPERATIONS[index % len(_OPERATIONS)]
    evidence_ids, operation, question = _task_for_operation(operation_name, categories, values, rng)
    image_relative = Path("images") / split / f"{index:06d}.png"
    provisional = ControlledRecord(
        record_id=f"{DATASET_VERSION}:{split}:{index:06d}",
        split=split,
        image_path=_relative_to_project(dataset_root / image_relative, project_root),
        image_sha256="pending",
        question=question,
        reference_answer="pending",
        chart_type=chart_type,
        image_size=IMAGE_SIZE,
        generation_seed=seed,
        categories=categories,
        values=values,
        palette_index=palette_index,
        evidence=(),
        gold_evidence_ids=(),
        gold_operation=operation,
    )
    evidence = _visual_evidence(provisional)
    provisional = replace(provisional, evidence=evidence, gold_evidence_ids=evidence_ids)
    answer = _gold_answer(provisional)
    final = replace(provisional, reference_answer=answer)
    image = render_chart(final)
    return replace(final, image_sha256=_sha256_bytes(image))


def render_chart(record: ControlledRecord) -> bytes:
    """Render a record from its frozen public generator fields."""

    image = Image.new("RGB", record.image_size, "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=16)
    small_font = ImageFont.load_default(size=13)
    title_font = ImageFont.load_default(size=20)
    plot = (72, 60, 530, 335)
    _draw_axes(draw, plot, font, small_font)
    colors = _PALETTES[record.palette_index]
    if record.chart_type == "bar":
        _draw_bars(draw, record, plot, colors, font, small_font)
    elif record.chart_type == "line":
        _draw_line(draw, record, plot, colors, font, small_font)
    else:
        raise ValueError(f"unsupported_chart_type:{record.chart_type}")
    title = "Quarterly Signal by Category"
    draw.text((72, 20), title, fill="#111827", font=title_font)
    draw.text((72, 380), "Synthetic controlled chart · units", fill="#374151", font=small_font)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=False)
    return buffer.getvalue()


def load_controlled_records(path: str | Path) -> list[ControlledRecord]:
    """Read one oracle manifest and fail on malformed lines."""

    records = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"invalid_controlled_manifest_line:{line_number}")
            records.append(ControlledRecord.from_dict(payload))
    return records


def audit_controlled_dataset(
    output_root: str | Path, *, expected_counts: dict[str, int] | None = None
) -> dict[str, Any]:
    """Perform deterministic render, oracle, split, and attack-set checks."""

    output = Path(output_root).resolve()
    counts = dict(DEFAULT_COUNTS if expected_counts is None else expected_counts)
    _validate_counts(counts)
    split_records: dict[str, list[ControlledRecord]] = {}
    render_mismatches = 0
    gold_failures = 0
    image_hashes: set[str] = set()
    duplicate_images = 0
    for split in ("train", "val", "test"):
        records = load_controlled_records(output / "manifests" / f"{split}_oracle.jsonl")
        if len(records) != counts[split]:
            raise ValueError(f"controlled_count_mismatch:{split}:{len(records)}:{counts[split]}")
        if any(record.split != split for record in records):
            raise ValueError(f"controlled_split_mismatch:{split}")
        for record in records:
            image_path = output.parent.parent / record.image_path
            if not image_path.is_file():
                raise FileNotFoundError(f"controlled_image_missing:{image_path}")
            actual = image_path.read_bytes()
            if _sha256_bytes(actual) != record.image_sha256 or actual != render_chart(record):
                render_mismatches += 1
            if record.image_sha256 in image_hashes:
                duplicate_images += 1
            image_hashes.add(record.image_sha256)
            verdict = verify_controlled_completion(record, render_controlled_completion(record))
            if not verdict.full_pass:
                gold_failures += 1
            _validate_visual_bounds(record)
        split_records[split] = records
    _check_prompt_views(output, split_records)
    _check_training_views(output, split_records["train"])
    attack_result = _attack_audit(split_records)
    decision = (
        "pass"
        if not any((render_mismatches, gold_failures, duplicate_images))
        and all(attack_result.values())
        else "stop"
    )
    return {
        "dataset_version": DATASET_VERSION,
        "generator_version": GENERATOR_VERSION,
        "counts": {split: len(records) for split, records in split_records.items()},
        "strict_minimums": {"train": 1500, "val": 200, "test": 500},
        "render_mismatches": render_mismatches,
        "gold_full_pass_failures": gold_failures,
        "duplicate_image_hashes": duplicate_images,
        "view_identities": {
            path.name: _identity(path) for path in sorted((output / "views").glob("*.jsonl"))
        },
        "attacks_rejected": attack_result,
        "decision": decision,
        "claim_boundary": (
            "pass supports controlled visual-evidence evaluation only; it does not relabel "
            "or replace natural ChartQA visual-grounding evidence"
        ),
    }


def _task_for_operation(
    name: str,
    categories: tuple[str, ...],
    values: tuple[int, ...],
    rng: random.Random,
) -> tuple[tuple[str, ...], Operation, str]:
    ids = tuple(f"mark-{index + 1}" for index in range(len(categories)))
    if name == "lookup":
        index = rng.randrange(len(categories))
        return (
            (ids[index],),
            Operation("lookup", (OperationArgument("ref", ids[index], True),)),
            f"What is the value for {categories[index]}?",
        )
    if name == "difference":
        first, second = rng.sample(range(len(categories)), 2)
        if values[first] < values[second]:
            first, second = second, first
        return (
            (ids[first], ids[second]),
            Operation(
                "difference",
                (
                    OperationArgument("ref", ids[first], True),
                    OperationArgument("ref", ids[second], True),
                ),
            ),
            f"What is the difference between {categories[first]} and {categories[second]}?",
        )
    if name == "sum":
        first, second = rng.sample(range(len(categories)), 2)
        return (
            (ids[first], ids[second]),
            Operation(
                "sum",
                (
                    OperationArgument("ref", ids[first], True),
                    OperationArgument("ref", ids[second], True),
                ),
            ),
            f"What is the combined value of {categories[first]} and {categories[second]}?",
        )
    if name == "average":
        pairs = [
            (left, right)
            for left in range(len(values))
            for right in range(left + 1, len(values))
            if (values[left] + values[right]) % 2 == 0
        ]
        first, second = rng.choice(pairs)
        return (
            (ids[first], ids[second]),
            Operation(
                "average",
                (
                    OperationArgument("ref", ids[first], True),
                    OperationArgument("ref", ids[second], True),
                ),
            ),
            f"What is the average value of {categories[first]} and {categories[second]}?",
        )
    if name == "argmax":
        return (
            ids,
            Operation("argmax", tuple(OperationArgument("ref", item, True) for item in ids)),
            "Which category has the greatest value?",
        )
    if name == "argmin":
        return (
            ids,
            Operation("argmin", tuple(OperationArgument("ref", item, True) for item in ids)),
            "Which category has the smallest value?",
        )
    if name == "compare":
        first, second = rng.sample(range(len(categories)), 2)
        return (
            (ids[first], ids[second]),
            Operation(
                "compare",
                (
                    OperationArgument("ref", ids[first], True),
                    OperationArgument("ref", ids[second], True),
                ),
            ),
            f"Is {categories[first]} greater or less than {categories[second]}?",
        )
    raise ValueError(f"unsupported_controlled_operation:{name}")


def _visual_evidence(record: ControlledRecord) -> tuple[VisualEvidence, ...]:
    boxes = _mark_boxes(record)
    return tuple(
        VisualEvidence(
            source_id=f"mark-{index + 1}",
            row=category,
            column="Signal",
            value=str(value),
            bbox=boxes[index],
        )
        for index, (category, value) in enumerate(
            zip(record.categories, record.values, strict=True)
        )
    )


def _gold_answer(record: ControlledRecord) -> str:
    by_id = {item.source_id: item for item in record.evidence}
    selected = tuple(
        _evidence_cell(by_id[argument.value]) for argument in record.gold_operation.arguments
    )
    outcome = execute(record.gold_operation, selected)
    if not outcome.success or outcome.value is None:
        raise ValueError(f"invalid_controlled_gold_operation:{record.record_id}")
    return outcome.value


def _draw_axes(
    draw: ImageDraw.ImageDraw,
    plot: tuple[int, int, int, int],
    font: ImageFont.ImageFont | ImageFont.FreeTypeFont,
    small_font: ImageFont.ImageFont | ImageFont.FreeTypeFont,
) -> None:
    left, top, right, bottom = plot
    draw.rectangle(plot, outline="#111827", width=2)
    for value in range(0, 101, 20):
        y = _value_y(value, top, bottom)
        draw.line((left, y, right, y), fill="#e5e7eb", width=1)
        draw.text((18, y - 7), str(value), fill="#374151", font=small_font)
    draw.text((8, 42), "Signal", fill="#111827", font=font)


def _draw_bars(
    draw: ImageDraw.ImageDraw,
    record: ControlledRecord,
    plot: tuple[int, int, int, int],
    colors: tuple[str, ...],
    font: ImageFont.ImageFont | ImageFont.FreeTypeFont,
    small_font: ImageFont.ImageFont | ImageFont.FreeTypeFont,
) -> None:
    left, top, right, bottom = plot
    width = right - left
    step = width / len(record.values)
    bar_width = int(step * 0.54)
    for index, (category, value) in enumerate(zip(record.categories, record.values, strict=True)):
        center = int(left + step * (index + 0.5))
        y = _value_y(value, top, bottom)
        draw.rectangle(
            (center - bar_width // 2, y, center + bar_width // 2, bottom), fill=colors[index]
        )
        draw.rectangle(
            (center - bar_width // 2, y, center + bar_width // 2, bottom), outline="#1f2937"
        )
        draw.text((center - 8, max(top, y - 20)), str(value), fill="#111827", font=font)
        label_x = center - len(category) * 4
        draw.text((label_x, bottom + 12), category, fill="#111827", font=small_font)


def _draw_line(
    draw: ImageDraw.ImageDraw,
    record: ControlledRecord,
    plot: tuple[int, int, int, int],
    colors: tuple[str, ...],
    font: ImageFont.ImageFont | ImageFont.FreeTypeFont,
    small_font: ImageFont.ImageFont | ImageFont.FreeTypeFont,
) -> None:
    left, top, right, bottom = plot
    width = right - left
    step = width / (len(record.values) - 1)
    points = [
        (int(left + step * index), _value_y(value, top, bottom))
        for index, value in enumerate(record.values)
    ]
    draw.line(points, fill="#111827", width=3)
    for index, ((x, y), category, value) in enumerate(
        zip(points, record.categories, record.values, strict=True)
    ):
        draw.ellipse((x - 7, y - 7, x + 7, y + 7), fill=colors[index], outline="#111827", width=2)
        draw.text((x - 8, max(top, y - 25)), str(value), fill="#111827", font=font)
        draw.text((x - len(category) * 4, bottom + 12), category, fill="#111827", font=small_font)


def _mark_boxes(record: ControlledRecord) -> tuple[tuple[int, int, int, int], ...]:
    left, top, right, bottom = (72, 60, 530, 335)
    if record.chart_type == "bar":
        step = (right - left) / len(record.values)
        bar_width = int(step * 0.54)
        return tuple(
            (
                int(left + step * (index + 0.5)) - bar_width // 2 - 3,
                max(top, _value_y(value, top, bottom) - 22),
                int(left + step * (index + 0.5)) + bar_width // 2 + 3,
                bottom + 1,
            )
            for index, value in enumerate(record.values)
        )
    if record.chart_type == "line":
        step = (right - left) / (len(record.values) - 1)
        return tuple(
            (
                int(left + step * index) - 16,
                max(top, _value_y(value, top, bottom) - 28),
                int(left + step * index) + 16,
                min(bottom, _value_y(value, top, bottom) + 16),
            )
            for index, value in enumerate(record.values)
        )
    raise ValueError(f"unsupported_chart_type:{record.chart_type}")


def _value_y(value: int, top: int, bottom: int) -> int:
    return int(bottom - (value / 100) * (bottom - top))


def _evidence_cell(item: VisualEvidence) -> EvidenceCell:
    from forgemm.data.schemas import EvidenceCell

    return EvidenceCell(item.source_id, item.row, item.column, item.value)


def _sft_row(record: ControlledRecord) -> dict[str, Any]:
    return {
        "sample_id": record.record_id,
        "messages": [
            {
                "role": "user",
                "content": (
                    "Answer the chart question using evidence cells with absolute pixel bboxes, "
                    "one executable operation, and the required XML blocks.\n"
                    f"Question: {record.question}"
                ),
            },
            {"role": "assistant", "content": render_controlled_completion(record)},
        ],
        "images": [record.image_path],
    }


def build_answer_sft_row(record: ControlledRecord) -> dict[str, Any]:
    """Build the answer-only E0 control from the identical image/question population."""

    return _answer_sft_row(record)


def _answer_sft_row(record: ControlledRecord) -> dict[str, Any]:
    return {
        "sample_id": record.record_id,
        "messages": [
            {
                "role": "user",
                "content": f"Answer the chart question concisely.\nQuestion: {record.question}",
            },
            {"role": "assistant", "content": record.reference_answer},
        ],
        "images": [record.image_path],
    }


def _grpo_row(record: ControlledRecord) -> dict[str, Any]:
    prompt = _sft_row(record)["messages"][0]
    return {
        "sample_id": record.record_id,
        "messages": [prompt],
        "images": [record.image_path],
        "controlled_gold": json.dumps(record.to_dict(), ensure_ascii=False, sort_keys=True),
    }


def _prompt_row(record: ControlledRecord) -> dict[str, Any]:
    return {
        "sample_id": record.record_id,
        "split": record.split,
        "messages": [_sft_row(record)["messages"][0]],
        "images": [record.image_path],
    }


def _attack_audit(records_by_split: dict[str, list[ControlledRecord]]) -> dict[str, bool]:
    result: dict[str, bool] = {}
    for split, records in records_by_split.items():
        record = records[0]
        gold = render_controlled_completion(record)
        first_box = list(_expected_first_bbox(record))
        box_text = ",".join(str(value) for value in first_box)
        wrong_box = gold.replace(f"bbox=[{box_text}]", "bbox=[0,0,1,1]", 1)
        wrong_answer = gold.replace(
            f"<answer>\n{record.reference_answer}\n</answer>", "<answer>\n0\n</answer>"
        )
        result[f"{split}:wrong_bbox"] = not verify_controlled_completion(
            record, wrong_box
        ).full_pass
        result[f"{split}:wrong_answer"] = not verify_controlled_completion(
            record, wrong_answer
        ).full_pass
        result[f"{split}:missing_evidence"] = not verify_controlled_completion(
            record, gold.replace("<evidence>", "<evidence>\n", 1).replace("\n</evidence>", "", 1)
        ).full_pass
    return result


def _expected_first_bbox(record: ControlledRecord) -> tuple[int, int, int, int]:
    return next(
        item.bbox for item in record.evidence if item.source_id == record.gold_evidence_ids[0]
    )


def _check_prompt_views(output: Path, records_by_split: dict[str, list[ControlledRecord]]) -> None:
    for split in ("val", "test"):
        rows = _load_jsonl(output / "views" / f"{split}_prompts.jsonl")
        if len(rows) != len(records_by_split[split]):
            raise ValueError(f"prompt_count_mismatch:{split}")
        forbidden = {"reference_answer", "controlled_gold", "evidence", "gold_operation"}
        if any(forbidden & set(row) for row in rows):
            raise ValueError(f"prompt_gold_leak:{split}")


def _check_training_views(output: Path, train_records: list[ControlledRecord]) -> None:
    expected = {
        "sft_train.jsonl",
        "answer_sft_train.jsonl",
        "grpo_train.jsonl",
    }
    expected_ids = [record.record_id for record in train_records]
    for name in expected:
        rows = _load_jsonl(output / "views" / name)
        if len(rows) != len(train_records):
            raise ValueError(f"training_view_count_mismatch:{name}")
        if [str(row.get("sample_id", "")) for row in rows] != expected_ids:
            raise ValueError(f"training_view_alignment_mismatch:{name}")


def _validate_visual_bounds(record: ControlledRecord) -> None:
    width, height = record.image_size
    for item in record.evidence:
        left, top, right, bottom = item.bbox
        if not (0 <= left < right <= width and 0 <= top < bottom <= height):
            raise ValueError(f"visual_bbox_out_of_bounds:{record.record_id}")


def _validate_counts(counts: dict[str, int]) -> None:
    if set(counts) != {"train", "val", "test"} or any(value <= 0 for value in counts.values()):
        raise ValueError("invalid_controlled_counts")


def _sample_seed(root_seed: int, split: str, index: int) -> int:
    material = f"{DATASET_VERSION}:{root_seed}:{split}:{index}".encode()
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big")


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _identity(path: Path) -> dict[str, Any]:
    return {"rows": len(_load_jsonl(path)), "sha256": _sha256_bytes(path.read_bytes())}


def _relative_to_project(path: Path, project_root: Path) -> str:
    try:
        return path.resolve().relative_to(project_root).as_posix()
    except ValueError as exc:
        raise ValueError(f"controlled_output_must_be_inside_project:{path}") from exc


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()
