"""Download DABench CSVs and build a stratified validation manifest (P8C)."""

from __future__ import annotations

import hashlib
import json
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import TypedDict

PROJECT = Path(__file__).resolve().parents[1]
REVISION = "b455d578e30fee513abd79936cbdf7a6de026cb5"
DATA_DIR = PROJECT / "evaluation" / "benchmarks" / "data" / "dabench" / REVISION
TABLES_DIR = DATA_DIR / "da-dev-tables"
QUESTIONS = DATA_DIR / "da-dev-questions.jsonl"
LABELS = DATA_DIR / "da-dev-labels.jsonl"
MANIFEST = PROJECT / "evaluation" / "benchmarks" / "manifests" / "dabench_validation35_v1.json"
BASE_URL = "https://huggingface.co/datasets/infiagent/DABench/resolve/main/da-dev-tables"

TARGET = {"easy": 12, "medium": 13, "hard": 10}


class DABenchQuestion(TypedDict):
    """One parsed record from da-dev-questions.jsonl."""

    level: str
    file_name: str
    id: int


def load_questions() -> list[DABenchQuestion]:
    with open(QUESTIONS, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def select_stratified(questions: list[DABenchQuestion]) -> list[DABenchQuestion]:
    by_level: dict[str, list[DABenchQuestion]] = defaultdict(list)
    for q in questions:
        by_level[q.get("level", "easy")].append(q)
    selected: list[DABenchQuestion] = []
    for level, count in TARGET.items():
        pool = by_level.get(level, [])
        seen_tables: set[str] = set()
        for q in pool:
            if q["file_name"] not in seen_tables or len(selected) < 20:
                selected.append(q)
                seen_tables.add(q["file_name"])
            if sum(1 for s in selected if s.get("level") == level) >= count:
                break
    return selected


def download_csv(filename: str) -> str | None:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    dest = TABLES_DIR / filename
    if dest.exists() and dest.stat().st_size > 0:
        return hashlib.sha256(dest.read_bytes()).hexdigest()
    url = f"{BASE_URL}/{filename}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = resp.read()
        dest.write_bytes(data)
        return hashlib.sha256(data).hexdigest()
    except Exception as e:
        print(f"  FAIL {filename}: {e}")
        return None


def main() -> int:
    questions = load_questions()
    selected = select_stratified(questions)
    print(f"Selected {len(selected)} questions:")
    for level in ("easy", "medium", "hard"):
        count = sum(1 for q in selected if q.get("level") == level)
        print(f"  {level}: {count}")

    csv_names = sorted({q["file_name"] for q in selected})
    print(f"\nNeed {len(csv_names)} CSV files")
    table_hashes: dict[str, str] = {}
    for i, name in enumerate(csv_names, 1):
        h = download_csv(name)
        if h:
            table_hashes[name] = h
            print(f"  [{i}/{len(csv_names)}] OK {name} ({h[:12]}...)")
        else:
            print(f"  [{i}/{len(csv_names)}] FAIL {name}")

    task_ids = [f"dabench-dev-{int(q['id']):04d}" for q in selected]
    manifest = {
        "benchmark_id": "infiagent_dabench_dev",
        "dataset_revision": REVISION,
        "split": "dev_validation35",
        "task_ids": task_ids,
        "tables": {k: v for k, v in table_hashes.items()},
        "levels": {q["file_name"]: q.get("level", "?") for q in selected},
        "network_policy": "deny",
        "temperature": 0.0,
        "random_seed": 7,
        "schema_version": 1,
    }
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"\nManifest: {MANIFEST}")
    print(f"Downloaded: {len(table_hashes)}/{len(csv_names)} CSVs")
    print(f"Tasks: {len(task_ids)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
