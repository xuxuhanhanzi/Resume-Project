"""Add a paired answer-only SFT control to an already frozen controlled dataset."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from forgemm.controlled.dataset import build_answer_sft_row, load_controlled_records  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=Path,
        default=PROJECT_ROOT / "datasets" / "ForgeMM-Controlled-v1-Lite",
    )
    args = parser.parse_args()
    target = args.dataset / "views" / "answer_sft_train.jsonl"
    if target.exists():
        raise FileExistsError(f"controlled_view_exists:{target}")
    records = load_controlled_records(args.dataset / "manifests" / "train_oracle.jsonl")
    with target.open("x", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(
                json.dumps(build_answer_sft_row(record), ensure_ascii=False, sort_keys=True) + "\n"
            )
    print(json.dumps({"rows": len(records), "output": str(target)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
