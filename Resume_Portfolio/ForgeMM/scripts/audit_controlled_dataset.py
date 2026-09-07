"""Audit a frozen ForgeMM-Controlled dataset before any SFT or GRPO run."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from forgemm.controlled.dataset import DEFAULT_COUNTS, audit_controlled_dataset  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=Path,
        default=PROJECT_ROOT / "datasets" / "ForgeMM-Controlled-v1-Lite",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train", type=int, default=DEFAULT_COUNTS["train"])
    parser.add_argument("--val", type=int, default=DEFAULT_COUNTS["val"])
    parser.add_argument("--test", type=int, default=DEFAULT_COUNTS["test"])
    args = parser.parse_args()
    result = audit_controlled_dataset(
        args.dataset,
        expected_counts={"train": args.train, "val": args.val, "test": args.test},
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if result["decision"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
