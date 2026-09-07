"""Build the frozen, generator-backed ForgeMM-Controlled v1-Lite dataset."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from forgemm.controlled.dataset import DEFAULT_COUNTS, create_controlled_dataset  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "datasets" / "ForgeMM-Controlled-v1-Lite",
    )
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument("--train", type=int, default=DEFAULT_COUNTS["train"])
    parser.add_argument("--val", type=int, default=DEFAULT_COUNTS["val"])
    parser.add_argument("--test", type=int, default=DEFAULT_COUNTS["test"])
    args = parser.parse_args()
    summary = create_controlled_dataset(
        args.output,
        project_root=args.project_root,
        counts={"train": args.train, "val": args.val, "test": args.test},
        root_seed=args.seed,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
