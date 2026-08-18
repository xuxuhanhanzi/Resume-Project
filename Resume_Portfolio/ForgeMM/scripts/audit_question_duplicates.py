"""Write the lightweight exact-question duplicate audit."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from forgemm.data.audit import audit_question_duplicates  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chartqa", type=Path, default=PROJECT_ROOT / "datasets" / "ChartQA")
    parser.add_argument(
        "--chartqapro",
        type=Path,
        default=PROJECT_ROOT / "datasets" / "ChartQAPro" / "chartqapro_test.parquet",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"duplicate_audit_exists:{args.output}")
    report = audit_question_duplicates(args.chartqa, args.chartqapro)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
