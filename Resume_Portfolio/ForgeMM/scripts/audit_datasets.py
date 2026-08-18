"""Run the full ForgeMM data audit and write an immutable JSON report."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from forgemm.data.audit import audit_chartqa, audit_chartqapro  # noqa: E402


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
    report = {
        "schema_version": "1.0.0",
        "chartqa": audit_chartqa(args.chartqa),
        "chartqapro": audit_chartqapro(args.chartqapro),
    }
    if args.output.exists():
        raise FileExistsError(f"audit_report_exists:{args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
