"""Build the final evidence-linked Stage 6 acceptance report."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from forgellm.evaluation.finalize import build_final_report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--quality", type=Path, default=Path("artifacts/stage06/quality_v4/report.json")
    )
    parser.add_argument(
        "--systems", type=Path, default=Path("artifacts/stage06/systems_v4/report.json")
    )
    parser.add_argument("--output", type=Path, default=Path("artifacts/stage06/final_v5"))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = build_final_report(
        project_root=Path.cwd(),
        quality_path=args.quality,
        systems_path=args.systems,
        output_dir=args.output,
    )
    print(
        json.dumps(
            {
                "automated_stage6_status": report["automated_stage6_status"],
                "schema_version": report["schema_version"],
                "status": "complete",
            },
            ensure_ascii=True,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
