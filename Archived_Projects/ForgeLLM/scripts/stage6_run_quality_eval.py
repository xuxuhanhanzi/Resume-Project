"""Run the frozen Stage 6 model-quality evaluation."""

from __future__ import annotations

import argparse
import json
import traceback
from collections.abc import Sequence
from pathlib import Path

from forgellm.evaluation.config import load_stage6_config
from forgellm.evaluation.quality import run_quality_evaluation


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/evaluation/stage6_final.toml"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/stage06/quality_v4"))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = run_quality_evaluation(
            load_stage6_config(args.config),
            project_root=Path.cwd(),
            output_dir=args.output,
        )
    except Exception as error:
        if args.output.is_dir():
            failure = args.output / "failure.json"
            if not failure.exists():
                failure.write_text(
                    json.dumps(
                        {
                            "error_type": type(error).__name__,
                            "message": str(error),
                            "traceback": traceback.format_exc(),
                        },
                        ensure_ascii=False,
                        indent=2,
                        sort_keys=True,
                    )
                    + "\n",
                    encoding="utf-8",
                    newline="\n",
                )
        raise
    print(
        json.dumps(
            {
                "frozen_cases": report["frozen_cases"],
                "run_fingerprint": report["run_fingerprint"],
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
