"""Run the frozen Stage 6 system benchmark matrix."""

from __future__ import annotations

import argparse
import json
import traceback
from collections.abc import Sequence
from pathlib import Path

from forgellm.evaluation.config import load_stage6_config
from forgellm.evaluation.system_runner import run_system_benchmark


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/evaluation/stage6_final.toml"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/stage06/systems_v4"))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = run_system_benchmark(
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
                "all_required_cells_complete": report["all_required_cells_complete"],
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
