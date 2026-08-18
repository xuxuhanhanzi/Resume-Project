"""Create the immutable Stage 6 case set and contamination audit."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from forgellm.evaluation.config import load_stage6_config
from forgellm.evaluation.preparation import prepare_stage6_data


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/evaluation/stage6_final.toml"))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    project_root = Path.cwd()
    config = load_stage6_config(args.config)
    report = prepare_stage6_data(config, project_root=project_root)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
