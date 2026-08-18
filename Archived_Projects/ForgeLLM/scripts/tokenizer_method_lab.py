"""Run the Stage 1 G1-B tokenizer method laboratory."""

from __future__ import annotations

import argparse
from pathlib import Path

from forgellm.tokenization.method_lab import run_method_lab


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--vocab-size", type=int, default=300)
    parser.add_argument("--min-pair-frequency", type=int, default=1)
    return parser


def main() -> int:
    args = _parser().parse_args()
    report = run_method_lab(
        args.train,
        args.evaluation,
        args.output,
        vocab_size=args.vocab_size,
        min_pair_frequency=args.min_pair_frequency,
    )
    print(f"method lab complete: {len(report['evaluation'])} algorithm paths -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
