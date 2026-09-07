"""Gate controlled GRPO using paired answer-only and structured validation inference."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from forgemm.controlled.dataset import load_controlled_records  # noqa: E402
from forgemm.evaluation.controlled_inference import evaluate_controlled_sft_gate  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--structured", type=Path, required=True)
    parser.add_argument("--oracle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-format", type=float, default=0.99)
    parser.add_argument("--noninferiority-pp", type=float, default=-0.01)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"controlled_gate_output_exists:{args.output}")
    baseline = _load(args.baseline)
    structured = _load(args.structured)
    result = evaluate_controlled_sft_gate(
        baseline,
        structured,
        load_controlled_records(args.oracle),
        min_format=args.min_format,
        noninferiority_pp=args.noninferiority_pp,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["decision"] == "advance" else 3


def _load(path: Path) -> list[dict[str, object]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"invalid_controlled_inference_row:{path}")
    return rows


if __name__ == "__main__":
    raise SystemExit(main())
