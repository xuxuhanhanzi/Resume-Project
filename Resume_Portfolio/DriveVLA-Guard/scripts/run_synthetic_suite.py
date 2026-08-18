from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from drivevla_guard.backends import SyntheticBackend  # noqa: E402
from drivevla_guard.config import load_config  # noqa: E402
from drivevla_guard.evaluation import EvaluationRunner, compare_results, write_summary  # noqa: E402
from drivevla_guard.io import write_jsonl  # noqa: E402
from drivevla_guard.pipeline import GuardPipeline  # noqa: E402
from drivevla_guard.synthetic import build_synthetic_manifest  # noqa: E402

VARIANTS = {
    "b0": "b0_fast_greedy.yaml",
    "b1": "b1_slow_greedy.yaml",
    "e1": "e1_candidates_only.yaml",
    "e2": "e2_rerank.yaml",
    "e3": "e3_router.yaml",
    "e4": "e4_guard.yaml",
}


def run_suite(output_root: Path, scenes: int, manifest_path: Path | None = None) -> dict[str, object]:
    output_root.mkdir(parents=True, exist_ok=True)
    manifest = manifest_path.resolve() if manifest_path else output_root / "manifest.jsonl"
    if manifest_path is None:
        if manifest.exists():
            raise FileExistsError(f"refusing to overwrite existing suite: {output_root}")
        write_jsonl(manifest, build_synthetic_manifest(scenes))

    summaries = {}
    for variant, filename in VARIANTS.items():
        run_dir = output_root / variant
        pipeline = GuardPipeline(SyntheticBackend(), load_config(PROJECT_ROOT / "configs" / filename))
        summary = EvaluationRunner(pipeline).run(
            manifest,
            run_dir / "predictions.jsonl",
            resume=False,
        )
        write_summary(run_dir / "summary.json", summary)
        summaries[variant] = summary

    comparisons = {}
    for name, baseline, candidate in (
        ("e2_vs_b0", "b0", "e2"),
        ("e3_vs_b0", "b0", "e3"),
        ("e4_vs_e2", "e2", "e4"),
    ):
        comparison = compare_results(
            output_root / baseline / "predictions.jsonl",
            output_root / candidate / "predictions.jsonl",
        )
        write_summary(output_root / f"{name}.json", comparison)
        comparisons[name] = comparison
    return {
        "metric_scope": "synthetic_proxy_not_navsim",
        "output_root": str(output_root.resolve()),
        "scenes": scenes,
        "summaries": summaries,
        "comparisons": comparisons,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the cross-platform B0/B1/E1-E4 synthetic suite")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--scenes", type=int, default=12)
    parser.add_argument("--manifest", help="use an existing frozen manifest instead of generating one")
    args = parser.parse_args()
    if args.scenes < 1:
        parser.error("--scenes must be positive")
    report = run_suite(
        Path(args.output_root),
        args.scenes,
        Path(args.manifest) if args.manifest else None,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
