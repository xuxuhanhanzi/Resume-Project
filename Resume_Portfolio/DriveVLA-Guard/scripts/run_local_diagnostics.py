from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from drivevla_guard.backends import SyntheticBackend  # noqa: E402
from drivevla_guard.config import GuardConfig, load_config  # noqa: E402
from drivevla_guard.evaluation import EvaluationRunner, write_summary  # noqa: E402
from drivevla_guard.io import write_jsonl  # noqa: E402
from drivevla_guard.pipeline import GuardPipeline  # noqa: E402
from drivevla_guard.synthetic import build_synthetic_manifest  # noqa: E402


def _run_variant(config: GuardConfig, manifest: Path, run_dir: Path) -> dict[str, Any]:
    run_dir.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    summary = EvaluationRunner(GuardPipeline(SyntheticBackend(), config)).run(
        manifest,
        run_dir / "predictions.jsonl",
        resume=False,
    )
    wall_seconds = time.perf_counter() - started
    summary = {
        **summary,
        "runner_wall_seconds": wall_seconds,
        "runner_throughput_scenes_per_second": summary["scenes"] / wall_seconds,
    }
    write_summary(run_dir / "summary.json", summary)
    return summary


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_diagnostics(output_root: Path, scenes: int) -> dict[str, Any]:
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite existing diagnostics: {output_root}")
    output_root.mkdir(parents=True)
    manifest = output_root / "manifest.jsonl"
    write_jsonl(manifest, build_synthetic_manifest(scenes))

    rerank_base = load_config(PROJECT_ROOT / "configs" / "e2_rerank.yaml")
    k_sweep: dict[str, dict[str, Any]] = {}
    for seed in (17, 42, 2026):
        for candidate_count in (1, 2, 4, 8):
            name = f"k{candidate_count}_seed{seed}"
            config = replace(
                rerank_base,
                generation=replace(
                    rerank_base.generation,
                    fast_candidates=candidate_count,
                    seed=seed,
                ),
            )
            k_sweep[name] = _run_variant(config, manifest, output_root / "k_sweep" / name)

    router_base = load_config(PROJECT_ROOT / "configs" / "e4_guard.yaml")
    threshold_sweep: dict[str, dict[str, Any]] = {}
    for threshold in (4.0, 8.0, 13.8, 16.0, 20.0):
        name = f"risk_{str(threshold).replace('.', 'p')}"
        config = replace(
            router_base,
            generation=replace(router_base.generation, fast_candidates=1, seed=17),
            router=replace(router_base.router, risk_threshold=threshold),
        )
        threshold_sweep[name] = _run_variant(config, manifest, output_root / "threshold_sweep" / name)

    report: dict[str, Any] = {
        "schema_version": 1,
        "experiment_scope": "local_synthetic_diagnostic_not_navsim",
        "claim_boundary": (
            "K/seed/threshold sensitivity and CPU pipeline stability only; declared candidate "
            "latency is synthetic and no official NAVSIM/PDMS claim is permitted"
        ),
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
        },
        "scenes": scenes,
        "manifest_sha256": _sha256(manifest),
        "k_sweep": k_sweep,
        "threshold_sweep": threshold_sweep,
    }
    write_summary(output_root / "diagnostic_summary.json", report)

    evidence = {}
    for path in sorted(output_root.rglob("*")):
        if path.is_file() and path.name != "evidence_manifest.json":
            evidence[path.relative_to(output_root).as_posix()] = {
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
    write_summary(
        output_root / "evidence_manifest.json",
        {"schema_version": 1, "algorithm": "sha256", "files": evidence},
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run local K/seed/router diagnostics on the synthetic guard backend"
    )
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--scenes", type=int, default=1200)
    arguments = parser.parse_args()
    if arguments.scenes < 1:
        parser.error("--scenes must be positive")
    report = run_diagnostics(Path(arguments.output_root), arguments.scenes)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
