from __future__ import annotations

import argparse
import json
from pathlib import Path

from .audit import audit_autovla, official_preflight, write_audit
from .backends import SyntheticBackend
from .config import load_config
from .evaluation import EvaluationRunner, compare_results, write_summary
from .evidence import freeze_evidence, verify_evidence
from .io import load_manifest, read_jsonl, write_jsonl
from .pipeline import GuardPipeline
from .synthetic import build_synthetic_manifest
from .visualization import render_result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="drivevla-guard", description="DriveVLA-Guard experiment CLI")
    commands = parser.add_subparsers(dest="command", required=True)

    synthetic = commands.add_parser("make-synthetic", help="create a deterministic smoke manifest")
    synthetic.add_argument("--output", required=True)
    synthetic.add_argument("--scenes", type=int, default=12)

    evaluate = commands.add_parser(
        "evaluate-synthetic", help="run the full pipeline with the synthetic backend"
    )
    evaluate.add_argument("--config", required=True)
    evaluate.add_argument("--manifest", required=True)
    evaluate.add_argument("--output", required=True)
    evaluate.add_argument("--summary", required=True)
    evaluate.add_argument("--no-resume", action="store_true")

    compare = commands.add_parser("compare", help="paired comparison for two result JSONL files")
    compare.add_argument("--baseline", required=True)
    compare.add_argument("--candidate", required=True)
    compare.add_argument("--output", required=True)

    visualize = commands.add_parser("visualize", help="render trajectory evidence from result JSONL")
    visualize.add_argument("--manifest", required=True)
    visualize.add_argument("--results", required=True)
    visualize.add_argument("--output-dir", required=True)
    visualize.add_argument("--limit", type=int, default=10)

    audit = commands.add_parser("audit-upstream", help="lock an AutoVLA checkout and license facts")
    audit.add_argument("--root", required=True)
    audit.add_argument("--output", required=True)

    preflight = commands.add_parser("official-preflight", help="check official-run paths and dependencies")
    preflight.add_argument("--autovla-root")
    preflight.add_argument("--checkpoint")
    preflight.add_argument("--qwen-model")
    preflight.add_argument("--model-config")
    preflight.add_argument("--json-data")
    preflight.add_argument("--sensor-data")
    preflight.add_argument("--metric-cache")
    preflight.add_argument("--min-vram-gb", type=float, default=24.0)
    preflight.add_argument("--expected-upstream-commit")
    preflight.add_argument("--output", required=True)

    freeze = commands.add_parser("freeze-evidence", help="freeze hashes for the validated synthetic run")
    freeze.add_argument("--root", default=".")
    freeze.add_argument("--output", default="artifacts/evidence_manifest.json")

    verify = commands.add_parser("verify-evidence", help="verify frozen synthetic evidence and hashes")
    verify.add_argument("--root", default=".")
    verify.add_argument("--snapshot", default="artifacts/evidence_manifest.json")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "make-synthetic":
        rows = build_synthetic_manifest(args.scenes)
        write_jsonl(args.output, rows)
        print(
            json.dumps({"output": str(Path(args.output).resolve()), "scenes": len(rows)}, ensure_ascii=False)
        )
        return 0
    if args.command == "evaluate-synthetic":
        config = load_config(args.config)
        pipeline = GuardPipeline(SyntheticBackend(), config)
        summary = EvaluationRunner(pipeline).run(args.manifest, args.output, resume=not args.no_resume)
        write_summary(args.summary, summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    if args.command == "compare":
        comparison = compare_results(args.baseline, args.candidate)
        write_summary(args.output, comparison)
        print(json.dumps(comparison, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    if args.command == "visualize":
        manifest = {context.scene_id: context for _, context in load_manifest(args.manifest)}
        rows = [row for row in read_jsonl(args.results) if row.get("status") == "ok"][: args.limit]
        output_dir = Path(args.output_dir)
        for row in rows:
            render_result(row, manifest[row["scene_id"]], output_dir / f"{row['scene_id']}.png")
        print(
            json.dumps({"output_dir": str(output_dir.resolve()), "rendered": len(rows)}, ensure_ascii=False)
        )
        return 0
    if args.command == "audit-upstream":
        payload = audit_autovla(args.root)
        write_audit(args.output, payload)
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
        return 0 if payload["ready"] else 2
    if args.command == "official-preflight":
        payload = official_preflight(
            {
                "autovla_root": args.autovla_root,
                "checkpoint": args.checkpoint,
                "qwen_model": args.qwen_model,
                "model_config": args.model_config,
                "json_data": args.json_data,
                "sensor_data": args.sensor_data,
                "metric_cache": args.metric_cache,
            },
            min_vram_gb=args.min_vram_gb,
            expected_upstream_commit=args.expected_upstream_commit,
        )
        write_audit(args.output, payload)
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
        return 0 if payload["ready"] else 2
    if args.command == "freeze-evidence":
        payload = freeze_evidence(args.root, args.output)
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    if args.command == "verify-evidence":
        payload = verify_evidence(args.root, args.snapshot)
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
        return 0 if payload["ready"] else 2
    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
