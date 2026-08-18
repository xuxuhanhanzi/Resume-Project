"""Command-line entry point for ForgeLLM."""

from __future__ import annotations

import argparse
import json
import platform
import sys
from collections.abc import Sequence
from pathlib import Path

from forgellm import __version__
from forgellm.config import ConfigError, load_run_config
from forgellm.data import DataConfigError, DataPipelineError, load_data_config, run_data_pipeline
from forgellm.runtime import initialize_run
from forgellm.tokenization import (
    TokenizerConfigError,
    TokenizerCorpusError,
    TokenizerError,
    load_tokenizer_config,
)
from forgellm.tokenization.workflow import (
    compare_with_huggingface_reference,
    evaluate_tokenizer_artifact,
    train_tokenizer_artifact,
)


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level command parser."""
    parser = argparse.ArgumentParser(
        prog="forgellm",
        description="Reproducible LLM data, training, evaluation, and serving workflows.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")

    subparsers = parser.add_subparsers(dest="command")
    subparsers.add_parser(
        "doctor",
        help="Print a minimal, non-sensitive runtime summary.",
    )
    init_run_parser = subparsers.add_parser(
        "init-run",
        help="Validate a config and initialize a provenance directory.",
    )
    init_run_parser.add_argument("--config", type=Path, required=True)
    init_run_parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"))
    init_run_parser.add_argument("--repo-root", type=Path, default=Path.cwd())

    data_parser = subparsers.add_parser(
        "data-pipeline",
        help="Run the deterministic JSONL data pipeline.",
    )
    data_parser.add_argument("--config", type=Path, required=True)
    data_parser.add_argument("--input", type=Path, required=True)
    data_parser.add_argument("--output-dir", type=Path, required=True)
    data_parser.add_argument("--source-name", required=True)
    data_parser.add_argument("--source-license", required=True)

    tokenizer_train_parser = subparsers.add_parser(
        "tokenizer-train",
        help="Train and freeze a deterministic byte-level BPE tokenizer.",
    )
    tokenizer_train_parser.add_argument("--config", type=Path, required=True)
    tokenizer_train_parser.add_argument("--input", type=Path, required=True)
    tokenizer_train_parser.add_argument("--output-dir", type=Path, required=True)
    tokenizer_train_parser.add_argument("--source-name", required=True)
    tokenizer_train_parser.add_argument("--source-license", required=True)

    tokenizer_evaluate_parser = subparsers.add_parser(
        "tokenizer-evaluate",
        help="Evaluate a tokenizer against the raw UTF-8 byte baseline.",
    )
    tokenizer_evaluate_parser.add_argument("--model", type=Path, required=True)
    tokenizer_evaluate_parser.add_argument("--input", type=Path, required=True)
    tokenizer_evaluate_parser.add_argument("--output", type=Path, required=True)
    tokenizer_evaluate_parser.add_argument("--encode-repeats", type=int, default=5)

    tokenizer_compare_parser = subparsers.add_parser(
        "tokenizer-compare",
        help="Compare ForgeLLM BPE with the optional Hugging Face reference.",
    )
    tokenizer_compare_parser.add_argument("--config", type=Path, required=True)
    tokenizer_compare_parser.add_argument("--model", type=Path, required=True)
    tokenizer_compare_parser.add_argument("--train-input", type=Path, required=True)
    tokenizer_compare_parser.add_argument("--evaluation-input", type=Path, required=True)
    tokenizer_compare_parser.add_argument("--output-dir", type=Path, required=True)
    tokenizer_compare_parser.add_argument("--encode-repeats", type=int, default=5)
    return parser


def _doctor_payload() -> dict[str, str]:
    """Return a stable, non-sensitive runtime summary."""
    return {
        "forgellm": __version__,
        "python": platform.python_version(),
        "platform": sys.platform,
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Run the ForgeLLM CLI and return a process exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "doctor":
        print(json.dumps(_doctor_payload(), sort_keys=True))
        return 0

    if args.command == "init-run":
        try:
            config = load_run_config(args.config)
            artifacts = initialize_run(config, args.artifacts_dir, args.repo_root)
        except (ConfigError, FileExistsError, OSError, ValueError) as error:
            print(f"forgellm: error: {error}", file=sys.stderr)
            return 2
        print(
            json.dumps(
                {"directory": str(artifacts.directory), "run_id": artifacts.run_id},
                sort_keys=True,
            )
        )
        return 0

    if args.command == "data-pipeline":
        try:
            data_config = load_data_config(args.config)
            result = run_data_pipeline(
                data_config,
                args.input,
                args.output_dir,
                source_name=args.source_name,
                source_license=args.source_license,
            )
        except (DataConfigError, DataPipelineError, FileExistsError, OSError) as error:
            print(f"forgellm: error: {error}", file=sys.stderr)
            return 2
        print(
            json.dumps(
                {
                    "input_records": result.input_records,
                    "manifest": str(result.manifest_path),
                    "rejected_records": result.rejected_records,
                    "retained_records": result.retained_records,
                },
                sort_keys=True,
            )
        )
        return 0

    if args.command == "tokenizer-train":
        try:
            tokenizer_config = load_tokenizer_config(args.config)
            training_result = train_tokenizer_artifact(
                tokenizer_config,
                args.input,
                args.output_dir,
                source_name=args.source_name,
                source_license=args.source_license,
            )
        except (
            FileExistsError,
            OSError,
            TokenizerConfigError,
            TokenizerCorpusError,
            TokenizerError,
            ValueError,
        ) as error:
            print(f"forgellm: error: {error}", file=sys.stderr)
            return 2
        print(
            json.dumps(
                {
                    "documents": training_result.documents,
                    "manifest": str(training_result.manifest_path),
                    "merge_count": training_result.merge_count,
                    "model": str(training_result.model_path),
                    "model_sha256": training_result.model_sha256,
                    "vocab_size": training_result.vocab_size,
                },
                sort_keys=True,
            )
        )
        return 0

    if args.command == "tokenizer-evaluate":
        try:
            evaluation = evaluate_tokenizer_artifact(
                args.model,
                args.input,
                args.output,
                encode_repeats=args.encode_repeats,
            )
        except (
            FileExistsError,
            OSError,
            TokenizerCorpusError,
            TokenizerError,
            ValueError,
        ) as error:
            print(f"forgellm: error: {error}", file=sys.stderr)
            return 2
        model_section = evaluation["model"]
        if not isinstance(model_section, dict):
            print("forgellm: error: invalid evaluation model section", file=sys.stderr)
            return 2
        print(
            json.dumps(
                {
                    "model_sha256": model_section["sha256"],
                    "output": str(args.output),
                    "schema_version": evaluation["schema_version"],
                },
                sort_keys=True,
            )
        )
        return 0

    if args.command == "tokenizer-compare":
        try:
            tokenizer_config = load_tokenizer_config(args.config)
            comparison = compare_with_huggingface_reference(
                tokenizer_config,
                args.model,
                args.train_input,
                args.evaluation_input,
                args.output_dir,
                encode_repeats=args.encode_repeats,
            )
        except (
            FileExistsError,
            ImportError,
            OSError,
            RuntimeError,
            TokenizerConfigError,
            TokenizerCorpusError,
            TokenizerError,
            ValueError,
        ) as error:
            print(f"forgellm: error: {error}", file=sys.stderr)
            return 2
        print(
            json.dumps(
                {
                    "output": str(args.output_dir / "comparison.json"),
                    "schema_version": comparison["schema_version"],
                },
                sort_keys=True,
            )
        )
        return 0

    parser.print_help()
    return 0


def entrypoint() -> None:
    """Console-script wrapper."""
    raise SystemExit(main())
