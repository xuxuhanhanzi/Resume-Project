"""Cross-platform development commands for ForgeLLM.

This Python entry point is authoritative. The Makefile only provides short aliases.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from collections.abc import Sequence

Command = tuple[str, ...]


def _commands() -> dict[str, tuple[Command, ...]]:
    python = sys.executable
    return {
        "format-check": ((python, "-m", "ruff", "format", "--check", "src", "tests", "scripts"),),
        "lint": ((python, "-m", "ruff", "check", "src", "tests", "scripts"),),
        "typecheck": ((python, "-m", "mypy"),),
        "unit": ((python, "-m", "pytest", "tests/unit"),),
        "integration": ((python, "-m", "pytest", "tests/integration"),),
        "smoke": ((python, "-m", "pytest", "tests/smoke"),),
        "test": ((python, "-m", "pytest"),),
    }


def build_parser() -> argparse.ArgumentParser:
    """Build the development command parser."""
    parser = argparse.ArgumentParser(description="Run ForgeLLM development checks.")
    parser.add_argument(
        "command",
        choices=(
            "check",
            "format-check",
            "integration",
            "lint",
            "smoke",
            "test",
            "typecheck",
            "unit",
        ),
    )
    return parser


def _run(label: str, command: Command) -> int:
    print(f"[forgellm-dev] {label}: {' '.join(command[1:])}", flush=True)
    completed = subprocess.run(command, check=False)
    if completed.returncode != 0:
        print(f"[forgellm-dev] {label}: failed ({completed.returncode})", file=sys.stderr)
    return completed.returncode


def main(argv: Sequence[str] | None = None) -> int:
    """Run one command group and stop at the first failure."""
    args = build_parser().parse_args(argv)
    commands = _commands()
    selected = (
        ("format-check", "lint", "typecheck", "test")
        if args.command == "check"
        else (args.command,)
    )

    for label in selected:
        for command in commands[label]:
            return_code = _run(label, command)
            if return_code != 0:
                return return_code
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
