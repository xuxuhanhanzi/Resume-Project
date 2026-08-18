"""Authoritative cross-platform quality commands for ForgeMM."""

from __future__ import annotations

import argparse
import subprocess
import sys
from collections.abc import Sequence

Command = tuple[str, ...]


def _commands() -> dict[str, Command]:
    python = sys.executable
    return {
        "format-check": (python, "-m", "ruff", "format", "--check", "src", "tests", "scripts"),
        "lint": (python, "-m", "ruff", "check", "src", "tests", "scripts"),
        "typecheck": (python, "-m", "mypy", "--no-incremental"),
        "test": (python, "-m", "pytest", "-p", "no:cacheprovider"),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=("check", "format-check", "lint", "typecheck", "test"),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    commands = _commands()
    selected = (
        ("format-check", "lint", "typecheck", "test")
        if args.command == "check"
        else (args.command,)
    )
    for label in selected:
        command = commands[label]
        print(f"[forgemm-dev] {label}: {' '.join(command[1:])}", flush=True)
        completed = subprocess.run(command, check=False)
        if completed.returncode:
            return completed.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
