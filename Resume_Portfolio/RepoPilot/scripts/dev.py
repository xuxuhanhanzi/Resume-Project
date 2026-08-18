"""Cross-platform development and quality entrypoint."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _run(arguments: list[str]) -> None:
    environment = os.environ.copy()
    source = str(ROOT / "src")
    environment["PYTHONPATH"] = (
        source
        if not environment.get("PYTHONPATH")
        else source + os.pathsep + environment["PYTHONPATH"]
    )
    subprocess.run(arguments, cwd=ROOT, env=environment, check=True)


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    command = arguments[0] if arguments else "check"
    commands = {
        "format": [sys.executable, "-m", "ruff", "format", "."],
        "format-check": [sys.executable, "-m", "ruff", "format", "--check", "."],
        "lint": [sys.executable, "-m", "ruff", "check", "."],
        "typecheck": [sys.executable, "-m", "mypy", "--no-incremental"],
        "test": [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider"],
        "unit": [sys.executable, "-m", "pytest", "-q", "tests/unit"],
        "integration": [sys.executable, "-m", "pytest", "-q", "tests/integration"],
        "safety": [sys.executable, "-m", "pytest", "-q", "tests/safety"],
        "smoke": [sys.executable, "-m", "pytest", "-q", "tests/e2e"],
    }
    if command == "check":
        for name in ("format-check", "lint", "typecheck", "test"):
            _run(commands[name])
        return 0
    selected = commands.get(command)
    if selected is None:
        print(f"unknown command: {command}", file=sys.stderr)
        return 2
    _run(selected)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
