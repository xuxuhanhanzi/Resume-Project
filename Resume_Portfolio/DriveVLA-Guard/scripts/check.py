from __future__ import annotations

import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def run(*command: str) -> None:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(PROJECT_ROOT / "src")
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=PROJECT_ROOT, env=environment, check=True)


def main() -> int:
    python = sys.executable
    run(python, "-m", "ruff", "format", "--check", "src", "tests", "scripts")
    run(python, "-m", "ruff", "check", "src", "tests", "scripts")
    run(python, "-m", "pytest", "-q")
    run(python, "-m", "compileall", "-q", "src", "tests", "scripts")
    run(python, "-m", "drivevla_guard.cli", "verify-evidence")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = PROJECT_ROOT / "artifacts" / "check_runs" / stamp
    run(python, "scripts/run_synthetic_suite.py", "--output-root", str(output), "--scenes", "12")
    print(f"release checks passed; fresh suite: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
