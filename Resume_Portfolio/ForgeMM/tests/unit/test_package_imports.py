from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def test_parser_imports_in_fresh_interpreter() -> None:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(Path(__file__).resolve().parents[2] / "src")
    result = subprocess.run(
        [sys.executable, "-c", "from forgemm.reasoning.parser import parse_prediction"],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert result.returncode == 0, result.stderr
