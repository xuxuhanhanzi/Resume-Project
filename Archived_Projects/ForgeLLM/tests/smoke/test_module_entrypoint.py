"""CPU-only smoke test for the module entry point."""

import json
import subprocess
import sys


def test_python_module_doctor() -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "forgellm", "doctor"],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["forgellm"]
    assert payload["python"]
    assert payload["platform"]
