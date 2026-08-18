"""CPU-only smoke test for the end-to-end data pipeline CLI."""

import json
import subprocess
import sys
from pathlib import Path


def test_data_pipeline_cli(tmp_path: Path) -> None:
    repo_root = Path(__file__).parents[2]
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "forgellm",
            "data-pipeline",
            "--config",
            str(repo_root / "configs" / "data" / "smoke.toml"),
            "--input",
            str(repo_root / "tests" / "fixtures" / "data" / "sample_documents.jsonl"),
            "--output-dir",
            str(tmp_path / "output"),
            "--source-name",
            "forgellm-test-fixture",
            "--source-license",
            "project-test-fixture",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=15,
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["input_records"] == 9
    assert payload["retained_records"] == 4
    assert payload["rejected_records"] == 5
    assert Path(payload["manifest"]).is_file()
