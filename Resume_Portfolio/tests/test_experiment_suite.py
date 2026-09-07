from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "experiment_suite.py"
SPEC = importlib.util.spec_from_file_location("experiment_suite", MODULE_PATH)
assert SPEC and SPEC.loader
SUITE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SUITE)


class ExperimentSuiteTests(unittest.TestCase):
    def test_manifest_is_valid_and_stage_ids_are_unique(self) -> None:
        manifest = SUITE.load_manifest(
            Path(__file__).resolve().parents[1] / "experiment-suite.json"
        )
        ids = [stage["id"] for stage in manifest["stages"]]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(len(ids), 9)

    def test_profile_and_project_filter_keeps_suite_preflight(self) -> None:
        manifest = SUITE.load_manifest(
            Path(__file__).resolve().parents[1] / "experiment-suite.json"
        )
        selected = SUITE.selected_stages(manifest, "run", "smoke", {"ForgeMM"})
        self.assertEqual([stage["id"] for stage in selected], ["forgemm-experiments"])
        selected = SUITE.selected_stages(manifest, "setup", "full", {"ForgeMM"})
        self.assertEqual(
            [stage["id"] for stage in selected], ["host-preflight", "forgemm-setup"]
        )

    def test_rejects_duplicate_stage(self) -> None:
        stage = {
            "id": "same",
            "project": "p",
            "phase": "run",
            "profiles": ["smoke"],
            "cwd": ".",
            "command": ["true"],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            path.write_text(json.dumps({"schema_version": 1, "stages": [stage, stage]}))
            with self.assertRaises(ValueError):
                SUITE.load_manifest(path)


if __name__ == "__main__":
    unittest.main()
