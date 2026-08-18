"""Prepare a fresh (uncontaminated) SWE-bench-Live holdout.

Selects N tasks that are NOT in the contaminated smoke3 set, clones each repo at
its ``base_commit`` into the workspace root, and writes:

  - ``fresh_public.jsonl``     public fields only (feeds the runner)
  - ``fresh_task_config.json`` image / allowed_paths / test_command

The eval image name follows the official SWE-bench-Live convention
``swebench/sweb.eval.x86_64.<instance_id>:latest``; ``docker pull`` it before
running the agent. Resolved-rate evaluation is performed separately by
``scripts/eval_swebench_resolved.py`` (applies the generated model patch +
gold test_patch, then runs the FAIL_TO_PASS tests inside the eval image).
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path
from typing import Any

import pandas as pd

CONTAMINATED = {
    "aws-cloudformation__cfn-lint-3767",
    "python-babel__babel-1141",
    "projectmesa__mesa-2394",
}
PUBLIC_FIELDS = [
    "base_commit",
    "commit_url",
    "created_at",
    "difficulty",
    "instance_id",
    "issue_numbers",
    "problem_statement",
    "pull_number",
    "repo",
]


def _test_files(test_patch: str) -> list[str]:
    return re.findall(r"^\+\+\+ b/(.+)$", test_patch or "", re.M)


def _image_name(instance_id: str) -> str:
    """Official SWE-bench-Live eval image name (``__`` -> ``_1776_``)."""
    return f"starryzhang/sweb.eval.x86_64.{instance_id.replace('__', '_1776_')}:latest"


def _first_test_file(test_ids: object) -> str:
    """First FAIL_TO_PASS entry, trimmed to its file path (no ``::Class::method``)."""
    if not test_ids:
        return "tests"
    first = test_ids[0] if isinstance(test_ids, (list, tuple)) else str(test_ids).split()[0]
    return str(first).split("::", 1)[0] or "tests"


def _to_jsonable(value: object) -> object:
    """Convert pandas/numpy scalars/arrays (e.g. Timestamp, ndarray) to JSON types."""
    if hasattr(value, "isoformat"):  # Timestamp / datetime
        return value.isoformat()
    if hasattr(value, "tolist"):  # numpy scalar or ndarray
        return _to_jsonable(value.tolist())
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", required=True)
    parser.add_argument("--count", type=int, default=5)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--workspace-root", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--clone", action="store_true")
    arguments = parser.parse_args()

    df = pd.read_parquet(arguments.parquet)
    fresh = df[~df["instance_id"].isin(CONTAMINATED)].reset_index(drop=True)
    fresh = fresh.sample(frac=1, random_state=arguments.seed).reset_index(drop=True)
    selected = fresh.head(arguments.count)

    out = Path(arguments.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    workspace = Path(arguments.workspace_root)
    workspace.mkdir(parents=True, exist_ok=True)

    public: list[dict[str, Any]] = []
    config: dict[str, dict[str, Any]] = {}
    for _, row in selected.iterrows():
        instance_id = str(row["instance_id"])
        public.append({key: _to_jsonable(row[key]) for key in PUBLIC_FIELDS if key in row})
        test_ids = row.get("FAIL_TO_PASS")
        if isinstance(test_ids, str):
            test_ids = re.split(r"[\s,]+", test_ids.strip()) if test_ids.strip() else []
        else:
            test_ids = list(test_ids) if test_ids is not None else []
        first_test = _first_test_file(test_ids)
        config[instance_id] = {
            "image": _image_name(instance_id),
            "allowed_paths": ["."],
            "test_command": ["pytest", "-q", first_test],
        }
        if arguments.clone:
            workdir = workspace / instance_id
            if not workdir.exists():
                url = f"https://github.com/{row['repo']}.git"
                subprocess.run(
                    ["git", "clone", "--filter=blob:none", "--no-checkout", url, str(workdir)],
                    check=True,
                )
                subprocess.run(
                    ["git", "-C", str(workdir), "checkout", str(row["base_commit"])],
                    check=True,
                )
    (out / "fresh_public.jsonl").write_text(
        "\n".join(json.dumps(record, ensure_ascii=False) for record in public) + "\n",
        encoding="utf-8",
    )
    (out / "fresh_task_config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps({"tasks": [r["instance_id"] for r in public], "n": len(public)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
