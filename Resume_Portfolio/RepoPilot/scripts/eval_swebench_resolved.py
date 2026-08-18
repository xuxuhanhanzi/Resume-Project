"""Compute a SWE-bench-Live resolved rate for a RepoPilot prediction file.

For each task in ``predictions.jsonl`` (produced by
``run_swebench_live_fresh_holdout.py``):

1. apply the generated ``model_patch`` to the workspace (base_commit),
2. apply the gold ``test_patch`` (NEVER shown to the agent),
3. run the FAIL_TO_PASS (and PASS_TO_PASS) test ids inside the task's eval image
   (networkless, bind-mounted workspace),
4. a task is *resolved* when every FAIL_TO_PASS test passes and no PASS_TO_PASS
   test regresses.

This is self-contained: it uses only the parquet metadata and the official eval
image, so no external SWE-bench harness is required.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path
from typing import Any

import pandas as pd

REVISION = "a637bd46829f3132e12938c8a0ca93173a977b8e"


def _run_docker_test(
    workspace: Path, image: str, test_ids: list[str], max_chars: int = 4000
) -> tuple[bool, str]:
    """Run pytest on ``test_ids`` inside the eval image.

    SWE-bench PASS_TO_PASS lists can contain hundreds of node ids; passing them
    all on one command line overflows the Windows CreateProcess limit (~32 KiB).
    So we chunk by accumulated length and require every chunk to pass.
    """
    if not test_ids:
        return True, ""
    chunks: list[list[str]] = []
    current: list[str] = []
    current_len = 0
    for tid in test_ids:
        extra = len(tid) + 1
        if current and current_len + extra > max_chars:
            chunks.append(current)
            current = []
            current_len = 0
        current.append(tid)
        current_len += extra
    if current:
        chunks.append(current)

    all_ok = True
    outputs: list[str] = []
    for index, chunk in enumerate(chunks):
        command = [
            "docker",
            "run",
            "--rm",
            "--network",
            "none",
            "--read-only",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "--pids-limit",
            "256",
            "--memory",
            "4g",
            "--cpus",
            "2.0",
            "--user",
            "65534:65534",
            "--tmpfs",
            "/tmp:rw,noexec,nosuid,size=64m",
            "--mount",
            f"type=bind,source={workspace.resolve()},target=/workspace",
            "--workdir",
            "/workspace",
            image,
            "pytest",
            "-q",
            *chunk,
        ]
        completed = subprocess.run(
            command, capture_output=True, text=True, encoding="utf-8", errors="replace"
        )
        ok = completed.returncode == 0
        all_ok = all_ok and ok
        status = "PASS" if ok else "FAIL"
        tail = (completed.stdout + completed.stderr)[-1500:]
        outputs.append(f"[chunk {index + 1}/{len(chunks)}] {status} ({len(chunk)} ids)\n{tail}")
    return all_ok, "\n".join(outputs)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--parquet", required=True)
    parser.add_argument("--workspace-root", required=True)
    parser.add_argument("--image-prefix", default="starryzhang/sweb.eval.x86_64")
    arguments = parser.parse_args()

    df = pd.read_parquet(arguments.parquet)
    meta = {row["instance_id"]: row for _, row in df.iterrows()}

    def _as_list(value: Any) -> list[str]:
        if isinstance(value, str):
            return re.split(r"[\s,]+", value.strip()) if value.strip() else []
        return list(value) if value is not None else []

    predictions = [
        json.loads(line)
        for line in Path(arguments.predictions).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    workspace_root = Path(arguments.workspace_root) / REVISION

    resolved = 0
    report = []
    for pred in predictions:
        instance_id = pred["instance_id"]
        model_patch = pred.get("model_patch", "")
        workspace = (workspace_root / instance_id).resolve(strict=True)
        row = meta[instance_id]
        image = f"{arguments.image_prefix}.{instance_id.replace('__', '_1776_')}:latest"
        fail_to_pass = _as_list(row["FAIL_TO_PASS"])
        pass_to_pass = _as_list(row.get("PASS_TO_PASS", []))
        ftp_ok = False
        ptp_ok = False
        ftp_out = ""
        ptp_out = ""
        try:
            # 1) model patch
            if model_patch.strip():
                subprocess.run(
                    ["git", "-c", "core.longpaths=true", "-C", str(workspace), "apply"],
                    input=model_patch,
                    text=True,
                    check=True,
                    capture_output=True,
                )
            # 2) gold test patch (evaluator-only; never entered the agent)
            test_patch = str(row.get("test_patch") or "")
            if test_patch.strip():
                subprocess.run(
                    ["git", "-c", "core.longpaths=true", "-C", str(workspace), "apply"],
                    input=test_patch,
                    text=True,
                    check=True,
                    capture_output=True,
                )
            # 3) run tests
            ftp_ok, ftp_out = _run_docker_test(workspace, image, list(fail_to_pass))
            ptp_ok, ptp_out = (
                (True, "")
                if not pass_to_pass
                else _run_docker_test(workspace, image, list(pass_to_pass))
            )
            task_resolved = bool(ftp_ok and ptp_ok)
        except subprocess.CalledProcessError as error:
            task_resolved = False
            ftp_out = (error.stdout or "") + (error.stderr or "")
            ptp_out = ""
        finally:
            # restore workspace to base_commit
            subprocess.run(
                ["git", "-c", "core.longpaths=true", "-C", str(workspace), "checkout", "--", "."],
                capture_output=True,
                text=True,
            )
            subprocess.run(
                ["git", "-c", "core.longpaths=true", "-C", str(workspace), "clean", "-fd", "."],
                capture_output=True,
                text=True,
            )
        if task_resolved:
            resolved += 1
        report.append(
            {
                "instance_id": instance_id,
                "resolved": task_resolved,
                "fail_to_pass_pass": ftp_ok,
                "pass_to_pass_pass": ptp_ok,
                "ftp_output_tail": ftp_out[-500:],
                "ptp_output_tail": ptp_out[-500:],
            }
        )
        print(f"{instance_id}: resolved={task_resolved}")

    summary = {
        "tasks": len(predictions),
        "resolved": resolved,
        "resolved_rate": resolved / len(predictions) if predictions else 0.0,
        "report": report,
    }
    out_path = Path(arguments.predictions).parent / "resolved_results.json"
    out_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "report"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
