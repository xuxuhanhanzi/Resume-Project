#!/usr/bin/env python3
"""Resumable, receipt-producing orchestrator for the four-project portfolio."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "experiment-suite.json"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def git_value(*args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(ROOT), *args],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else "unavailable"


def load_manifest(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema_version") != 1 or not isinstance(data.get("stages"), list):
        raise ValueError(
            "experiment manifest must have schema_version=1 and a stages list"
        )
    ids: set[str] = set()
    for stage in data["stages"]:
        required = {"id", "project", "phase", "profiles", "cwd", "command"}
        if not required.issubset(stage) or stage["id"] in ids:
            raise ValueError(f"invalid or duplicate stage: {stage!r}")
        if stage["phase"] not in {"setup", "run"}:
            raise ValueError(f"invalid phase for {stage['id']}")
        if not isinstance(stage["command"], list) or not stage["command"]:
            raise ValueError(f"stage {stage['id']} must use a non-empty argv list")
        ids.add(stage["id"])
    return data


def command_for(stage: dict[str, Any], profile: str) -> list[str]:
    return [str(item).replace("{profile}", profile) for item in stage["command"]]


def fingerprint(manifest_path: Path, stage: dict[str, Any], command: list[str]) -> str:
    payload = {
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "git_commit": git_value("rev-parse", "HEAD"),
        "stage": stage,
        "command": command,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def selected_stages(
    manifest: dict[str, Any], phase: str, profile: str, projects: set[str] | None
) -> list[dict[str, Any]]:
    phases = {"setup", "run"} if phase in {"all", "list"} else {phase}
    return [
        stage
        for stage in manifest["stages"]
        if stage["phase"] in phases
        and profile in stage["profiles"]
        and (
            not projects or stage["project"] in projects or stage["project"] == "suite"
        )
    ]


def run_stage(
    stage: dict[str, Any],
    command: list[str],
    receipt_path: Path,
    log_path: Path,
    fp: str,
) -> int:
    cwd = (ROOT / stage["cwd"]).resolve()
    if ROOT not in (cwd, *cwd.parents):
        raise ValueError(f"stage cwd escapes portfolio root: {cwd}")
    env = os.environ.copy()
    env.update(
        {
            "EXPERIMENT_SUITE_ROOT": str(ROOT),
            "EXPERIMENT_RUN_ID": receipt_path.parents[1].name,
            "PYTHONUNBUFFERED": "1",
        }
    )
    receipt: dict[str, Any] = {
        "schema_version": 1,
        "stage_id": stage["id"],
        "project": stage["project"],
        "phase": stage["phase"],
        "fingerprint": fp,
        "git_commit": git_value("rev-parse", "HEAD"),
        "git_dirty": bool(git_value("status", "--porcelain")),
        "cwd": str(cwd),
        "command": command,
        "started_at": utc_now(),
        "status": "running",
    }
    atomic_json(receipt_path, receipt)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8", errors="replace") as log:
        header = f"[{receipt['started_at']}] $ {shlex.join(command)}\n"
        print(header, end="")
        log.write(header)
        try:
            process = subprocess.Popen(
                command,
                cwd=cwd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            )
            assert process.stdout is not None
            for line in process.stdout:
                print(line, end="")
                log.write(line)
            code = process.wait()
        except FileNotFoundError as exc:
            message = f"launcher not found: {exc}\n"
            print(message, end="", file=sys.stderr)
            log.write(message)
            code = 127
        except KeyboardInterrupt:
            if "process" in locals():
                process.terminate()
            code = 130
    receipt.update(
        {
            "finished_at": utc_now(),
            "exit_code": code,
            "status": "passed" if code == 0 else "failed",
            "log": str(log_path),
        }
    )
    atomic_json(receipt_path, receipt)
    return code


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("setup", "run", "all", "status", "list"))
    parser.add_argument("--profile", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--run-id", help="resume namespace (default: <profile>-latest)")
    parser.add_argument(
        "--project", action="append", help="project name; repeat to select several"
    )
    parser.add_argument("--from-stage", help="skip stages before this id")
    parser.add_argument(
        "--rerun", action="store_true", help="rerun successful matching receipts"
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest_path = args.manifest.resolve()
    manifest = load_manifest(manifest_path)
    run_id = args.run_id or f"{args.profile}-latest"
    run_root = ROOT / "artifacts" / "experiment-suite" / run_id
    projects = set(args.project) if args.project else None

    if args.phase == "status":
        receipts = sorted((run_root / "receipts").glob("*.json"))
        if not receipts:
            print(f"No receipts found for run-id={run_id}")
            return 1
        for path in receipts:
            item = json.loads(path.read_text(encoding="utf-8"))
            print(
                f"{item['stage_id']:<28} {item['status']:<8} exit={item.get('exit_code', '-')}"
            )
        return 0

    stages = selected_stages(manifest, args.phase, args.profile, projects)
    if args.from_stage:
        ids = [stage["id"] for stage in stages]
        if args.from_stage not in ids:
            raise SystemExit(
                f"--from-stage {args.from_stage!r} is not in the selected plan"
            )
        stages = stages[ids.index(args.from_stage) :]

    print(f"Portfolio: {ROOT}")
    print(f"Run: {run_id} | profile={args.profile} | phase={args.phase}")
    print(f"Host: {platform.platform()} | commit={git_value('rev-parse', 'HEAD')}")
    for index, stage in enumerate(stages, 1):
        print(f"{index:02d}. {stage['id']} ({stage['project']}/{stage['phase']})")
    if args.phase == "list" or args.dry_run:
        return 0

    run_root.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {
        "schema_version": 1,
        "run_id": run_id,
        "profile": args.profile,
        "phase": args.phase,
        "git_commit": git_value("rev-parse", "HEAD"),
        "started_at": utc_now(),
        "stages": [],
        "status": "running",
    }
    atomic_json(run_root / "summary.json", summary)

    for stage in stages:
        command = command_for(stage, args.profile)
        fp = fingerprint(manifest_path, stage, command)
        receipt_path = run_root / "receipts" / f"{stage['id']}.json"
        log_path = run_root / "logs" / f"{stage['id']}.log"
        previous = None
        if receipt_path.exists():
            previous = json.loads(receipt_path.read_text(encoding="utf-8"))
        if (
            not args.rerun
            and previous
            and previous.get("status") == "passed"
            and previous.get("fingerprint") == fp
        ):
            print(f"SKIP {stage['id']}: matching successful receipt")
            summary["stages"].append({"id": stage["id"], "status": "skipped-resume"})
            atomic_json(run_root / "summary.json", summary)
            continue

        print(f"START {stage['id']}")
        code = run_stage(stage, command, receipt_path, log_path, fp)
        summary["stages"].append(
            {"id": stage["id"], "status": "passed" if code == 0 else "failed"}
        )
        atomic_json(run_root / "summary.json", summary)
        if code != 0:
            summary.update(
                {
                    "status": "failed",
                    "failed_stage": stage["id"],
                    "finished_at": utc_now(),
                }
            )
            atomic_json(run_root / "summary.json", summary)
            print(
                f"FAILED {stage['id']} (exit {code}); fix the logged cause and rerun the same command.",
                file=sys.stderr,
            )
            return code
        print(f"PASS {stage['id']}")

    summary.update({"status": "passed", "finished_at": utc_now()})
    atomic_json(run_root / "summary.json", summary)
    print(f"All selected stages passed. Summary: {run_root / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
