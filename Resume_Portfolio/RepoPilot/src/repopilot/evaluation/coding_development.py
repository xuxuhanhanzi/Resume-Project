"""Versioned synthetic development fixtures for repair-loop diagnostics.

This module deliberately has no relation to a scored benchmark or holdout.  A
fixture is materialized into a fresh artifact directory before a run; the
project workspace, existing artifacts, and any user source tree are never used
as the agent's writable workspace.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from time import monotonic
from typing import Any

from repopilot.core.budgets import RunBudget
from repopilot.core.contracts import AgentState, RunStatus
from repopilot.evaluation.coding_funnel import (
    CodingFunnelSummary,
    load_coding_funnel_records,
    summarize_coding_funnel,
)
from repopilot.security.redaction import redact_json_value, redact_text
from repopilot.task import PublicTaskSpec

_MANIFEST_NAME = "coding_development_v1.json"
_HOLDOUT_REGISTRY_NAME = "coding_release_holdout_v1.json"
_MAX_CASES = 20
_MAX_FILE_CHARACTERS = 24_000
_RUN_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}")
_MANIFEST_KEYS = {"schema_version", "suite_id", "cases"}
_CASE_KEYS = {"task_id", "problem_statement", "source_path", "source", "test_path", "test"}
_LOCALIZATION_TOOLS = frozenset({"list_files", "read_file", "search_text", "find_symbol"})
_RECEIPT_NAME = "run.receipt.json"
_OUTCOMES_NAME = "outcomes.redacted.jsonl"
_MAX_FAILURE_REASON = 512
_MAX_RECEIPT_BYTES = 1_000_000
_MAX_RECEIPT_TEXT = 512
_RECEIPT_KEYS = frozenset(
    {
        "schema_version",
        "kind",
        "suite_id",
        "manifest_sha256",
        "run_id",
        "provider",
        "model",
        "selected_case_ids",
        "cases",
        "elapsed_seconds",
        "notes",
        "execution_profile",
        "outcomes_sha256",
    }
)
_RECEIPT_REQUIRED_KEYS = _RECEIPT_KEYS - {"execution_profile", "outcomes_sha256"}
_RECEIPT_CASE_KEYS = frozenset(
    {
        "task_id",
        "status",
        "failure_reason",
        "iterations",
        "tool_calls",
        "input_tokens",
        "output_tokens",
        "elapsed_seconds",
    }
)
_EXECUTION_PROFILE_REQUIRED_KEYS = frozenset(
    {
        "repopilot_version",
        "auto_finalize_after_successful_test",
        "task_budget",
        "tool_surface",
    }
)
_EXECUTION_PROFILE_KEYS = _EXECUTION_PROFILE_REQUIRED_KEYS | {"fixture_verification_command"}
_TASK_BUDGET_KEYS = frozenset(
    {"max_iterations", "max_tool_calls", "max_total_tokens", "max_wall_seconds"}
)
_RECEIPT_STATUSES = frozenset({status.value for status in RunStatus} | {"runtime_error"})


@dataclass(frozen=True, slots=True)
class CodingDevelopmentCase:
    """One public, small, intentionally failing source/test fixture."""

    task_id: str
    problem_statement: str
    source_path: str
    source: str
    test_path: str
    test: str

    @classmethod
    def from_mapping(cls, raw: object, *, index: int) -> CodingDevelopmentCase:
        if not isinstance(raw, dict) or set(raw) != _CASE_KEYS:
            raise ValueError(
                f"development case {index} must contain exactly: {', '.join(sorted(_CASE_KEYS))}"
            )
        values = {key: raw[key] for key in _CASE_KEYS}
        if not all(isinstance(value, str) for value in values.values()):
            raise ValueError(f"development case {index} fields must all be strings")
        task_id = str(values["task_id"]).strip()
        prompt = str(values["problem_statement"]).strip()
        if not task_id or not prompt or not _RUN_ID.fullmatch(task_id):
            raise ValueError(
                f"development case {index} has an invalid task_id or problem statement"
            )
        source_path = _relative_path(str(values["source_path"]), field="source_path")
        test_path = _relative_path(str(values["test_path"]), field="test_path")
        if source_path == test_path:
            raise ValueError(f"development case {index} source_path and test_path must differ")
        source = str(values["source"])
        test = str(values["test"])
        if (
            not source.strip()
            or not test.strip()
            or max(len(source), len(test)) > _MAX_FILE_CHARACTERS
        ):
            raise ValueError(f"development case {index} has empty or oversized fixture text")
        return cls(task_id, prompt, source_path, source, test_path, test)


@dataclass(frozen=True, slots=True)
class CodingDevelopmentManifest:
    """A frozen, public-only fixture collection."""

    schema_version: int
    suite_id: str
    cases: tuple[CodingDevelopmentCase, ...]

    @classmethod
    def load(cls, path: Path | None = None) -> CodingDevelopmentManifest:
        source = (path or default_coding_development_manifest()).resolve(strict=True)
        try:
            raw = json.loads(source.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError(f"could not load coding development manifest: {error}") from error
        if not isinstance(raw, dict) or set(raw) != _MANIFEST_KEYS:
            raise ValueError(
                "coding development manifest must contain exactly: "
                + ", ".join(sorted(_MANIFEST_KEYS))
            )
        schema_version = raw["schema_version"]
        suite_id = raw["suite_id"]
        cases = raw["cases"]
        if schema_version != 1 or not isinstance(suite_id, str) or not _RUN_ID.fullmatch(suite_id):
            raise ValueError(
                "coding development manifest has an unsupported version or invalid suite_id"
            )
        if not isinstance(cases, list) or not 1 <= len(cases) <= _MAX_CASES:
            raise ValueError(f"coding development manifest must contain 1-{_MAX_CASES} cases")
        parsed = tuple(
            CodingDevelopmentCase.from_mapping(item, index=index)
            for index, item in enumerate(cases, 1)
        )
        if len({case.task_id for case in parsed}) != len(parsed):
            raise ValueError("coding development manifest contains duplicate task_id values")
        return cls(schema_version, suite_id, parsed)

    def plan(self) -> dict[str, object]:
        """Return public metadata only; never materialize or execute a fixture."""
        return {
            "suite_id": self.suite_id,
            "schema_version": self.schema_version,
            "cases": [
                {
                    "task_id": case.task_id,
                    "source_path": case.source_path,
                    "test_path": case.test_path,
                }
                for case in self.cases
            ],
            "notes": [
                "Synthetic public development fixtures only; this is not a capability benchmark.",
                "No holdout task, hidden test, or model patch is included.",
                "A run materializes fresh artifact workspaces and does not modify the project.",
            ],
        }


def default_coding_development_manifest() -> Path:
    """Locate the bundled manifest in both editable and wheel installations."""
    return Path(__file__).with_name("fixtures") / _MANIFEST_NAME


def coding_release_holdout_plan() -> dict[str, object]:
    """Load only the non-sensitive, immutable release-holdout contract.

    This registry deliberately contains no task identity, source, hidden test,
    gold patch, evaluator output, or model prompt.  It makes the separation
    between public development diagnostics and the existing external official
    evaluator explicit without leaking evaluator material into an agent turn.
    """

    path = Path(__file__).with_name("fixtures") / _HOLDOUT_REGISTRY_NAME
    try:
        raw: Any = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"could not load coding release holdout registry: {error}") from error
    required = {
        "schema_version",
        "suite_id",
        "kind",
        "task_count",
        "visibility",
        "selection",
        "evaluator",
        "execution_requirements",
        "evidence_paths",
        "release_rule",
    }
    if not isinstance(raw, dict) or set(raw) != required:
        raise ValueError("coding release holdout registry has an invalid schema")
    if (
        raw["schema_version"] != 1
        or raw["kind"] != "external_immutable_holdout"
        or raw["task_count"] != 5
        or not isinstance(raw["suite_id"], str)
        or not _RUN_ID.fullmatch(raw["suite_id"])
    ):
        raise ValueError("coding release holdout registry has invalid identity fields")
    for key in ("visibility", "selection", "evaluator", "release_rule"):
        if not isinstance(raw[key], str) or not raw[key].strip() or len(raw[key]) > 1_000:
            raise ValueError(f"coding release holdout registry has invalid {key}")
    for key in ("execution_requirements", "evidence_paths"):
        values = raw[key]
        if (
            not isinstance(values, list)
            or not 1 <= len(values) <= 16
            or not all(isinstance(value, str) and 1 <= len(value) <= 500 for value in values)
        ):
            raise ValueError(f"coding release holdout registry has invalid {key}")
    return {
        "schema_version": 1,
        "suite_id": raw["suite_id"],
        "kind": raw["kind"],
        "task_count": raw["task_count"],
        "visibility": raw["visibility"],
        "selection": raw["selection"],
        "evaluator": raw["evaluator"],
        "execution_requirements": list(raw["execution_requirements"]),
        "evidence_paths": list(raw["evidence_paths"]),
        "release_rule": raw["release_rule"],
    }


def validate_run_id(value: str) -> str:
    """Keep artifact materialization names traversal-free and bounded."""
    if not _RUN_ID.fullmatch(value):
        raise ValueError("run id must contain 1-64 letters, digits, '.', '_' or '-'")
    return value


def materialize_coding_development_cases(
    manifest: CodingDevelopmentManifest,
    *,
    artifacts: Path,
    run_id: str,
    max_cases: int,
) -> tuple[PublicTaskSpec, ...]:
    """Create one new, synthetic workspace per selected case without overwriting data."""
    if not 1 <= max_cases <= _MAX_CASES:
        raise ValueError(f"max_cases must be between 1 and {_MAX_CASES}")
    selected = manifest.cases[:max_cases]
    run_root = artifacts.resolve() / "coding_development" / "runs" / validate_run_id(run_id)
    if run_root.exists():
        raise ValueError(
            f"coding development run already exists: {run_root}; choose a new --run-id"
        )
    run_root.mkdir(parents=True, exist_ok=False)
    specs: list[PublicTaskSpec] = []
    for case in selected:
        workspace = run_root / case.task_id / "workspace"
        source = workspace / case.source_path
        test = workspace / case.test_path
        source.parent.mkdir(parents=True, exist_ok=True)
        test.parent.mkdir(parents=True, exist_ok=True)
        source.write_text(case.source, encoding="utf-8", newline="")
        test.write_text(case.test, encoding="utf-8", newline="")
        specs.append(
            PublicTaskSpec(
                task_id=case.task_id,
                workspace=workspace,
                problem_statement=case.problem_statement,
                allowed_paths=("src", "tests"),
                visible_tests=(case.test_path,),
                test_command=(sys.executable, "-m", "pytest", "-q"),
                max_changed_files=2,
                budget=RunBudget(
                    max_iterations=8,
                    max_tool_calls=20,
                    max_total_tokens=12_000,
                    max_wall_seconds=180.0,
                ),
                trusted_fixture=True,
            )
        )
    return tuple(specs)


def development_record(task_id: str, state: AgentState) -> dict[str, object]:
    """Derive a redacted funnel record from durable runtime observations."""
    localized = False
    patch_applied = False
    for message in state.messages:
        if message.role != "tool" or message.name is None:
            continue
        try:
            raw = json.loads(message.content)
        except json.JSONDecodeError:
            continue
        if not isinstance(raw, dict) or raw.get("ok") is not True:
            continue
        if message.name in _LOCALIZATION_TOOLS:
            localized = True
        elif message.name == "apply_patch":
            patch_applied = True
    # The sentinel represents a successful patch operation without storing its
    # contents in a report that is intended to be shareable.
    return {
        "task_id": task_id,
        "localized": localized,
        "patch": "applied" if patch_applied else "",
        "patch_applied": patch_applied,
        "verification_passed": state.status is RunStatus.COMPLETED,
    }


def development_case_receipt(
    task_id: str,
    state: AgentState | None,
    *,
    elapsed_seconds: float,
    runtime_error: BaseException | None = None,
) -> dict[str, object]:
    """Create a redacted, payload-free outcome receipt for one fixture.

    ``outcomes.redacted.jsonl`` intentionally remains the small stable funnel
    contract.  This separate receipt preserves enough operational detail to
    distinguish a model/budget/runtime stop without persisting model messages,
    tool inputs, tool output, patches, or test logs.
    """

    reason: str | None
    if runtime_error is not None:
        reason = _bounded_reason(runtime_error)
        status = "runtime_error"
    elif state is not None:
        reason = _bounded_reason(state.failure_reason) if state.failure_reason else None
        status = state.status.value
    else:
        reason = "runtime produced no state"
        status = "runtime_error"
    return {
        "task_id": task_id,
        "status": status,
        "failure_reason": reason,
        "iterations": state.iteration if state is not None else 0,
        "tool_calls": state.tool_calls if state is not None else 0,
        "input_tokens": state.input_tokens if state is not None else 0,
        "output_tokens": state.output_tokens if state is not None else 0,
        "elapsed_seconds": round(max(elapsed_seconds, 0.0), 3),
    }


def development_run_receipt(
    manifest: CodingDevelopmentManifest,
    *,
    run_id: str,
    provider: str,
    model: str,
    selected_case_ids: tuple[str, ...],
    cases: list[dict[str, object]],
    elapsed_seconds: float,
    execution_profile: dict[str, object] | None = None,
    outcomes_sha256: str | None = None,
) -> dict[str, object]:
    """Return comparable, local-only metadata for a synthetic diagnostic run."""

    receipt: dict[str, object] = {
        "schema_version": 1,
        "kind": "synthetic_coding_development_receipt",
        "suite_id": manifest.suite_id,
        "manifest_sha256": _manifest_sha256(manifest),
        "run_id": validate_run_id(run_id),
        "provider": provider,
        "model": model,
        "selected_case_ids": list(selected_case_ids),
        "cases": cases,
        "elapsed_seconds": round(max(elapsed_seconds, 0.0), 3),
        "notes": [
            "Public synthetic development diagnostics only; not a scored benchmark.",
            "The receipt excludes prompts, patches, tool payloads, test logs, and credentials.",
            "Compare only runs with the same suite digest, selected case IDs, provider, and model.",
        ],
    }
    if execution_profile is not None:
        receipt["execution_profile"] = execution_profile
    if outcomes_sha256 is not None:
        receipt["outcomes_sha256"] = outcomes_sha256
    return receipt


def write_development_records(path: Path, records: list[dict[str, object]]) -> CodingFunnelSummary:
    """Persist redacted JSONL once, then validate it with the funnel contract."""
    if path.exists():
        raise ValueError(f"development record already exists: {path}")
    payload = "".join(
        json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n" for record in records
    )
    path.write_text(payload, encoding="utf-8", newline="")
    return summarize_coding_funnel(load_coding_funnel_records(path))


def development_outcomes_sha256(path: Path) -> str:
    """Return the exact byte digest of a validated redacted outcome file."""

    try:
        source = path.resolve(strict=True)
        load_coding_funnel_records(source)
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
    except (OSError, ValueError) as error:
        raise ValueError(f"could not hash coding development outcomes: {error}") from error
    return digest


def write_development_receipt(path: Path, receipt: dict[str, object]) -> None:
    """Persist one redacted run receipt without replacing an earlier baseline."""

    if path.name != _RECEIPT_NAME:
        raise ValueError(f"development receipt must be named {_RECEIPT_NAME}")
    if path.exists():
        raise ValueError(f"development receipt already exists: {path}")
    redacted = redact_json_value(_validated_development_receipt(receipt))
    path.write_text(
        json.dumps(redacted, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="",
    )


def inspect_development_run(run_directory: Path) -> dict[str, object]:
    """Read a completed development run locally and suggest evidence-led next steps.

    This inspection never creates a provider, launches a command, or reads raw
    runtime traces.  It deliberately bases every recommendation on an observed
    funnel category, so follow-up work cannot claim an improvement before a
    comparable diagnostic exists.
    """

    receipt, summary = _load_development_run(run_directory)
    return {
        "receipt": redact_json_value(receipt),
        "summary": summary.to_dict(),
        "next_actions": _funnel_next_actions(summary),
        "note": (
            "Recommendations are diagnostic follow-ups, not proof of a capability improvement. "
            "Make one bounded change, add a regression test, then repeat the same fixture set."
        ),
    }


def verify_development_run(run_directory: Path) -> dict[str, object]:
    """Verify a local receipt/outcome binding without creating a provider."""

    receipt, summary = _load_development_run(run_directory, verify_outcomes_digest=False)
    expected = receipt.get("outcomes_sha256")
    actual = development_outcomes_sha256(run_directory.resolve(strict=True) / _OUTCOMES_NAME)
    bound = isinstance(expected, str)
    return {
        "receipt": _comparison_identity(receipt),
        "integrity": {
            "outcomes_sha256": actual,
            "expected_outcomes_sha256": expected,
            "bound": bound,
            "verified": bound and actual == expected,
            "classification": (
                "bound_verified"
                if bound and actual == expected
                else "bound_mismatch"
                if bound
                else "legacy_unbound"
            ),
        },
        "reproducibility": {
            "execution_profile_recorded": isinstance(receipt.get("execution_profile"), dict),
            "selected_case_count": len(_receipt_task_ids(receipt)),
            "verified_case_count": summary.verification_passed,
        },
        "note": (
            "This validates stored local artifact integrity, not a benchmark result or a "
            "general capability claim. Legacy receipts remain readable but are unbound."
        ),
    }


def development_reliability_dashboard(run_directories: tuple[Path, ...]) -> dict[str, object]:
    """Summarize bounded local diagnostic receipts without overstating evidence.

    The bundled fixtures are intentionally public development fixtures.  This
    dashboard makes their integrity and observed categories easy to inspect,
    while always refusing to call them a holdout or a general capability score.
    It neither constructs a provider nor reads raw traces, prompts, patches, or
    hidden evaluator content.
    """

    if not 1 <= len(run_directories) <= 50:
        raise ValueError("development reliability dashboard requires 1-50 run directories")
    runs: list[dict[str, object]] = []
    total_cases = 0
    verified_cases = 0
    integrity_failures = 0
    for supplied in run_directories:
        receipt, summary = _load_development_run(supplied, verify_outcomes_digest=False)
        directory = supplied.resolve(strict=True)
        expected = receipt.get("outcomes_sha256")
        actual = development_outcomes_sha256(directory / _OUTCOMES_NAME)
        verified = isinstance(expected, str) and actual == expected
        integrity = (
            "bound_verified"
            if verified
            else "bound_mismatch"
            if isinstance(expected, str)
            else "legacy_unbound"
        )
        total_cases += len(summary.records)
        verified_cases += summary.verification_passed
        if integrity != "bound_verified":
            integrity_failures += 1
        runs.append(
            {
                "directory": str(directory),
                "identity": _comparison_identity(receipt),
                "integrity": integrity,
                "categories": dict(sorted(summary.categories.items())),
                "case_count": len(summary.records),
                "verified_case_count": summary.verification_passed,
            }
        )
    return {
        "schema_version": 1,
        "kind": "coding_development_reliability_dashboard",
        "runs": runs,
        "aggregate": {
            "run_count": len(runs),
            "case_count": total_cases,
            "verified_case_count": verified_cases,
            "receipts_without_verified_outcome_binding": integrity_failures,
        },
        "evidence_boundary": {
            "suite": "public synthetic development fixtures",
            "release_holdout_required": True,
            "capability_score_claimed": False,
            "note": (
                "A release decision also requires a separately frozen, unseen holdout under "
                "its official evaluator. This dashboard is development evidence only."
            ),
        },
    }


def compare_development_runs(
    baseline_directory: Path, candidate_directory: Path
) -> dict[str, object]:
    """Compare two local synthetic runs without claiming a general capability gain.

    A comparison is fully controlled only when both receipts specify an identical
    execution profile in addition to the fixture, provider, model, and case set.
    Older receipts without that field remain reportable but are explicitly
    labelled directional.
    """

    baseline, baseline_summary = _load_development_run(baseline_directory)
    candidate, candidate_summary = _load_development_run(candidate_directory)
    comparable_fields = (
        "suite_id",
        "manifest_sha256",
        "provider",
        "model",
        "selected_case_ids",
    )
    field_matches = {field: baseline[field] == candidate[field] for field in comparable_fields}
    baseline_profile = baseline.get("execution_profile")
    candidate_profile = candidate.get("execution_profile")
    profiles_present = isinstance(baseline_profile, dict) and isinstance(candidate_profile, dict)
    profile_match = profiles_present and baseline_profile == candidate_profile
    fully_controlled = all(field_matches.values()) and profile_match
    baseline_categories = {record.task_id: record.category for record in baseline_summary.records}
    candidate_categories = {record.task_id: record.category for record in candidate_summary.records}
    task_ids = _receipt_task_ids(baseline)
    case_changes = [
        {
            "task_id": task_id,
            "baseline_category": baseline_categories[task_id],
            "candidate_category": candidate_categories[task_id],
        }
        for task_id in task_ids
    ]
    categories = sorted(set(baseline_summary.categories) | set(candidate_summary.categories))
    return {
        "baseline": _comparison_identity(baseline),
        "candidate": _comparison_identity(candidate),
        "comparability": {
            "field_matches": field_matches,
            "execution_profiles_present": profiles_present,
            "execution_profile_match": profile_match,
            "fully_controlled": fully_controlled,
            "classification": "controlled" if fully_controlled else "directional_only",
            "note": (
                "This is a public synthetic diagnostic comparison, not a benchmark or a "
                "general capability claim. Missing or changed execution profiles make the "
                "comparison directional only."
            ),
        },
        "funnel_delta": {
            category: candidate_summary.categories.get(category, 0)
            - baseline_summary.categories.get(category, 0)
            for category in categories
        },
        "case_changes": case_changes,
    }


def _load_development_run(
    run_directory: Path,
    *,
    verify_outcomes_digest: bool = True,
) -> tuple[dict[str, object], CodingFunnelSummary]:
    try:
        directory = run_directory.resolve(strict=True)
    except OSError as error:
        raise ValueError(f"could not resolve coding development run path: {error}") from error
    if not directory.is_dir():
        raise ValueError("coding development run path must be a directory")
    receipt = _load_development_receipt(directory / _RECEIPT_NAME)
    summary = summarize_coding_funnel(load_coding_funnel_records(directory / _OUTCOMES_NAME))
    receipt_ids = _receipt_task_ids(receipt)
    outcome_ids = [record.task_id for record in summary.records]
    if receipt_ids != outcome_ids:
        raise ValueError("coding development receipt and funnel outcomes disagree on task IDs")
    expected_digest = receipt.get("outcomes_sha256")
    if expected_digest is not None and verify_outcomes_digest:
        actual_digest = development_outcomes_sha256(directory / _OUTCOMES_NAME)
        if actual_digest != expected_digest:
            raise ValueError("coding development receipt outcomes_sha256 does not match outcomes")
    return receipt, summary


def _load_development_receipt(path: Path) -> dict[str, object]:
    try:
        source = path.resolve(strict=True)
        if source.stat().st_size > _MAX_RECEIPT_BYTES:
            raise ValueError(f"coding development receipt exceeds {_MAX_RECEIPT_BYTES:,} bytes")
        raw: Any = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"could not load coding development receipt: {error}") from error
    return _validated_development_receipt(raw)


def _validated_development_receipt(raw: object) -> dict[str, object]:
    """Accept only the bounded, payload-free receipt contract used by reports."""

    if not isinstance(raw, dict):
        raise ValueError("coding development receipt must be a JSON object")
    keys = frozenset(raw)
    unsupported = keys - _RECEIPT_KEYS
    missing = _RECEIPT_REQUIRED_KEYS - keys
    if unsupported or missing:
        details: list[str] = []
        if unsupported:
            details.append("unsupported fields: " + ", ".join(sorted(unsupported)))
        if missing:
            details.append("missing fields: " + ", ".join(sorted(missing)))
        raise ValueError("coding development receipt has " + "; ".join(details))
    if raw["schema_version"] != 1 or raw["kind"] != "synthetic_coding_development_receipt":
        raise ValueError("coding development receipt has an unsupported schema")
    suite_id = _receipt_text(raw["suite_id"], field="suite_id", pattern=True)
    manifest_sha256 = raw["manifest_sha256"]
    if not isinstance(manifest_sha256, str) or not re.fullmatch(r"[a-f0-9]{64}", manifest_sha256):
        raise ValueError("coding development receipt has an invalid manifest_sha256")
    run_id = validate_run_id(_receipt_text(raw["run_id"], field="run_id", pattern=True))
    provider = _receipt_text(raw["provider"], field="provider", pattern=True)
    model = _receipt_text(raw["model"], field="model")
    selected_case_ids = _receipt_id_list(raw["selected_case_ids"], field="selected_case_ids")
    raw_cases = raw["cases"]
    if not isinstance(raw_cases, list) or not 1 <= len(raw_cases) <= _MAX_CASES:
        raise ValueError(f"coding development receipt must contain 1-{_MAX_CASES} cases")
    cases = [_validated_receipt_case(case, index=index) for index, case in enumerate(raw_cases, 1)]
    case_ids = [str(case["task_id"]) for case in cases]
    if selected_case_ids != case_ids:
        raise ValueError(
            "coding development receipt case IDs must match selected_case_ids in order"
        )
    notes = raw["notes"]
    if not isinstance(notes, list) or not 1 <= len(notes) <= 5:
        raise ValueError("coding development receipt notes must contain 1-5 entries")
    safe_notes = [_receipt_text(note, field="notes") for note in notes]
    receipt: dict[str, object] = {
        "schema_version": 1,
        "kind": "synthetic_coding_development_receipt",
        "suite_id": suite_id,
        "manifest_sha256": manifest_sha256,
        "run_id": run_id,
        "provider": provider,
        "model": model,
        "selected_case_ids": selected_case_ids,
        "cases": cases,
        "elapsed_seconds": _receipt_elapsed(raw["elapsed_seconds"], field="elapsed_seconds"),
        "notes": safe_notes,
    }
    if "execution_profile" in raw:
        receipt["execution_profile"] = _validated_execution_profile(raw["execution_profile"])
    if "outcomes_sha256" in raw:
        outcomes_sha256 = raw["outcomes_sha256"]
        if not isinstance(outcomes_sha256, str) or not re.fullmatch(
            r"[a-f0-9]{64}", outcomes_sha256
        ):
            raise ValueError("coding development receipt has an invalid outcomes_sha256")
        receipt["outcomes_sha256"] = outcomes_sha256
    return receipt


def _validated_receipt_case(raw: object, *, index: int) -> dict[str, object]:
    if not isinstance(raw, dict) or frozenset(raw) != _RECEIPT_CASE_KEYS:
        raise ValueError(f"coding development receipt case {index} has an invalid schema")
    status = raw["status"]
    if not isinstance(status, str) or status not in _RECEIPT_STATUSES:
        raise ValueError(f"coding development receipt case {index} has an invalid status")
    reason = raw["failure_reason"]
    if reason is not None and (not isinstance(reason, str) or len(reason) > _MAX_FAILURE_REASON):
        raise ValueError(f"coding development receipt case {index} has an invalid failure_reason")
    return {
        "task_id": _receipt_text(raw["task_id"], field=f"cases[{index}].task_id", pattern=True),
        "status": status,
        "failure_reason": reason,
        "iterations": _receipt_count(raw["iterations"], field=f"cases[{index}].iterations"),
        "tool_calls": _receipt_count(raw["tool_calls"], field=f"cases[{index}].tool_calls"),
        "input_tokens": _receipt_count(raw["input_tokens"], field=f"cases[{index}].input_tokens"),
        "output_tokens": _receipt_count(
            raw["output_tokens"], field=f"cases[{index}].output_tokens"
        ),
        "elapsed_seconds": _receipt_elapsed(
            raw["elapsed_seconds"], field=f"cases[{index}].elapsed_seconds"
        ),
    }


def _validated_execution_profile(raw: object) -> dict[str, object]:
    if not isinstance(raw, dict):
        raise ValueError("coding development receipt execution_profile has an invalid schema")
    profile_keys = frozenset(raw)
    if profile_keys - _EXECUTION_PROFILE_KEYS or _EXECUTION_PROFILE_REQUIRED_KEYS - profile_keys:
        raise ValueError("coding development receipt execution_profile has an invalid schema")
    budget = raw["task_budget"]
    if not isinstance(budget, dict) or frozenset(budget) != _TASK_BUDGET_KEYS:
        raise ValueError("coding development receipt task_budget has an invalid schema")
    surface = raw["tool_surface"]
    if not isinstance(surface, list) or not 1 <= len(surface) <= 16:
        raise ValueError("coding development receipt tool_surface must contain 1-16 tools")
    tools = [_receipt_text(item, field="tool_surface", pattern=True) for item in surface]
    if len(set(tools)) != len(tools):
        raise ValueError("coding development receipt tool_surface contains duplicate tools")
    auto_finalize = raw["auto_finalize_after_successful_test"]
    if not isinstance(auto_finalize, bool):
        raise ValueError("coding development receipt auto-finalize setting must be boolean")
    profile: dict[str, object] = {
        "repopilot_version": _receipt_text(raw["repopilot_version"], field="repopilot_version"),
        "auto_finalize_after_successful_test": auto_finalize,
        "task_budget": {
            "max_iterations": _receipt_count(budget["max_iterations"], field="max_iterations"),
            "max_tool_calls": _receipt_count(budget["max_tool_calls"], field="max_tool_calls"),
            "max_total_tokens": _receipt_count(
                budget["max_total_tokens"], field="max_total_tokens"
            ),
            "max_wall_seconds": _receipt_elapsed(
                budget["max_wall_seconds"], field="max_wall_seconds"
            ),
        },
        "tool_surface": tools,
    }
    if "fixture_verification_command" in raw:
        command = raw["fixture_verification_command"]
        if command != ["python", "-m", "pytest", "-q"]:
            raise ValueError(
                "coding development receipt has an invalid fixture verification command"
            )
        profile["fixture_verification_command"] = list(command)
    return profile


def _receipt_id_list(raw: object, *, field: str) -> list[str]:
    if not isinstance(raw, list) or not 1 <= len(raw) <= _MAX_CASES:
        raise ValueError(f"coding development receipt {field} must contain 1-{_MAX_CASES} task IDs")
    task_ids = [_receipt_text(item, field=field, pattern=True) for item in raw]
    if len(set(task_ids)) != len(task_ids):
        raise ValueError(f"coding development receipt {field} contains duplicate task IDs")
    return task_ids


def _receipt_task_ids(receipt: dict[str, object]) -> list[str]:
    selected = receipt["selected_case_ids"]
    assert isinstance(selected, list)
    assert all(isinstance(item, str) for item in selected)
    return list(selected)


def _receipt_text(value: object, *, field: str, pattern: bool = False) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > _MAX_RECEIPT_TEXT:
        raise ValueError(f"coding development receipt {field} must be non-empty bounded text")
    text = value.strip()
    if pattern and not _RUN_ID.fullmatch(text):
        raise ValueError(f"coding development receipt {field} contains invalid characters")
    return text


def _receipt_count(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"coding development receipt {field} must be a non-negative integer")
    return value


def _receipt_elapsed(value: object, *, field: str) -> float | int:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
        raise ValueError(f"coding development receipt {field} must be a non-negative number")
    return value


def _comparison_identity(receipt: dict[str, object]) -> dict[str, object]:
    return {
        field: receipt[field]
        for field in (
            "run_id",
            "suite_id",
            "manifest_sha256",
            "provider",
            "model",
            "selected_case_ids",
        )
    }


def measure_elapsed(started_at: float) -> float:
    """Keep elapsed-time capture testable without exposing a clock in receipts."""

    return monotonic() - started_at


def _manifest_sha256(manifest: CodingDevelopmentManifest) -> str:
    canonical = {
        "schema_version": manifest.schema_version,
        "suite_id": manifest.suite_id,
        "cases": [
            {
                "task_id": case.task_id,
                "problem_statement": case.problem_statement,
                "source_path": case.source_path,
                "source": case.source,
                "test_path": case.test_path,
                "test": case.test,
            }
            for case in manifest.cases
        ],
    }
    encoded = json.dumps(canonical, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _bounded_reason(value: object) -> str:
    text = redact_text(str(value)).replace("\r", " ").replace("\n", " ").strip()
    return text[:_MAX_FAILURE_REASON] or "runtime error"


def _funnel_next_actions(summary: CodingFunnelSummary) -> list[str]:
    """Map observed categories to a small, deliberately non-automatic P17 backlog."""

    actions: list[str] = []
    category_actions = {
        "localization_failed": (
            "Inspect the local trace for the missing discovery step; consider a narrow "
            "read/search-before-edit runtime correction."
        ),
        "empty_patch": (
            "Inspect the final model turn and tool trace for an omitted apply_patch; improve "
            "the implementation prompt or tool-selection guard only with a regression test."
        ),
        "patch_apply_failed": (
            "Inspect the local patch-conflict receipt; prioritize fresh-read/conflict recovery "
            "before retrying an edit."
        ),
        "verification_failed": (
            "Inspect the persisted local verification result; prioritize one bounded "
            "test-repair loop and a regression test for the observed failure."
        ),
    }
    for category, action in category_actions.items():
        count = summary.categories.get(category, 0)
        if count:
            actions.append(f"{category}: {count} case(s). {action}")
    if not actions:
        actions.append(
            "All selected fixtures verified; freeze this receipt as a baseline before changing "
            "runtime behavior."
        )
    return actions


def _relative_path(value: str, *, field: str) -> str:
    normalized = value.replace("\\", "/")
    candidate = PurePosixPath(normalized)
    if not normalized or candidate.is_absolute() or ".." in candidate.parts:
        raise ValueError(f"{field} must be a non-empty traversal-free relative path")
    return str(candidate)
