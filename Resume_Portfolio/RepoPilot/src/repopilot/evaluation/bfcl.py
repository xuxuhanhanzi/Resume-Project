"""Narrow, offline BFCL-v4 tool-call validity smoke evaluator for R3.

It intentionally does not claim BFCL AST/executable accuracy: that requires
the upstream package and category-specific state engines.  Instead it records
the part directly exercised by R3's B0/B1 comparison: whether a local model's
call shape is recoverably accepted by a typed contract when function schemas
are exposed only in text (B0) or through the native Ollama tools field (B1).
"""

from __future__ import annotations

import argparse
import json
import platform
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

from repopilot.core.contracts import Permission, ToolCall, ToolSpec
from repopilot.evidence.protocol import (
    DatasetInstance,
    ExperimentReceipt,
    FrozenManifest,
    build_grouped_split_manifest,
    canonical_sha256,
    sha256_file,
)
from repopilot.tools.contracts import ContractDecision, RecoveryCode, StageScopedToolContracts

PromptVariant = Literal["loose_text_json", "typed_native_tools"]
_VARIANTS: tuple[PromptVariant, ...] = ("loose_text_json", "typed_native_tools")
_SPLITS = ("development", "validation", "final_holdout")
_SINGLE_TURN_CATEGORIES = ("simple_python", "parallel", "irrelevance")
_ALL_CATEGORIES = (*_SINGLE_TURN_CATEGORIES, "multi_turn_base")
_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class BFCLInstance:
    """One frozen BFCL prompt with its public function descriptions."""

    instance_id: str
    category: str
    question: str
    functions: tuple[dict[str, Any], ...]


@dataclass(frozen=True, slots=True)
class BFCLDataset:
    """A validated, source-digested selection of BFCL v4 category files."""

    root: Path
    instances: dict[str, BFCLInstance]
    sha256: str


def load_bfcl_dataset(root: Path) -> BFCLDataset:
    """Read the four plan-named BFCL v4 input categories without ground truth."""

    instances: dict[str, BFCLInstance] = {}
    digests: list[dict[str, str]] = []
    for category in _ALL_CATEGORIES:
        path = root / f"BFCL_v4_{category}.json"
        if not path.is_file():
            raise ValueError(f"missing BFCL category file: {path}")
        digests.append({"path": path.name, "sha256": sha256_file(path)})
        for raw in _jsonl(path):
            instance_id = _required_string(raw, "id", path)
            if instance_id in instances:
                raise ValueError(f"duplicate BFCL test ID: {instance_id}")
            turns = raw.get("question")
            if not isinstance(turns, list) or not turns or not isinstance(turns[0], list):
                raise ValueError(f"invalid BFCL question field in {path}: {instance_id}")
            first_turn = turns[0]
            if not first_turn or not isinstance(first_turn[-1], dict):
                raise ValueError(f"invalid BFCL first conversation turn: {instance_id}")
            question = _required_string(first_turn[-1], "content", path)
            functions = raw.get("function", [])
            if not isinstance(functions, list) or not all(
                isinstance(item, dict) for item in functions
            ):
                raise ValueError(f"invalid BFCL function descriptions in {path}: {instance_id}")
            instances[instance_id] = BFCLInstance(
                instance_id=instance_id,
                category=category,
                question=question,
                functions=tuple(cast(dict[str, Any], item) for item in functions),
            )
    return BFCLDataset(
        root=root,
        instances=instances,
        sha256=canonical_sha256(sorted(digests, key=str)),
    )


def build_bfcl_manifest(dataset: BFCLDataset) -> FrozenManifest:
    """Freeze all plan categories before selecting the single-turn smoke subset."""

    return build_grouped_split_manifest(
        dataset_name="BFCL-v4/R3-plan-categories",
        dataset_version="gorilla@6ea57973c7a6097fd7c5915698c54c17c5b1b6c8",
        dataset_sha256=dataset.sha256,
        instances=[
            DatasetInstance(
                instance_id=item.instance_id,
                group_ids=(f"bfcl:{item.instance_id}",),
                stratum=item.category,
            )
            for item in dataset.instances.values()
        ],
        notes=(
            "The manifest includes simple_python, parallel, irrelevance, and multi_turn_base.",
            "The R3 smoke runner rejects multi_turn_base because it needs BFCL's state engine; "
            "the omission is reported rather than scored as a failure or a pass.",
        ),
    )


def load_manifest(path: Path) -> FrozenManifest:
    raw = cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))
    return FrozenManifest(
        schema_version=int(raw["schema_version"]),
        dataset_name=str(raw["dataset_name"]),
        dataset_version=str(raw["dataset_version"]),
        dataset_sha256=str(raw["dataset_sha256"]),
        split_algorithm=str(raw["split_algorithm"]),
        assignments={name: tuple(raw["assignments"][name]) for name in _SPLITS},
        strata_counts={name: dict(raw["strata_counts"][name]) for name in _SPLITS},
        notes=tuple(raw.get("notes", [])),
    )


def run_single_turn_smoke(
    dataset: BFCLDataset,
    manifest: FrozenManifest,
    *,
    split: Literal["development", "validation"],
    variant: PromptVariant,
    limit: int | None = None,
) -> dict[str, object]:
    """Measure recoverable schema acceptance for the explicit single-turn subset."""

    if manifest.dataset_sha256 != dataset.sha256:
        raise ValueError("manifest dataset hash does not match BFCL input files")
    if variant not in _VARIANTS:
        raise ValueError(f"unsupported R3 variant: {variant}")
    selected = [dataset.instances[instance_id] for instance_id in manifest.assignments[split]]
    unsupported = [
        item.instance_id for item in selected if item.category not in _SINGLE_TURN_CATEGORIES
    ]
    runnable = [item for item in selected if item.category in _SINGLE_TURN_CATEGORIES]
    if limit is not None:
        if limit <= 0:
            raise ValueError("limit must be positive when provided")
        runnable = runnable[:limit]
    records: list[dict[str, object]] = []
    for item in runnable:
        started = time.perf_counter()
        try:
            calls, raw = _generate_calls(item, variant)
            decision = _contract_decision(item, calls)
            error = None
        except RuntimeError as exc:
            calls, raw, decision, error = [], "", None, str(exc)
        elapsed = (time.perf_counter() - started) * 1000.0
        is_irrelevance = item.category == "irrelevance"
        relevance_correct = (not calls) if is_irrelevance else None
        records.append(
            {
                "id": item.instance_id,
                "category": item.category,
                "latency_ms": elapsed,
                "calls": calls,
                "raw_response": raw,
                "contract_allowed": None if decision is None else decision.allowed,
                "recovery_code": (
                    None if decision is None or decision.code is None else decision.code.value
                ),
                "error": error,
                "irrelevance_correct": relevance_correct,
            }
        )
    return {
        "schema_version": 1,
        "dataset": "BFCL-v4/R3-plan-categories",
        "dataset_sha256": dataset.sha256,
        "manifest_sha256": manifest.sha256,
        "split": split,
        "variant": variant,
        "runnable_instances": len(records),
        "unsupported_multi_turn_instances": len(unsupported),
        "summary": _summarize(records),
        "records": records,
        "notes": [
            "This is a local B0/B1 contract-validity smoke, not an official BFCL "
            "AST/executable score.",
            "Stage-scoped A is intentionally not mapped onto arbitrary BFCL tools; "
            "using gold paths to invent stages would contaminate the comparison.",
        ],
    }


def _generate_calls(item: BFCLInstance, variant: PromptVariant) -> tuple[list[dict[str, Any]], str]:
    if variant == "typed_native_tools":
        response = _ollama_chat(
            messages=[{"role": "user", "content": item.question}], tools=list(item.functions)
        )
        message = response.get("message")
        if not isinstance(message, dict):
            raise RuntimeError("Ollama response has no message object")
        raw = str(message.get("content", ""))
        calls = message.get("tool_calls", [])
        if not isinstance(calls, list):
            raise RuntimeError("Ollama native tool_calls is not a list")
        converted: list[dict[str, Any]] = []
        for call in calls:
            if not isinstance(call, dict) or not isinstance(call.get("function"), dict):
                raise RuntimeError("Ollama native tool call has an invalid shape")
            function = cast(dict[str, Any], call["function"])
            name = function.get("name")
            arguments = function.get("arguments")
            if not isinstance(name, str) or not isinstance(arguments, dict):
                raise RuntimeError("Ollama native tool call lacks name or object arguments")
            converted.append({"name": name, "arguments": arguments})
        return converted, raw
    prompt = (
        "Return only a JSON array.  Each item must contain exactly `name` and `arguments`; "
        "arguments must be a JSON object.  Return [] if no listed function applies.\n\n"
        f"Functions:\n{json.dumps(item.functions, ensure_ascii=False)}\n\n"
        f"User request:\n{item.question}"
    )
    response = _ollama_chat(messages=[{"role": "user", "content": prompt}], tools=None)
    message = response.get("message")
    if not isinstance(message, dict) or not isinstance(message.get("content"), str):
        raise RuntimeError("Ollama loose-text response has no string content")
    raw = message["content"]
    try:
        parsed = json.loads(_FENCE.sub("", raw.strip()).strip())
    except json.JSONDecodeError as exc:
        raise RuntimeError("loose-text response is not JSON") from exc
    if not isinstance(parsed, list) or not all(isinstance(call, dict) for call in parsed):
        raise RuntimeError("loose-text response is not an array of call objects")
    calls = [cast(dict[str, Any], call) for call in parsed]
    if any(set(call) != {"name", "arguments"} for call in calls):
        raise RuntimeError("loose-text call does not contain exactly name and arguments")
    if any(
        not isinstance(call["name"], str) or not isinstance(call["arguments"], dict)
        for call in calls
    ):
        raise RuntimeError("loose-text call has invalid name or arguments type")
    return calls, raw


def _contract_decision(item: BFCLInstance, calls: list[dict[str, Any]]) -> ContractDecision | None:
    if item.category == "irrelevance":
        return _single_decision(item, calls) if calls else None
    if not calls:
        return _denied_empty_call()
    return _single_decision(item, calls)


def _single_decision(item: BFCLInstance, calls: list[dict[str, Any]]) -> ContractDecision:
    specs = tuple(_tool_spec(function) for function in item.functions)
    contracts = StageScopedToolContracts(
        specs,
        allowlist={"execute": frozenset(spec.name for spec in specs)},
    )
    for index, call in enumerate(calls):
        name = call.get("name")
        arguments = call.get("arguments")
        if not isinstance(name, str) or not isinstance(arguments, dict):
            return _denied_empty_call()
        decision = contracts.validate(ToolCall(str(index), name, arguments), stage="execute")
        if not decision.allowed:
            return decision
    return _allowed_decision()


def _tool_spec(function: Mapping[str, Any]) -> ToolSpec:
    name = function.get("name")
    parameters = function.get("parameters")
    if not isinstance(name, str) or not isinstance(parameters, dict):
        raise RuntimeError("BFCL function description lacks name or parameters")
    return ToolSpec(
        name=name,
        description=str(function.get("description", "")),
        input_schema=_normalise_schema(parameters),
        permission=Permission.READ,
    )


def _normalise_schema(schema: Mapping[str, Any]) -> dict[str, Any]:
    aliases = {"dict": "object", "float": "number", "list": "array", "tuple": "array"}
    result: dict[str, Any] = {}
    for key, value in schema.items():
        if key == "type" and isinstance(value, str):
            result[key] = aliases.get(value, value)
        elif key == "properties" and isinstance(value, dict):
            result[key] = {
                name: _normalise_schema(child) if isinstance(child, dict) else child
                for name, child in value.items()
            }
        elif key == "items" and isinstance(value, dict):
            result[key] = _normalise_schema(value)
        else:
            result[key] = value
    result.setdefault("additionalProperties", False)
    return result


def _allowed_decision() -> ContractDecision:
    return ContractDecision(True)


def _denied_empty_call() -> ContractDecision:
    return ContractDecision(False, RecoveryCode.SCHEMA_INVALID, "a relevant task needs a tool call")


def _ollama_chat(
    *, messages: list[dict[str, str]], tools: list[dict[str, Any]] | None
) -> dict[str, Any]:
    payload: dict[str, object] = {
        "model": "qwen2.5:1.5b",
        "messages": messages,
        "stream": False,
        "options": {"temperature": 0, "num_predict": 128},
    }
    if tools is not None:
        payload["tools"] = [{"type": "function", "function": function} for function in tools]
    request = urllib.request.Request(
        "http://127.0.0.1:11434/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            raw = json.loads(response.read())
    except urllib.error.URLError as exc:
        raise RuntimeError("Ollama qwen2.5:1.5b endpoint is unavailable") from exc
    if not isinstance(raw, dict):
        raise RuntimeError("Ollama response is not a JSON object")
    return cast(dict[str, Any], raw)


def _summarize(records: list[dict[str, object]]) -> dict[str, object]:
    errors = [record for record in records if record["error"] is not None]
    decisions = [
        record["contract_allowed"] for record in records if record["contract_allowed"] is not None
    ]
    irrelevance = [
        record["irrelevance_correct"]
        for record in records
        if record["irrelevance_correct"] is not None
    ]
    recovery_counts = Counter(
        cast(str, record["recovery_code"])
        for record in records
        if record["recovery_code"] is not None
    )
    latencies = sorted(cast(float, record["latency_ms"]) for record in records)
    return {
        "transport_or_parse_error_rate": len(errors) / len(records) if records else None,
        "contract_valid_rate": sum(bool(value) for value in decisions) / len(decisions)
        if decisions
        else None,
        "irrelevance_correct_rate": sum(bool(value) for value in irrelevance) / len(irrelevance)
        if irrelevance
        else None,
        "recovery_codes": dict(sorted(recovery_counts.items())),
        "latency_ms_p50": _percentile(latencies, 0.5) if latencies else None,
        "latency_ms_p95": _percentile(latencies, 0.95) if latencies else None,
    }


def _percentile(values: list[float], quantile: float) -> float:
    if len(values) == 1:
        return values[0]
    position = (len(values) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    fraction = position - lower
    return values[lower] * (1.0 - fraction) + values[upper] * fraction


def _jsonl(path: Path) -> Iterable[dict[str, Any]]:
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid BFCL JSON at {path}:{number}") from exc
        if not isinstance(raw, dict):
            raise ValueError(f"BFCL record is not an object at {path}:{number}")
        yield cast(dict[str, Any], raw)


def _required_string(raw: Mapping[str, Any], key: str, path: Path) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"missing non-empty {key!r} in {path}")
    return value


def _git_value(args: list[str], fallback: str) -> str:
    completed = subprocess.run(args, capture_output=True, check=False, text=True)
    if completed.returncode == 0 and completed.stdout.strip():
        return completed.stdout.strip()
    return fallback


def _write_run(
    dataset: BFCLDataset,
    manifest: FrozenManifest,
    *,
    split: Literal["development", "validation"],
    variant: PromptVariant,
    output: Path,
    limit: int | None,
) -> None:
    receipt_path = output.with_suffix(".receipt.json")
    if output.exists() or receipt_path.exists():
        raise ValueError("refusing to overwrite R3 raw output or receipt")
    result = run_single_turn_smoke(dataset, manifest, split=split, variant=variant, limit=limit)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    receipt = ExperimentReceipt(
        schema_version=1,
        run_id=output.stem,
        variant=variant,
        manifest_sha256=manifest.sha256,
        source_commit=_git_value(["git", "rev-parse", "HEAD"], "unknown"),
        environment={
            "platform": platform.platform(),
            "python": sys.version.split()[0],
            "model": "qwen2.5:1.5b",
        },
        fixed_controls={
            "temperature": 0,
            "num_predict": 128,
            "request_timeout_seconds": 60,
            "single_turn_categories": list(_SINGLE_TURN_CATEGORIES),
            "limit": limit,
            "official_bfcl_ast_or_state_evaluator": False,
        },
        raw_output_sha256=sha256_file(output),
        notes=(
            "Local contract-validity smoke only; no official BFCL score is implied.",
            "multi_turn_base and stage-scoped A are outside this runner's valid scope.",
        ),
    )
    receipt.write_once(receipt_path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build-manifest")
    build.add_argument("--data-root", type=Path, required=True)
    build.add_argument("--manifest", type=Path, required=True)
    run = commands.add_parser("run-single-turn-smoke")
    run.add_argument("--data-root", type=Path, required=True)
    run.add_argument("--manifest", type=Path, required=True)
    run.add_argument("--split", choices=("development", "validation"), required=True)
    run.add_argument("--variant", choices=_VARIANTS, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--limit", type=int)
    args = parser.parse_args(argv)
    dataset = load_bfcl_dataset(args.data_root)
    if args.command == "build-manifest":
        manifest = build_bfcl_manifest(dataset)
        manifest.write_once(args.manifest)
        print(
            json.dumps({"manifest": str(args.manifest), "sha256": manifest.sha256}, sort_keys=True)
        )
        return 0
    manifest = load_manifest(args.manifest)
    _write_run(
        dataset,
        manifest,
        split=args.split,
        variant=args.variant,
        output=args.output,
        limit=args.limit,
    )
    print(
        json.dumps({"output": str(args.output), "sha256": sha256_file(args.output)}, sort_keys=True)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
