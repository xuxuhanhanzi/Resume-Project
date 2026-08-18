"""Bounded latency, throughput, memory and KV-cache benchmark matrix."""

from __future__ import annotations

import gc
import importlib.metadata
import json
import platform
import statistics
import time
from pathlib import Path
from typing import cast

import torch
from torch import Tensor

from forgellm.evaluation.cases import read_cases
from forgellm.evaluation.config import Stage6Config
from forgellm.evaluation.hf_runner import generate_case, load_evaluation_stack
from forgellm.evaluation.identity import code_revision
from forgellm.evaluation.schema import EvaluationCase, canonical_sha256
from forgellm.evaluation.systems import (
    approximate_decode_tokens_per_second,
    summarize_timings,
    theoretical_kv_cache_bytes,
)
from forgellm.structured_logging import JsonValue


def _config_int(config: object, name: str) -> int:
    value = getattr(config, name, None)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise RuntimeError(f"model config has no positive integer {name}")
    return value


def _prompt_batch(tokenizer: object, *, batch_size: int, length: int) -> Tensor:
    encode = getattr(tokenizer, "encode", None)
    if not callable(encode):
        raise RuntimeError("tokenizer has no encode method")
    seed_raw = encode(
        "System benchmark prompt with deterministic tokens and no sampled content. ",
        add_special_tokens=False,
    )
    if not isinstance(seed_raw, list) or not seed_raw:
        raise RuntimeError("could not construct system benchmark prompt")
    seed = [int(value) for value in seed_raw]
    tokens = (seed * ((length + len(seed) - 1) // len(seed)))[:length]
    return torch.tensor([tokens] * batch_size, dtype=torch.long, device="cuda")


@torch.no_grad()
def _time_generate(
    model: object,
    input_ids: Tensor,
    *,
    new_tokens: int,
    pad_token_id: int,
) -> float:
    generate = getattr(model, "generate", None)
    if not callable(generate):
        raise RuntimeError("model has no generate method")
    torch.cuda.synchronize(input_ids.device)
    started = time.perf_counter()
    generate(
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        do_sample=False,
        min_new_tokens=new_tokens,
        max_new_tokens=new_tokens,
        eos_token_id=None,
        pad_token_id=pad_token_id,
        use_cache=True,
    )
    torch.cuda.synchronize(input_ids.device)
    return time.perf_counter() - started


def _benchmark_policy(
    config: Stage6Config,
    *,
    project_root: Path,
    model_key: str,
    adapter_path: Path | None,
    effort_cases: list[EvaluationCase],
) -> dict[str, JsonValue]:
    torch.cuda.reset_peak_memory_stats()
    load_started = time.perf_counter()
    stack = load_evaluation_stack(
        model_id=config.model.model_id,
        revision=config.model.revision,
        tokenizer_path=project_root / config.model.sft_adapter_path,
        adapter_path=adapter_path,
    )
    torch.cuda.synchronize()
    load_seconds = time.perf_counter() - load_started
    load_peak_allocated = int(torch.cuda.max_memory_allocated())
    model_config = getattr(stack.model, "config", None)
    layers = _config_int(model_config, "num_hidden_layers")
    hidden_size = _config_int(model_config, "hidden_size")
    attention_heads = _config_int(model_config, "num_attention_heads")
    key_value_heads = _config_int(model_config, "num_key_value_heads")
    if hidden_size % attention_heads:
        raise RuntimeError("hidden size is not divisible by attention heads")
    head_dim = hidden_size // attention_heads
    pad_token_id = getattr(stack.tokenizer, "pad_token_id", None)
    if not isinstance(pad_token_id, int):
        raise RuntimeError("tokenizer has no pad token ID")
    cells: list[JsonValue] = []
    for batch_size in config.systems.batch_sizes:
        for prompt_length in config.systems.prompt_lengths:
            input_ids = _prompt_batch(stack.tokenizer, batch_size=batch_size, length=prompt_length)
            for _ in range(config.systems.warmups):
                _time_generate(
                    stack.model,
                    input_ids,
                    new_tokens=1,
                    pad_token_id=pad_token_id,
                )
                _time_generate(
                    stack.model,
                    input_ids,
                    new_tokens=config.systems.max_new_tokens,
                    pad_token_id=pad_token_id,
                )
            torch.cuda.reset_peak_memory_stats(input_ids.device)
            first = [
                _time_generate(
                    stack.model,
                    input_ids,
                    new_tokens=1,
                    pad_token_id=pad_token_id,
                )
                for _ in range(config.systems.repeats)
            ]
            total = [
                _time_generate(
                    stack.model,
                    input_ids,
                    new_tokens=config.systems.max_new_tokens,
                    pad_token_id=pad_token_id,
                )
                for _ in range(config.systems.repeats)
            ]
            first_summary = summarize_timings(first)
            total_summary = summarize_timings(total)
            decode_rate = approximate_decode_tokens_per_second(
                batch_size=batch_size,
                generated_tokens=config.systems.max_new_tokens,
                total_seconds=total_summary.median_seconds,
                first_token_seconds=first_summary.median_seconds,
            )
            cell: dict[str, JsonValue] = {
                "batch_size": batch_size,
                "prompt_tokens": prompt_length,
                "generated_tokens_per_sequence": config.systems.max_new_tokens,
                "time_to_first_token_seconds": cast(JsonValue, first_summary.as_dict()),
                "end_to_end_seconds": cast(JsonValue, total_summary.as_dict()),
                "approximate_decode_tokens_per_second": decode_rate,
                "end_to_end_tokens_per_second": (
                    batch_size * config.systems.max_new_tokens / total_summary.median_seconds
                ),
                "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(input_ids.device)),
                "theoretical_kv_cache_bytes": theoretical_kv_cache_bytes(
                    layers=layers,
                    batch_size=batch_size,
                    sequence_length=prompt_length + config.systems.max_new_tokens,
                    key_value_heads=key_value_heads,
                    head_dim=head_dim,
                    bytes_per_element=2,
                ),
                "kv_cache_boundary": "theoretical BF16 K+V only; allocator overhead excluded",
            }
            cells.append(cell)
            print(
                f"stage6_system model={model_key} batch={batch_size} prompt={prompt_length}",
                flush=True,
            )
    effort_audit: JsonValue = None
    if model_key == "q1":
        effort_rows: list[JsonValue] = []
        effort_fingerprint = canonical_sha256(
            {
                "audit": "stage6-q1-effort-v1",
                "budgets": list(config.systems.effort_token_budgets),
                "case_ids": [case.case_id for case in effort_cases],
            }
        )
        elapsed_by_budget: dict[int, list[float]] = {
            budget: [] for budget in config.systems.effort_token_budgets
        }
        passes_by_budget: dict[int, int] = {
            budget: 0 for budget in config.systems.effort_token_budgets
        }
        for case in effort_cases:
            row: dict[str, JsonValue] = {"case_id": case.case_id}
            for budget in config.systems.effort_token_budgets:
                generation, behavior = generate_case(
                    stack,
                    case,
                    run_fingerprint=effort_fingerprint,
                    model_key="q1",
                    max_length=config.quality.max_length,
                    max_new_tokens=budget,
                )
                elapsed_by_budget[budget].append(generation.elapsed_seconds)
                passes_by_budget[budget] += int(behavior.all_constraints_passed)
                row[str(budget)] = {
                    "generation": generation.as_dict(),
                    "behavior": cast(JsonValue, behavior.as_dict()),
                }
            effort_rows.append(row)
        effort_audit = {
            "cases": len(effort_cases),
            "budgets": {
                str(budget): {
                    "strict_success_rate": passes_by_budget[budget] / len(effort_cases),
                    "median_elapsed_seconds": statistics.median(elapsed_by_budget[budget]),
                }
                for budget in config.systems.effort_token_budgets
            },
            "rows": effort_rows,
            "boundary": (
                "Token budget is a compute/output allowance, not a guarantee of deeper reasoning."
            ),
        }
    report: dict[str, JsonValue] = {
        "model_key": model_key,
        "adapter_sha256": stack.adapter_sha256,
        "cold_load_seconds": load_seconds,
        "cold_load_peak_allocated_bytes": load_peak_allocated,
        "matrix_cells": cells,
        "matrix_complete": len(cells)
        == len(config.systems.batch_sizes) * len(config.systems.prompt_lengths),
        "timing_boundary": (
            "Single local GPU process; TTFT and full generation measured separately."
        ),
        "effort_audit": effort_audit,
    }
    del stack
    gc.collect()
    torch.cuda.empty_cache()
    return report


def run_system_benchmark(
    config: Stage6Config,
    *,
    project_root: Path,
    output_dir: Path,
) -> dict[str, JsonValue]:
    """Benchmark Q0-Q2 under one exact matrix and exclude Q3 capability claims."""
    if output_dir.exists():
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True)
    paths: dict[str, Path | None] = {
        "q0": None,
        "q1": project_root / config.model.sft_adapter_path,
        "q2": project_root / config.model.dpo_adapter_path,
    }
    effort_cases = [
        case
        for case in read_cases(project_root / config.data.cases_path)
        if case.task_type != "robustness"
    ][:16]
    reports = {
        key: _benchmark_policy(
            config,
            project_root=project_root,
            model_key=key,
            adapter_path=path,
            effort_cases=effort_cases,
        )
        for key, path in paths.items()
    }
    report: dict[str, JsonValue] = {
        "schema_version": "forgellm-stage6-system-report-v1",
        "code_revision": code_revision(project_root),
        "environment": {
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "transformers": importlib.metadata.version("transformers"),
            "peft": importlib.metadata.version("peft"),
            "cuda_runtime": str(torch.version.cuda),
            "gpu": torch.cuda.get_device_name(0),
            "precision": "bfloat16",
            "external_cost_usd": 0,
        },
        "models": cast(JsonValue, reports),
        "q3": "excluded: one optimizer step is a pipeline audit, not a serving candidate",
        "matrix": {
            "batch_sizes": list(config.systems.batch_sizes),
            "prompt_lengths": list(config.systems.prompt_lengths),
            "max_new_tokens": config.systems.max_new_tokens,
            "effort_token_budgets": list(config.systems.effort_token_budgets),
            "warmups": config.systems.warmups,
            "repeats": config.systems.repeats,
        },
        "all_required_cells_complete": all(
            cast(bool, value["matrix_complete"]) for value in reports.values()
        ),
    }
    path = output_dir / "report.json"
    path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return report
