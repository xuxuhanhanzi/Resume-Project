"""Orchestrate the complete frozen Stage 6 quality evaluation."""

from __future__ import annotations

import gc
import importlib.metadata
import json
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import torch

from forgellm.alignment.schema import read_preference_jsonl
from forgellm.evaluation.behavioral import BehaviorResult, aggregate_behavior
from forgellm.evaluation.cases import read_cases
from forgellm.evaluation.config import Stage6Config
from forgellm.evaluation.hf_runner import (
    evaluate_expected_nll,
    evaluate_preference_pairs,
    evaluate_retention_nll,
    generate_case,
    load_evaluation_stack,
)
from forgellm.evaluation.identity import (
    code_revision,
    directory_sha256,
    file_sha256,
)
from forgellm.evaluation.judge import (
    build_blind_comparisons,
    position_consistency,
    write_blind_package,
)
from forgellm.evaluation.schema import (
    EvaluationCase,
    EvaluationManifest,
    GenerationRecord,
    GenerationSettings,
    ModelIdentity,
    canonical_sha256,
)
from forgellm.evaluation.stage3_runner import evaluate_stage3_checkpoint
from forgellm.evaluation.statistics import paired_bootstrap_interval, wilson_interval
from forgellm.structured_logging import JsonValue


@dataclass(frozen=True, slots=True)
class EvaluatedPolicy:
    """Outputs and rule metrics kept together for paired analysis."""

    generations: tuple[GenerationRecord, ...]
    behaviors: dict[str, BehaviorResult]
    report: dict[str, JsonValue]


def _write_json(path: Path, value: JsonValue) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _write_jsonl(path: Path, values: list[dict[str, JsonValue]]) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.write_text(
        "".join(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n" for value in values),
        encoding="utf-8",
        newline="\n",
    )


def _behavior_summary(
    cases: list[EvaluationCase], behaviors: dict[str, BehaviorResult]
) -> dict[str, JsonValue]:
    sections: dict[str, JsonValue] = {}
    groups = {
        "all": cases,
        "original": [case for case in cases if case.task_type != "robustness"],
        "correctness": [case for case in cases if case.task_type == "correctness"],
        "preference": [case for case in cases if case.task_type == "preference"],
        "robustness": [case for case in cases if case.task_type == "robustness"],
    }
    for name, selected in groups.items():
        results = [behaviors[case.case_id] for case in selected]
        aggregate = aggregate_behavior(results)
        successes = sum(item.all_constraints_passed for item in results)
        section: dict[str, JsonValue] = {key: value for key, value in aggregate.items()}
        section["constraint_pass_wilson_95"] = cast(
            JsonValue, wilson_interval(successes, len(results)).as_dict()
        )
        sections[name] = section
    robustness_pairs = [case for case in cases if case.parent_case_id is not None]
    stable = sum(
        behaviors[case.case_id].all_constraints_passed
        == behaviors[cast(str, case.parent_case_id)].all_constraints_passed
        for case in robustness_pairs
    )
    sections["robustness_pair_consistency"] = {
        "pairs": len(robustness_pairs),
        "rate": stable / len(robustness_pairs),
    }
    return sections


def _evaluate_policy(
    *,
    config: Stage6Config,
    project_root: Path,
    model_key: str,
    adapter_path: Path | None,
    tokenizer_sha256: str,
    cases: list[EvaluationCase],
    manifest: EvaluationManifest,
) -> EvaluatedPolicy:
    stack = load_evaluation_stack(
        model_id=config.model.model_id,
        revision=config.model.revision,
        tokenizer_path=project_root / config.model.sft_adapter_path,
        adapter_path=adapter_path,
    )
    generations: list[GenerationRecord] = []
    behaviors: dict[str, BehaviorResult] = {}
    for index, case in enumerate(cases, start=1):
        record, behavior = generate_case(
            stack,
            case,
            run_fingerprint=manifest.fingerprint(),
            model_key=model_key,
            max_length=config.quality.max_length,
            max_new_tokens=config.quality.max_new_tokens,
        )
        generations.append(record)
        behaviors[case.case_id] = behavior
        if index == 1 or index % 16 == 0:
            print(f"stage6_quality model={model_key} cases={index}/{len(cases)}", flush=True)
    expected = evaluate_expected_nll(
        stack,
        cases,
        tokenizer_sha256=tokenizer_sha256,
        max_length=config.quality.max_length,
    )
    retention = evaluate_retention_nll(
        stack,
        project_root / config.data.retention_test_path,
        tokenizer_sha256=tokenizer_sha256,
        limit=config.quality.retention_documents,
        max_length=config.quality.max_length,
    )
    preferences = evaluate_preference_pairs(
        stack,
        read_preference_jsonl(project_root / config.data.preference_test_path),
        max_length=config.quality.max_length,
    )
    report: dict[str, JsonValue] = {
        "behavior": _behavior_summary(cases, behaviors),
        "expected_response_language_modeling": cast(JsonValue, expected.as_dict()),
        "retention_language_modeling": cast(JsonValue, retention.as_dict()),
        "preference_pairs": preferences,
        "adapter_sha256": stack.adapter_sha256,
    }
    del stack
    gc.collect()
    torch.cuda.empty_cache()
    return EvaluatedPolicy(tuple(generations), behaviors, report)


def _build_manifest(
    config: Stage6Config,
    *,
    project_root: Path,
    preparation: dict[str, object],
) -> EvaluationManifest:
    sft = project_root / config.model.sft_adapter_path
    dpo = project_root / config.model.dpo_adapter_path
    grpo = project_root / config.model.grpo_adapter_path
    tokenizer_files = [
        sft / "tokenizer.json",
        dpo / "tokenizer.json",
        grpo / "tokenizer.json",
    ]
    tokenizer_hashes = [file_sha256(path) for path in tokenizer_files]
    if len(set(tokenizer_hashes)) != 1:
        raise RuntimeError("Q0-Q3 tokenizer identities differ")
    tokenizer_sha256 = tokenizer_hashes[0]
    qwen = (config.model.model_id, config.model.revision, tokenizer_sha256)
    stage3_tokenizer = project_root / config.model.stage3_tokenizer_path
    stage3_checkpoint = project_root / config.model.stage3_checkpoint_path
    models = (
        ModelIdentity("q0", *qwen, None),
        ModelIdentity("q1", *qwen, directory_sha256(sft)),
        ModelIdentity("q2", *qwen, directory_sha256(dpo)),
        ModelIdentity("q3", *qwen, directory_sha256(grpo)),
        ModelIdentity(
            "m3",
            "forgellm/decoderlm-stage3",
            f"checkpoint-sha256:{file_sha256(stage3_checkpoint)}",
            file_sha256(stage3_tokenizer),
            None,
        ),
    )
    sources_raw = preparation.get("source_data_sha256")
    cases_sha = preparation.get("cases_sha256")
    if not isinstance(sources_raw, dict) or not isinstance(cases_sha, str):
        raise ValueError("preparation report has invalid source identities")
    source_hashes = cast(dict[str, JsonValue], sources_raw)
    return EvaluationManifest(
        schema_version="forgellm-stage6-evaluation-v1",
        run_name=config.quality.run_name,
        models=models,
        cases_sha256=cases_sha,
        source_data_sha256=source_hashes,
        generation=GenerationSettings(
            max_new_tokens=config.quality.max_new_tokens,
            do_sample=False,
            temperature=0.0,
            top_p=1.0,
            seed=config.quality.seed,
            chat_template_sha256=canonical_sha256(
                {
                    "renderer": "forgellm.render_qwen_messages-v1",
                    "assistant_prefix": "<|im_start|>assistant\\n",
                }
            ),
        ),
        code_revision=code_revision(project_root),
    )


def _comparison_report(
    baseline: EvaluatedPolicy,
    candidate: EvaluatedPolicy,
    cases: list[EvaluationCase],
    *,
    baseline_key: str,
    candidate_key: str,
    config: Stage6Config,
) -> dict[str, JsonValue]:
    originals = [case for case in cases if case.task_type != "robustness"]
    baseline_values = [
        float(baseline.behaviors[case.case_id].all_constraints_passed) for case in originals
    ]
    candidate_values = [
        float(candidate.behaviors[case.case_id].all_constraints_passed) for case in originals
    ]
    interval = paired_bootstrap_interval(
        baseline_values,
        candidate_values,
        resamples=config.quality.bootstrap_resamples,
        seed=config.quality.seed,
    )
    return {
        "comparison": f"{baseline_key}_to_{candidate_key}",
        "metric": "original_constraint_pass_rate",
        "candidate_minus_baseline_paired_bootstrap_95": cast(JsonValue, interval.as_dict()),
        "interpretation_rule": (
            "Evidence of improvement requires the entire interval above zero; "
            "otherwise inconclusive."
        ),
        "conclusion": "improved" if interval.lower > 0 else "inconclusive_or_not_improved",
    }


def run_quality_evaluation(
    config: Stage6Config,
    *,
    project_root: Path,
    output_dir: Path,
) -> dict[str, JsonValue]:
    """Run Q0/Q1/Q2, audit Q3, evaluate M3, and write all raw evidence."""
    if output_dir.exists():
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True)
    cases = read_cases(project_root / config.data.cases_path)
    preparation_raw = cast(
        object,
        json.loads(
            (project_root / config.data.cases_path)
            .with_name("preparation_report.json")
            .read_text(encoding="utf-8")
        ),
    )
    if not isinstance(preparation_raw, dict):
        raise ValueError("preparation report must be an object")
    preparation = cast(dict[str, object], preparation_raw)
    manifest = _build_manifest(config, project_root=project_root, preparation=preparation)
    _write_json(output_dir / "manifest.json", manifest.as_dict())
    tokenizer_sha256 = manifest.models[0].tokenizer_sha256
    adapter_paths: dict[str, Path | None] = {
        "q0": None,
        "q1": project_root / config.model.sft_adapter_path,
        "q2": project_root / config.model.dpo_adapter_path,
    }
    policies: dict[str, EvaluatedPolicy] = {}
    for model_key, adapter_path in adapter_paths.items():
        policies[model_key] = _evaluate_policy(
            config=config,
            project_root=project_root,
            model_key=model_key,
            adapter_path=adapter_path,
            tokenizer_sha256=tokenizer_sha256,
            cases=cases,
            manifest=manifest,
        )
    raw_generations: list[dict[str, JsonValue]] = []
    raw_behaviors: list[dict[str, JsonValue]] = []
    for model_key, policy in policies.items():
        raw_generations.extend(record.as_dict() for record in policy.generations)
        raw_behaviors.extend(
            {
                "model_key": model_key,
                "case_id": case_id,
                "metrics": cast(JsonValue, behavior.as_dict()),
            }
            for case_id, behavior in sorted(policy.behaviors.items())
        )
    _write_jsonl(output_dir / "raw_generations.jsonl", raw_generations)
    _write_jsonl(output_dir / "behavior_metrics.jsonl", raw_behaviors)

    comparisons = [
        _comparison_report(
            policies["q0"],
            policies["q1"],
            cases,
            baseline_key="q0",
            candidate_key="q1",
            config=config,
        ),
        _comparison_report(
            policies["q1"],
            policies["q2"],
            cases,
            baseline_key="q1",
            candidate_key="q2",
            config=config,
        ),
    ]
    q1_outputs = {record.case_id: record.response for record in policies["q1"].generations}
    q2_outputs = {record.case_id: record.response for record in policies["q2"].generations}
    prompts = {
        case.case_id: "\n".join(f"{message.role}: {message.content}" for message in case.prompt)
        for case in cases
    }
    blind = build_blind_comparisons(
        q1_outputs,
        q2_outputs,
        prompts,
        left_model="q1",
        right_model="q2",
        count=config.judge.blind_cases,
        seed=config.quality.seed,
    )
    write_blind_package(output_dir / "blind_review", blind)
    position = position_consistency(
        [
            (policies["q1"].behaviors[case.case_id], policies["q2"].behaviors[case.case_id])
            for case in cases
        ]
    )
    q3_report_path = project_root / config.model.grpo_adapter_path
    q3_report_path = q3_report_path.parent / "report.json"
    q3_report = cast(object, json.loads(q3_report_path.read_text(encoding="utf-8")))
    if not isinstance(q3_report, dict):
        raise ValueError("Q3 report must be an object")
    q3_training = q3_report.get("training")
    if not isinstance(q3_training, dict):
        raise ValueError("Q3 report has no training object")
    m3 = evaluate_stage3_checkpoint(
        config_path=project_root / config.model.stage3_config_path,
        checkpoint_path=project_root / config.model.stage3_checkpoint_path,
        tokenizer_path=project_root / config.model.stage3_tokenizer_path,
        validation_path=project_root / config.data.retention_test_path,
    )
    report: dict[str, JsonValue] = {
        "schema_version": "forgellm-stage6-quality-report-v1",
        "run_fingerprint": manifest.fingerprint(),
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
        "frozen_cases": len(cases),
        "models": {key: policy.report for key, policy in policies.items()},
        "q3_pipeline_audit": {
            "report_path": q3_report_path.relative_to(project_root).as_posix(),
            "report_sha256": file_sha256(q3_report_path),
            "optimizer_steps": cast(JsonValue, q3_training.get("optimizer_steps")),
            "accepted_scope": "one-step on-policy pipeline audit",
            "excluded_scope": "Q3 is not ranked as a quality-improved model",
        },
        "m3_language_modeling": m3,
        "paired_comparisons": cast(JsonValue, comparisons),
        "rule_judge_position_audit": cast(JsonValue, position),
        "blind_review": {
            "comparisons": len(blind),
            "public_file": "blind_review/blind_review.jsonl",
            "html_file": "blind_review/blind_review.html",
            "private_key": "blind_review/private_key.jsonl",
            "status": "awaiting learner ratings",
        },
        "contamination": cast(JsonValue, preparation.get("contamination")),
        "metric_boundary": (
            "No composite score. PPL is compared only inside one tokenizer identity; "
            "M3 remains separate."
        ),
    }
    _write_json(output_dir / "report.json", report)
    return report
