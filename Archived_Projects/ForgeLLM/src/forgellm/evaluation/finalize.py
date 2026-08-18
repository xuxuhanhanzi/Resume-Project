"""Build the evidence-linked Stage 6 acceptance report without a total score."""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

from forgellm.evaluation.identity import code_revision, file_sha256
from forgellm.evaluation.report import EvidenceClaim, audit_claim_evidence, model_acceptance
from forgellm.structured_logging import JsonValue


def _read_object(path: Path) -> dict[str, object]:
    raw = cast(object, json.loads(path.read_text(encoding="utf-8")))
    if not isinstance(raw, dict):
        raise ValueError(f"report is not an object: {path}")
    return cast(dict[str, object], raw)


def _mapping(parent: dict[str, object], key: str) -> dict[str, object]:
    value = parent.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"missing report mapping: {key}")
    return cast(dict[str, object], value)


def _number(parent: dict[str, object], key: str) -> float:
    value = parent.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"missing report number: {key}")
    return float(value)


def _write_json(path: Path, value: JsonValue) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def build_final_report(
    *,
    project_root: Path,
    quality_path: Path,
    systems_path: Path,
    output_dir: Path,
) -> dict[str, JsonValue]:
    """Join quality/system evidence and keep automated vs learner gates distinct."""
    project_root = project_root.resolve()
    quality_path = quality_path.resolve()
    systems_path = systems_path.resolve()
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True)
    quality = _read_object(quality_path)
    systems = _read_object(systems_path)
    quality_models = _mapping(quality, "models")
    system_models = _mapping(systems, "models")
    q0_retention = _number(
        _mapping(cast(dict[str, object], quality_models["q0"]), "retention_language_modeling"),
        "loss_per_token",
    )
    decisions: dict[str, JsonValue] = {}
    for model_key in ("q0", "q1", "q2"):
        quality_model = cast(dict[str, object], quality_models[model_key])
        behavior = _mapping(quality_model, "behavior")
        original = _mapping(behavior, "original")
        all_cases = _mapping(behavior, "all")
        retention = _mapping(quality_model, "retention_language_modeling")
        system_model = cast(dict[str, object], system_models[model_key])
        matrix_complete = system_model.get("matrix_complete") is True
        decisions[model_key] = cast(
            JsonValue,
            model_acceptance(
                strict_success_rate=_number(original, "constraint_pass_rate"),
                minimum_strict_success_rate=0.5,
                retention_loss_ratio=_number(retention, "loss_per_token") / q0_retention,
                maximum_retention_loss_ratio=1.1,
                repetition_rate=_number(all_cases, "mean_character_8gram_repetition"),
                maximum_repetition_rate=0.25,
                system_matrix_completed=matrix_complete,
            ),
        )
    contamination = quality.get("contamination")
    if not isinstance(contamination, dict):
        raise ValueError("quality report has no contamination audit")
    exact_matches = contamination.get("exact_matches")
    if isinstance(exact_matches, bool) or not isinstance(exact_matches, int):
        raise ValueError("contamination exact-match count is invalid")
    quality_relative = quality_path.relative_to(project_root).as_posix()
    systems_relative = systems_path.relative_to(project_root).as_posix()
    manifest_relative = quality_path.with_name("manifest.json").relative_to(project_root).as_posix()
    generations_relative = (
        quality_path.with_name("raw_generations.jsonl").relative_to(project_root).as_posix()
    )
    claims = [
        EvidenceClaim(
            "stage6-protocol",
            "The frozen Q0-Q2 quality protocol ran on 64 cases per policy.",
            (quality_relative, manifest_relative, generations_relative),
            "This validates the local evaluation process, not broad model capability.",
        ),
        EvidenceClaim(
            "stage6-systems",
            "The required local latency/memory matrix was completed for Q0-Q2.",
            (systems_relative,),
            "Results describe this hardware/software environment only.",
        ),
        EvidenceClaim(
            "stage6-q3-boundary",
            "Q3 is accepted only as a one-step on-policy pipeline audit.",
            (
                "artifacts/stage05/qwen_grpo_one_step_v2/report.json",
                "artifacts/stage05/qwen_grpo_one_step_v2/rollouts.jsonl",
            ),
            "No GRPO capability-improvement claim is made.",
        ),
    ]
    claim_audit = audit_claim_evidence(claims, project_root=project_root)
    system_complete = systems.get("all_required_cells_complete") is True
    automated_gates = {
        "evaluation_identity_frozen": quality.get("run_fingerprint") is not None,
        "raw_generation_count_is_192": sum(
            1 for _ in quality_path.with_name("raw_generations.jsonl").open(encoding="utf-8")
        )
        == 192,
        "contamination_has_no_exact_match": exact_matches == 0,
        "system_matrix_complete": system_complete,
        "claim_evidence_complete": claim_audit["passed"] is True,
        "q3_scope_is_pipeline_only": True,
    }
    report: dict[str, JsonValue] = {
        "schema_version": "forgellm-stage6-final-acceptance-v1",
        "report_builder_code_revision": code_revision(project_root),
        "quality_report": {
            "path": quality_relative,
            "sha256": file_sha256(quality_path),
        },
        "system_report": {
            "path": systems_relative,
            "sha256": file_sha256(systems_path),
        },
        "thresholds": {
            "minimum_strict_success_rate": 0.5,
            "maximum_retention_loss_ratio_to_q0": 1.1,
            "maximum_mean_character_8gram_repetition": 0.25,
            "system_matrix_required": True,
        },
        "model_decisions": decisions,
        "automated_gates": cast(JsonValue, automated_gates),
        "automated_stage6_status": ("passed" if all(automated_gates.values()) else "not_passed"),
        "learner_gate": {
            "status": "pending",
            "requirements": [
                "Complete the 30-pair blind review without opening private_key.jsonl.",
                "Explain PPL/BPB comparability and Wilson/paired-bootstrap conclusions.",
                "Complete the Stage 6 learner acceptance station.",
            ],
        },
        "claim_evidence_audit": cast(JsonValue, claim_audit),
        "decision_boundary": (
            "No composite score; automated implementation acceptance and learner "
            "acceptance are separate."
        ),
    }
    _write_json(output_dir / "claim_evidence.json", cast(JsonValue, claim_audit))
    _write_json(output_dir / "report.json", report)
    markdown = "# Stage 6 最终评测卡\n\n"
    markdown += f"- 自动化状态：`{report['automated_stage6_status']}`\n"
    markdown += "- 学习者验收：`pending`\n"
    markdown += "- 综合总分：不使用\n\n"
    markdown += "## 模型门禁\n\n"
    for key, value in decisions.items():
        decision = cast(dict[str, object], value)
        markdown += f"- {key}: {decision['status']}；gates={decision['gates']}\n"
    markdown += "\n## 结论边界\n\n"
    markdown += (
        "Stage 6 验收的是本地、冻结协议下的证据链。它不能推出通用能力提升；"
        "Q3 仅证明一次 on-policy 更新链路可运行。\n"
    )
    card_path = output_dir / "evaluation_card.md"
    card_path.write_text(markdown, encoding="utf-8", newline="\n")
    return report
