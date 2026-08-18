"""Deterministic blind-review packages and position-consistency audits."""

from __future__ import annotations

import hashlib
import html
import json
from dataclasses import dataclass
from pathlib import Path

from forgellm.evaluation.behavioral import BehaviorResult
from forgellm.structured_logging import JsonValue


@dataclass(frozen=True, slots=True)
class BlindComparison:
    """One anonymized A/B comparison with an auditable hidden mapping."""

    comparison_id: str
    case_id: str
    prompt: str
    response_a: str
    response_b: str
    model_a: str
    model_b: str

    def public_dict(self) -> dict[str, JsonValue]:
        """Exclude the hidden model mapping from the reviewer file."""
        return {
            "case_id": self.case_id,
            "comparison_id": self.comparison_id,
            "prompt": self.prompt,
            "response_a": self.response_a,
            "response_b": self.response_b,
            "rubric": {
                "clarity": None,
                "correctness": None,
                "harmfulness": None,
                "instruction_following": None,
                "repetition": None,
                "winner": None,
            },
        }

    def private_dict(self) -> dict[str, str]:
        """Return the hidden key stored separately from reviewer data."""
        return {
            "case_id": self.case_id,
            "comparison_id": self.comparison_id,
            "model_a": self.model_a,
            "model_b": self.model_b,
        }


def build_blind_comparisons(
    left_outputs: dict[str, str],
    right_outputs: dict[str, str],
    prompts: dict[str, str],
    *,
    left_model: str,
    right_model: str,
    count: int,
    seed: int,
) -> list[BlindComparison]:
    """Select and orient pairs by SHA-256 rather than mutable file order."""
    shared = sorted(set(left_outputs) & set(right_outputs) & set(prompts))
    if count <= 0 or len(shared) < count or left_model == right_model:
        raise ValueError("blind comparison inputs are invalid")
    ranked = sorted(
        shared,
        key=lambda case_id: hashlib.sha256(f"{seed}:{case_id}".encode()).hexdigest(),
    )[:count]
    comparisons: list[BlindComparison] = []
    for index, case_id in enumerate(ranked):
        digest = hashlib.sha256(f"orientation:{seed}:{case_id}".encode()).digest()
        swap = digest[0] % 2 == 1
        model_a, model_b = (right_model, left_model) if swap else (left_model, right_model)
        response_a = right_outputs[case_id] if swap else left_outputs[case_id]
        response_b = left_outputs[case_id] if swap else right_outputs[case_id]
        comparisons.append(
            BlindComparison(
                comparison_id=f"blind-{index:03d}",
                case_id=case_id,
                prompt=prompts[case_id],
                response_a=response_a,
                response_b=response_b,
                model_a=model_a,
                model_b=model_b,
            )
        )
    return comparisons


def deterministic_rule_judge(left: BehaviorResult, right: BehaviorResult) -> str:
    """Compare frozen rule metrics without reading model identity or position."""
    left_key = (
        int(left.all_constraints_passed),
        int(left.exact_match),
        -left.character_8gram_repetition,
        -left.token_3gram_repetition,
        -left.response_tokens,
    )
    right_key = (
        int(right.all_constraints_passed),
        int(right.exact_match),
        -right.character_8gram_repetition,
        -right.token_3gram_repetition,
        -right.response_tokens,
    )
    if left_key == right_key:
        return "tie"
    return "left" if left_key > right_key else "right"


def position_consistency(
    pairs: list[tuple[BehaviorResult, BehaviorResult]],
) -> dict[str, float | int]:
    """Audit whether swapping A/B changes the rule judge's semantic winner."""
    if not pairs:
        raise ValueError("position audit requires at least one pair")
    consistent = 0
    ties = 0
    for left, right in pairs:
        forward = deterministic_rule_judge(left, right)
        reverse = deterministic_rule_judge(right, left)
        if forward == "tie":
            ties += 1
            consistent += int(reverse == "tie")
        else:
            expected_reverse = "right" if forward == "left" else "left"
            consistent += int(reverse == expected_reverse)
    return {
        "pairs": len(pairs),
        "position_consistency_rate": consistent / len(pairs),
        "tie_rate": ties / len(pairs),
    }


def write_blind_package(
    output_dir: Path,
    comparisons: list[BlindComparison],
) -> tuple[Path, Path, Path]:
    """Write reviewer JSONL/HTML and a separate private answer key."""
    if output_dir.exists():
        raise FileExistsError(f"blind package directory already exists: {output_dir}")
    output_dir.mkdir(parents=True)
    public_path = output_dir / "blind_review.jsonl"
    key_path = output_dir / "private_key.jsonl"
    html_path = output_dir / "blind_review.html"
    public_path.write_text(
        "".join(
            json.dumps(item.public_dict(), ensure_ascii=False, sort_keys=True) + "\n"
            for item in comparisons
        ),
        encoding="utf-8",
        newline="\n",
    )
    key_path.write_text(
        "".join(
            json.dumps(item.private_dict(), ensure_ascii=False, sort_keys=True) + "\n"
            for item in comparisons
        ),
        encoding="utf-8",
        newline="\n",
    )
    rows = []
    for item in comparisons:
        rows.append(
            "<section><h2>"
            + html.escape(item.comparison_id)
            + " · "
            + html.escape(item.case_id)
            + "</h2><h3>Prompt</h3><pre>"
            + html.escape(item.prompt)
            + "</pre><h3>A</h3><pre>"
            + html.escape(item.response_a)
            + "</pre><h3>B</h3><pre>"
            + html.escape(item.response_b)
            + "</pre><p>Winner: A / B / Tie · correctness · instruction following · "
            + "repetition · clarity · harmfulness</p></section>"
        )
    html_path.write_text(
        "<!doctype html><meta charset='utf-8'>"
        "<title>ForgeLLM Stage 6 blind review</title>"
        "<style>body{font-family:sans-serif;max-width:960px;margin:auto}"
        "section{border-bottom:1px solid #ccc;padding:1rem}"
        "pre{white-space:pre-wrap;background:#f6f6f6;padding:1rem}</style>" + "".join(rows),
        encoding="utf-8",
        newline="\n",
    )
    return public_path, key_path, html_path
