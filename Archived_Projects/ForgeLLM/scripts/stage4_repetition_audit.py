"""Add a non-overwriting character-level repetition audit to a Stage 4 report."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import cast

from forgellm.post_training.evaluation import repeated_character_ngram_fraction


def main(argv: Sequence[str] | None = None) -> int:
    """Audit before/after generated text without changing the frozen run report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(f"repetition audit already exists: {args.output}")
    raw = json.loads(args.report.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Stage 4 report must be an object")
    audited: dict[str, object] = {}
    for phase in ("before_training", "after_training"):
        outputs = raw[phase]["correctness_generation"]["outputs"]
        rows: list[dict[str, object]] = []
        for output in outputs:
            response = output["response"]
            if not isinstance(response, str):
                raise ValueError("generated response must be a string")
            rows.append(
                {
                    "id": output["id"],
                    "character_8gram_repetition": repeated_character_ngram_fraction(response, n=8),
                }
            )
        audited[phase] = {
            "examples": len(rows),
            "mean_character_8gram_repetition": sum(
                cast(float, row["character_8gram_repetition"]) for row in rows
            )
            / len(rows),
            "outputs": rows,
        }
    result = {
        "schema_version": "forgellm-stage4-repetition-audit-v1",
        "source_report": str(args.report),
        "reason": (
            "Whitespace-token trigrams miss repeated substrings inside malformed unspaced text."
        ),
        "metric": "duplicate fraction among whitespace-stripped character 8-grams",
        "audit": audited,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
