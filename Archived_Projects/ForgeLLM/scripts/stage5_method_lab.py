"""Run the bounded Stage 5 RM/PG/PPO/GRPO/frontier method lab."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import cast

from forgellm.alignment.method_lab import run_method_lab
from forgellm.alignment.schema import read_preference_jsonl


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run Stage 5 tiny reward, policy-gradient and modern-method experiments."
    )
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    train = cast(Path, args.train)
    output = cast(Path, args.output)
    report = run_method_lab(read_preference_jsonl(train))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report["all_gates_passed"] is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
