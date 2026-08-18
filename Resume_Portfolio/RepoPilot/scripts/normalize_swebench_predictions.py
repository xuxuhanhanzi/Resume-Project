"""Normalize local SWE-bench prediction metadata for the official evaluator."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--model-name", required=True)
    arguments = parser.parse_args()

    if arguments.input.resolve() == arguments.output.resolve():
        raise ValueError("output must differ from input so raw evidence is retained")
    records: list[dict[str, object]] = []
    for line in arguments.input.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        if not isinstance(record, dict):
            raise ValueError("prediction record must be an object")
        record["model_name_or_path"] = arguments.model_name
        records.append(record)
    arguments.output.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps({"output": str(arguments.output), "records": len(records)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
