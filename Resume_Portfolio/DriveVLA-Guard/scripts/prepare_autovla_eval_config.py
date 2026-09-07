#!/usr/bin/env python3
"""Render the upstream AutoVLA YAML with portable absolute asset paths."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--qwen-model", type=Path, required=True)
    parser.add_argument("--codebook", type=Path, required=True)
    parser.add_argument("--json-data", type=Path, required=True)
    parser.add_argument("--sensor-data", type=Path, required=True)
    parser.add_argument("--metric-cache", type=Path, required=True)
    args = parser.parse_args()
    payload = yaml.safe_load(args.template.read_text(encoding="utf-8"))
    payload["model"]["pretrained_model_path"] = str(args.qwen_model.resolve())
    payload["model"]["codebook_cache_path"] = str(args.codebook.resolve())
    payload["data"]["val"].update(
        {
            "metric_cache_path": str(args.metric_cache.resolve()),
            "json_dataset_path": str(args.json_data.resolve()),
            "sensor_data_path": str(args.sensor_data.resolve()),
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    print(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
