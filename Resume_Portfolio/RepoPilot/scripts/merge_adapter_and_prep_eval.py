#!/usr/bin/env python3
"""R4 — P10 adapter downstream-eval preparation.

Merges the P10 QLoRA/DPO LoRA adapter(s) into a base HF model, saves the
merged model, writes an Ollama Modelfile, and prints the DABench eval command.

IMPORTANT HONESTY NOTE (see docs/STATUS.md R4):
  The P10 adapters were trained on **Qwen/Qwen2.5-1.5B** (see
  artifacts/training/p10_*/adapter/adapter_config.json). They CANNOT be merged
  into or applied to the qwen2.5:7b model that the DABench baseline uses — the
  architectures differ (hidden size, layer count, vocab projection). R4 therefore
  evaluates a *separate 1.5B configuration* (1.5B base vs 1.5B+adapter on DABench),
  NOT an improvement of the 7B baseline (71.4%).

PREREQUISITES (currently blocked by `network=deny`):
  - The base model Qwen/Qwen2.5-1.5B must be available LOCALLY (cached HF dir or
    safetensors). With network denied it cannot be downloaded.
  - To produce a GGUF for Ollama you additionally need llama.cpp's
    `convert_hf_to_gguf.py` (not installed here; fetching it is also network-blocked).
    Alternative: serve the merged HF model with any OpenAI-compatible server and
    point the DABench runner at it via `--base-url`.

This script does NOT pull from the network. Pass --base-model as a local path
once the weights are available.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--base-model",
        default="Qwen/Qwen2.5-1.5B",
        help="Local path or HF id of the BASE model. MUST be 1.5B to match the adapter.",
    )
    p.add_argument(
        "--sft-adapter",
        default="artifacts/training/p10_qlora/adapter",
        help="P10 QLoRA SFT adapter dir.",
    )
    p.add_argument(
        "--dpo-adapter",
        default="artifacts/training/p10_dpo/adapter",
        help="P10 DPO adapter dir (applied on top of SFT when --apply-dpo is set).",
    )
    p.add_argument(
        "--apply-dpo",
        action="store_true",
        help="Also merge the DPO adapter on top of the SFT-merged weights.",
    )
    p.add_argument(
        "--out-dir",
        default="artifacts/training/r4_merged_1p5b",
        help="Where to save the merged HF model.",
    )
    p.add_argument(
        "--ollama-model-name",
        default="repopilot-qwen2.5-1p5b-adapter",
        help="Name to register the merged model under in Ollama.",
    )
    p.add_argument(
        "--convert-script",
        default=None,
        help="Path to llama.cpp convert_hf_to_gguf.py (optional; for GGUF). If omitted, "
        "the script only merges to HF and prints the Ollama/serve instructions.",
    )
    p.add_argument(
        "--gguf-out",
        default=None,
        help="Output GGUF path (used only with --convert-script).",
    )
    return p.parse_args()


def merge(base_model: str, sft_adapter: str, dpo_adapter: str | None, out_dir: Path) -> None:
    # Imported lazily so the script can be inspected/reviewed without torch loaded.
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"[R4] loading base: {base_model}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype="auto",
        device_map="auto",
    )
    print(f"[R4] merging SFT adapter: {sft_adapter}", flush=True)
    model = PeftModel.from_pretrained(model, sft_adapter)
    model = model.merge_and_unload()
    if dpo_adapter:
        print(f"[R4] merging DPO adapter on top: {dpo_adapter}", flush=True)
        model = PeftModel.from_pretrained(model, dpo_adapter)
        model = model.merge_and_unload()
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    print(f"[R4] merged model saved to {out_dir}", flush=True)


def to_gguf(convert_script: str, out_dir: Path, gguf_out: Path) -> None:
    print(f"[R4] converting to GGUF via {convert_script}", flush=True)
    subprocess.run(
        [sys.executable, convert_script, str(out_dir), "--outfile", str(gguf_out)],
        check=True,
    )
    print(f"[R4] GGUF written to {gguf_out}", flush=True)


def write_modelfile(gguf_out: Path, model_name: str, out_dir: Path) -> Path:
    # Prefer GGUF if produced, else serve the HF dir directly (Ollama can build
    # from an HF directory too, but a GGUF is the portable path).
    from_path = gguf_out if gguf_out.exists() else out_dir
    mf = out_dir.parent / f"Modelfile.{model_name}"
    mf.write_text(f"FROM {from_path}\n", encoding="utf-8")
    return mf


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    gguf_out = Path(args.gguf_out) if args.gguf_out else out_dir.with_suffix(".gguf")

    merge(args.base_model, args.sft_adapter, args.dpo_adapter if args.apply_dpo else None, out_dir)

    if args.convert_script:
        to_gguf(args.convert_script, out_dir, gguf_out)
    mf = write_modelfile(gguf_out, args.ollama_model_name, out_dir)

    print("\n=== R4 next steps (operator) ===")
    print(f"1. Register in Ollama:  ollama create {args.ollama_model_name} -f {mf}")
    print("2. Run DABench 3x (base config):")
    for i in (1, 2, 3):
        print(
            f"   .venv/Scripts/python scripts/run_dabench_agent_smoke.py "
            f"--run-id r4_1p5b_adapter_rep{i} --model {args.ollama_model_name} "
            f"--model-revision <digest> "
            f"--manifest evaluation/benchmarks/manifests/dabench_validation35_v2.json"
        )
    print("3. Register the 3 runs in docs/evidence/artifact_index.json and commit.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
