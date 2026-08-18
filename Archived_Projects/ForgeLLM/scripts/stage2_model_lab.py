"""Run bounded Stage 2 GPU/reference experiments and write one JSON artifact."""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import torch
from torch import Tensor
from torch.nn import functional as F

from forgellm.model.attention import (
    causal_attention_mask,
    manual_scaled_dot_product_attention,
)
from forgellm.model.config import ModelConfig
from forgellm.model.decoder import DecoderLM, next_token_loss
from forgellm.model.frontier_attention import MultiHeadLatentAttention
from forgellm.model.frontier_layers import SparseMoE
from forgellm.model.systems import (
    Int8WeightOnlyLinear,
    benchmark_callable,
    online_softmax_blockwise_attention,
)


def _attention_experiment(device: torch.device) -> dict[str, object]:
    torch.manual_seed(60)
    query = torch.randn(1, 4, 128, 32, device=device)
    key = torch.randn(1, 4, 128, 32, device=device)
    value = torch.randn(1, 4, 128, 32, device=device)
    mask = causal_attention_mask(128, 128, past_length=0, device=device)

    def manual() -> Tensor:
        return manual_scaled_dot_product_attention(query, key, value, allowed_mask=mask)

    def sdpa() -> Tensor:
        return F.scaled_dot_product_attention(query, key, value, is_causal=True)

    def tiled() -> Tensor:
        return online_softmax_blockwise_attention(
            query,
            key,
            value,
            query_block_size=32,
            key_block_size=32,
        )

    with torch.inference_mode():
        manual_output = manual()
        sdpa_output = sdpa()
        tiled_output = tiled()
        manual_benchmark = benchmark_callable(manual, warmup=3, repetitions=20)
        sdpa_benchmark = benchmark_callable(sdpa, warmup=3, repetitions=20)
        tiled_benchmark = benchmark_callable(tiled, warmup=1, repetitions=3)
    return {
        "shape": list(query.shape),
        "manual_vs_sdpa_max_abs": float((manual_output - sdpa_output).abs().max()),
        "tiled_vs_sdpa_max_abs": float((tiled_output - sdpa_output).abs().max()),
        "manual_ms": manual_benchmark.mean_milliseconds,
        "sdpa_ms": sdpa_benchmark.mean_milliseconds,
        "python_tiled_ms": tiled_benchmark.mean_milliseconds,
        "claim_boundary": (
            "Python tiled code demonstrates online softmax; it is not a fused kernel."
        ),
    }


def _decoder_smoke(device: torch.device) -> dict[str, object]:
    torch.manual_seed(61)
    config = ModelConfig(
        vocab_size=320,
        d_model=256,
        n_layers=7,
        n_heads=8,
        n_kv_heads=2,
        ffn_hidden_dim=768,
        max_seq_len=128,
        attention_backend="sdpa",
        qk_norm=True,
    )
    model = DecoderLM(config).to(device)
    input_ids = torch.randint(0, config.vocab_size, (2, 64), device=device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    output = model(input_ids)
    loss = next_token_loss(output.logits, input_ids)
    loss.backward()  # type: ignore[no-untyped-call]
    peak_mib = torch.cuda.max_memory_allocated() / (1024 * 1024) if device.type == "cuda" else None
    return {
        "parameters": model.parameter_count(),
        "logits_shape": list(output.logits.shape),
        "loss": float(loss.detach()),
        "gradients_finite": all(
            parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
            for parameter in model.parameters()
        ),
        "peak_memory_mib": peak_mib,
    }


def _architecture_experiment(device: torch.device) -> dict[str, object]:
    mla = MultiHeadLatentAttention(
        d_model=64,
        n_heads=4,
        q_rank=16,
        kv_rank=12,
        nope_dim=12,
        rope_dim=4,
        value_dim=16,
        max_seq_len=64,
    ).to(device)
    moe = SparseMoE(64, 96, num_experts=8, top_k=2, shared_expert=True).to(device)
    hidden = torch.randn(2, 32, 64, device=device)
    _, cache = mla(hidden, use_cache=True)
    routed = moe(hidden)
    assert cache is not None
    return {
        "mla_cache_elements_per_token": mla.compressed_cache_elements_per_token,
        "mha_cache_elements_per_token": mla.mha_cache_elements_per_token,
        "mla_cache_ratio": (
            mla.compressed_cache_elements_per_token / mla.mha_cache_elements_per_token
        ),
        "moe_expert_counts": cast(list[int], routed.expert_counts.cpu().tolist()),
        "moe_total_assignments": int(routed.expert_counts.sum()),
        "moe_auxiliary_loss": float(routed.auxiliary_loss.detach()),
    }


def _quantization_experiment(device: torch.device) -> dict[str, object]:
    torch.manual_seed(62)
    source = torch.nn.Linear(256, 128).to(device).eval()
    quantized = Int8WeightOnlyLinear(source).to(device).eval()
    inputs = torch.randn(4, 16, 256, device=device)
    with torch.inference_mode():
        expected = source(inputs)
        actual = quantized(inputs)
    fp32_bytes = source.weight.numel() * source.weight.element_size()
    return {
        "max_abs_error": float((actual - expected).abs().max()),
        "mean_abs_error": float((actual - expected).abs().mean()),
        "fp32_weight_bytes": fp32_bytes,
        "int8_reference_weight_bytes": quantized.stored_weight_bytes,
        "storage_ratio": quantized.stored_weight_bytes / fp32_bytes,
    }


def _compile_experiment(device: torch.device) -> dict[str, object]:
    config = ModelConfig(
        vocab_size=32,
        d_model=32,
        n_layers=1,
        n_heads=4,
        n_kv_heads=2,
        ffn_hidden_dim=64,
        max_seq_len=16,
        attention_backend="sdpa",
    )
    model = DecoderLM(config).to(device).eval()
    inputs = torch.randint(0, 32, (2, 8), device=device)
    expected = model(inputs).logits
    compiled = torch.compile(model, backend="eager", fullgraph=True)
    actual = compiled(inputs).logits
    return {
        "fullgraph_eager_max_abs_error": float((actual - expected).abs().max()),
        "inductor_status": "blocked: official environment has no working Triton installation",
        "claim_boundary": "Graph capture is verified; Inductor speedup is not verified on Windows.",
    }


def main() -> int:
    """Run the bounded experiment matrix and write a deterministic report location."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    report = {
        "environment": {
            "torch": torch.__version__,
            "compiled_cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        },
        "decoder_smoke": _decoder_smoke(device),
        "attention": _attention_experiment(device),
        "architecture": _architecture_experiment(device),
        "quantization": _quantization_experiment(device),
        "compile": _compile_experiment(device),
    }
    output = Path("artifacts/stage02/stage2_model_lab_report.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(output.resolve())
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
