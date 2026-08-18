"""Graph-capture regression test for the complete decoder."""

import torch

from forgellm.model.config import ModelConfig
from forgellm.model.decoder import DecoderLM


def test_decoder_is_captured_as_one_torch_compile_graph() -> None:
    torch.manual_seed(51)
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
    model = DecoderLM(config).eval()
    input_ids = torch.randint(0, 32, (2, 8))
    expected = model(input_ids).logits

    compiled = torch.compile(model, backend="eager", fullgraph=True)
    actual = compiled(input_ids).logits

    torch.testing.assert_close(actual, expected)
