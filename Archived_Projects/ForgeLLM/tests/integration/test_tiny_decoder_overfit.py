"""A tiny-set overfit gate for the complete Decoder-only model."""

import torch

from forgellm.model.config import ModelConfig
from forgellm.model.decoder import DecoderLM, next_token_loss


def test_tiny_decoder_can_overfit_a_repeated_sequence() -> None:
    torch.manual_seed(11)
    config = ModelConfig(
        vocab_size=16,
        d_model=32,
        n_layers=1,
        n_heads=4,
        n_kv_heads=2,
        ffn_hidden_dim=64,
        max_seq_len=16,
    )
    model = DecoderLM(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.03, weight_decay=0.0)
    input_ids = torch.tensor([[1, 4, 7, 2, 1, 4, 7, 2]]).repeat(4, 1)

    initial_loss = float(next_token_loss(model(input_ids).logits, input_ids).detach())
    final_loss = initial_loss
    for _ in range(120):
        optimizer.zero_grad(set_to_none=True)
        loss = next_token_loss(model(input_ids).logits, input_ids)
        loss.backward()  # type: ignore[no-untyped-call]
        optimizer.step()
        final_loss = float(loss.detach())

    assert final_loss < 0.02
    assert final_loss < initial_loss * 0.01
