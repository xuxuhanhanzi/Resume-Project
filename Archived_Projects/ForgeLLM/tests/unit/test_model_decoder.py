"""Decoder, loss, persistence, generation, and gradient tests."""

import torch

from forgellm.model.config import ModelConfig
from forgellm.model.decoder import DecoderLM, next_token_loss
from forgellm.model.generation import GenerationConfig, filter_logits, generate


def tiny_model_config(*, tie_embeddings: bool = True) -> ModelConfig:
    return ModelConfig(
        vocab_size=24,
        d_model=32,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        ffn_hidden_dim=64,
        max_seq_len=24,
        tie_embeddings=tie_embeddings,
    )


def test_decoder_forward_backward_and_weight_tying() -> None:
    torch.manual_seed(5)
    model = DecoderLM(tiny_model_config())
    input_ids = torch.randint(0, 24, (2, 10))

    output = model(input_ids)
    training_output = model(input_ids, return_hidden_states=True)
    loss = next_token_loss(output.logits, input_ids)
    loss.backward()  # type: ignore[no-untyped-call]

    assert output.logits.shape == (2, 10, 24)
    assert output.hidden_states is None
    assert training_output.hidden_states is not None
    assert training_output.hidden_states.shape == (2, 10, 32)
    assert model.lm_head.weight.data_ptr() == model.token_embedding.weight.data_ptr()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.requires_grad]
    assert gradients
    assert all(gradient is not None for gradient in gradients)
    assert all(
        bool(torch.isfinite(gradient).all()) for gradient in gradients if gradient is not None
    )


def test_next_token_loss_uses_shifted_targets() -> None:
    input_ids = torch.tensor([[2, 1, 3]])
    logits = torch.full((1, 3, 4), -20.0)
    logits[0, 0, 1] = 20.0
    logits[0, 1, 3] = 20.0

    assert float(next_token_loss(logits, input_ids)) < 1e-6


def test_decoder_state_dict_round_trip_preserves_logits() -> None:
    torch.manual_seed(6)
    source = DecoderLM(tiny_model_config()).eval()
    restored = DecoderLM(tiny_model_config()).eval()
    input_ids = torch.randint(0, 24, (2, 7))

    restored.load_state_dict(source.state_dict())

    torch.testing.assert_close(source(input_ids).logits, restored(input_ids).logits)


def test_decoder_cache_matches_full_forward() -> None:
    torch.manual_seed(7)
    model = DecoderLM(tiny_model_config()).eval()
    input_ids = torch.randint(0, 24, (2, 9))
    full_logits = model(input_ids).logits
    cache = None
    pieces = []

    for position in range(input_ids.size(1)):
        output = model(input_ids[:, position : position + 1], cache=cache, use_cache=True)
        cache = output.cache
        pieces.append(output.logits)

    torch.testing.assert_close(torch.cat(pieces, dim=1), full_logits, rtol=3e-5, atol=3e-6)


def test_generation_greedy_and_zero_new_tokens() -> None:
    model = DecoderLM(tiny_model_config()).eval()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    prompt = torch.tensor([[1, 2, 3]])

    generated = generate(model, prompt, GenerationConfig(max_new_tokens=4))

    assert generated.tolist() == [[1, 2, 3, 0, 0, 0, 0]]
    assert torch.equal(generate(model, prompt, GenerationConfig(max_new_tokens=0)), prompt)


def test_top_k_and_top_p_filters_keep_expected_candidates() -> None:
    logits = torch.tensor([[4.0, 3.0, 2.0, 1.0]])

    top_k = filter_logits(logits, top_k=2, top_p=None)
    top_p = filter_logits(logits, top_k=None, top_p=0.8)

    assert torch.isfinite(top_k).tolist() == [[True, True, False, False]]
    assert torch.isfinite(top_p).tolist() == [[True, True, False, False]]
