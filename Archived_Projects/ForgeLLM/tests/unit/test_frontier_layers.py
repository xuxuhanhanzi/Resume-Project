"""Tests for modern MoE, residual, MTP, and Muon reference modules."""

import torch

from forgellm.model.frontier_layers import (
    AttentionResiduals,
    ManifoldHyperConnectionLite,
    MultiTokenPredictionHead,
    SparseMoE,
    hash_expert_indices,
    multi_token_prediction_loss,
    sinkhorn_doubly_stochastic,
)
from forgellm.model.optim import Muon, zeropower_via_newton_schulz5


def test_sparse_moe_routes_each_token_to_top_k_and_backpropagates() -> None:
    torch.manual_seed(30)
    module = SparseMoE(16, 24, num_experts=4, top_k=2, shared_expert=True)
    hidden = torch.randn(2, 5, 16, requires_grad=True)

    output = module(hidden)
    loss = output.hidden_states.square().mean() + 0.01 * output.auxiliary_loss
    loss.backward()

    assert output.hidden_states.shape == hidden.shape
    assert output.selected_experts.shape == (2, 5, 2)
    assert int(output.expert_counts.sum()) == 20
    assert hidden.grad is not None and bool(torch.isfinite(hidden.grad).all())


def test_hash_moe_routing_is_deterministic_and_drives_reference_experts() -> None:
    token_ids = torch.tensor([[0, 1, 2, 3, 4, 5]])
    first = hash_expert_indices(token_ids, 4)
    second = hash_expert_indices(token_ids, 4)
    module = SparseMoE(8, 12, num_experts=4, top_k=1, shared_expert=False)

    output = module(torch.randn(1, 6, 8), token_ids=token_ids, use_hash_routing=True)

    assert torch.equal(first, second)
    assert torch.equal(output.selected_experts.squeeze(-1), first)
    assert int(output.expert_counts.sum()) == token_ids.numel()


def test_sinkhorn_matrix_is_nonnegative_and_doubly_stochastic() -> None:
    matrix = sinkhorn_doubly_stochastic(torch.randn(4, 4), iterations=20)

    assert bool((matrix >= 0).all())
    torch.testing.assert_close(matrix.sum(dim=0), torch.ones(4), atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(matrix.sum(dim=1), torch.ones(4), atol=1e-5, rtol=1e-5)


def test_mhc_lite_preserves_stream_shape_and_constrained_mixing() -> None:
    module = ManifoldHyperConnectionLite(stream_count=3, d_model=8)
    streams = torch.randn(2, 4, 3, 8)
    branch = torch.randn(2, 4, 8)

    output = module(streams, branch)

    assert output.shape == streams.shape
    torch.testing.assert_close(
        module.mixing_matrix().sum(dim=0), torch.ones(3), atol=1e-5, rtol=1e-5
    )


def test_attention_residual_weights_form_a_layer_probability_distribution() -> None:
    torch.manual_seed(31)
    module = AttentionResiduals(d_model=12, attention_dim=6)
    query = torch.randn(2, 5, 12)
    history = torch.randn(2, 5, 4, 12)

    selected, weights = module(query, history)

    assert selected.shape == query.shape
    assert weights.shape == (2, 5, 4)
    torch.testing.assert_close(weights.sum(dim=-1), torch.ones(2, 5))


def test_multi_token_prediction_loss_uses_each_future_offset() -> None:
    token_ids = torch.tensor([[0, 1, 2, 3, 4]])
    logits = torch.full((1, 5, 2, 5), -20.0)
    for position in range(4):
        logits[0, position, 0, token_ids[0, position + 1]] = 20.0
    for position in range(3):
        logits[0, position, 1, token_ids[0, position + 2]] = 20.0

    loss, per_offset = multi_token_prediction_loss(logits, token_ids)
    head = MultiTokenPredictionHead(d_model=8, vocab_size=5, num_future_tokens=2)

    assert float(loss) < 1e-6
    assert per_offset.shape == (2,)
    assert head(torch.randn(2, 5, 8)).shape == (2, 5, 2, 5)


def test_newton_schulz_makes_singular_values_more_uniform() -> None:
    torch.manual_seed(32)
    matrix = torch.randn(8, 8) * torch.linspace(0.2, 3.0, 8)

    before = torch.linalg.svdvals(matrix)
    after = torch.linalg.svdvals(zeropower_via_newton_schulz5(matrix))

    assert float(after.std() / after.mean()) < float(before.std() / before.mean())
    assert bool(torch.isfinite(after).all())


def test_muon_step_reduces_a_matrix_quadratic() -> None:
    torch.manual_seed(33)
    parameter = torch.nn.Parameter(torch.randn(6, 6))
    optimizer = Muon([parameter], lr=0.05, momentum=0.0, nesterov=False)
    before = float(parameter.square().sum().detach())

    parameter.square().sum().backward()  # type: ignore[no-untyped-call]
    optimizer.step()
    after = float(parameter.square().sum().detach())

    assert after < before
