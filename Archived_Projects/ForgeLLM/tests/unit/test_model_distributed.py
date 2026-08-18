"""Pure correctness tests for the DDP gradient averaging contract."""

import pytest
import torch

from forgellm.model.distributed import average_gradient_tensors


def test_average_gradient_tensors_matches_manual_replica_mean() -> None:
    gradients = [torch.tensor([1.0, 3.0]), torch.tensor([3.0, 7.0])]

    torch.testing.assert_close(average_gradient_tensors(gradients), torch.tensor([2.0, 5.0]))


def test_average_gradient_tensors_rejects_mismatched_shapes() -> None:
    with pytest.raises(ValueError, match="same shape"):
        average_gradient_tensors([torch.ones(2), torch.ones(3)])
