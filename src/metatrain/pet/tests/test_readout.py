import pytest
import torch

from metatrain.pet.modules.readout import LinearReadout


def _reference(readout, features, group_idx):
    """Naive conditioned linear map, gathering each row's weight and bias explicitly."""
    weight = readout.weight[group_idx]  # (n_rows, out, in)
    bias = readout.bias[group_idx]  # (n_rows, out)
    if features.dim() == 2:
        return torch.einsum("ri,roi->ro", features, weight) + bias
    return torch.einsum("rci,roi->rco", features, weight) + bias.unsqueeze(1)


def test_unconditioned_matches_torch_linear():
    """Check an unconditioned readout loads the state dict of ``torch.nn.Linear``."""
    readout = LinearReadout(4, 3)
    reference = torch.nn.Linear(4, 3)
    readout.load_state_dict(reference.state_dict())

    features = torch.randn(5, 4)
    torch.testing.assert_close(
        readout(features, torch.zeros(5, dtype=torch.long)), reference(features)
    )


@pytest.mark.parametrize("n_columns", [None, 6])
def test_conditioned_matches_reference(n_columns):
    readout = LinearReadout(8, 2, n_groups=3)

    shape = (7, 8) if n_columns is None else (7, n_columns, 8)
    features = torch.randn(shape)
    group_idx = torch.tensor([2, 0, 2, 2, 0, 0, 2])  # group 1 is absent

    torch.testing.assert_close(
        readout(features, group_idx, torch.tensor([0, 2])),
        _reference(readout, features, group_idx),
    )


@pytest.mark.parametrize("n_groups", [None, 3])
def test_bias_scale_pools_exactly(n_groups):
    """Check that, with ``bias_scale``, the readout of a weighted sum of features is
    the weighted sum of the readouts of each feature."""
    readout = LinearReadout(8, 2, n_groups=n_groups).to(torch.float64)
    features = torch.randn(7, 5, 8, dtype=torch.float64)
    weights = torch.rand(7, 5, dtype=torch.float64)
    group_idx = torch.tensor([2, 0, 2, 2, 0, 0, 2]) % (n_groups or 1)
    present_groups = torch.unique(group_idx)

    outputs = readout(features, group_idx, present_groups)
    expected = (outputs * weights[..., None]).sum(1)
    pooled = (features * weights[..., None]).sum(1)
    actual = readout(
        pooled, group_idx, present_groups, bias_scale=weights.sum(1, keepdim=True)
    )
    torch.testing.assert_close(actual, expected)
