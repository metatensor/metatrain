import pytest
import torch

from metatrain.utils.readout import LinearReadout


def _reference(readout, features, group_idx):
    """Naive gated linear map, gathering each row's weight and bias explicitly."""
    weight = readout.weight[group_idx]  # (n_rows, out, in)
    bias = readout.bias[group_idx]  # (n_rows, out)
    if features.dim() == 2:
        return torch.einsum("ri,roi->ro", features, weight) + bias
    return torch.einsum("rci,roi->rco", features, weight) + bias.unsqueeze(1)


def test_ungated_matches_torch_linear():
    """An ungated readout has the parameter names and initialization of
    ``torch.nn.Linear``, so that it can replace it without changing checkpoints."""
    torch.manual_seed(0)
    readout = LinearReadout(4, 3)
    torch.manual_seed(0)
    reference = torch.nn.Linear(4, 3)

    assert readout.state_dict().keys() == reference.state_dict().keys()
    torch.testing.assert_close(readout.weight, reference.weight, rtol=0, atol=0)
    torch.testing.assert_close(readout.bias, reference.bias, rtol=0, atol=0)

    features = torch.randn(5, 4)
    torch.testing.assert_close(
        readout(features, torch.zeros(5, dtype=torch.long)), reference(features)
    )

    assert "bias" not in LinearReadout(4, 3, bias=False).state_dict()


@pytest.mark.parametrize("n_columns", [None, 6])
@pytest.mark.parametrize("out_features", [2, 16])  # dense and grouped paths
def test_gated_matches_reference(n_columns, out_features):
    readout = LinearReadout(8, out_features, n_groups=3)
    assert readout.weight.shape == (3, out_features, 8)
    assert readout.grouped == (out_features == 16)

    shape = (7, 8) if n_columns is None else (7, n_columns, 8)
    features = torch.randn(shape)
    group_idx = torch.tensor([2, 0, 2, 2, 0, 0, 2])  # group 1 is absent

    expected = _reference(readout, features, group_idx)
    torch.testing.assert_close(readout(features, group_idx), expected)
    torch.testing.assert_close(
        readout(features, group_idx, torch.tensor([0, 2])), expected
    )


def test_present_groups_gradients():
    """Restricting the dense path to the present groups gives the same gradients."""
    readout = LinearReadout(4, 3, n_groups=6)
    features = torch.randn(9, 4)
    group_idx = torch.tensor([4, 1, 1, 4, 5, 1, 4, 5, 1])

    expected = torch.autograd.grad(
        readout(features, group_idx).sum(), list(readout.parameters())
    )
    actual = torch.autograd.grad(
        readout(features, group_idx, torch.unique(group_idx)).sum(),
        list(readout.parameters()),
    )
    for a, e in zip(actual, expected, strict=True):
        torch.testing.assert_close(a, e)


@pytest.mark.parametrize("n_groups", [None, 3])
@pytest.mark.parametrize("out_features", [2, 16])  # dense and grouped paths
def test_bias_scale_pools_exactly(n_groups, out_features):
    """With ``bias_scale``, the readout of a weighted sum of features equals the
    weighted sum of the readouts of each feature."""
    readout = LinearReadout(8, out_features, n_groups=n_groups).to(torch.float64)
    features = torch.randn(7, 5, 8, dtype=torch.float64)
    weights = torch.rand(7, 5, dtype=torch.float64)
    group_idx = torch.tensor([2, 0, 2, 2, 0, 0, 2]) % (n_groups or 1)

    expected = (readout(features, group_idx) * weights[..., None]).sum(1)
    pooled = (features * weights[..., None]).sum(1)
    actual = readout(pooled, group_idx, bias_scale=weights.sum(1, keepdim=True))
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("n_groups", [None, 3])
def test_torchscript(n_groups):
    readout = LinearReadout(8, 16, n_groups=n_groups)
    scripted = torch.jit.script(readout)
    group_idx = torch.randint(0, n_groups or 1, (6,))
    for features in [torch.randn(6, 8), torch.randn(6, 5, 8)]:
        torch.testing.assert_close(
            scripted(features, group_idx), readout(features, group_idx)
        )
    bias_scale = torch.rand(6, 1)
    features = torch.randn(6, 8)
    torch.testing.assert_close(
        scripted(features, group_idx, None, bias_scale),
        readout(features, group_idx, bias_scale=bias_scale),
    )
