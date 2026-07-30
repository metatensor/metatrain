"""Tests for the general-purpose readout utilities (:mod:`metatrain.utils.readout`)."""

import pytest
import torch

from metatrain.utils.readout import LinearReadout, MoEReadout


def test_ungated_matches_torch_linear():
    readout = LinearReadout(4, 3, bias=True)
    reference = torch.nn.Linear(4, 3)
    reference.weight.data = readout.weight.data.clone()
    reference.bias.data = readout.bias.data.clone()

    x = torch.randn(5, 4)
    ignored = torch.zeros(5, dtype=torch.long)
    torch.testing.assert_close(readout(x, ignored), reference(x))


def test_bias_defaults_off_and_is_absent():
    readout = LinearReadout(4, 3)
    assert readout.bias is None
    assert "bias" not in readout.state_dict()


@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("n_columns", [None, 6])
def test_gated_matches_per_row_reference(bias, n_columns):
    """The dense-all-groups path must agree with the naive per-row weight gather."""
    n_groups, n_rows = 3, 7
    readout = LinearReadout(4, 2, n_groups=n_groups, bias=bias)
    assert tuple(readout.weight.shape) == (n_groups, 2, 4)

    shape = (n_rows, 4) if n_columns is None else (n_rows, n_columns, 4)
    x = torch.randn(*shape)
    group_idx = torch.randint(0, n_groups, (n_rows,))

    weight = readout.weight[group_idx]  # (n_rows, out, in)
    expected = torch.matmul(
        x if n_columns is not None else x.unsqueeze(1), weight.transpose(-2, -1)
    )
    if bias:
        expected = expected + readout.bias[group_idx].unsqueeze(1)
    if n_columns is None:
        expected = expected.squeeze(1)

    torch.testing.assert_close(readout(x, group_idx), expected)


@pytest.mark.parametrize("bias", [False, True])
def test_gated_paths_agree(bias):
    """The dense and grouped algorithms must give the same result."""
    n_groups, n_rows = 5, 40
    dense = LinearReadout(16, 2, n_groups=n_groups, bias=bias)
    grouped = LinearReadout(16, 2, n_groups=n_groups, bias=bias)
    assert not dense.grouped  # 5 * 2 <= 16
    with torch.no_grad():
        grouped.weight.copy_(dense.weight)
        if bias:
            grouped.bias.copy_(dense.bias)
    grouped.grouped = True  # force the other path on identical weights

    x = torch.randn(n_rows, 16)
    group_idx = torch.randint(0, n_groups, (n_rows,))
    torch.testing.assert_close(dense(x, group_idx), grouped(x, group_idx))

    # An absent group (no rows) must not break the grouped path.
    group_idx = torch.zeros(n_rows, dtype=torch.long)
    torch.testing.assert_close(dense(x, group_idx), grouped(x, group_idx))


def test_algorithm_choice_is_static_and_shape_independent():
    """The path is fixed by (n_groups, in, out) alone, never by the batch.

    The rule is ``n_groups * out_features > in_features`` -> grouped.
    """
    # Few groups onto a narrow property axis (the common readout): dense.
    assert not LinearReadout(512, 8, n_groups=7).grouped
    assert not LinearReadout(512, 64, n_groups=4).grouped
    # Wide outputs: grouped.
    assert LinearReadout(512, 128, n_groups=100).grouped
    assert LinearReadout(512, 32, n_groups=20).grouped
    # Many groups tips the balance even for a narrow output.
    assert LinearReadout(512, 8, n_groups=100).grouped
    # Ungated readouts never take the grouped path.
    assert not LinearReadout(512, 128).grouped
    # The choice does not depend on how many rows are passed in.
    readout = LinearReadout(512, 8, n_groups=7)
    for n_rows in (1, 20, 20000):
        readout(torch.randn(n_rows, 512), torch.zeros(n_rows, dtype=torch.long))
        assert not readout.grouped


def test_grouped_path_falls_back_to_dense_for_3d_features():
    """Sorting scales with the column axis, so 3-D input must stay on dense."""
    readout = LinearReadout(4, 16, n_groups=8)
    assert readout.grouped
    x = torch.randn(6, 5, 4)
    group_idx = torch.randint(0, 8, (6,))
    # Would raise if the grouped path were taken: it only handles 2-D.
    out = readout(x, group_idx)
    assert out.shape == (6, 5, 16)
    torch.testing.assert_close(out, readout._forward_dense(x, group_idx))


def test_gated_rows_only_see_their_own_group():
    """Perturbing another group's weights must not change a row's prediction."""
    readout = LinearReadout(4, 2, n_groups=3)
    x = torch.randn(5, 4)
    group_idx = torch.zeros(5, dtype=torch.long)

    before = readout(x, group_idx)
    with torch.no_grad():
        readout.weight[1:] += 10.0
    torch.testing.assert_close(readout(x, group_idx), before)


def test_ntk_parametrization_applies_input_scaling():
    readout = LinearReadout(16, 3, ntk_parametrization=True)
    assert readout.bias is None
    x = torch.randn(5, 16)
    torch.testing.assert_close(
        readout(x, torch.zeros(5, dtype=torch.long)),
        (x @ readout.weight.t()) * 16 ** (-0.5),
    )


@pytest.mark.parametrize("n_groups", [None, 3])
@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("out_features", [2, 16])  # dense and grouped paths
def test_linear_readout_torchscript(n_groups, bias, out_features):
    readout = LinearReadout(4, out_features, n_groups=n_groups, bias=bias)
    scripted = torch.jit.script(readout)
    x = torch.randn(6, 4)
    group_idx = torch.randint(0, n_groups or 1, (6,))
    torch.testing.assert_close(readout(x, group_idx), scripted(x, group_idx))
    x3 = torch.randn(6, 5, 4)
    torch.testing.assert_close(readout(x3, group_idx), scripted(x3, group_idx))


@pytest.mark.parametrize("bias", [False, True])
def test_moe_readout_bias_and_torchscript(bias):
    readout = MoEReadout(
        4,
        3,
        n_groups=5,
        num_experts=4,
        num_routed_experts=3,
        num_topk_experts=2,
        bias=bias,
    )
    for expert in list(readout.routed_experts) + list(readout.shared_experts):
        assert (expert.bias is not None) == bias

    scripted = torch.jit.script(readout)
    x = torch.randn(6, 4)
    group_idx = torch.randint(0, 5, (6,))
    torch.testing.assert_close(readout(x, group_idx), scripted(x, group_idx))


def test_moe_readout_invalid_hypers():
    with pytest.raises(ValueError, match="num_routed_experts must be >= 1"):
        MoEReadout(4, 3, 5, num_experts=4, num_routed_experts=0, num_topk_experts=1)
    with pytest.raises(ValueError, match="exceeds"):
        MoEReadout(4, 3, 5, num_experts=2, num_routed_experts=3, num_topk_experts=1)
    with pytest.raises(ValueError, match="num_topk_experts"):
        MoEReadout(4, 3, 5, num_experts=4, num_routed_experts=3, num_topk_experts=4)
