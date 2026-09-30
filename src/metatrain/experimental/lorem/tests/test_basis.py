"""Bernstein radial basis, Racah normalization and paper defaults."""

import torch

from metatrain.experimental.lorem.modules.radial import bernstein_basis, binomial_row
from metatrain.experimental.lorem.modules.spherical import to_racah


def test_bernstein_is_zero_outside_cutoff_and_partition_of_unity():
    cutoff = 4.0
    r = torch.tensor([-0.2, 0.0, 1.0, 4.0, 4.2])
    values = bernstein_basis(r, cutoff, binomial_row(5).to(torch.float32))
    assert values.shape == (5, 5)
    torch.testing.assert_close(values[0], torch.zeros(5), atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(values[-1], torch.zeros(5), atol=1e-6, rtol=1e-6)
    inside = values[1:4].sum(dim=-1)
    torch.testing.assert_close(inside, torch.ones(3), atol=1e-5, rtol=1e-5)


def test_e3x_racah_scales_l0_to_one():
    y00 = torch.full((2, 1), 0.28209479177387814)
    racah = to_racah(y00, max_degree=0)
    torch.testing.assert_close(racah, torch.ones(2, 1), atol=1e-5, rtol=1e-5)


def test_e3x_racah_l1_has_unit_cartesian_coeffs():
    c1 = 0.4886025119029199
    sh = torch.tensor([[c1, 0.0, 0.0]])
    racah = to_racah(torch.cat([torch.zeros(1, 1), sh], dim=-1), max_degree=1)
    torch.testing.assert_close(racah[0, 1], torch.tensor(1.0), atol=1e-5, rtol=1e-5)


def test_paper_defaults():
    from metatrain.utils.architectures import get_default_hypers

    hypers = get_default_hypers("experimental.lorem")["model"]
    assert hypers["num_species"] == 8
    assert hypers["num_message_passing"] == 0
    assert hypers["equivariant_message_passing"] is True
    assert "cutoff_width" not in hypers
    assert "long_range" not in hypers
