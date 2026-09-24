import math

import pytest
import torch

from metatrain.experimental.lorem.modules.tensor_dense import TensorDense


def _rotate_z(theta: float) -> torch.Tensor:
    return torch.tensor(
        [
            [math.cos(theta), -math.sin(theta), 0.0],
            [math.sin(theta), math.cos(theta), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )


def _rotate_l1(values: torch.Tensor, rotation: torch.Tensor) -> torch.Tensor:
    """Rotate real SH ℓ=1 channels ordered (y, z, x)."""
    xyz = torch.stack([values[:, 2], values[:, 0], values[:, 1]], dim=-1)
    xyz_rot = xyz @ rotation.T
    return torch.stack([xyz_rot[:, 1], xyz_rot[:, 2], xyz_rot[:, 0]], dim=-1)


def test_tensor_dense_scalar_is_rotation_invariant():
    layer = TensorDense(
        in_features=2,
        out_features=1,
        in_max_degree=1,
        out_max_degree=0,
        include_pseudotensors=False,
    )
    layer.eval()
    spherical = torch.randn(4, 4, 2)
    rotation = _rotate_z(math.pi / 5.0)
    rotated = spherical.clone()
    for feat in range(2):
        rotated[:, 1:4, feat] = _rotate_l1(spherical[:, 1:4, feat], rotation)

    out = layer(spherical)[:, 0, 0]
    out_rot = layer(rotated)[:, 0, 0]
    torch.testing.assert_close(out, out_rot, atol=1e-5, rtol=1e-5)


def test_tensor_dense_preserves_shape():
    layer = TensorDense(3, 4, in_max_degree=2, out_max_degree=2)
    output = layer(torch.randn(5, 9, 3))
    assert output.shape == (5, 9, 4)


@pytest.mark.parametrize("max_degree", [0, 1, 2])
def test_tensor_dense_zero_atoms(max_degree):
    n_lm = (max_degree + 1) ** 2
    layer = TensorDense(2, 2, in_max_degree=max_degree, out_max_degree=max_degree)
    output = layer(torch.zeros(0, n_lm, 2))
    assert output.shape == (0, n_lm, 2)
