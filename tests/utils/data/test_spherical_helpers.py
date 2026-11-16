import itertools

import metatensor.torch as mts
import pytest
import torch
from featomic.torch.clebsch_gordan._coefficients import (
    calculate_cg_coefficients,
)
from metatensor.torch import Labels, TensorBlock, TensorMap

from metatrain.utils.data.spherical_helpers import (
    couple_tensor_blocks,
    uncouple_tensor_blocks,
)


@pytest.fixture(params=["cpu", "cuda"])
def device(request):
    device = request.param
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    return device


@pytest.fixture(params=[32, 64])
def dtype(request):
    return torch.float32 if request.param == 32 else torch.float64


@pytest.fixture
def coupled_tensormap(device, dtype):
    """
    Simple coupled tensormap coming from a (l=1)x(l=1) block.

    In the coupled representation, this has 3 blocks (l=0, l=1, l=2).
    """
    # All blocks come from the same uncoupled block.
    properties = Labels(
        names=["l_1", "l_2", "n_1", "n_2"],
        values=torch.tensor([[1, 1, 0, 0]]),
    )

    # We have 2 samples, corresponding to 2 atoms in the same system.
    samples = Labels(names=["system", "atom"], values=torch.tensor([[0, 0], [0, 1]]))

    # Define the values for each block in the coupled representation.
    # One of the samples has only an invariant part,
    # while the other has also l=1 and l=2 parts.
    scalars = torch.tensor([[1], [2]])
    l1s = torch.tensor([[0, 0, 0], [1, 2, 3]])
    l2s = torch.tensor([[0, 0, 0, 0, 0], [4, 5, 6, 7, 8]])

    return TensorMap(
        keys=Labels(
            names=["o3_lambda", "o3_sigma"],
            values=torch.tensor([[0, 1], [1, -1], [2, 1]]),
        ),
        blocks=[
            TensorBlock(
                values=scalars.reshape(2, 1, 1),
                samples=samples,
                components=[Labels(names=["o3_mu"], values=torch.tensor([[0]]))],
                properties=properties,
            ),
            TensorBlock(
                values=l1s.reshape(2, 3, 1),
                samples=samples,
                components=[
                    Labels(names=["o3_mu"], values=torch.tensor([[-1], [0], [1]]))
                ],
                properties=properties,
            ),
            TensorBlock(
                values=l2s.reshape(2, 5, 1),
                samples=samples,
                components=[
                    Labels(names=["o3_mu"], values=torch.arange(-2, 3).reshape(-1, 1))
                ],
                properties=properties,
            ),
        ],
    ).to(device, dtype=dtype)


@pytest.fixture
def big_uncoupled_tensormap(device, dtype):
    """
    Random uncoupled tensormap.

    It is used to test a roundtrip of coupling and uncoupling.

    It has random keys and samples to verify that the operations
    are not affected by these and they propagate them correctly.
    """
    # Set random seed for reproducibility
    torch.manual_seed(0)

    n1s = torch.arange(3)
    n2s = torch.arange(4)

    properties = Labels(
        names=["n_1", "n_2"],
        values=torch.tensor([[n1, n2] for n1 in n1s for n2 in n2s]),
    )

    samples = Labels(
        names=["system", "atom", "something"],
        values=torch.randint(0, 10, (7, 3)),
    )

    l_values = torch.tensor(list(itertools.product(range(3), range(4))))

    return TensorMap(
        keys=Labels(
            names=[
                "o3_lambda_1",
                "o3_lambda_2",
                "o3_sigma_1",
                "o3_sigma_2",
                "some",
                "extra",
                "keys",
            ],
            values=torch.concat(
                [
                    l_values,
                    torch.ones((len(l_values), 2), dtype=torch.int64),
                    torch.randint(0, 10, (len(l_values), 3)),
                ],
                dim=1,
            ),
        ),
        blocks=[
            TensorBlock(
                values=torch.randn(
                    (len(samples), (2 * l1 + 1), (2 * l2 + 1), len(properties)),
                ),
                samples=samples,
                components=[
                    Labels(
                        names=["o3_mu_1"],
                        values=torch.arange(-l1, l1 + 1).reshape(-1, 1),
                    ),
                    Labels(
                        names=["o3_mu_2"],
                        values=torch.arange(-l2, l2 + 1).reshape(-1, 1),
                    ),
                ],
                properties=properties,
            )
            for l1, l2 in l_values
        ],
    ).to(device, dtype=dtype)


@pytest.fixture
def cg_coeffs(device, dtype):
    """
    Returns the Clebsch-Gordan coefficients for the coupling of two irreps
    with angular momenta ``o3_lambda_1`` and ``o3_lambda_2`` to an irrep
    with angular momentum ``o3_lambda``.
    """
    return calculate_cg_coefficients(
        8,
        cg_backend="python-dense",
        arrays_backend="torch",
        dtype=dtype,
        device=device,
    )


def test_uncouple(coupled_tensormap, cg_coeffs):
    """Tests the uncoupling of a simple coupled tensormap with l=1xl=1."""
    uncoupled = uncouple_tensor_blocks(coupled_tensormap, cg_coeffs)

    # We are uncoupling a l=1xl=1 block, which in the coupled representation
    # has 3 blocks (l=0, l=1, l=2).
    assert len(coupled_tensormap) == 3
    assert len(uncoupled) == 1

    # Check that the keys now reflect the uncoupled representation
    assert uncoupled.keys.names[:4] == [
        "o3_lambda_1",
        "o3_lambda_2",
        "o3_sigma_1",
        "o3_sigma_2",
    ]

    # Check that the block is correct.
    uncoupled_block = uncoupled.block(0).to("cpu")

    coupled_samples = coupled_tensormap.block(0).samples
    assert uncoupled_block.samples == coupled_samples

    uncoup_components = uncoupled_block.components
    assert len(uncoup_components) == 2
    assert uncoup_components[0].names == ["o3_mu_1"]
    assert len(uncoup_components[0].values) == 3
    assert torch.all(uncoup_components[0].values.ravel() == torch.tensor([-1, 0, 1]))
    assert uncoup_components[1].names == ["o3_mu_2"]
    assert len(uncoup_components[1].values) == 3
    assert torch.all(uncoup_components[1].values.ravel() == torch.tensor([-1, 0, 1]))

    assert uncoupled_block.properties.names == ["n_1", "n_2"]
    assert len(uncoupled_block.properties.values) == 1
    assert torch.all(uncoupled_block.properties.values.ravel() == torch.tensor([0, 0]))

    # The first atom has only an invariant part, so the uncoupled block is diagonal.
    expected = torch.tensor(
        [
            [
                [-0.5774, 0.0000, 0.0000],
                [0.0000, -0.5774, 0.0000],
                [0.0000, 0.0000, -0.5774],
            ],
            [
                [-9.2610, 1.4142, 4.2426],
                [5.6569, 3.7443, 4.2426],
                [1.4142, 5.6569, 2.0527],
            ],
        ],
        dtype=uncoupled_block.values.dtype,
        device=uncoupled_block.values.device,
    ).unsqueeze(-1)

    assert torch.allclose(uncoupled_block.values, expected, atol=1e-4)


def test_roundtrip(big_uncoupled_tensormap, cg_coeffs, dtype):
    coupled = couple_tensor_blocks(big_uncoupled_tensormap, cg_coeffs)
    recovered = uncouple_tensor_blocks(coupled, cg_coeffs)

    # In order to compare the two tensormaps we sort the samples,
    # because the operations does not preserve the order of the samples.
    # Should we make them preserve the order?
    recovered = mts.sort(recovered, axes=["samples"])
    big_uncoupled_tensormap = mts.sort(big_uncoupled_tensormap, axes=["samples"])

    tolerance = 1e-12 if dtype == torch.float64 else 1e-6
    mts.allclose_raise(recovered, big_uncoupled_tensormap, atol=tolerance)


def test_jit_script_couple():
    torch.jit.script(couple_tensor_blocks)


def test_jit_script_uncouple():
    torch.jit.script(uncouple_tensor_blocks)
