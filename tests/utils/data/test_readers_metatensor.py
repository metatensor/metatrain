import metatensor.torch as mts
import numpy as np
import pytest
import torch
from metatensor.torch import Labels, TensorBlock, TensorMap
from omegaconf import OmegaConf

from metatrain.utils.data.readers.metatensor import (
    read_energy,
    read_generic,
    read_systems,
)


@pytest.fixture
def energy_tensor_map():
    return TensorMap(
        keys=Labels.single(),
        blocks=[
            TensorBlock(
                values=torch.rand(2, 1, dtype=torch.float64),
                samples=Labels(
                    names=["system"],
                    values=torch.tensor([[0], [1]], dtype=torch.int32),
                ),
                components=[],
                properties=Labels.range("energy", 1),
            )
        ],
    )


@pytest.fixture
def scalar_tensor_map():
    return TensorMap(
        keys=Labels.single(),
        blocks=[
            TensorBlock(
                values=torch.rand(3, 10, dtype=torch.float64),
                samples=Labels(
                    names=["system", "atom"],
                    values=torch.tensor([[0, 0], [0, 1], [1, 0]], dtype=torch.int32),
                ),
                components=[],
                properties=Labels.range("scalar", 10),
            )
        ],
    )


@pytest.fixture
def spherical_tensor_map():
    return TensorMap(
        keys=Labels(
            names=["o3_lambda", "o3_sigma"],
            values=torch.tensor([[0, 1], [2, 1]]),
        ),
        blocks=[
            TensorBlock(
                values=torch.rand(2, 1, 1, dtype=torch.float64),
                samples=Labels(
                    names=["system"],
                    values=torch.tensor([[0], [1]], dtype=torch.int32),
                ),
                components=[
                    Labels(
                        names=["o3_mu"],
                        values=torch.arange(0, 1, dtype=torch.int32).reshape(-1, 1),
                    ),
                ],
                properties=Labels.range("spherical", 1),
            ),
            TensorBlock(
                values=torch.rand(2, 5, 1, dtype=torch.float64),
                samples=Labels(
                    names=["system"],
                    values=torch.tensor([[0], [1]], dtype=torch.int32),
                ),
                components=[
                    Labels(
                        names=["o3_mu"],
                        values=torch.arange(-2, 3, dtype=torch.int32).reshape(-1, 1),
                    ),
                ],
                properties=Labels.range("spherical", 1),
            ),
        ],
    )


@pytest.fixture
def cartesian_tensor_map():
    return TensorMap(
        keys=Labels.single(),
        blocks=[
            TensorBlock(
                values=torch.rand(2, 3, 3, 1, dtype=torch.float64),
                samples=Labels(
                    names=["system"],
                    values=torch.tensor([[0], [1]], dtype=torch.int32),
                ),
                components=[
                    Labels(
                        names=["xyz_1"],
                        values=torch.arange(0, 3, dtype=torch.int32).reshape(-1, 1),
                    ),
                    Labels(
                        names=["xyz_2"],
                        values=torch.arange(0, 3, dtype=torch.int32).reshape(-1, 1),
                    ),
                ],
                properties=Labels.range("cartesian", 1),
            ),
        ],
    )


def test_read_systems():
    with pytest.raises(NotImplementedError):
        read_systems("foo.mts")


def test_read_energy(tmpdir, energy_tensor_map):
    conf = {
        "quantity": "energy",
        "read_from": "energy.mts",
        "reader": "metatensor",
        "key": "true_energy",
        "unit": "eV",
        "type": "scalar",
        "per_atom": False,
        "num_subtargets": 1,
        "forces": False,
        "stress": False,
        "virial": False,
    }

    with tmpdir.as_cwd():
        mts.save("energy.mts", energy_tensor_map)
        tensor_maps, _ = read_energy("energy", OmegaConf.create(conf))

    tensor_map = mts.join(tensor_maps, axis="samples", remove_tensor_name=True)
    assert mts.equal(tensor_map, energy_tensor_map)


def test_read_generic_scalar(tmpdir, scalar_tensor_map):
    conf = {
        "quantity": "generic",
        "read_from": "generic.mts",
        "reader": "metatensor",
        "keys": ["scalar"],
        "per_atom": True,
        "unit": "unit",
        "type": "scalar",
        "num_subtargets": 10,
    }

    with tmpdir.as_cwd():
        mts.save("generic.mts", scalar_tensor_map)
        tensor_maps, _ = read_generic("generic", OmegaConf.create(conf))

    tensor_map = mts.join(tensor_maps, axis="samples", remove_tensor_name=True)
    assert mts.equal(tensor_map, scalar_tensor_map)


def test_read_generic_spherical(tmpdir, spherical_tensor_map):
    conf = {
        "quantity": "generic",
        "read_from": "generic.mts",
        "reader": "metatensor",
        "keys": ["o3_lambda", "o3_sigma"],
        "per_atom": False,
        "unit": "unit",
        "type": {
            "spherical": {
                "irreps": [
                    {"o3_lambda": 0, "o3_sigma": 1},
                    {"o3_lambda": 2, "o3_sigma": 1},
                ],
            },
        },
        "num_subtargets": 1,
    }

    with tmpdir.as_cwd():
        mts.save("generic.mts", spherical_tensor_map)
        tensor_maps, _ = read_generic("generic", OmegaConf.create(conf))

    tensor_map = mts.join(tensor_maps, axis="samples", remove_tensor_name=True)
    assert mts.equal(tensor_map, spherical_tensor_map)


def test_read_generic_cartesian(tmpdir, cartesian_tensor_map):
    conf = {
        "quantity": "generic",
        "read_from": "generic.mts",
        "reader": "metatensor",
        "keys": ["cartesian"],
        "per_atom": False,
        "unit": "unit",
        "type": {
            "cartesian": {
                "rank": 2,
            },
        },
        "num_subtargets": 1,
    }

    with tmpdir.as_cwd():
        mts.save("generic.mts", cartesian_tensor_map)
        tensor_maps, _ = read_generic("generic", OmegaConf.create(conf))

    tensor_map = mts.join(tensor_maps, axis="samples", remove_tensor_name=True)

    assert mts.equal(tensor_map, cartesian_tensor_map)


def test_read_errors(tmpdir, energy_tensor_map, scalar_tensor_map):
    with tmpdir.as_cwd():
        mts.save("energy.mts", energy_tensor_map)

    conf = {
        "quantity": "energy",
        "read_from": "energy.mts",
        "reader": "metatensor",
        "key": "true_energy",
        "unit": "eV",
        "type": "scalar",
        "per_atom": False,
        "num_subtargets": 1,
        "forces": False,
        "stress": False,
        "virial": False,
    }

    numpy_array = np.zeros((2, 2))

    with tmpdir.as_cwd():
        np.save("numpy_array.mts", numpy_array)
        conf["read_from"] = "numpy_array.mts"
        with pytest.raises(ValueError, match="Failed to read"):
            read_energy("energy", OmegaConf.create(conf))
        conf["read_from"] = "energy.mts"

        conf["forces"] = True
        with pytest.raises(ValueError, match="Unexpected gradients"):
            read_energy("energy", OmegaConf.create(conf))
        conf["forces"] = False

        mts.save("scalar.mts", scalar_tensor_map)

        conf["read_from"] = "scalar.mts"
        with pytest.raises(ValueError, match="Unexpected samples"):
            read_generic("scalar", OmegaConf.create(conf))
