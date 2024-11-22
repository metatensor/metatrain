import metatensor.torch
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
                properties=Labels.range("properties", 10),
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
                properties=Labels.range("properties", 1),
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
                properties=Labels.range("properties", 1),
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
                properties=Labels.range("properties", 1),
            ),
        ],
    )


def test_read_systems():
    with pytest.raises(NotImplementedError):
        read_systems("foo.npz")


def test_read_energy(monkeypatch, tmpdir, energy_tensor_map):
    monkeypatch.chdir(tmpdir)

    metatensor.torch.save(
        "energy.npz",
        energy_tensor_map,
    )

    conf = {
        "quantity": "energy",
        "read_from": "energy.npz",
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

    tensor_maps, target_info = read_energy(OmegaConf.create(conf))

    tensor_map = metatensor.torch.join(
        tensor_maps, axis="samples", remove_tensor_name=True
    )
    assert metatensor.torch.equal(tensor_map, energy_tensor_map)


def test_read_generic_scalar(monkeypatch, tmpdir, scalar_tensor_map):
    monkeypatch.chdir(tmpdir)

    metatensor.torch.save(
        "generic.npz",
        scalar_tensor_map,
    )

    conf = {
        "quantity": "generic",
        "read_from": "generic.npz",
        "reader": "metatensor",
        "keys": ["scalar"],
        "per_atom": True,
        "unit": "unit",
        "type": "scalar",
        "num_subtargets": 10,
    }

    tensor_maps, target_info = read_generic(OmegaConf.create(conf))

    tensor_map = metatensor.torch.join(
        tensor_maps, axis="samples", remove_tensor_name=True
    )
    assert metatensor.torch.equal(tensor_map, scalar_tensor_map)


def test_read_generic_spherical(monkeypatch, tmpdir, spherical_tensor_map):
    monkeypatch.chdir(tmpdir)

    metatensor.torch.save(
        "generic.npz",
        spherical_tensor_map,
    )

    conf = {
        "quantity": "generic",
        "read_from": "generic.npz",
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

    tensor_maps, target_info = read_generic(OmegaConf.create(conf))

    tensor_map = metatensor.torch.join(
        tensor_maps, axis="samples", remove_tensor_name=True
    )
    assert metatensor.torch.equal(tensor_map, spherical_tensor_map)


def test_read_generic_cartesian(monkeypatch, tmpdir, cartesian_tensor_map):
    monkeypatch.chdir(tmpdir)

    metatensor.torch.save(
        "generic.npz",
        cartesian_tensor_map,
    )

    conf = {
        "quantity": "generic",
        "read_from": "generic.npz",
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

    tensor_maps, target_info = read_generic(OmegaConf.create(conf))

    print(tensor_maps)

    tensor_map = metatensor.torch.join(
        tensor_maps, axis="samples", remove_tensor_name=True
    )
    print(tensor_map)
    print(cartesian_tensor_map)
    assert metatensor.torch.equal(tensor_map, cartesian_tensor_map)


def test_read_errors(monkeypatch, tmpdir, energy_tensor_map, scalar_tensor_map):
    monkeypatch.chdir(tmpdir)

    metatensor.torch.save(
        "energy.npz",
        energy_tensor_map,
    )

    conf = {
        "quantity": "energy",
        "read_from": "energy.npz",
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
    np.save("numpy_array.npz", numpy_array)
    conf["read_from"] = "numpy_array.npz"
    with pytest.raises(ValueError, match="Failed to read"):
        read_energy(OmegaConf.create(conf))
    conf["read_from"] = "energy.npz"

    conf["forces"] = True
    with pytest.raises(ValueError, match="Unexpected gradients"):
        read_energy(OmegaConf.create(conf))
    conf["forces"] = False

    metatensor.torch.save(
        "scalar.npz",
        scalar_tensor_map,
    )

    conf["read_from"] = "scalar.npz"
    with pytest.raises(ValueError, match="Unexpected samples"):
        read_generic(OmegaConf.create(conf))
