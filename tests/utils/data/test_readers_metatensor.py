import metatensor.torch
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
                values=torch.rand(1, 1, dtype=torch.float64),
                samples=Labels(
                    names=["system"],
                    values=torch.tensor([[0]], dtype=torch.int32),
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
                values=torch.rand(2, 10, dtype=torch.float64),
                samples=Labels(
                    names=["system", "atom"],
                    values=torch.tensor([[0, 0], [0, 1]], dtype=torch.int32),
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
                values=torch.rand(1, 1, 1, dtype=torch.float64),
                samples=Labels(
                    names=["system"],
                    values=torch.tensor([[0]], dtype=torch.int32),
                ),
                components=[
                    Labels(
                        names=["o3_mu"],
                        values=torch.arange(0, 1, dtype=torch.int32).reshape(-1, 1),
                    ),
                ],
                properties=Labels.single(),
            ),
            TensorBlock(
                values=torch.rand(1, 5, 1, dtype=torch.float64),
                samples=Labels(
                    names=["system"],
                    values=torch.tensor([[0]], dtype=torch.int32),
                ),
                components=[
                    Labels(
                        names=["o3_mu"],
                        values=torch.arange(-2, 3, dtype=torch.int32).reshape(-1, 1),
                    ),
                ],
                properties=Labels.single(),
            ),
        ],
    )


@pytest.fixture
def cartesian_tensor_map():
    return TensorMap(
        keys=Labels.single(),
        blocks=[
            TensorBlock(
                values=torch.rand(1, 3, 3, 1, dtype=torch.float64),
                samples=Labels(
                    names=["system"],
                    values=torch.tensor([[0]], dtype=torch.int32),
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
                properties=Labels.single(),
            ),
        ],
    )


def test_read_systems():
    with pytest.raises(NotImplementedError):
        read_systems("foo.mts")


def test_read_energy(monkeypatch, tmpdir, energy_tensor_map):
    monkeypatch.chdir(tmpdir)

    torch.save(
        [energy_tensor_map, energy_tensor_map],
        "energy.mts",
    )

    conf = {
        "quantity": "energy",
        "read_from": "energy.mts",
        "reader": "metatensor",
        "key": "true_energy",
        "unit": "eV",
        "type": "scalar",
        "per_atom": False,
        "num_properties": 1,
        "forces": False,
        "stress": False,
        "virial": False,
    }

    tensor_maps, target_info = read_energy(OmegaConf.create(conf))

    for tensor_map in tensor_maps:
        assert metatensor.torch.equal(tensor_map, energy_tensor_map)


def test_read_generic_scalar(monkeypatch, tmpdir, scalar_tensor_map):
    monkeypatch.chdir(tmpdir)

    torch.save(
        [scalar_tensor_map, scalar_tensor_map],
        "generic.mts",
    )

    conf = {
        "quantity": "generic",
        "read_from": "generic.mts",
        "reader": "metatensor",
        "keys": ["scalar"],
        "per_atom": True,
        "unit": "unit",
        "type": "scalar",
        "num_properties": 10,
    }

    tensor_maps, target_info = read_generic(OmegaConf.create(conf))

    for tensor_map in tensor_maps:
        assert metatensor.torch.equal(tensor_map, scalar_tensor_map)


def test_read_generic_spherical(monkeypatch, tmpdir, spherical_tensor_map):
    monkeypatch.chdir(tmpdir)

    torch.save(
        [spherical_tensor_map, spherical_tensor_map],
        "generic.mts",
    )

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
        "num_properties": 1,
    }

    tensor_maps, target_info = read_generic(OmegaConf.create(conf))

    for tensor_map in tensor_maps:
        assert metatensor.torch.equal(tensor_map, spherical_tensor_map)


def test_read_generic_cartesian(monkeypatch, tmpdir, cartesian_tensor_map):
    monkeypatch.chdir(tmpdir)

    torch.save(
        [cartesian_tensor_map, cartesian_tensor_map],
        "generic.mts",
    )

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
        "num_properties": 1,
    }

    tensor_maps, target_info = read_generic(OmegaConf.create(conf))

    for tensor_map in tensor_maps:
        assert metatensor.torch.equal(tensor_map, cartesian_tensor_map)
