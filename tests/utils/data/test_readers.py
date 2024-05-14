"""Test correct type and metadata of readers. Correct values will be checked
within the tests for each fileformat."""

import logging

import ase
import ase.io
import pytest
import torch
from metatensor.torch import Labels
from omegaconf import OmegaConf
from targets.test_targets_ase import ase_system, ase_systems

from metatensor.models.utils.data.readers import (
    read_energy,
    read_forces,
    read_stress,
    read_systems,
    read_targets,
    read_virial,
)


@pytest.mark.parametrize("fileformat", (None, ".xyz", ".extxyz"))
def test_read_systems(fileformat, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    filename = "systems.xyz"
    systems = ase_systems()
    ase.io.write(filename, systems)

    results = read_systems(filename, fileformat=fileformat, dtype=torch.float16)

    assert isinstance(results, list)
    assert len(results) == len(systems)
    for system, result in zip(systems, results):
        assert isinstance(result, torch.ScriptObject)

        torch.testing.assert_close(
            result.positions, torch.tensor(system.positions, dtype=torch.float16)
        )
        torch.testing.assert_close(
            result.types, torch.tensor([1, 1], dtype=torch.int32)
        )


def test_read_systems_unknown_fileformat():
    with pytest.raises(ValueError, match="fileformat '.bar' is not supported"):
        read_systems("foo.bar")


@pytest.mark.parametrize("fileformat", (None, ".xyz", ".extxyz"))
def test_read_energies(fileformat, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    filename = "systems.xyz"
    systems = ase_systems()
    ase.io.write(filename, systems)

    results = read_energy(
        filename, fileformat=fileformat, target_value="true_energy", dtype=torch.float16
    )

    assert type(results) is list
    assert len(results) == len(systems)
    for i_system, result in enumerate(results):
        assert result.values.dtype is torch.float16
        assert result.samples.names == ["system"]
        assert result.samples.values == torch.tensor([[i_system]])
        assert result.properties == Labels("energy", torch.tensor([[0]]))


@pytest.mark.parametrize("fileformat", (None, ".xyz", ".extxyz"))
def test_read_forces(fileformat, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    filename = "systems.xyz"
    systems = ase_systems()
    ase.io.write(filename, systems)

    results = read_forces(
        filename, fileformat=fileformat, target_value="forces", dtype=torch.float16
    )

    assert type(results) is list
    assert len(results) == len(systems)
    for i_system, result in enumerate(results):
        assert result.values.dtype is torch.float16
        assert result.samples.names == ["sample", "system", "atom"]
        assert torch.all(result.samples["sample"] == torch.tensor(0))
        assert torch.all(result.samples["system"] == torch.tensor(i_system))
        assert result.components == [Labels(["xyz"], torch.arange(3).reshape(-1, 1))]
        assert result.properties == Labels("energy", torch.tensor([[0]]))


@pytest.mark.parametrize("reader", [read_stress, read_virial])
@pytest.mark.parametrize("fileformat", (None, ".xyz", ".extxyz"))
def test_read_stress_virial(reader, fileformat, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    filename = "systems.xyz"
    systems = ase_systems()
    ase.io.write(filename, systems)

    results = reader(
        filename, fileformat=fileformat, target_value="stress-3x3", dtype=torch.float16
    )

    assert type(results) is list
    assert len(results) == len(systems)
    components = [
        Labels(["xyz_1"], torch.arange(3).reshape(-1, 1)),
        Labels(["xyz_2"], torch.arange(3).reshape(-1, 1)),
    ]
    for result in results:
        assert result.values.dtype is torch.float16
        assert result.samples.names == ["sample"]
        assert result.samples.values == torch.tensor([[0]])
        assert result.components == components
        assert result.properties == Labels("energy", torch.tensor([[0]]))


@pytest.mark.parametrize("reader", [read_energy, read_forces, read_stress, read_virial])
def test_reader_unknown_fileformat(reader):
    with pytest.raises(ValueError, match="fileformat '.bar' is not supported"):
        reader("foo.bar", target_value="baz")


STRESS_VIRIAL_DICT = {
    "read_from": "systems.xyz",
    "file_format": ".xyz",
    "key": "stress-3x3",
}


@pytest.mark.parametrize(
    "stress_dict, virial_dict",
    [[STRESS_VIRIAL_DICT, False], [False, STRESS_VIRIAL_DICT]],
)
def test_read_targets(stress_dict, virial_dict, monkeypatch, tmp_path, caplog):
    monkeypatch.chdir(tmp_path)

    filename = "systems.xyz"
    systems = ase_systems()
    ase.io.write(filename, systems)

    energy_section = {
        "quantity": "energy",
        "read_from": filename,
        "file_format": ".xyz",
        "key": "true_energy",
        "forces": {"read_from": filename, "file_format": ".xyz", "key": "forces"},
        "stress": stress_dict,
        "virial": virial_dict,
    }

    conf = {
        "energy": energy_section,
        "energy2": energy_section,
    }

    caplog.set_level(logging.INFO)
    result = read_targets(OmegaConf.create(conf), dtype=torch.float16)

    assert any(["Forces found" in rec.message for rec in caplog.records])

    if stress_dict:
        assert any(["Stress found" in rec.message for rec in caplog.records])
    if virial_dict:
        assert any(["Virial found" in rec.message for rec in caplog.records])

    for target_name, target_list in result.items():
        conf[target_name]

        assert type(target_list) is list
        for target in target_list:
            assert target.keys == Labels(["_"], torch.tensor([[0]]))

            result_block = target.block()
            assert result_block.values.dtype is torch.float16
            assert result_block.samples.names == ["system"]
            assert result_block.properties == Labels("energy", torch.tensor([[0]]))

            pos_grad = result_block.gradient("positions")
            assert pos_grad.values.dtype is torch.float16
            assert pos_grad.samples.names == ["sample", "system", "atom"]
            assert pos_grad.components == [
                Labels(["xyz"], torch.arange(3).reshape(-1, 1))
            ]
            assert pos_grad.properties == Labels("energy", torch.tensor([[0]]))

            disp_grad = result_block.gradient("strain")
            components = [
                Labels(["xyz_1"], torch.arange(3).reshape(-1, 1)),
                Labels(["xyz_2"], torch.arange(3).reshape(-1, 1)),
            ]
            assert disp_grad.values.dtype is torch.float16
            assert disp_grad.samples.names == ["sample"]
            assert disp_grad.components == components
            assert disp_grad.properties == Labels("energy", torch.tensor([[0]]))


@pytest.mark.parametrize(
    "stress_dict, virial_dict",
    [[STRESS_VIRIAL_DICT, False], [False, STRESS_VIRIAL_DICT]],
)
def test_read_targets_warnings(stress_dict, virial_dict, monkeypatch, tmp_path, caplog):
    monkeypatch.chdir(tmp_path)

    filename = "systems.xyz"
    systems = ase_system()

    # Delete gradient sections
    systems.info.pop("stress-3x3")
    systems.info.pop("stress-9")
    systems.arrays.pop("forces")

    ase.io.write(filename, systems)

    energy_section = {
        "quantity": "energy",
        "read_from": filename,
        "file_format": ".xyz",
        "key": "true_energy",
        "forces": {"read_from": filename, "file_format": ".xyz", "key": "forces"},
        "stress": stress_dict,
        "virial": virial_dict,
    }

    conf = {"energy": energy_section}

    caplog.set_level(logging.WARNING)
    read_targets(OmegaConf.create(conf))  # , slice_samples_by="system")

    assert any(["No Forces found" in rec.message for rec in caplog.records])

    if stress_dict:
        assert any(["No Stress found" in rec.message for rec in caplog.records])
    if virial_dict:
        assert any(["No Virial found" in rec.message for rec in caplog.records])


def test_read_targets_error(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    filename = "systems.xyz"
    systems = ase_system()
    ase.io.write(filename, systems)

    energy_section = {
        "quantity": "energy",
        "read_from": filename,
        "file_format": ".xyz",
        "key": "true_energy",
        "forces": {"read_from": filename, "file_format": ".xyz", "key": "forces"},
        "stress": True,
        "virial": True,
    }

    conf = {"energy": energy_section}

    with pytest.raises(
        ValueError,
        match="stress and virial at the same time",
    ):
        read_targets(OmegaConf.create(conf))


def test_unsopprted_quantity():
    conf = {
        "energy": {
            "quantity": "foo",
        }
    }

    with pytest.raises(
        ValueError,
        match="Quantity: 'foo' is not supported. Choose 'energy'.",
    ):
        read_targets(OmegaConf.create(conf))
