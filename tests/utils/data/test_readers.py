"""Test correct type and metadata of readers. Correct values will be checked
within the tests for each fileformat"""

import logging

import ase
import ase.io
import pytest
import torch
from metatensor.torch import Labels
from omegaconf import OmegaConf
from targets.test_targets_ase import ase_system

from metatensor.models.utils.data.readers import (
    read_energy,
    read_forces,
    read_stress,
    read_structures,
    read_targets,
    read_virial,
)


@pytest.mark.parametrize("fileformat", (None, ".xyz"))
def test_read_structures(fileformat, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    filename = "structures.xyz"
    structures = ase_system()
    ase.io.write(filename, structures)

    result = read_structures(filename, fileformat=fileformat)

    assert isinstance(result, list)
    assert len(result) == 1
    assert isinstance(result[0], torch.ScriptObject)

    torch.testing.assert_close(result[0].positions, torch.tensor(structures.positions))
    torch.testing.assert_close(
        result[0].species, torch.tensor([1, 1], dtype=torch.int32)
    )


def test_read_structures_unknown_fileformat():
    with pytest.raises(ValueError, match="fileformat '.bar' is not supported"):
        read_structures("foo.bar")


@pytest.mark.parametrize("fileformat", (None, ".xyz"))
def test_read_energies(fileformat, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    filename = "structures.xyz"
    structures = ase_system()
    ase.io.write(filename, structures)

    result = read_energy(filename, fileformat=fileformat, target_value="true_energy")

    assert result.samples.names == ["structure"]
    assert result.properties == Labels.single()


@pytest.mark.parametrize("fileformat", (None, ".xyz"))
def test_read_forces(fileformat, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    filename = "structures.xyz"
    structures = ase_system()
    ase.io.write(filename, structures)

    result = read_forces(filename, fileformat=fileformat, target_value="forces")

    assert result.samples.names == ["sample", "structure", "atom"]
    assert result.components == [Labels(["direction"], torch.arange(3).reshape(-1, 1))]
    assert result.properties == Labels.single()


@pytest.mark.parametrize("reader", [read_stress, read_virial])
@pytest.mark.parametrize("fileformat", (None, ".xyz"))
def test_read_stress_virial(reader, fileformat, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    filename = "structures.xyz"
    structures = ase_system()
    ase.io.write(filename, structures)

    result = reader(filename, fileformat=fileformat, target_value="stress-3x3")

    components = [
        Labels(["cell_vector"], torch.arange(3).reshape(-1, 1)),
        Labels(["coordinate"], torch.arange(3).reshape(-1, 1)),
    ]

    assert result.samples.names == ["sample"]
    assert result.components == components
    assert result.properties == Labels.single()


@pytest.mark.parametrize("reader", [read_energy, read_forces, read_stress, read_virial])
def test_reader_unknown_fileformat(reader):
    with pytest.raises(ValueError, match="fileformat '.bar' is not supported"):
        reader("foo.bar", target_value="baz")


STRESS_VIRIAL_DICT = {
    "read_from": "structures.xyz",
    "file_format": ".xyz",
    "key": "stress-3x3",
}


@pytest.mark.parametrize(
    "stress_dict, virial_dict",
    [[STRESS_VIRIAL_DICT, False], [False, STRESS_VIRIAL_DICT]],
)
def test_read_targets(stress_dict, virial_dict, monkeypatch, tmp_path, caplog):
    monkeypatch.chdir(tmp_path)

    filename = "structures.xyz"
    structures = ase_system()
    ase.io.write(filename, structures)

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
    result = read_targets(OmegaConf.create(conf))  # , slice_samples_by="structure")

    assert any(["Forces found" in rec.message for rec in caplog.records])

    if stress_dict:
        assert any(["Stress found" in rec.message for rec in caplog.records])
    if virial_dict:
        assert any(["Virial found" in rec.message for rec in caplog.records])

    for target_name, target in result.items():
        conf[target_name]

        assert target.keys == Labels(["lambda", "sigma"], torch.tensor([(0, 1)]))

        # Use blocks() to ensure that there is only a single block
        result_block = target.block()
        assert result_block.samples.names == ["structure"]
        assert result_block.properties == Labels.single()

        pos_grad = result_block.gradient("positions")
        assert pos_grad.samples.names == ["sample", "structure", "atom"]
        assert pos_grad.components == [
            Labels(["direction"], torch.arange(3).reshape(-1, 1))
        ]
        assert pos_grad.properties == Labels.single()

        disp_grad = result_block.gradient("displacement")
        components = [
            Labels(["cell_vector"], torch.arange(3).reshape(-1, 1)),
            Labels(["coordinate"], torch.arange(3).reshape(-1, 1)),
        ]

        assert disp_grad.samples.names == ["sample"]
        assert disp_grad.components == components
        assert disp_grad.properties == Labels.single()


@pytest.mark.parametrize(
    "stress_dict, virial_dict",
    [[STRESS_VIRIAL_DICT, False], [False, STRESS_VIRIAL_DICT]],
)
def test_read_targets_warnings(stress_dict, virial_dict, monkeypatch, tmp_path, caplog):
    monkeypatch.chdir(tmp_path)

    filename = "structures.xyz"
    structures = ase_system()

    # Delete gradient sections
    structures.info.pop("stress-3x3")
    structures.info.pop("stress-9")
    structures.arrays.pop("forces")

    ase.io.write(filename, structures)

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
    read_targets(OmegaConf.create(conf))  # , slice_samples_by="structure")

    assert any(["Forces not found" in rec.message for rec in caplog.records])

    if stress_dict:
        assert any(["Stress not found" in rec.message for rec in caplog.records])
    if virial_dict:
        assert any(["Virial not found" in rec.message for rec in caplog.records])


def test_read_targets_error(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    filename = "structures.xyz"
    structures = ase_system()
    ase.io.write(filename, structures)

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
        # TODO: slice by structure in the reader
        read_targets(OmegaConf.create(conf))  # , slice_samples_by="structure")


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
        # TODO: slice by structure in the reader
        read_targets(OmegaConf.create(conf))  # , slice_samples_by="structure")
