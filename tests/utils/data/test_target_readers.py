import ase.io
import pytest
import torch
from test_structure_readers import ase_system

from metatensor.models.utils.data import read_targets
from metatensor.models.utils.data.readers.targets import read_ase


@pytest.mark.parametrize("fileformat", (None, ".xyz"))
def test_read_targets(fileformat, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    filename = "structures.xyz"
    structures = ase_system()
    ase.io.write(filename, structures)

    result = read_targets(
        filename, fileformat=fileformat, target_values=["true_energy", "dipole_moment"]
    )

    assert isinstance(result, dict)

    torch.testing.assert_close(
        result["true_energy"].block().values, torch.tensor([[42.0]])
    )
    torch.testing.assert_close(
        result["dipole_moment"].block().values, torch.tensor([[10.0]])
    )


def test_read_target_unknown_fileformat():
    with pytest.raises(ValueError, match="fileformat '.bar' is not supported"):
        read_targets("foo.bar", target_values="foo")


def test_read_ase(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    filename = "structures.xyz"

    structures = ase_system()
    ase.io.write(filename, structures)

    result = read_ase(filename, target_values=["true_energy", "dipole_moment"])

    assert isinstance(result, dict)

    torch.testing.assert_close(
        result["true_energy"].block().values, torch.tensor([[42.0]])
    )
    torch.testing.assert_close(
        result["dipole_moment"].block().values, torch.tensor([[10.0]])
    )
