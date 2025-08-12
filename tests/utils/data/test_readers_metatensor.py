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
        read_systems("foo.mts")


def test_read_systems_diskdataset_zip(tmp_path):
    """Test reading DiskDataset .zip files with read_systems function."""
    import ase
    import zipfile
    from metatomic.torch import systems_to_torch
    from metatrain.utils.data.writers import DiskDatasetWriter
    
    # Create test systems using ASE
    system1 = ase.Atoms('H2', positions=[[0, 0, 0], [0, 0, 0.74]], cell=[2, 2, 2], pbc=True)
    system2 = ase.Atoms('H2', positions=[[0, 0, 0], [0, 0, 0.75]], cell=[2, 2, 2], pbc=True)
    ase_systems = [system1, system2]
    
    # Convert to torch systems
    torch_systems = [systems_to_torch(sys, dtype=torch.float64) for sys in ase_systems]
    
    # Create a DiskDataset .zip file
    zip_path = tmp_path / "test_dataset.zip"
    
    # Create zip with DiskDataset structure manually
    with zipfile.ZipFile(zip_path, "w") as zip_file:
        for i, system in enumerate(torch_systems):
            # Save system.mta file for each index
            with zip_file.open(f"{i}/system.mta", "w") as f:
                import metatomic.torch as mta
                mta.save(f, system.to("cpu").to(torch.float64))
    
    # Test reading the zip file
    result_systems = read_systems(str(zip_path))
    
    # Verify results
    assert isinstance(result_systems, list)
    assert len(result_systems) == 2
    
    # Check that systems are read in correct order and have correct properties
    for i, (original, result) in enumerate(zip(torch_systems, result_systems)):
        assert isinstance(result, torch.ScriptObject)
        
        # Check positions
        torch.testing.assert_close(result.positions, original.positions)
        
        # Check types (atomic numbers)
        torch.testing.assert_close(result.types, original.types)
        
        # Check cell
        torch.testing.assert_close(result.cell, original.cell)


def test_read_systems_empty_zip(tmp_path):
    """Test reading an empty .zip file."""
    import zipfile
    
    # Create empty zip file
    zip_path = tmp_path / "empty.zip"
    with zipfile.ZipFile(zip_path, "w") as zip_file:
        pass  # Empty zip
    
    # Should return empty list for empty zip
    result = read_systems(str(zip_path))
    assert isinstance(result, list)
    assert len(result) == 0


def test_read_systems_non_diskdataset_zip(tmp_path):
    """Test reading a .zip file that doesn't contain DiskDataset structure."""
    import zipfile
    
    # Create zip with non-DiskDataset structure
    zip_path = tmp_path / "other.zip"
    with zipfile.ZipFile(zip_path, "w") as zip_file:
        zip_file.writestr("some_file.txt", "hello world")
        zip_file.writestr("another/file.data", "some data")
    
    # Should return empty list for non-DiskDataset zip
    result = read_systems(str(zip_path))
    assert isinstance(result, list)
    assert len(result) == 0


def test_read_systems_zip_ordering(tmp_path):
    """Test that systems are read in correct numerical order."""
    import ase
    import zipfile
    from metatomic.torch import systems_to_torch
    
    # Create test systems with different positions to distinguish them
    positions = [
        [[0, 0, 0], [0, 0, 0.74]],  # system 0
        [[0, 0, 0], [0, 0, 0.75]],  # system 1 
        [[0, 0, 0], [0, 0, 0.76]],  # system 2
    ]
    
    ase_systems = [
        ase.Atoms('H2', positions=pos, cell=[2, 2, 2], pbc=True) 
        for pos in positions
    ]
    torch_systems = [systems_to_torch(sys, dtype=torch.float64) for sys in ase_systems]
    
    # Create zip with indices in non-sequential order to test sorting
    zip_path = tmp_path / "test_ordering.zip"
    with zipfile.ZipFile(zip_path, "w") as zip_file:
        # Write systems in order: 2, 0, 1 (intentionally out of order)
        for idx, system in [(2, torch_systems[2]), (0, torch_systems[0]), (1, torch_systems[1])]:
            with zip_file.open(f"{idx}/system.mta", "w") as f:
                import metatomic.torch as mta
                mta.save(f, system.to("cpu").to(torch.float64))
    
    # Read systems
    result_systems = read_systems(str(zip_path))
    
    # Verify they are returned in correct numerical order (0, 1, 2)
    assert len(result_systems) == 3
    for i, (original, result) in enumerate(zip(torch_systems, result_systems)):
        torch.testing.assert_close(result.positions, original.positions)
        # Verify specific positions to ensure correct ordering
        expected_z_coord = 0.74 + i * 0.01  # 0.74, 0.75, 0.76
        assert abs(result.positions[1, 2].item() - expected_z_coord) < 1e-10


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
        tensor_maps, _ = read_energy(OmegaConf.create(conf))

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
        tensor_maps, _ = read_generic(OmegaConf.create(conf))

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
        tensor_maps, _ = read_generic(OmegaConf.create(conf))

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
        tensor_maps, _ = read_generic(OmegaConf.create(conf))

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
            read_energy(OmegaConf.create(conf))
        conf["read_from"] = "energy.mts"

        conf["forces"] = True
        with pytest.raises(ValueError, match="Unexpected gradients"):
            read_energy(OmegaConf.create(conf))
        conf["forces"] = False

        mts.save("scalar.mts", scalar_tensor_map)

        conf["read_from"] = "scalar.mts"
        with pytest.raises(ValueError, match="Unexpected samples"):
            read_generic(OmegaConf.create(conf))
