from pathlib import Path
from typing import Dict, List, Tuple

import metatensor.torch as mts
import numpy as np
import pytest
import torch
from metatensor.torch import Labels, TensorBlock, TensorMap
from metatomic.torch import ModelCapabilities, ModelOutput, System

from metatrain.utils.data.readers.ase import read
from metatrain.utils.data.writers import (
    ASEWriter,
    DiskDatasetWriter,
    MemMapWriter,
    MetatensorWriter,
    get_writer,
)


def systems_capabilities_predictions(
    cell: torch.tensor = None,
) -> Tuple[List[System], ModelCapabilities, Dict[str, TensorMap]]:
    if cell is None:
        cell = torch.zeros((3, 3))

    systems = 2 * [
        System(
            types=torch.tensor([1, 1]),
            positions=torch.tensor([[0, 0, 0], [0, 0, 0.74]]),
            cell=cell,
            pbc=torch.logical_not(torch.all(cell == 0, dim=1)),
        ),
    ]

    # Create a mock TensorMap for predictions
    n_systems = len(systems)
    values = torch.tensor([[1.0], [2.0]])
    block = TensorBlock(
        values=values.reshape(-1, 1),
        samples=Labels(["system"], torch.arange(n_systems).reshape(-1, 1)),
        components=[],
        properties=Labels(["energy"], torch.tensor([(0,)])),
    )
    block.add_gradient(
        "positions",
        TensorBlock(
            values=torch.arange(12, dtype=torch.get_default_dtype()).reshape(4, 3, 1),
            samples=Labels(
                ["sample", "atom"], torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]])
            ),
            components=[Labels.range("xyz", 3)],
            properties=Labels(["energy"], torch.tensor([(0,)])),
        ),
    )
    if not torch.all(cell == 0):
        block.add_gradient(
            "strain",
            TensorBlock(
                values=torch.arange(18, dtype=torch.get_default_dtype()).reshape(
                    2, 3, 3, 1
                ),
                samples=Labels(["sample"], torch.tensor([0, 1]).reshape(-1, 1)),
                components=[Labels.range("xyz_1", 3), Labels.range("xyz_2", 3)],
                properties=Labels(["energy"], torch.tensor([(0,)])),
            ),
        )

    predictions = {"energy": TensorMap(Labels.single(), [block])}

    capabilities = ModelCapabilities(
        length_unit="angstrom",
        outputs={"energy": ModelOutput(quantity="energy", unit="kcal/mol")},
        interaction_range=1.0,
        dtype="float32",
    )

    return systems, capabilities, predictions


def test_ASEWriter(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    systems, capabilities, predictions = systems_capabilities_predictions()

    filename = "test_output.xyz"

    writer = ASEWriter(filename, capabilities=capabilities)
    writer.write(systems, predictions)
    writer.finish()

    # Read the file and verify its contents
    frames = read(filename, index=":")
    assert len(frames) == len(systems)
    for i, atoms in enumerate(frames):
        assert atoms.info["energy"] == float(predictions["energy"].block().values[i, 0])
        assert all(atoms.pbc == 3 * [False])


def test_write_components_and_properties_xyz(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    systems, _, _ = systems_capabilities_predictions()

    capabilities = ModelCapabilities(
        length_unit="angstrom",
        outputs={"energy": ModelOutput(quantity="dos", unit="")},
        interaction_range=1.0,
        dtype="float32",
    )

    predictions = {
        "dos": TensorMap(
            keys=Labels.single(),
            blocks=[
                TensorBlock(
                    values=torch.rand(2, 3, 100),
                    samples=Labels(["system"], torch.tensor([[0], [1]])),
                    components=[
                        Labels.range("xyz", 3),
                    ],
                    properties=Labels(
                        ["property"],
                        torch.arange(100, dtype=torch.int32).reshape(-1, 1),
                    ),
                )
            ],
        )
    }

    filename = "test_output.xyz"

    writer = ASEWriter(filename, capabilities=capabilities)
    writer.write(systems, predictions)
    writer.finish()

    # Read the file and verify its contents
    frames = read(filename, index=":")
    assert len(frames) == len(systems)
    for atoms in frames:
        assert atoms.info["dos"].shape == (3, 100)


def test_write_components_and_properties_xyz_per_atom(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    systems, _, _ = systems_capabilities_predictions()

    capabilities = ModelCapabilities(
        length_unit="angstrom",
        outputs={"energy": ModelOutput(quantity="dos", unit="", sample_kind="atom")},
        interaction_range=1.0,
        dtype="float32",
    )

    predictions = {
        "dos": TensorMap(
            keys=Labels.single(),
            blocks=[
                TensorBlock(
                    values=torch.rand(4, 3, 100),
                    samples=Labels(
                        ["system", "atom"],
                        torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]]),
                    ),
                    components=[
                        Labels.range("xyz", 3),
                    ],
                    properties=Labels(
                        ["property"],
                        torch.arange(100, dtype=torch.int32).reshape(-1, 1),
                    ),
                )
            ],
        )
    }

    filename = "test_output.xyz"

    writer = ASEWriter(filename, capabilities=capabilities)
    writer.write(systems, predictions)
    writer.finish()

    # Read the file and verify its contents
    frames = read(filename, index=":")
    assert len(frames) == len(systems)
    for atoms in frames:
        assert atoms.arrays["dos"].shape == (2, 300)


def test_write_xyz_cell(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    cell_expected = torch.eye(3)
    systems, capabilities, predictions = systems_capabilities_predictions(
        cell=cell_expected
    )

    filename = "test_output.xyz"

    writer = ASEWriter(filename, capabilities=capabilities)
    writer.write(systems, predictions)
    writer.finish()

    # Read the file and verify its contents
    frames = read(filename, index=":")
    for i, atoms in enumerate(frames):
        cell_actual = torch.tensor(atoms.cell.array, dtype=cell_expected.dtype)
        torch.testing.assert_close(cell_actual, cell_expected)
        assert all(atoms.pbc == 3 * [True])
        assert atoms.info["energy"] == float(predictions["energy"].block().values[i, 0])
        assert atoms.arrays["forces"].shape == (2, 3)
        assert atoms.info["stress"].shape == (3, 3)
        assert atoms.info["virial"].shape == (3, 3)


@pytest.mark.parametrize(
    "filename",
    ("test_output.xyz", "test_output.mts", "test_output.zip", "test_output.npy"),
)
@pytest.mark.parametrize("fileformat", (None, "same_as_filename"))
@pytest.mark.parametrize("cell", (None, torch.eye(3)))
def test_write_predictions(filename, fileformat, cell, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    systems, capabilities, predictions = systems_capabilities_predictions(cell=cell)

    if fileformat == "same_as_filename" or fileformat is None:
        fileformat = "." + filename.split(".")[1]

    try:
        writer = get_writer(filename, capabilities=capabilities)
    except KeyError:
        raise ValueError(f"fileformat '{fileformat}' is not supported")

    writer.write(systems, predictions)
    writer.finish()

    if filename.endswith(".xyz"):
        frames = read(filename, index=":")
        assert len(frames) == len(systems)
        for i, frame in enumerate(frames):
            assert frame.info["energy"] == float(
                predictions["energy"].block().values[i, 0]
            )
            assert frame.arrays["forces"].shape == (2, 3)
            if cell is not None:
                assert frame.info["stress"].shape == (3, 3)

    elif filename.endswith(".mts"):
        tensormap = mts.load(filename.split(".")[0] + "_energy.mts")
        assert tensormap.block().values.shape == (2, 1)
        assert tensormap.block().gradient("positions").values.shape == (4, 3, 1)
        if cell is not None:
            assert tensormap.block().gradient("strain").values.shape == (2, 3, 3, 1)

    elif filename.endswith(".npy"):
        energy = np.load("test_output_energy.npy", mmap_mode="r")
        assert energy.shape == (2, 1)
        assert energy.dtype == np.float32
        forces = np.load("test_output_forces.npy", mmap_mode="r")
        assert forces.shape == (4, 3, 1)
        raw_positions_gradient = (
            predictions["energy"].block().gradient("positions").values.numpy()
        )
        np.testing.assert_array_almost_equal(forces, -raw_positions_gradient)
        if cell is not None:
            # strain gradient → "virial"/"stress" via to_external_name for
            # energy quantity
            raw_strain_gradient = (
                predictions["energy"].block().gradient("strain").values.numpy()
            )
            virial = np.load("test_output_virial.npy", mmap_mode="r")
            assert virial.shape == (2, 3, 3, 1)
            np.testing.assert_array_almost_equal(virial, -raw_strain_gradient)

            cell_volume = torch.det(cell).item()
            stress = np.load("test_output_stress.npy", mmap_mode="r")
            assert stress.shape == (2, 3, 3, 1)
            np.testing.assert_array_almost_equal(
                stress, raw_strain_gradient / cell_volume
            )
    else:
        ValueError("This test only does `.xyz` and `.mts`")


def test_write_predictions_unknown_fileformat():
    with pytest.raises(ValueError, match="fileformat '.bar' is not supported"):
        get_writer("foo.bar", capabilities=None)


def test_write_disk_dataset_non_contiguous(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    systems, _, _ = systems_capabilities_predictions(cell=None)

    # create a non-contiguous TensorMap
    vals = torch.rand(100, 2, 3)
    non_contig_vals = vals.permute(1, 2, 0)
    assert not non_contig_vals.is_contiguous()
    block = TensorBlock(
        values=non_contig_vals,
        samples=Labels(["system"], torch.tensor([[0], [1]])),
        components=[
            Labels.range("xyz", 3),
        ],
        properties=Labels(
            ["property"],
            torch.arange(100, dtype=torch.int32).reshape(-1, 1),
        ),
    )
    predictions = {"dos": TensorMap(keys=Labels.single(), blocks=[block])}
    assert not mts.is_contiguous(predictions["dos"])

    # write to DiskDatasetWriter
    new_dataset = DiskDatasetWriter("test_output.zip")
    new_dataset.write(systems, predictions)
    new_dataset.finish()


def test_ase_writer_streams_to_disk(monkeypatch, tmp_path):
    """ASEWriter should write to disk in write(), not accumulate in memory."""
    monkeypatch.chdir(tmp_path)

    systems, capabilities, predictions = systems_capabilities_predictions()
    filename = "test_streaming.xyz"

    writer = ASEWriter(filename, capabilities=capabilities)

    # first batch
    writer.write(systems, predictions)
    # writer should NOT hold accumulated data
    assert not hasattr(writer, "_systems") or len(getattr(writer, "_systems", [])) == 0
    assert not hasattr(writer, "_preds") or len(getattr(writer, "_preds", [])) == 0

    # second batch
    writer.write(systems, predictions)
    assert not hasattr(writer, "_systems") or len(getattr(writer, "_systems", [])) == 0
    assert not hasattr(writer, "_preds") or len(getattr(writer, "_preds", [])) == 0

    writer.finish()

    # verify all 4 structures present and correct
    frames = read(filename, index=":")
    assert len(frames) == 4
    for i, atoms in enumerate(frames):
        assert atoms.info["energy"] == float(
            predictions["energy"].block().values[i % 2, 0]
        )


def test_metatensor_writer_streams_to_disk(monkeypatch, tmp_path):
    """MetatensorWriter should save to temp files in write(), not accumulate."""
    monkeypatch.chdir(tmp_path)

    systems, capabilities, predictions = systems_capabilities_predictions()
    filename = "test_streaming.mts"

    writer = MetatensorWriter(filename, capabilities=capabilities)

    # first batch
    writer.write(systems, predictions)
    assert not hasattr(writer, "_systems") or len(getattr(writer, "_systems", [])) == 0
    assert not hasattr(writer, "_preds") or len(getattr(writer, "_preds", [])) == 0

    # second batch
    writer.write(systems, predictions)
    assert not hasattr(writer, "_systems") or len(getattr(writer, "_systems", [])) == 0
    assert not hasattr(writer, "_preds") or len(getattr(writer, "_preds", [])) == 0

    writer.finish()

    # verify output has all 4 systems with correct system label offsets
    tensormap = mts.load("test_streaming_energy.mts")
    assert tensormap.block().values.shape == (4, 1)
    system_labels = tensormap.block().samples.column("system").tolist()
    assert system_labels == [0, 1, 2, 3]
    assert tensormap.block().gradient("positions").values.shape == (8, 3, 1)


def test_disk_dataset_writer_multi_batch_gradients(monkeypatch, tmp_path):
    """DiskDatasetWriter with gradients must not corrupt gradient sample indices.

    Regression test for the bug where ``_split_tensormaps`` applied the
    ``istart_system`` offset to the gradient "sample" column (a positional
    reference into the parent block) instead of leaving it at 0.  On batches
    after the first (``istart_system > 0``), this produced an out-of-range
    sample index and metatensor raised::

        RuntimeError: invalid value for the 'sample' dimension in gradient
        samples: we got N, but the values contain 1 samples
    """
    monkeypatch.chdir(tmp_path)

    systems, capabilities, predictions = systems_capabilities_predictions()

    writer = DiskDatasetWriter("test_multi_batch.zip", capabilities=capabilities)
    # First batch (istart_system=0) -- always worked
    writer.write(systems, predictions)
    # Second batch (istart_system=2) -- previously crashed on gradients
    writer.write(systems, predictions)
    writer.finish()


def test_disk_dataset_writer_multi_batch_strain_gradients(monkeypatch, tmp_path):
    """Same regression test but with strain gradients (samples=["sample"] only).

    Strain gradients have a single "sample" column, so the eye-matrix offset
    hit them too. Exercises both positions (["sample", "atom"]) and strain
    (["sample"]) gradient layouts in a single run.
    """
    monkeypatch.chdir(tmp_path)

    systems, capabilities, predictions = systems_capabilities_predictions(
        cell=torch.eye(3)
    )

    writer = DiskDatasetWriter("test_strain.zip", capabilities=capabilities)
    writer.write(systems, predictions)
    writer.write(systems, predictions)
    writer.finish()


# ============================================================================
# MemMapWriter tests
# ============================================================================


def test_MemMapWriter(tmp_path):
    """MemMapWriter writes one float32 .npy file per target with correct values."""
    systems, capabilities, predictions = systems_capabilities_predictions()

    out = tmp_path / "preds.npy"
    writer = MemMapWriter(out, capabilities=capabilities)
    writer.write(systems, predictions)
    writer.finish()

    energy = np.load(tmp_path / "preds_energy.npy", mmap_mode="r")
    assert energy.shape == (2, 1)
    assert energy.dtype == np.float32
    expected = predictions["energy"].block().values.numpy().astype(np.float32)
    np.testing.assert_array_almost_equal(energy, expected)


def test_MemMapWriter_gradient_naming(tmp_path):
    """Gradient files use to_external_name: positions -> forces, strain -> virial."""
    cell = torch.eye(3)
    systems, capabilities, predictions = systems_capabilities_predictions(cell=cell)

    out = tmp_path / "preds.npy"
    writer = MemMapWriter(out, capabilities=capabilities)
    writer.write(systems, predictions)
    writer.finish()

    raw_positions_gradient = (
        predictions["energy"].block().gradient("positions").values.numpy()
    )
    raw_strain_gradient = (
        predictions["energy"].block().gradient("strain").values.numpy()
    )
    cell_volume = torch.det(cell).item()

    # positions gradient of energy -> negated "forces"
    forces = np.load(tmp_path / "preds_forces.npy", mmap_mode="r")
    assert forces.shape == (4, 3, 1)
    assert forces.dtype == np.float32
    np.testing.assert_array_almost_equal(forces, -raw_positions_gradient)

    # strain gradient of energy -> negated "virial" and volume-normalized "stress"
    virial = np.load(tmp_path / "preds_virial.npy", mmap_mode="r")
    assert virial.shape == (2, 3, 3, 1)
    assert virial.dtype == np.float32
    np.testing.assert_array_almost_equal(virial, -raw_strain_gradient)

    stress = np.load(tmp_path / "preds_stress.npy", mmap_mode="r")
    assert stress.shape == (2, 3, 3, 1)
    assert stress.dtype == np.float32
    np.testing.assert_array_almost_equal(stress, raw_strain_gradient / cell_volume)

    # internal-named files must NOT exist
    assert not (tmp_path / "preds_energy_positions_gradients.npy").exists()
    assert not (tmp_path / "preds_energy_strain_gradients.npy").exists()


def test_MemMapWriter_virial_nan_without_cell(tmp_path):
    """Systems without a valid cell get NaN virial/stress, matching ASEWriter."""
    systems, capabilities, predictions = systems_capabilities_predictions(
        cell=torch.eye(3)
    )
    # Give the two systems independent cells: `systems_capabilities_predictions`
    # returns the *same* System object twice, so replace one entry outright
    # instead of mutating its cell in place (which would affect both).
    systems = [
        systems[0],
        System(
            types=systems[0].types,
            positions=systems[0].positions,
            cell=torch.zeros(3, 3),
            pbc=torch.tensor([False, False, False]),
        ),
    ]

    out = tmp_path / "preds.npy"
    writer = MemMapWriter(out, capabilities=capabilities)
    writer.write(systems, predictions)
    writer.finish()

    virial = np.load(tmp_path / "preds_virial.npy", mmap_mode="r")
    stress = np.load(tmp_path / "preds_stress.npy", mmap_mode="r")
    assert np.all(np.isfinite(virial[0]))
    assert np.all(np.isnan(virial[1]))
    assert np.all(np.isfinite(stress[0]))
    assert np.all(np.isnan(stress[1]))


def test_MemMapWriter_multi_batch(tmp_path):
    """Multiple write() calls are concatenated correctly by finish()."""
    systems, capabilities, predictions = systems_capabilities_predictions()

    out = tmp_path / "preds.npy"
    writer = MemMapWriter(out, capabilities=capabilities)
    writer.write(systems, predictions)  # batch 0: systems [0, 1]
    writer.write(systems, predictions)  # batch 1: systems [0, 1] (local indexing)
    writer.finish()

    energy = np.load(tmp_path / "preds_energy.npy", mmap_mode="r")
    assert energy.shape == (4, 1)  # 2 batches × 2 systems
    expected = np.tile(
        predictions["energy"].block().values.numpy().astype(np.float32), (2, 1)
    )
    np.testing.assert_array_almost_equal(energy, expected)


def test_MemMapWriter_float32_output(tmp_path):
    """Output is always float32 regardless of the input tensor dtype."""
    systems, capabilities, predictions = systems_capabilities_predictions()
    # predictions come from systems_capabilities_predictions as float32 by default;
    # force float64 to check the downcast path
    block64 = TensorBlock(
        values=predictions["energy"].block().values.to(torch.float64),
        samples=predictions["energy"].block().samples,
        components=predictions["energy"].block().components,
        properties=predictions["energy"].block().properties,
    )
    predictions64 = {"energy": TensorMap(Labels.single(), [block64])}

    out = tmp_path / "preds.npy"
    writer = MemMapWriter(out, capabilities=capabilities)
    writer.write(systems, predictions64)
    writer.finish()

    energy = np.load(tmp_path / "preds_energy.npy", mmap_mode="r")
    assert energy.dtype == np.float32


def test_MemMapWriter_creates_output_directory(tmp_path):
    """MemMapWriter creates the parent directory if it does not exist yet."""
    out = tmp_path / "nested" / "subdir" / "preds.npy"
    systems, capabilities, predictions = systems_capabilities_predictions()

    writer = MemMapWriter(out, capabilities=capabilities)
    writer.write(systems, predictions)
    writer.finish()

    assert (tmp_path / "nested" / "subdir").is_dir()
    assert (tmp_path / "nested" / "subdir" / "preds_energy.npy").exists()


def test_MemMapWriter_multi_block_raises(tmp_path):
    """MemMapWriter raises ValueError for TensorMaps with more than one block."""
    systems, capabilities, _ = systems_capabilities_predictions()

    multi_block = TensorMap(
        keys=Labels(["l"], torch.tensor([[0], [1]])),
        blocks=[
            TensorBlock(
                values=torch.tensor([[1.0]]),
                samples=Labels(["system"], torch.tensor([[0]])),
                components=[],
                properties=Labels(["p"], torch.tensor([[0]])),
            ),
            TensorBlock(
                values=torch.tensor([[2.0]]),
                samples=Labels(["system"], torch.tensor([[1]])),
                components=[],
                properties=Labels(["p"], torch.tensor([[0]])),
            ),
        ],
    )

    out = tmp_path / "preds.npy"
    writer = MemMapWriter(out, capabilities=capabilities)
    with pytest.raises(ValueError, match="single-block"):
        writer.write(systems, {"energy": multi_block})


def test_MemMapWriter_output_is_memory_mappable(tmp_path):
    """Output .npy files can be opened with mmap_mode='r' without loading all data."""
    systems, capabilities, predictions = systems_capabilities_predictions()

    out = tmp_path / "preds.npy"
    writer = MemMapWriter(out, capabilities=capabilities)
    writer.write(systems, predictions)
    writer.finish()

    energy_path = tmp_path / "preds_energy.npy"
    arr = np.load(energy_path, mmap_mode="r")
    assert isinstance(arr, np.memmap)
    assert arr.shape == (2, 1)


def test_get_writer_preserves_parent_directory(tmp_path):
    """get_writer passes the full path (including subdirectory) to the writer."""
    subdir = tmp_path / "final_evaluation"
    writer = get_writer(subdir / "train_predictions.npy")
    assert isinstance(writer, MemMapWriter)
    assert Path(writer.filename).parent == subdir

    writer_xyz = get_writer(subdir / "train_predictions.xyz")
    assert isinstance(writer_xyz, ASEWriter)
    assert Path(writer_xyz.filename).parent == subdir
