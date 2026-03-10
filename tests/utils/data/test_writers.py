from typing import Dict, List, Tuple

import metatensor.torch as mts
import pytest
import torch
from metatensor.torch import Labels, TensorBlock, TensorMap
from metatomic.torch import ModelCapabilities, ModelOutput, System

from metatrain.utils.data.readers.ase import read
from metatrain.utils.data.writers import (
    ASEWriter,
    DiskDatasetWriter,
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
        outputs={"energy": ModelOutput(quantity="dos", unit="", per_atom=True)},
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
    "filename", ("test_output.xyz", "test_output.mts", "test_output.zip")
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
