from pathlib import Path

import metatensor.torch as metatensor
import torch
from metatensor.learn.data import Dataset
from metatensor.torch import Labels
from omegaconf import OmegaConf

from metatensor.models.utils.data import (
    collate_fn,
    get_all_species,
    read_structures,
    read_targets,
)


RESOURCES_PATH = Path(__file__).parent.resolve() / ".." / ".." / "resources"


def test_dataset():
    """Tests the readers and the dataset class."""

    structures = read_structures(RESOURCES_PATH / "qm9_reduced_100.xyz")

    filename = str(RESOURCES_PATH / "qm9_reduced_100.xyz")
    conf = {
        "energy": {
            "quantity": "energy",
            "read_from": filename,
            "file_format": ".xyz",
            "key": "U0",
            "forces": False,
            "stress": False,
            "virial": False,
        }
    }
    targets = read_targets(OmegaConf.create(conf))  # , slice_samples_by="structure")
    # TODO: use this when targets are sliced in the reader
    # dataset = Dataset(structure=structures, energy=targets["energy"])

    # TODO: change the readers to provide the targets as a list of TensorMaps
    # for each sample, not a single TensorMap. This then aligns with the
    # paradigm set by `metatensor-learn`. In the meantime, slice the targets to
    # per-structure TensorMaps.
    targets_sliced = {"energy": []}
    for structure_idx in range(len(structures)):
        targets_sliced["energy"].append(
            metatensor.slice(
                targets["energy"],
                axis="samples",
                labels=Labels(
                    names=["structure"],
                    values=torch.tensor([structure_idx]).reshape(-1, 1),
                ),
            )
        )
    dataset = Dataset(structure=structures, energy=targets_sliced["energy"])

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=10, collate_fn=collate_fn
    )

    for batch in dataloader:
        assert batch[1]["energy"].block().values.shape == (10, 1)


def test_species_list():
    """Tests that the species list is correctly computed with get_all_species."""

    structures = read_structures(RESOURCES_PATH / "qm9_reduced_100.xyz")
    conf = {
        "energy": {
            "quantity": "energy",
            "read_from": str(RESOURCES_PATH / "qm9_reduced_100.xyz"),
            "file_format": ".xyz",
            "key": "U0",
            "forces": False,
            "stress": False,
            "virial": False,
        }
    }
    targets = read_targets(OmegaConf.create(conf))  # , slice_samples_by="structure")
    # TODO: use this when targets are sliced in the reader
    # dataset = Dataset(structure=structures, energy=targets["energy"])

    # TODO: change the readers to provide the targets as a list of TensorMaps
    # for each sample, not a single TensorMap. This then aligns with the
    # paradigm set by `metatensor-learn`. In the meantime, slice the targets to
    # per-structure TensorMaps.
    targets_sliced = {"energy": []}
    for structure_idx in range(len(structures)):
        targets_sliced["energy"].append(
            metatensor.slice(
                targets["energy"],
                axis="samples",
                labels=Labels(
                    names=["structure"],
                    values=torch.tensor([structure_idx]).reshape(-1, 1),
                ),
            )
        )
    dataset = Dataset(structure=structures, energy=targets_sliced["energy"])

    assert get_all_species(dataset) == [1, 6, 7, 8]
