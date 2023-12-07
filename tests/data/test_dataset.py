from pathlib import Path

import torch

from metatensor.models.utils.data import (
    Dataset,
    collate_fn,
    read_structures,
    read_targets,
)


RESOURCES_PATH = Path(__file__).parent.resolve() / ".." / "resources"


def test_dataset():
    """Tests the readers and the dataset class."""

    structures = read_structures(RESOURCES_PATH / "qm9_reduced_100.xyz")
    targets = read_targets(RESOURCES_PATH / "qm9_reduced_100.xyz", "U0")

    dataset = Dataset(structures, targets)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=10, collate_fn=collate_fn
    )

    for batch in dataloader:
        assert batch[1]["U0"].block().values.shape == (10, 1)


def test_species_list():
    """Tests that the species list is correctly computed."""

    structures = read_structures(RESOURCES_PATH / "qm9_reduced_100.xyz")
    targets = read_targets(RESOURCES_PATH / "qm9_reduced_100.xyz", "U0")

    dataset = Dataset(structures, targets)

    assert dataset.all_species == [1, 6, 7, 8]
