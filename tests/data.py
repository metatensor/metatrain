import os
import torch

from metatensor.models.utils.data import Dataset, collate_fn, read_structures, read_targets, get_all_species


def test_dataset():
    """Tests the readers and the dataset class."""

    dataset_path = os.path.join(os.path.dirname(__file__), "data/qm9_reduced_100.xyz")

    structures = read_structures(dataset_path)
    targets = read_targets(dataset_path, "U0")

    dataset = Dataset(structures, targets)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=10, collate_fn=collate_fn)

    for batch in dataloader:
        assert batch[1]["U0"].block().values.shape == (10, 1)


def test_species_list():
    """Tests that the species list is correctly computed."""

    dataset_path = os.path.join(os.path.dirname(__file__), "data/qm9_reduced_100.xyz")

    structures = read_structures(dataset_path)
    targets = read_targets(dataset_path, "U0")

    dataset = Dataset(structures, targets)
    species_list = get_all_species(dataset)

    assert species_list == [1, 6, 7, 8]
