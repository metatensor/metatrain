import torch

from metatensor_models.utils.data import Dataset, collate_fn, read_structures, read_targets


def test_dataset():
    """Tests the readers and the dataset class."""

    structures = read_structures("data/qm9_reduced_100.xyz")
    targets = read_targets("data/qm9_reduced_100.xyz", "U0")

    dataset = Dataset(structures, targets)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=10, collate_fn=collate_fn)

    for batch in dataloader:
        assert batch[1]["U0"].block().values.shape == (10, 1)
