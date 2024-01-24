from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from metatensor.models.utils.data import (
    Dataset,
    collate_fn,
    combine_dataloaders,
    read_structures,
    read_targets,
)


RESOURCES_PATH = Path(__file__).parent.resolve() / ".." / ".." / "resources"

np.random.seed(0)


def test_without_shuffling():
    """Tests combining dataloaders without shuffling."""

    structures = read_structures(RESOURCES_PATH / "qm9_reduced_100.xyz")

    conf = {
        "U0": {
            "quantity": "energy",
            "read_from": RESOURCES_PATH / "qm9_reduced_100.xyz",
            "file_format": ".xyz",
            "key": "U0",
            "forces": False,
            "stress": False,
            "virial": False,
        }
    }
    targets = read_targets(OmegaConf.create(conf))

    dataset = Dataset(structures, targets)
    dataloader_qm9 = torch.utils.data.DataLoader(
        dataset, batch_size=10, collate_fn=collate_fn
    )
    # will yield 10 batches of 10

    structures = read_structures(RESOURCES_PATH / "alchemical_reduced_10.xyz")

    conf = {
        "free_energy": {
            "quantity": "energy",
            "read_from": RESOURCES_PATH / "alchemical_reduced_10.xyz",
            "file_format": ".xyz",
            "key": "free_energy",
            "forces": False,
            "stress": False,
            "virial": False,
        }
    }
    targets = read_targets(OmegaConf.create(conf))

    dataset = Dataset(structures, targets)
    dataloader_alchemical = torch.utils.data.DataLoader(
        dataset, batch_size=2, collate_fn=collate_fn
    )
    # will yield 5 batches of 2

    combined_dataloader = combine_dataloaders(
        [dataloader_qm9, dataloader_alchemical], shuffle=False
    )

    assert len(combined_dataloader) == 15
    for i_batch, batch in enumerate(combined_dataloader):
        if i_batch < 10:
            assert batch[1]["U0"].block().values.shape == (10, 1)
        else:
            assert batch[1]["free_energy"].block().values.shape == (2, 1)


def test_with_shuffling():
    """Tests combining dataloaders with shuffling."""
    # WARNING: this test might fail if the random seed is changed,
    # with a probability of 1/(15 5) = 1/3003

    structures = read_structures(RESOURCES_PATH / "qm9_reduced_100.xyz")

    conf = {
        "U0": {
            "quantity": "energy",
            "read_from": RESOURCES_PATH / "qm9_reduced_100.xyz",
            "file_format": ".xyz",
            "key": "U0",
            "forces": False,
            "stress": False,
            "virial": False,
        }
    }
    targets = read_targets(OmegaConf.create(conf))

    dataset = Dataset(structures, targets)
    dataloader_qm9 = torch.utils.data.DataLoader(
        dataset, batch_size=10, collate_fn=collate_fn
    )
    # will yield 10 batches of 10

    structures = read_structures(RESOURCES_PATH / "alchemical_reduced_10.xyz")

    conf = {
        "free_energy": {
            "quantity": "energy",
            "read_from": RESOURCES_PATH / "alchemical_reduced_10.xyz",
            "file_format": ".xyz",
            "key": "free_energy",
            "forces": False,
            "stress": False,
            "virial": False,
        }
    }
    targets = read_targets(OmegaConf.create(conf))

    dataset = Dataset(structures, targets)
    dataloader_alchemical = torch.utils.data.DataLoader(
        dataset, batch_size=2, collate_fn=collate_fn
    )
    # will yield 5 batches of 2

    combined_dataloader = combine_dataloaders(
        [dataloader_qm9, dataloader_alchemical], shuffle=True
    )

    assert len(combined_dataloader) == 15

    qm9_batch_count = 0
    alchemical_batch_count = 0
    original_ordering = ["qm9"] * 10 + ["alchemical"] * 5
    actual_ordering = []

    for batch in combined_dataloader:
        if "U0" in batch[1]:
            qm9_batch_count += 1
            assert batch[1]["U0"].block().values.shape == (10, 1)
            actual_ordering.append("qm9")
        else:
            alchemical_batch_count += 1
            assert batch[1]["free_energy"].block().values.shape == (2, 1)
            actual_ordering.append("alchemical")

    assert qm9_batch_count == 10
    assert alchemical_batch_count == 5
    assert actual_ordering != original_ordering
