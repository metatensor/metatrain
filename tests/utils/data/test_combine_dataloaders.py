from pathlib import Path

import numpy as np
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from metatensor.models.utils.data import (
    CombinedDataLoader,
    Dataset,
    collate_fn,
    read_systems,
    read_targets,
)


RESOURCES_PATH = Path(__file__).parents[2] / "resources"

np.random.seed(0)


def test_without_shuffling():
    """Tests combining dataloaders without shuffling."""

    systems = read_systems(RESOURCES_PATH / "qm9_reduced_100.xyz")

    conf = {
        "mtm::U0": {
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
    dataset = Dataset({"system": systems, "mtm::U0": targets["mtm::U0"]})
    dataloader_qm9 = DataLoader(dataset, batch_size=10, collate_fn=collate_fn)
    # will yield 10 batches of 10

    systems = read_systems(RESOURCES_PATH / "alchemical_reduced_10.xyz")

    conf = {
        "mtm::free_energy": {
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
    dataset = Dataset(
        {"system": systems, "mtm::free_energy": targets["mtm::free_energy"]}
    )
    dataloader_alchemical = DataLoader(dataset, batch_size=2, collate_fn=collate_fn)
    # will yield 5 batches of 2

    combined_dataloader = CombinedDataLoader(
        [dataloader_qm9, dataloader_alchemical], shuffle=False
    )

    assert len(combined_dataloader) == 15
    for i_batch, batch in enumerate(combined_dataloader):
        if i_batch < 10:
            assert batch[1]["mtm::U0"].block().values.shape == (10, 1)
        else:
            assert batch[1]["mtm::free_energy"].block().values.shape == (2, 1)


def test_with_shuffling():
    """Tests combining dataloaders with shuffling."""
    # WARNING: this test might fail if the random seed is changed,
    # with a probability of 1/(15 5) = 1/3003

    systems = read_systems(RESOURCES_PATH / "qm9_reduced_100.xyz")

    conf = {
        "mtm::U0": {
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
    dataset = Dataset({"system": systems, "mtm::U0": targets["mtm::U0"]})
    dataloader_qm9 = DataLoader(
        dataset, batch_size=10, collate_fn=collate_fn, shuffle=True
    )
    # will yield 10 batches of 10

    systems = read_systems(RESOURCES_PATH / "alchemical_reduced_10.xyz")

    conf = {
        "mtm::free_energy": {
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
    dataset = Dataset(
        {"system": systems, "mtm::free_energy": targets["mtm::free_energy"]}
    )
    dataloader_alchemical = DataLoader(
        dataset, batch_size=2, collate_fn=collate_fn, shuffle=True
    )
    # will yield 5 batches of 2

    combined_dataloader = CombinedDataLoader(
        [dataloader_qm9, dataloader_alchemical], shuffle=True
    )

    assert len(combined_dataloader) == 15

    qm9_batch_count = 0
    alchemical_batch_count = 0
    original_ordering = ["qm9"] * 10 + ["alchemical"] * 5
    actual_ordering = []
    qm9_samples = []
    alchemical_samples = []

    for batch in combined_dataloader:
        if "mtm::U0" in batch[1]:
            qm9_batch_count += 1
            assert batch[1]["mtm::U0"].block().values.shape == (10, 1)
            actual_ordering.append("qm9")
            qm9_samples.append(batch[1]["mtm::U0"].block().samples.column("system"))
        else:
            alchemical_batch_count += 1
            assert batch[1]["mtm::free_energy"].block().values.shape == (2, 1)
            actual_ordering.append("alchemical")
            alchemical_samples.append(
                batch[1]["mtm::free_energy"].block().samples.column("system")
            )

    assert qm9_batch_count == 10
    assert alchemical_batch_count == 5
    assert actual_ordering != original_ordering

    qm9_samples = [int(item) for sublist in qm9_samples for item in sublist]
    alchemical_samples = [
        int(item) for sublist in alchemical_samples for item in sublist
    ]
    assert set(qm9_samples) == set(range(100))
    assert set(alchemical_samples) == set(range(10))
