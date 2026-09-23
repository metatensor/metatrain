"""Tests for the shared dataloader-building utilities in
``metatrain.utils.data.dataloaders``, covering ``max_atoms_per_batch`` packing
end-to-end.

Architectures each used to carry their own ``test_train_max_atoms_per_batch``
regression test exercising this through a full model training. Since
``build_train_dataloaders``/``build_val_dataloaders`` are the single shared
implementation every architecture trainer calls into, one direct test here is
enough: it removes the duplication without losing coverage.
"""

from pathlib import Path

import pytest
import torch
from metatensor.learn.data import Dataset
from omegaconf import OmegaConf
from torch.utils.data import DistributedSampler

from metatrain.utils.data import build_train_dataloaders, build_val_dataloaders
from metatrain.utils.data.dataset import (
    CollateFn,
    fork_is_available,
    unpack_batch,
)
from metatrain.utils.data.readers import read_systems, read_targets
from metatrain.utils.data.samplers import MaxAtomDistributedBatchSampler


RESOURCES_PATH = Path(__file__).resolve().parents[2] / "resources"
DATASET_PATH = str(RESOURCES_PATH / "qm9_reduced_100.xyz")


def _build_dataset() -> Dataset:
    systems = read_systems(DATASET_PATH)
    conf = {
        "energy": {
            "quantity": "energy",
            "read_from": DATASET_PATH,
            "reader": "ase",
            "key": "U0",
            "unit": "eV",
            "type": "scalar",
            "sample_kind": "system",
            "num_subtargets": 1,
            "forces": False,
            "stress": False,
            "virial": False,
        }
    }
    targets, _ = read_targets(OmegaConf.create(conf))
    return Dataset.from_dict({"system": systems, "energy": targets["energy"]})


def test_max_atoms_per_batch_end_to_end():
    """``max_atoms_per_batch``/``min_atoms_per_batch`` pack batches by atom
    count instead of a fixed ``batch_size``, and the resulting ``DataLoader``
    iterates into correctly collated batches that respect those bounds."""
    dataset = _build_dataset()
    collate_fn = CollateFn(target_keys=["energy"])

    train_dataloaders, epoch_samplers = build_train_dataloaders(
        train_datasets=[dataset],
        train_distributed_samplers=[None],
        collate_fn_train=collate_fn,
        batch_size=5,
        max_atoms_per_batch=20,
        min_atoms_per_batch=1,
        num_workers=0,
    )
    assert len(train_dataloaders) == 1
    assert len(epoch_samplers) == 1

    batch_sampler = train_dataloaders[0].batch_sampler
    assert isinstance(batch_sampler, MaxAtomDistributedBatchSampler)
    assert batch_sampler is epoch_samplers[0]

    # Every packed batch respects the configured bounds, and small structures
    # get combined into the same batch (as opposed to silently falling back
    # to one structure per batch).
    packed_batches = batch_sampler.all_batches
    assert len(packed_batches) > 0
    for batch in packed_batches:
        atom_count = sum(len(dataset[i].system) for i in batch)
        assert 1 <= atom_count <= 20
    assert any(len(batch) > 1 for batch in packed_batches)

    # The DataLoader must actually be iterable end-to-end with real collation:
    # every emitted batch decodes back into systems within the atom bounds.
    n_batches = 0
    n_systems_seen = 0
    for raw_batch in train_dataloaders[0]:
        systems, targets, _ = unpack_batch(raw_batch)
        assert sum(len(s) for s in systems) <= 20
        assert "energy" in targets
        n_batches += 1
        n_systems_seen += len(systems)
    assert n_batches == len(packed_batches)
    assert n_systems_seen == len(dataset)

    # build_val_dataloaders mirrors the same packing logic.
    val_dataloaders = build_val_dataloaders(
        val_datasets=[dataset],
        val_distributed_samplers=[None],
        collate_fn_val=collate_fn,
        batch_size=5,
        max_atoms_per_batch=20,
        num_workers=0,
    )
    assert len(val_dataloaders) == 1
    val_batch_sampler = val_dataloaders[0].batch_sampler
    assert isinstance(val_batch_sampler, MaxAtomDistributedBatchSampler)
    for batch in val_batch_sampler.all_batches:
        atom_count = sum(len(dataset[i].system) for i in batch)
        assert atom_count <= 20


def test_iterating_a_loader_does_not_shift_the_global_rng():
    """How often a loader is iterated must not change what is drawn next.

    A ``DataLoader`` on the default generator draws a worker seed from the
    global RNG every time an iterator is created, which leaked the number of
    passes (and, through ``get_num_workers``, the host's core count) into the
    random numbers the training loop drew afterwards.
    """
    dataset = _build_dataset()
    dataloaders = build_val_dataloaders(
        val_datasets=[dataset],
        val_distributed_samplers=[None],
        collate_fn_val=CollateFn(target_keys=["energy"]),
        batch_size=5,
        max_atoms_per_batch=None,
        num_workers=0,
    )

    def draw_after(passes: int) -> float:
        torch.manual_seed(0)
        for _ in range(passes):
            for _ in dataloaders[0]:
                pass
        return torch.rand(1).item()

    assert draw_after(1) == draw_after(2)


@pytest.mark.skipif(
    not fork_is_available(), reason="parallel data loading requires fork"
)
def test_persistent_workers():
    """Workers stay alive between epochs, and ``set_epoch`` still reshuffles:
    every epoch sees every sample exactly once, in a different order."""
    dataset = _build_dataset()
    dataloaders, samplers = build_train_dataloaders(
        train_datasets=[dataset],
        train_distributed_samplers=[
            DistributedSampler(dataset, num_replicas=1, rank=0, shuffle=True)
        ],
        collate_fn_train=CollateFn(target_keys=["energy"]),
        batch_size=10,
        max_atoms_per_batch=None,
        min_atoms_per_batch=0,
        num_workers=1,
    )
    dataloader = dataloaders[0]
    assert dataloader.persistent_workers

    # the energies identify the structures of this qm9 subset uniquely
    orders = []
    for epoch in range(2):
        samplers[0].set_epoch(epoch)
        orders.append(
            [
                value
                for batch in dataloader
                for value in unpack_batch(batch)[1]["energy"]
                .block()
                .values.flatten()
                .tolist()
            ]
        )

    assert len(orders[0]) == len(dataset)
    assert sorted(orders[0]) == sorted(orders[1])
    assert orders[0] != orders[1]
