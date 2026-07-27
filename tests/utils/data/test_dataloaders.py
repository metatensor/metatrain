"""Tests for the shared dataloader-building utilities in
``metatrain.utils.data.dataloaders``, covering ``max_atoms_per_batch`` packing
end-to-end and the pickle boundary of spawned dataloader workers.

Architectures each used to carry their own ``test_train_max_atoms_per_batch``
regression test exercising this through a full model training. Since
``build_train_dataloaders``/``build_val_dataloaders`` are the single shared
implementation every architecture trainer calls into, one direct test here is
enough: it removes the duplication without losing coverage.
"""

import pickle
from pathlib import Path

import torch.utils.data
from metatensor.learn.data import Dataset
from metatomic.torch import NeighborListOptions
from omegaconf import OmegaConf

from metatrain.utils.data import build_train_dataloaders, build_val_dataloaders
from metatrain.utils.data.dataloaders import datasets_are_disk_backed
from metatrain.utils.data.dataset import CollateFn, DiskDataset, unpack_batch
from metatrain.utils.data.readers import read_systems, read_targets
from metatrain.utils.data.samplers import MaxAtomDistributedBatchSampler
from metatrain.utils.neighbor_lists import get_system_with_neighbor_lists_transform


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


def _collate_with_nl_transform() -> CollateFn:
    return CollateFn(
        target_keys=["energy"],
        callables=[
            get_system_with_neighbor_lists_transform(
                [NeighborListOptions(cutoff=5.0, full_list=True, strict=True)]
            ),
        ],
    )


def test_collate_fn_with_transforms_is_picklable():
    """Collate transforms must stay picklable: spawned workers receive the
    collate function by pickle."""
    dataset = _build_dataset()

    reloaded = pickle.loads(pickle.dumps(_collate_with_nl_transform()))

    batch = [dataset[i] for i in range(4)]
    systems, targets, _ = unpack_batch(reloaded(batch))
    assert "energy" in targets
    assert all(len(s.known_neighbor_lists()) == 1 for s in systems)


def test_dataloader_spawn_workers():
    """A full epoch with spawn-started workers exercises everything crossing
    the worker pickle boundary (dataset, collate function, transforms)."""
    dataset = _build_dataset()

    train_dataloaders, _ = build_train_dataloaders(
        train_datasets=[dataset],
        train_distributed_samplers=[None],
        collate_fn_train=_collate_with_nl_transform(),
        batch_size=25,
        max_atoms_per_batch=None,
        min_atoms_per_batch=1,
        num_workers=2,
        multiprocessing_context="spawn",
    )

    n_systems_seen = 0
    for raw_batch in train_dataloaders[0]:
        systems, targets, _ = unpack_batch(raw_batch)
        assert "energy" in targets
        assert all(len(s.known_neighbor_lists()) == 1 for s in systems)
        n_systems_seen += len(systems)
    assert n_systems_seen == len(dataset)


def test_datasets_are_disk_backed():
    """Only disk-backed datasets opt in to spawned workers."""
    in_memory = _build_dataset()
    disk = DiskDataset(RESOURCES_PATH / "spherical_disk_dataset.zip")
    assert datasets_are_disk_backed([disk])
    assert datasets_are_disk_backed([torch.utils.data.Subset(disk, [0])])
    assert not datasets_are_disk_backed([in_memory])
    assert not datasets_are_disk_backed([disk, in_memory])
