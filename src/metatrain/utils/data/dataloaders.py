from typing import Any, List, Optional, Tuple, Union

import torch.utils.data
from torch.utils.data import DataLoader, DistributedSampler

from .dataset import CollateFn, Dataset
from .samplers import MaxAtomDistributedBatchSampler


DatasetLike = Union[Dataset, torch.utils.data.Subset]


def build_train_dataloaders(
    train_datasets: List[DatasetLike],
    train_distributed_samplers: List[Optional[DistributedSampler]],
    collate_fn_train: CollateFn,
    batch_size: int,
    max_atoms_per_batch: Optional[int],
    min_atoms_per_batch: int,
    num_workers: int,
) -> Tuple[List[DataLoader], List[Any]]:
    """Build one ``DataLoader`` per training dataset.

    If ``max_atoms_per_batch`` is set, each dataset is packed with a
    :class:`MaxAtomDistributedBatchSampler` (shuffled, ``drop_last=True``).
    Otherwise, a fixed ``batch_size`` is used, sharded via
    ``train_distributed_samplers`` when distributed training is active.

    :param train_datasets: Training datasets, one ``DataLoader`` is built per dataset.
    :param train_distributed_samplers: Per-dataset ``DistributedSampler`` (or ``None``
        for non-distributed training). Its ``num_replicas``/``rank`` are reused to
        shard :class:`MaxAtomDistributedBatchSampler` when ``max_atoms_per_batch``
        is set.
    :param collate_fn_train: Collate function for training batches.
    :param batch_size: Fixed batch size, used only when ``max_atoms_per_batch``
        is ``None``.
    :param max_atoms_per_batch: If set, pack batches by atom count instead of by a
        fixed number of structures.
    :param min_atoms_per_batch: Minimum total atom count for a packed batch to be
        kept. Only used when ``max_atoms_per_batch`` is set.
    :param num_workers: Number of ``DataLoader`` workers.
    :return: A tuple ``(dataloaders, epoch_samplers)``. ``epoch_samplers`` contains
        every sampler (``DistributedSampler`` or ``MaxAtomDistributedBatchSampler``)
        that must have ``set_epoch()`` called on it before each epoch.
    """
    dataloaders: List[DataLoader] = []
    epoch_samplers: List[Any] = []
    for train_dataset, train_sampler in zip(
        train_datasets, train_distributed_samplers, strict=True
    ):
        if max_atoms_per_batch is not None:
            batch_sampler = MaxAtomDistributedBatchSampler(
                dataset=train_dataset,
                max_atoms=max_atoms_per_batch,
                min_atoms=min_atoms_per_batch,
                num_replicas=(
                    train_sampler.num_replicas if train_sampler is not None else 1
                ),
                rank=train_sampler.rank if train_sampler is not None else 0,
                shuffle=True,
                drop_last=True,
            )
            epoch_samplers.append(batch_sampler)
            dataloaders.append(
                DataLoader(
                    dataset=train_dataset,
                    batch_sampler=batch_sampler,
                    collate_fn=collate_fn_train,
                    num_workers=num_workers,
                )
            )
        else:
            if len(train_dataset) < batch_size:
                raise ValueError(
                    f"A training dataset has fewer samples "
                    f"({len(train_dataset)}) than the batch size "
                    f"({batch_size}). "
                    "Please reduce the batch size."
                )
            if train_sampler is not None:
                epoch_samplers.append(train_sampler)
            dataloaders.append(
                DataLoader(
                    dataset=train_dataset,
                    batch_size=batch_size,
                    sampler=train_sampler,
                    shuffle=(train_sampler is None),
                    drop_last=(train_sampler is None),
                    collate_fn=collate_fn_train,
                    num_workers=num_workers,
                )
            )
    return dataloaders, epoch_samplers


def build_val_dataloaders(
    val_datasets: List[DatasetLike],
    val_distributed_samplers: List[Optional[DistributedSampler]],
    collate_fn_val: CollateFn,
    batch_size: int,
    max_atoms_per_batch: Optional[int],
    num_workers: int,
) -> List[DataLoader]:
    """Build one ``DataLoader`` per validation dataset.

    Mirrors :func:`build_train_dataloaders`, but without shuffling, ``drop_last``,
    or a ``min_atoms_per_batch`` bound (validation should cover every sample). A
    validation dataset smaller than ``batch_size`` is not an error: ``DataLoader``
    simply yields one smaller batch, so unlike training there is no size
    constraint to enforce here.

    :param val_datasets: Validation datasets, one ``DataLoader`` is built per dataset.
    :param val_distributed_samplers: Per-dataset ``DistributedSampler`` (or ``None``
        for non-distributed training). Its ``num_replicas``/``rank`` are reused to
        shard :class:`MaxAtomDistributedBatchSampler` when ``max_atoms_per_batch``
        is set.
    :param collate_fn_val: Collate function for validation batches.
    :param batch_size: Fixed batch size, used only when ``max_atoms_per_batch``
        is ``None``.
    :param max_atoms_per_batch: If set, pack batches by atom count instead of by a
        fixed number of structures.
    :param num_workers: Number of ``DataLoader`` workers.
    :return: One ``DataLoader`` per dataset in ``val_datasets``.
    """
    dataloaders: List[DataLoader] = []
    for val_dataset, val_sampler in zip(
        val_datasets, val_distributed_samplers, strict=True
    ):
        if max_atoms_per_batch is not None:
            batch_sampler = MaxAtomDistributedBatchSampler(
                dataset=val_dataset,
                max_atoms=max_atoms_per_batch,
                num_replicas=(
                    val_sampler.num_replicas if val_sampler is not None else 1
                ),
                rank=val_sampler.rank if val_sampler is not None else 0,
                shuffle=False,
            )
            dataloaders.append(
                DataLoader(
                    dataset=val_dataset,
                    batch_sampler=batch_sampler,
                    collate_fn=collate_fn_val,
                    num_workers=num_workers,
                )
            )
        else:
            dataloaders.append(
                DataLoader(
                    dataset=val_dataset,
                    batch_size=batch_size,
                    sampler=val_sampler,
                    shuffle=False,
                    drop_last=False,
                    collate_fn=collate_fn_val,
                    num_workers=num_workers,
                )
            )
    return dataloaders
