import logging
import multiprocessing
from typing import Any, List, Optional, Tuple, Union

import torch.multiprocessing
import torch.utils.data
from torch.utils.data import DataLoader, DistributedSampler

from .dataset import (
    CollateFn,
    Dataset,
    DiskDataset,
    MemmapDataset,
    get_num_workers,
)
from .samplers import MaxAtomDistributedBatchSampler


DatasetLike = Union[Dataset, torch.utils.data.Subset]


def resolve_dataloader_workers(
    requested_num_workers: Optional[int],
    device: torch.device,
    datasets: List[DatasetLike],
) -> Tuple[int, Optional[str]]:
    """Resolve the dataloader worker count and start method for a trainer.

    Workers are started with the ``spawn`` method when training on CUDA
    (forking after CUDA initialization is unsupported by PyTorch), but only
    when every dataset is disk-backed: spawned workers receive their dataset
    by pickling, which would copy an in-memory dataset into every worker.

    :param requested_num_workers: The ``num_workers`` hyperparameter, or
        ``None`` to choose automatically.
    :param device: The device the model is trained on.
    :param datasets: All datasets the dataloaders will read from.
    :return: The number of workers and the multiprocessing context to pass to
        the dataloader builders (``"spawn"`` or ``None``).
    """
    if requested_num_workers is None:
        num_workers = get_num_workers()
        logging.info(
            "Number of workers for data-loading not provided and chosen "
            f"automatically. Using {num_workers} workers."
        )
    else:
        num_workers = requested_num_workers

    multiprocessing_context = (
        "spawn"
        if (
            num_workers > 0
            and device.type == "cuda"
            and datasets_are_disk_backed(datasets)
        )
        else None
    )
    return num_workers, multiprocessing_context


def datasets_are_disk_backed(datasets: List[DatasetLike]) -> bool:
    """Whether every dataset reads its samples from disk (and therefore
    pickles small, making it cheap to send to spawned workers).

    :param datasets: The datasets the dataloaders will read from.
    :return: True if every dataset is disk- or memmap-backed.
    """
    for dataset in datasets:
        while isinstance(dataset, torch.utils.data.Subset):
            dataset = dataset.dataset
        if not isinstance(dataset, (DiskDataset, MemmapDataset)):
            return False
    return True


def ensure_spawn_safe_sharing(multiprocessing_context: Optional[str]) -> None:
    """Make PyTorch's tensor sharing compatible with spawned workers.

    The default ``"file_descriptor"`` sharing strategy passes tensors as open
    file descriptors, which spawned worker processes cannot inherit
    (``ValueError: bad value(s) in fds_to_keep``), and some container setups
    additionally restrict ``/dev/shm``, on which it relies. The
    ``"file_system"`` strategy works in both cases.

    :param multiprocessing_context: The worker start method about to be used,
        or ``None`` for the platform default (which is also not fork on e.g.
        macOS, Windows, and in children of ``torch.multiprocessing.spawn``).
    """
    effective = multiprocessing_context or multiprocessing.get_start_method(
        allow_none=False
    )
    if effective != "fork":
        torch.multiprocessing.set_sharing_strategy("file_system")


def build_train_dataloaders(
    train_datasets: List[DatasetLike],
    train_distributed_samplers: List[Optional[DistributedSampler]],
    collate_fn_train: CollateFn,
    batch_size: int,
    max_atoms_per_batch: Optional[int],
    min_atoms_per_batch: int,
    num_workers: int,
    multiprocessing_context: Optional[str] = None,
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
    :param multiprocessing_context: Worker start method (e.g. ``"spawn"``), or
        ``None`` for the PyTorch default.
    :return: A tuple ``(dataloaders, epoch_samplers)``. ``epoch_samplers`` contains
        every sampler (``DistributedSampler`` or ``MaxAtomDistributedBatchSampler``)
        that must have ``set_epoch()`` called on it before each epoch.
    """
    if num_workers > 0:
        ensure_spawn_safe_sharing(multiprocessing_context)

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
                    multiprocessing_context=multiprocessing_context,
                    persistent_workers=(num_workers > 0),
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
                    multiprocessing_context=multiprocessing_context,
                    persistent_workers=(num_workers > 0),
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
    multiprocessing_context: Optional[str] = None,
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
    :param multiprocessing_context: Worker start method (e.g. ``"spawn"``), or
        ``None`` for the PyTorch default.
    :return: One ``DataLoader`` per dataset in ``val_datasets``.
    """
    if num_workers > 0:
        ensure_spawn_safe_sharing(multiprocessing_context)

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
                    multiprocessing_context=multiprocessing_context,
                    persistent_workers=(num_workers > 0),
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
                    multiprocessing_context=multiprocessing_context,
                    persistent_workers=(num_workers > 0),
                )
            )
    return dataloaders
