"""Batch samplers for atom-count-aware batching.

The design of :class:`MaxAtomDistributedBatchSampler` is based on the
``MaxAtomDistributedBatchSampler`` from the fairchem library:

    https://github.com/facebookresearch/fairchem

Copyright (c) Meta Platforms, Inc. and affiliates.
Licensed under the MIT License.
"""

import logging
import math
from typing import Iterator, List, Sequence, Union

import numpy as np
import torch
import torch.utils.data

# Each batch is an int64 numpy array. Storing ``np.ndarray`` instead of Python lists
# of ints cuts the memory of ``MaxAtomDistributedBatchSampler.all_batches`` by
# roughly 4x for large datasets (8 bytes per int64 vs ~28+ bytes for a boxed
# Python int plus a list slot).
Batch = np.ndarray


logger = logging.getLogger(__name__)


def _get_num_atoms(dataset: torch.utils.data.Dataset, i: int) -> int:
    """Return atom count for sample ``i``, resolving ``Subset`` wrappers.

    :param dataset: The dataset to query.
    :param i: The sample index.
    :return: The number of atoms in sample ``i``.
    """
    if isinstance(dataset, torch.utils.data.Subset):
        return _get_num_atoms(dataset.dataset, dataset.indices[i])
    if hasattr(dataset, "get_num_atoms"):
        return dataset.get_num_atoms(i)
    raise TypeError(
        f"Dataset of type {type(dataset).__name__} does not support "
        "get_num_atoms(). Only MemmapDataset (and Subsets thereof) is "
        "currently supported with max_atoms_per_batch."
    )


def _greedy_pack(
    indices: Union[Sequence[int], np.ndarray],
    atom_counts: Union[Sequence[int], np.ndarray],
    max_atoms: int,
    min_atoms: int = 0,
) -> List[Batch]:
    """Greedily pack ``indices`` into batches where total atoms <= ``max_atoms``.

    Single structures that alone exceed ``max_atoms`` are skipped with a warning.
    Completed batches whose total atom count falls below ``min_atoms`` are dropped.

    Implementation is vectorised: oversized structures are removed with a numpy
    mask, and batch break points are found via ``np.cumsum`` + ``np.searchsorted``,
    so the inner loop iterates once per *batch* (not once per structure).

    :param indices: Dataset indices to pack.
    :param atom_counts: Atom count for each index (parallel to ``indices``).
    :param max_atoms: Maximum total atoms allowed per batch.
    :param min_atoms: Minimum total atoms required to keep a batch.
    :return: List of batches, each batch being an int64 numpy array of dataset
        indices.
    """
    indices_arr = np.asarray(indices, dtype=np.int64)
    counts_arr = np.asarray(atom_counts, dtype=np.int64)
    if indices_arr.shape != counts_arr.shape:
        raise ValueError(
            f"indices and atom_counts must have the same shape, got "
            f"{indices_arr.shape} vs {counts_arr.shape}"
        )

    n_total = indices_arr.size
    if n_total == 0:
        return []

    oversized_mask = counts_arr > max_atoms
    n_oversized = int(oversized_mask.sum())
    if n_oversized:
        # Per-structure warning (preserves prior behaviour for callers grepping
        # logs). The cost is bounded by the oversized count, not n_total.
        over_indices = indices_arr[oversized_mask]
        over_counts = counts_arr[oversized_mask]
        for over_idx, over_n in zip(over_indices.tolist(), over_counts.tolist()):
            logger.warning(
                f"Structure {over_idx} has {over_n} atoms which exceeds "
                f"max_atoms_per_batch ({max_atoms}). Skipping this structure."
            )
        keep = ~oversized_mask
        indices_arr = indices_arr[keep]
        counts_arr = counts_arr[keep]

    n = indices_arr.size
    if n == 0:
        return []

    cumsum = np.cumsum(counts_arr)

    # Walk batch boundaries. Each iteration is a single ``searchsorted`` call,
    # so the loop runs once per batch — typically orders of magnitude fewer
    # iterations than the structure count. We slice (and copy) ``indices_arr``
    # so each batch is a small standalone int64 array; this lets the (filtered)
    # parent array be freed once packing is done.
    batches: List[Batch] = []
    n_dropped_batches = 0
    n_dropped_structures = 0
    start = 0
    while start < n:
        offset = int(cumsum[start - 1]) if start > 0 else 0
        end = int(np.searchsorted(cumsum, offset + max_atoms, side="right"))
        # Every remaining element has count <= max_atoms after filtering, so
        # at least one element fits in the current batch.
        if end <= start:
            end = start + 1
        batch_atoms = int(cumsum[end - 1]) - offset
        if batch_atoms >= min_atoms:
            batches.append(indices_arr[start:end].copy())
        else:
            n_dropped_batches += 1
            n_dropped_structures += end - start
        start = end

    n_skipped = n_oversized + n_dropped_structures
    if n_skipped > 0:
        logger.info(
            f"Greedy packing: {n_skipped}/{n_total} structures will be skipped "
            f"per epoch ({n_oversized} oversized, {n_dropped_structures} in "
            f"{n_dropped_batches} batches below min_atoms={min_atoms})."
        )

    return batches


class MaxAtomDistributedBatchSampler(torch.utils.data.Sampler):
    """Distributed batch sampler that packs structures greedily up to ``max_atoms``.

    Structure-to-batch packing is performed **once at construction** using a
    hardcoded seed (stable across runs, ranks, and restarts). Each epoch, the
    *order* in which batches are presented to each rank is reshuffled
    deterministically from the epoch number, mirroring the fairchem
    ``MaxAtomDistributedBatchSampler`` design.

    :param dataset: The dataset to sample from. Must support ``get_num_atoms(i)``
        (currently only ``MemmapDataset`` and ``Subset`` wrappers thereof).
    :param max_atoms: Maximum total number of atoms across all structures in a batch.
    :param num_replicas: Number of distributed processes (world size).
    :param rank: Rank of the current process.
    :param shuffle: Whether to shuffle batch presentation order each epoch.
    :param drop_last: If ``True``, drop tail batches so the count is evenly divisible
        by ``num_replicas`` (no padding/repetition). If ``False``, repeat batches from
        the front to pad.
    :param min_atoms: Minimum total number of atoms required for a batch to be kept.
        Batches whose total atom count falls below this threshold are discarded during
        packing. Defaults to ``0`` (no minimum).
    """

    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        max_atoms: int,
        num_replicas: int = 1,
        rank: int = 0,
        shuffle: bool = True,
        drop_last: bool = False,
        min_atoms: int = 0,
    ) -> None:
        self.dataset = dataset
        self.max_atoms = max_atoms
        self.min_atoms = min_atoms
        self.num_replicas = num_replicas
        self.rank = rank
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.epoch = 0

        n = len(dataset)
        # Fast path: avoid a Python loop over millions of structures.
        # MemmapDataset exposes get_all_atom_counts() which returns np.diff(na)
        # in one vectorised operation.
        inner = (
            dataset.dataset if isinstance(dataset, torch.utils.data.Subset) else dataset
        )
        if hasattr(inner, "get_all_atom_counts"):
            all_counts = np.asarray(inner.get_all_atom_counts(), dtype=np.int64)
            if isinstance(dataset, torch.utils.data.Subset):
                subset_idx = np.asarray(dataset.indices, dtype=np.int64)
                self._atom_counts = all_counts[subset_idx]
            else:
                self._atom_counts = all_counts
        else:
            self._atom_counts = np.fromiter(
                (_get_num_atoms(dataset, i) for i in range(n)),
                dtype=np.int64,
                count=n,
            )

        # Pack once at init; only batch *order* changes each epoch.
        self.all_batches: List[Batch] = self._build_batches()

        if len(self.all_batches) < self.num_replicas:
            raise ValueError(
                f"Only {len(self.all_batches)} batches were packed but "
                f"num_replicas={self.num_replicas}. Increase the dataset size or "
                "reduce max_atoms."
            )

        if self.drop_last and len(self.all_batches) % self.num_replicas != 0:
            self.num_samples = math.floor(len(self.all_batches) / self.num_replicas)
        else:
            self.num_samples = math.ceil(len(self.all_batches) / self.num_replicas)
        self.total_size = self.num_samples * self.num_replicas

    def set_epoch(self, epoch: int) -> None:
        """Set the epoch for deterministic shuffling. Call before each epoch.

        :param epoch: The current epoch number.
        """
        self.epoch = epoch

    def _build_batches(self) -> List[Batch]:
        """Pack structures into batches (called once at init).

        :return: List of batches, each batch being an int64 numpy array of
            dataset indices.
        """
        n = len(self.dataset)
        if self.shuffle:
            # Hardcoded seed so packing is identical across runs, ranks, and
            # restarts regardless of the global RNG state.
            rng = np.random.default_rng(0)
            indices = rng.permutation(n)
        else:
            indices = np.arange(n, dtype=np.int64)
        atom_counts = self._atom_counts[indices]
        return _greedy_pack(indices, atom_counts, self.max_atoms, self.min_atoms)

    def __iter__(self) -> Iterator[Batch]:
        # Shuffle batch presentation order per epoch. We use a local generator keyed
        # solely on ``self.epoch`` so that all ranks agree on the order without
        # requiring the global RNG state to be synchronised across processes.
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.epoch)
            batch_indices = torch.randperm(len(self.all_batches), generator=g).tolist()
        else:
            batch_indices = list(range(len(self.all_batches)))

        if not self.drop_last:
            # Pad to total_size, wrapping multiple times if needed.
            padding_size = self.total_size - len(batch_indices)
            if padding_size <= len(batch_indices):
                batch_indices += batch_indices[:padding_size]
            else:
                batch_indices += (
                    batch_indices * math.ceil(padding_size / len(batch_indices))
                )[:padding_size]
        else:
            batch_indices = batch_indices[: self.total_size]

        assert len(batch_indices) == self.total_size

        # Assign to this rank via interleaved striding.
        batch_indices = batch_indices[self.rank : self.total_size : self.num_replicas]
        assert len(batch_indices) == self.num_samples

        batch_slice = [self.all_batches[i] for i in batch_indices]
        return iter(batch_slice)

    def __len__(self) -> int:
        return self.num_samples


class MaxAtomBatchSampler(MaxAtomDistributedBatchSampler):
    """Single-process version of :class:`MaxAtomDistributedBatchSampler`.

    Convenience wrapper that fixes ``num_replicas=1`` and ``rank=0``.

    :param dataset: The dataset to sample from.
    :param max_atoms: Maximum total atoms per batch.
    :param shuffle: Whether to shuffle batch order each epoch.
    :param drop_last: Whether to drop the last incomplete batch.
    :param min_atoms: Minimum total atoms required to keep a batch.
    """

    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        max_atoms: int,
        shuffle: bool = True,
        drop_last: bool = False,
        min_atoms: int = 0,
    ) -> None:
        super().__init__(
            dataset=dataset,
            max_atoms=max_atoms,
            num_replicas=1,
            rank=0,
            shuffle=shuffle,
            drop_last=drop_last,
            min_atoms=min_atoms,
        )
