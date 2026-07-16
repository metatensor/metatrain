"""Batch samplers for atom-count-aware batching.

The design of :class:`MaxAtomDistributedBatchSampler` is based on the
``MaxAtomDistributedBatchSampler`` from the fairchem library:

    https://github.com/facebookresearch/fairchem

Copyright (c) Meta Platforms, Inc. and affiliates.
Licensed under the MIT License.
"""

import logging
import math
from typing import Iterator, List, Tuple

import numpy as np
import torch
import torch.utils.data


logger = logging.getLogger(__name__)


def _get_num_atoms(dataset: torch.utils.data.Dataset, i: int) -> int:
    """Return atom count for sample ``i``, resolving ``Subset`` wrappers.

    Datasets that expose a fast ``get_num_atoms(i)`` (e.g. ``MemmapDataset``) are
    queried directly. Any other dataset is assumed to yield samples with a
    ``system`` field (e.g. ``metatensor.learn.data.Dataset``, ``DiskDataset``),
    and the atom count is read off that ``System`` via ``len()``.

    :param dataset: The dataset to query.
    :param i: The sample index.
    :return: The number of atoms in sample ``i``.
    """
    if isinstance(dataset, torch.utils.data.Subset):
        return _get_num_atoms(dataset.dataset, dataset.indices[i])
    if hasattr(dataset, "get_num_atoms"):
        return dataset.get_num_atoms(i)
    system = getattr(dataset[i], "system", None)
    if system is not None:
        return len(system)
    raise TypeError(
        f"Dataset of type {type(dataset).__name__} does not support "
        "get_num_atoms() and its samples do not expose a 'system' field. "
        "max_atoms_per_batch requires one of the two."
    )


def _pack_batches_csr(
    indices: np.ndarray,
    atom_counts: np.ndarray,
    max_atoms: int,
    min_atoms: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Greedily pack ``indices`` into batches and return CSR arrays.

    Single structures whose own atom count exceeds ``max_atoms`` are dropped with
    a warning. Completed batches whose total atom count falls below ``min_atoms``
    are dropped (their structures are skipped for the epoch).

    Implementation is vectorised: oversized structures are removed with a numpy
    mask and batch break points are found via ``np.cumsum`` + ``np.searchsorted``,
    so the inner loop runs once per batch (not once per structure).

    :param indices: Dataset indices to pack.
    :param atom_counts: Atom count for each index (parallel to ``indices``).
    :param max_atoms: Maximum total atoms allowed per batch.
    :param min_atoms: Minimum total atoms required to keep a batch.
    :return: ``(flat_indices, offsets)`` int64 arrays such that batch ``i``
        is ``flat_indices[offsets[i]:offsets[i + 1]]``. ``offsets`` has length
        ``num_batches + 1`` and starts at 0.
    """
    indices_arr = np.asarray(indices, dtype=np.int64)
    counts_arr = np.asarray(atom_counts, dtype=np.int64)
    if indices_arr.shape != counts_arr.shape:
        raise ValueError(
            f"indices and atom_counts must have the same shape, got "
            f"{indices_arr.shape} vs {counts_arr.shape}"
        )

    empty_csr = (np.empty(0, dtype=np.int64), np.zeros(1, dtype=np.int64))

    n_total = indices_arr.size
    if n_total == 0:
        return empty_csr

    oversized_mask = counts_arr > max_atoms
    n_oversized = int(oversized_mask.sum())
    if n_oversized:
        # Per-structure warning preserves existing log surface; cost is bounded
        # by the oversized count, not n_total.
        over_indices = indices_arr[oversized_mask].tolist()
        over_counts = counts_arr[oversized_mask].tolist()
        for over_idx, over_n in zip(over_indices, over_counts, strict=True):
            logger.warning(
                f"Structure {over_idx} has {over_n} atoms which exceeds "
                f"max_atoms_per_batch ({max_atoms}). Skipping this structure."
            )
        keep = ~oversized_mask
        indices_arr = indices_arr[keep]
        counts_arr = counts_arr[keep]

    n = indices_arr.size
    if n == 0:
        return empty_csr

    cumsum = np.cumsum(counts_arr)

    # Walk batch boundaries. Each iteration is a single ``searchsorted`` call
    # — typically orders of magnitude fewer iterations than the structure count.
    kept_starts: List[int] = []
    kept_ends: List[int] = []
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
            kept_starts.append(start)
            kept_ends.append(end)
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

    if not kept_starts:
        return empty_csr

    starts_arr = np.asarray(kept_starts, dtype=np.int64)
    ends_arr = np.asarray(kept_ends, dtype=np.int64)
    lengths = ends_arr - starts_arr

    offsets = np.empty(lengths.size + 1, dtype=np.int64)
    offsets[0] = 0
    np.cumsum(lengths, out=offsets[1:])

    # Common case (no min_atoms drops): kept batches cover ``indices_arr``
    # contiguously, so we can hand it back without an extra concat.
    if (
        starts_arr[0] == 0
        and ends_arr[-1] == n
        and bool(np.array_equal(starts_arr[1:], ends_arr[:-1]))
    ):
        flat_indices = np.ascontiguousarray(indices_arr)
    else:
        flat_indices = np.concatenate(
            [indices_arr[s:e] for s, e in zip(kept_starts, kept_ends, strict=True)]
        )

    return flat_indices, offsets


class MaxAtomDistributedBatchSampler(torch.utils.data.Sampler):
    """Distributed batch sampler that packs structures greedily up to ``max_atoms``.

    Structure-to-batch packing is performed **once at construction** using a
    hardcoded seed (stable across runs, ranks, and restarts). Each epoch, the
    *order* in which batches are presented to each rank is reshuffled
    deterministically from the epoch number, mirroring the fairchem
    ``MaxAtomDistributedBatchSampler`` design.

    Batches are stored in CSR (compressed-sparse-row) form — two int64 numpy
    arrays, ``_batch_indices`` (flat indices of every kept structure) and
    ``_batch_offsets`` (per-batch start offsets). This avoids the per-batch
    Python-object overhead of a list-of-lists, and — more importantly —
    removes the long-lived Python objects that would otherwise be refcount-
    touched by ``DataLoader`` worker forks (each touch dirties a page and
    triggers copy-on-write, exhausting ``/dev/shm`` for large datasets).
    ``__iter__`` materialises each batch as a Python ``list[int]`` on demand,
    so workers only ever see short-lived lists.

    :param dataset: The dataset to sample from. Either supports ``get_num_atoms(i)``
        directly (e.g. ``MemmapDataset``) or yields samples with a ``system`` field
        (e.g. ``metatensor.learn.data.Dataset``, ``DiskDataset``); ``Subset``
        wrappers of either are also supported.
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
                atom_counts = all_counts[subset_idx]
            else:
                atom_counts = all_counts
        else:
            atom_counts = np.fromiter(
                (_get_num_atoms(dataset, i) for i in range(n)),
                dtype=np.int64,
                count=n,
            )

        # Pack once at init; only batch *order* changes each epoch.
        self._batch_indices, self._batch_offsets = self._build_batches_csr(atom_counts)
        # ``atom_counts`` is no longer needed; let GC reclaim it before workers
        # fork so we don't COW it into every worker.
        del atom_counts

        num_batches = self._batch_offsets.size - 1
        if num_batches < self.num_replicas:
            raise ValueError(
                f"Only {num_batches} batches were packed but "
                f"num_replicas={self.num_replicas}. Increase the dataset size or "
                "reduce max_atoms."
            )

        if self.drop_last and num_batches % self.num_replicas != 0:
            self.num_samples = math.floor(num_batches / self.num_replicas)
        else:
            self.num_samples = math.ceil(num_batches / self.num_replicas)
        self.total_size = self.num_samples * self.num_replicas

    def set_epoch(self, epoch: int) -> None:
        """Set the epoch for deterministic shuffling. Call before each epoch.

        :param epoch: The current epoch number.
        """
        self.epoch = epoch

    def _build_batches_csr(
        self, atom_counts: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Pack structures into batches and return as CSR numpy arrays.

        :param atom_counts: int64 array of atom counts, one per structure (in
            the dataset's natural order).
        :return: ``(flat_indices, offsets)`` arrays describing the packed
            batches. Batch ``i`` contains the dataset indices in
            ``flat_indices[offsets[i]:offsets[i + 1]]``.
        """
        n = atom_counts.size
        if self.shuffle:
            # Hardcoded seed so packing is identical across runs, ranks, and
            # restarts regardless of the global RNG state.
            rng = np.random.default_rng(0)
            indices = rng.permutation(n)
        else:
            indices = np.arange(n, dtype=np.int64)
        return _pack_batches_csr(
            indices, atom_counts[indices], self.max_atoms, self.min_atoms
        )

    @property
    def all_batches(self) -> List[List[int]]:
        """All packed batches as a list of Python lists (for inspection/tests).

        Production iteration should go through ``__iter__``; this property
        materialises every batch and is intended for debugging.
        """
        offsets = self._batch_offsets
        return [
            self._batch_indices[offsets[i] : offsets[i + 1]].tolist()
            for i in range(offsets.size - 1)
        ]

    def _get_batch(self, i: int) -> List[int]:
        """Return batch ``i`` as a Python list of indices.

        The returned list is short-lived: it's built per ``__iter__`` call and
        passed straight into the ``DataLoader`` worker, so no long-lived list
        objects accumulate in the parent process for forks to COW.

        :param i: Index of the batch to return.
        :return: Dataset indices belonging to batch ``i``.
        """
        return self._batch_indices[
            self._batch_offsets[i] : self._batch_offsets[i + 1]
        ].tolist()

    def __iter__(self) -> Iterator[List[int]]:
        num_batches = self._batch_offsets.size - 1
        # Shuffle batch presentation order per epoch. We use a local generator
        # keyed solely on ``self.epoch`` so that all ranks agree on the order
        # without requiring the global RNG state to be synchronised.
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.epoch)
            batch_indices = torch.randperm(num_batches, generator=g).tolist()
        else:
            batch_indices = list(range(num_batches))

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

        return (self._get_batch(i) for i in batch_indices)

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
