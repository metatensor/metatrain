"""Batch samplers for atom-count-aware batching."""

import logging
import math
from typing import Iterator, List

import numpy as np
import torch
import torch.utils.data


logger = logging.getLogger(__name__)


def _get_num_atoms(dataset: torch.utils.data.Dataset, i: int) -> int:
    """Return atom count for sample ``i``, resolving ``Subset`` wrappers.

    :param dataset: The dataset to query.
    :param i: Index of the sample.
    :return: Number of atoms in sample ``i``.
    """
    if isinstance(dataset, torch.utils.data.Subset):
        return _get_num_atoms(dataset.dataset, dataset.indices[i])
    if hasattr(dataset, "get_num_atoms"):
        return dataset.get_num_atoms(i)
    # Fallback for in-memory datasets (metatensor.learn.data.Dataset)
    sample = dataset[i]
    if hasattr(sample, "_asdict"):
        sample = sample._asdict()
    if isinstance(sample, dict) and "system" in sample:
        return len(sample["system"])
    raise TypeError(
        f"Dataset of type {type(dataset).__name__} does not support "
        "get_num_atoms(). Ensure the dataset class implements get_num_atoms(i)."
    )


def _greedy_pack(
    indices: List[int],
    atom_counts: List[int],
    max_atoms: int,
) -> List[List[int]]:
    """Greedily pack ``indices`` into batches where total atoms <= ``max_atoms``.

    Single structures that alone exceed ``max_atoms`` are skipped with a warning.

    :param indices: Sample indices to pack.
    :param atom_counts: Atom count for each index.
    :param max_atoms: Maximum atoms per batch.
    :return: List of batches, each a list of indices.
    """
    batches: List[List[int]] = []
    current_batch: List[int] = []
    current_atoms = 0

    for idx, n in zip(indices, atom_counts, strict=True):
        if n > max_atoms:
            logger.warning(
                f"Structure {idx} has {n} atoms which exceeds max_atoms_per_batch "
                f"({max_atoms}). Skipping this structure."
            )
            continue
        if current_atoms + n > max_atoms and current_batch:
            batches.append(current_batch)
            current_batch = []
            current_atoms = 0
        current_batch.append(idx)
        current_atoms += n

    if current_batch:
        batches.append(current_batch)

    return batches


def _get_atom_counts(dataset: torch.utils.data.Dataset) -> np.ndarray:
    """Get atom counts for all samples, using vectorised path if available.

    :param dataset: The dataset to query.
    :return: Array of atom counts per sample.
    """
    inner = dataset.dataset if isinstance(dataset, torch.utils.data.Subset) else dataset
    if hasattr(inner, "get_all_atom_counts"):
        all_counts = inner.get_all_atom_counts()
        if isinstance(dataset, torch.utils.data.Subset):
            return all_counts[np.array(dataset.indices, dtype=np.int64)]
        return all_counts
    return np.array(
        [_get_num_atoms(dataset, i) for i in range(len(dataset))], dtype=np.int64
    )


class MaxAtomBatchSampler(torch.utils.data.Sampler):
    """Single-process batch sampler that packs structures greedily up to
    ``max_atoms`` total atoms per batch.

    Packing is done once at construction. Each epoch, only the batch
    presentation order is reshuffled.

    :param dataset: The dataset to sample from. Must support ``get_num_atoms(i)``.
    :param max_atoms: Maximum total atoms per batch.
    :param shuffle: Whether to shuffle batch order each epoch.
    :param seed: Base random seed.
    """

    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        max_atoms: int,
        shuffle: bool = True,
        seed: int = 0,
    ) -> None:
        self.dataset = dataset
        self.max_atoms = max_atoms
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

        atom_counts = _get_atom_counts(dataset)
        self._batch_indices, self._batch_offsets = self._build_batches_csr(atom_counts)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def _build_batches_csr(self, atom_counts: np.ndarray) -> tuple:
        indices = np.arange(len(self.dataset), dtype=np.int64)
        if self.shuffle:
            rng = np.random.default_rng(self.seed)
            rng.shuffle(indices)
        counts = atom_counts[indices]
        batches = _greedy_pack(indices.tolist(), counts.tolist(), self.max_atoms)
        flat = np.concatenate([np.asarray(b, dtype=np.int64) for b in batches])
        lengths = np.array([len(b) for b in batches], dtype=np.int64)
        offsets = np.zeros(len(lengths) + 1, dtype=np.int64)
        np.cumsum(lengths, out=offsets[1:])
        return flat, offsets

    def _get_batch(self, i: int) -> List[int]:
        return self._batch_indices[
            self._batch_offsets[i] : self._batch_offsets[i + 1]
        ].tolist()

    def __iter__(self) -> Iterator[List[int]]:
        num_batches = len(self._batch_offsets) - 1
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            batch_order = torch.randperm(num_batches, generator=g).tolist()
        else:
            batch_order = list(range(num_batches))
        for i in batch_order:
            yield self._get_batch(i)

    def __len__(self) -> int:
        return len(self._batch_offsets) - 1


class MaxAtomDistributedBatchSampler(torch.utils.data.Sampler):
    """Distributed batch sampler that packs structures greedily up to ``max_atoms``.

    Structure-to-batch packing is performed once at construction using ``seed``
    (stable across epochs). Each epoch, the order in which batches are presented
    to each rank is reshuffled using ``seed + epoch``, mirroring the fairchem
    ``MaxAtomDistributedBatchSampler`` design.

    Batch data is stored in CSR (compressed-sparse-row) numpy arrays rather than
    Python lists to avoid fork-induced copy-on-write pressure from Python's
    garbage collector traversing reference-counted objects in worker processes.

    :param dataset: The dataset to sample from. Must support ``get_num_atoms(i)``.
    :param max_atoms: Maximum total atoms per batch.
    :param num_replicas: Number of distributed processes (world size).
    :param rank: Rank of the current process.
    :param shuffle: Whether to shuffle batch presentation order each epoch.
    :param seed: Base random seed.
    :param drop_last: If True, drop tail batches so count is evenly divisible
        by ``num_replicas``.
    """

    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        max_atoms: int,
        num_replicas: int = 1,
        rank: int = 0,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
    ) -> None:
        self.dataset = dataset
        self.max_atoms = max_atoms
        self.num_replicas = num_replicas
        self.rank = rank
        self.shuffle = shuffle
        self.seed = seed
        self.drop_last = drop_last
        self.epoch = 0

        atom_counts = _get_atom_counts(dataset)
        self._batch_indices, self._batch_offsets = self._build_batches_csr(atom_counts)

        num_batches = len(self._batch_offsets) - 1
        if num_batches < self.num_replicas:
            raise ValueError(
                f"Only {num_batches} batches were packed but "
                f"num_replicas={self.num_replicas}. Increase the dataset size or "
                "increase max_atoms_per_batch."
            )

        if self.drop_last and num_batches % self.num_replicas != 0:
            self.num_samples = math.floor(num_batches / self.num_replicas)
        else:
            self.num_samples = math.ceil(num_batches / self.num_replicas)
        self.total_size = self.num_samples * self.num_replicas

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def _build_batches_csr(self, atom_counts: np.ndarray) -> tuple:
        indices = np.arange(len(self.dataset), dtype=np.int64)
        if self.shuffle:
            rng = np.random.default_rng(self.seed)
            rng.shuffle(indices)
        counts = atom_counts[indices]
        batches = _greedy_pack(indices.tolist(), counts.tolist(), self.max_atoms)
        flat = np.concatenate([np.asarray(b, dtype=np.int64) for b in batches])
        lengths = np.array([len(b) for b in batches], dtype=np.int64)
        offsets = np.zeros(len(lengths) + 1, dtype=np.int64)
        np.cumsum(lengths, out=offsets[1:])
        return flat, offsets

    def _get_batch(self, i: int) -> List[int]:
        return self._batch_indices[
            self._batch_offsets[i] : self._batch_offsets[i + 1]
        ].tolist()

    def __iter__(self) -> Iterator[List[int]]:
        num_batches = len(self._batch_offsets) - 1

        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            batch_indices = torch.randperm(num_batches, generator=g).tolist()
        else:
            batch_indices = list(range(num_batches))

        # Pad or truncate to total_size
        if not self.drop_last:
            padding_size = self.total_size - len(batch_indices)
            if padding_size <= len(batch_indices):
                batch_indices += batch_indices[:padding_size]
            else:
                batch_indices += (
                    batch_indices * math.ceil(padding_size / len(batch_indices))
                )[:padding_size]
        else:
            batch_indices = batch_indices[: self.total_size]

        # Subsample for this rank (interleaved striding)
        rank_indices = batch_indices[self.rank :: self.num_replicas]

        for i in rank_indices:
            yield self._get_batch(i)

    def __len__(self) -> int:
        return self.num_samples
