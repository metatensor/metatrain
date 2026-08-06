"""Byte-budgeted caching for collate transforms.

Collate transforms sometimes attach expensive per-system data to a batch
(e.g. two-centre metric matrices for density losses). That data often depends
only on the unaugmented geometry, which is identical every epoch, so it can be
cached across epochs inside the dataloader workers: the workers are persistent
(``persistent_workers``), so a process-level cache survives from one epoch to
the next.

This module provides the generic pieces of such a cache: an LRU tensor cache
bounded by a total byte budget, the environment-variable override for that
budget, and a helper to read a batch's native system ids from the
``mtt::aux::system_index`` extra data — the stable cache key.
"""

from __future__ import annotations

import os
from collections import OrderedDict
from typing import Dict, List, Optional

import torch
from metatensor.torch import TensorMap


DEFAULT_COLLATE_CACHE_MAX_BYTES = 2 * 1024**3


def collate_cache_max_bytes() -> int:
    """Per-process byte budget for collate-transform caches.

    Overridable via ``METATRAIN_COLLATE_CACHE_MAX_BYTES``. The budget applies
    per dataloader worker process: the total host-memory footprint is
    ``budget × num_workers × ranks_per_node``.

    :return: The byte budget.
    """
    return int(
        os.environ.get(
            "METATRAIN_COLLATE_CACHE_MAX_BYTES", DEFAULT_COLLATE_CACHE_MAX_BYTES
        )
    )


class ByteBudgetCache:
    """LRU tensor cache bounded by total byte size.

    Unbounded caching is not an option here: on large datasets (e.g. >1M
    systems) per-system cached tensors exceed host memory long before an epoch
    completes, so once the budget is reached the least-recently-used entries
    are evicted and the cache degrades to recomputation. Entries larger than
    the whole budget are not cached.

    :param max_bytes: Total byte budget for cached tensors.
    """

    def __init__(self, max_bytes: int) -> None:
        self._data: "OrderedDict[tuple, torch.Tensor]" = OrderedDict()
        self._bytes = 0
        self._max_bytes = max_bytes

    def get(self, key: tuple) -> Optional[torch.Tensor]:
        tensor = self._data.get(key)
        if tensor is not None:
            self._data.move_to_end(key)
        return tensor

    def put(self, key: tuple, tensor: torch.Tensor) -> None:
        nbytes = tensor.numel() * tensor.element_size()
        if nbytes > self._max_bytes:
            return
        existing = self._data.pop(key, None)
        if existing is not None:
            self._bytes -= existing.numel() * existing.element_size()
        self._data[key] = tensor
        self._bytes += nbytes
        while self._bytes > self._max_bytes:
            _, evicted = self._data.popitem(last=False)
            self._bytes -= evicted.numel() * evicted.element_size()


def batch_system_ids(extra: Dict[str, TensorMap]) -> Optional[List[int]]:
    """Native per-system ids of a batch, in batch order, if available.

    :param extra: The batch's extra-data dictionary.
    :return: One id per system, or ``None`` when the batch carries none.
    """
    index_map = extra.get("mtt::aux::system_index")
    if index_map is None:
        return None
    return [int(v) for v in index_map[0].values[:, 0]]
