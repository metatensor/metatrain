"""Tests for ``metatrain.utils.data.byte_budget_cache``."""

import torch
from metatensor.torch import Labels, TensorBlock, TensorMap

from metatrain.utils.data.byte_budget_cache import (
    DEFAULT_COLLATE_CACHE_MAX_BYTES,
    ByteBudgetCache,
    batch_system_ids,
    collate_cache_max_bytes,
)


def test_byte_budget_bounds_memory():
    # The cache must stay within its byte budget on datasets too large to
    # fit, evicting least-recently-used entries instead of growing without
    # bound (an unbounded cache OOMs multi-node runs on >1M-system datasets).
    t = torch.zeros(100, dtype=torch.float32)  # 400 bytes each
    cache = ByteBudgetCache(max_bytes=1000)  # fits 2 entries

    cache.put(("aux", 0), t)
    cache.put(("aux", 1), t)
    assert cache.get(("aux", 0)) is not None
    assert cache.get(("aux", 1)) is not None

    # 3rd entry exceeds the budget -> the least recently used entry (id 0,
    # touched before id 1 by the gets above) is evicted
    cache.put(("aux", 2), t)
    assert cache.get(("aux", 0)) is None  # evicted (least recently used)
    assert cache.get(("aux", 1)) is not None
    assert cache.get(("aux", 2)) is not None

    # re-putting an existing key must not double-count its bytes
    cache.put(("aux", 1), t)
    assert cache.get(("aux", 2)) is not None

    # an entry larger than the whole budget is never cached
    big = torch.zeros(1000, dtype=torch.float32)  # 4000 bytes
    cache.put(("aux", 3), big)
    assert cache.get(("aux", 3)) is None


def test_budget_env_override(monkeypatch):
    assert collate_cache_max_bytes() == DEFAULT_COLLATE_CACHE_MAX_BYTES
    monkeypatch.setenv("METATRAIN_COLLATE_CACHE_MAX_BYTES", "12345")
    assert collate_cache_max_bytes() == 12345


def _system_index(ids) -> TensorMap:
    """The collated ``mtt::aux::system_index`` extra data, as DiskDataset
    emits it (one float64 id per system)."""
    ids = torch.tensor(ids)
    return TensorMap(
        keys=Labels(["_"], torch.tensor([[0]])),
        blocks=[
            TensorBlock(
                values=ids.reshape(-1, 1).to(torch.float64),
                samples=Labels(
                    names=["system"],
                    values=torch.arange(len(ids)).reshape(-1, 1),
                ),
                components=[],
                properties=Labels(["_"], torch.tensor([[0]])),
            )
        ],
    )


def test_batch_system_ids():
    extra = {"mtt::aux::system_index": _system_index([7, 4])}
    assert batch_system_ids(extra) == [7, 4]


def test_batch_system_ids_absent():
    # Batches without native system ids (e.g. from in-memory datasets) have
    # no stable cache key: callers must fall back to recomputing.
    assert batch_system_ids({}) is None
