"""Tests for the metric-matrix cache in ``metatrain.utils.pyscf_loss``.

These are separate from ``test_pyscf_loss.py`` because the cache is pure
bookkeeping around ``compute_metric_matrix``: none of it needs PySCF, so it
must keep running in environments where that optional dependency (and hence
the whole PySCF-guarded test module) is absent. The generic cache behaviour
(byte budget, LRU eviction, system-id extraction) is covered in
``tests/utils/data/test_byte_budget_cache.py``.
"""

import pytest
import torch
from metatensor.torch import Labels, TensorBlock, TensorMap
from metatomic.torch import System

import metatrain.utils.pyscf_loss as pyscf_loss
from metatrain.utils.pyscf_loss import (
    _metric_matrices_transform,
    metric_matrix_name,
    unpack_metric_matrices,
)


def _system(n_atoms: int) -> System:
    return System(
        types=torch.full((n_atoms,), 1),
        positions=torch.rand((n_atoms, 3), dtype=torch.float64),
        cell=torch.zeros((3, 3), dtype=torch.float64),
        pbc=torch.tensor([False, False, False]),
    )


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


@pytest.fixture
def counted_compute(monkeypatch):
    """Replace ``compute_metric_matrix`` with a PySCF-free stand-in that
    counts its calls, and reset the process-wide cache around the test."""
    calls = []

    def fake_compute(system, aux_basis, metric):
        calls.append((aux_basis, metric))
        n = len(system)
        return torch.eye(n, dtype=torch.float64)

    monkeypatch.setattr(pyscf_loss, "compute_metric_matrix", fake_compute)
    monkeypatch.setattr(pyscf_loss, "_METRIC_MATRIX_CACHE", None)
    return calls


def test_transform_caches_across_batches(counted_compute):
    # With native system ids in the batch, a second pass over the same
    # systems (i.e. the next epoch, in the same persistent worker) must be
    # served entirely from the cache.
    systems = [_system(2), _system(3)]
    extra = {"mtt::aux::system_index": _system_index([7, 4])}

    _, _, extra = _metric_matrices_transform(
        {"mtt::density": "some-basis"}, "overlap", systems, {}, extra
    )
    matrices = unpack_metric_matrices(
        extra[metric_matrix_name("mtt::density", "overlap")]
    )
    assert [m.shape for m in matrices] == [(2, 2), (3, 3)]
    assert len(counted_compute) == 2

    _metric_matrices_transform(
        {"mtt::density": "some-basis"}, "overlap", systems, {}, dict(extra)
    )
    assert len(counted_compute) == 2  # cache hits, no recomputation

    # A different metric must not reuse the overlap matrices.
    _metric_matrices_transform(
        {"mtt::density": "some-basis"}, "coulomb", systems, {}, dict(extra)
    )
    assert len(counted_compute) == 4


def test_transform_recomputes_without_system_ids(counted_compute):
    # Batches without native system ids have no stable cache key: every pass
    # recomputes, which is slow but never serves a wrong matrix.
    systems = [_system(2)]

    _metric_matrices_transform(
        {"mtt::density": "some-basis"}, "overlap", systems, {}, {}
    )
    _metric_matrices_transform(
        {"mtt::density": "some-basis"}, "overlap", systems, {}, {}
    )
    assert len(counted_compute) == 2
