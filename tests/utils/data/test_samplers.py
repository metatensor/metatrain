"""Tests for MaxAtom batch samplers."""

import logging

import numpy as np
import pytest
import torch
import torch.utils.data
from torch.utils.data import DataLoader

from metatrain.utils.data.dataset import MemmapDataset
from metatrain.utils.data.samplers import (
    MaxAtomBatchSampler,
    MaxAtomDistributedBatchSampler,
    _greedy_pack,
)


# ---------------------------------------------------------------------------
# Minimal fake dataset with get_num_atoms support
# ---------------------------------------------------------------------------


class _FakeDataset(torch.utils.data.Dataset):
    """Fake dataset whose atom counts are set explicitly."""

    def __init__(self, atom_counts):
        self._atom_counts = atom_counts

    def __len__(self):
        return len(self._atom_counts)

    def __getitem__(self, i):
        return i

    def get_num_atoms(self, i):
        return self._atom_counts[i]


class _DatasetNoGetNumAtoms(torch.utils.data.Dataset):
    def __len__(self):
        return 5

    def __getitem__(self, i):
        return i


# ---------------------------------------------------------------------------
# _greedy_pack unit tests
# ---------------------------------------------------------------------------


def test_greedy_pack_basic():
    """All batches respect the max_atoms limit."""
    indices = list(range(10))
    atom_counts = [3] * 10  # 10 structures, 3 atoms each
    batches = _greedy_pack(indices, atom_counts, max_atoms=9)
    # Each batch can hold at most 3 structures (3*3=9)
    for batch in batches:
        assert sum(atom_counts[i] for i in batch) <= 9
    # All indices appear exactly once
    assert sorted(sum(batches, [])) == list(range(10))


def test_greedy_pack_all_fit_one_batch():
    indices = list(range(4))
    atom_counts = [2, 2, 2, 2]
    batches = _greedy_pack(indices, atom_counts, max_atoms=8)
    assert len(batches) == 1
    assert sorted(batches[0]) == [0, 1, 2, 3]


def test_greedy_pack_min_atoms_drops_small_batches():
    """Batches whose total atom count is below min_atoms are dropped."""
    # 5 structures: sizes [3, 1, 3, 1, 3]. With max_atoms=3, greedy packs into
    # batches [3], [1], [3], [1], [3]. min_atoms=2 drops the two size-1 batches.
    indices = [0, 1, 2, 3, 4]
    atom_counts = [3, 1, 3, 1, 3]
    batches = _greedy_pack(indices, atom_counts, max_atoms=3, min_atoms=2)
    assert len(batches) == 3
    for batch in batches:
        assert sum(atom_counts[i] for i in batch) >= 2


def test_max_atom_batch_sampler_min_atoms_drops_small_batches():
    """MaxAtomBatchSampler with min_atoms drops batches below the threshold."""
    # 5 structures: sizes [3, 1, 3, 1, 3]. max_atoms=3 forces one structure per
    # batch; min_atoms=2 discards the two singleton batches with 1 atom each.
    atom_counts = [3, 1, 3, 1, 3]
    ds = _FakeDataset(atom_counts)
    sampler = MaxAtomBatchSampler(ds, max_atoms=3, min_atoms=2, shuffle=False)
    batches = list(sampler)
    assert len(batches) == 3
    for batch in batches:
        assert sum(atom_counts[i] for i in batch) >= 2


def test_greedy_pack_min_atoms_zero_unchanged():
    """min_atoms=0 (default) keeps all batches, including the final partial one."""
    indices = list(range(5))
    atom_counts = [3, 3, 3, 3, 1]  # last batch has only 1 atom
    batches_default = _greedy_pack(indices, atom_counts, max_atoms=3)
    batches_explicit = _greedy_pack(indices, atom_counts, max_atoms=3, min_atoms=0)
    assert batches_default == batches_explicit
    # The last singleton batch is kept
    assert len(batches_default) == 5


def test_greedy_pack_oversized_structure_skipped(caplog):
    """Structures larger than max_atoms are skipped with a warning."""
    indices = [0, 1, 2]
    atom_counts = [3, 100, 3]  # index 1 is too big
    with caplog.at_level(logging.WARNING, logger="metatrain.utils.data.samplers"):
        batches = _greedy_pack(indices, atom_counts, max_atoms=9)
    assert "Structure 1" in caplog.text
    all_indices = sorted(sum(batches, []))
    assert all_indices == [0, 2]


def test_greedy_pack_variable_sizes():
    """Greedy packing fills batches as tightly as possible."""
    indices = [0, 1, 2, 3]
    atom_counts = [5, 3, 4, 2]
    batches = _greedy_pack(indices, atom_counts, max_atoms=8)
    for batch in batches:
        assert sum(atom_counts[i] for i in batch) <= 8
    assert sorted(sum(batches, [])) == [0, 1, 2, 3]


# ---------------------------------------------------------------------------
# MaxAtomBatchSampler (single-process) unit tests
# ---------------------------------------------------------------------------


def test_single_process_all_samples_covered():
    """All samples appear exactly once per epoch."""
    atom_counts = [3] * 20
    ds = _FakeDataset(atom_counts)
    sampler = MaxAtomBatchSampler(ds, max_atoms=9, shuffle=False)
    all_indices = sorted(sum(list(sampler), []))
    assert all_indices == list(range(20))


def test_single_process_no_shuffle_produces_sequential_batches():
    """With shuffle=False and uniform atom counts the batches are sequential."""
    n, n_atoms = 12, 3
    ds = _FakeDataset([n_atoms] * n)
    batch_size = 3  # max_atoms = 3*3 = 9
    sampler = MaxAtomBatchSampler(ds, max_atoms=n_atoms * batch_size, shuffle=False)
    batches = list(sampler)
    expected = [list(range(i, i + batch_size)) for i in range(0, n, batch_size)]
    assert batches == expected


def test_single_process_atom_limit_respected():
    """No batch exceeds max_atoms."""
    atom_counts = [1, 5, 2, 4, 3, 6, 1]
    ds = _FakeDataset(atom_counts)
    sampler = MaxAtomBatchSampler(ds, max_atoms=7, shuffle=False)
    for batch in sampler:
        total = sum(atom_counts[i] for i in batch)
        assert total <= 7


def test_set_epoch_changes_order():
    """Different epochs produce different batch ordering."""
    atom_counts = [2] * 30
    ds = _FakeDataset(atom_counts)
    sampler = MaxAtomBatchSampler(ds, max_atoms=6, shuffle=True)

    sampler.set_epoch(0)
    batches_e0 = list(sampler)
    sampler.set_epoch(1)
    batches_e1 = list(sampler)

    # The first batches should differ (with high probability for 30 structures)
    assert batches_e0 != batches_e1


def test_same_epoch_same_order():
    """Same epoch always produces the same batches (deterministic)."""
    atom_counts = [2] * 20
    ds = _FakeDataset(atom_counts)
    sampler = MaxAtomBatchSampler(ds, max_atoms=6, shuffle=True)

    sampler.set_epoch(3)
    run1 = list(sampler)
    sampler.set_epoch(3)
    run2 = list(sampler)
    assert run1 == run2


def test_unsupported_dataset_raises():
    """Constructing the sampler on a dataset without get_num_atoms raises TypeError."""
    ds = _DatasetNoGetNumAtoms()
    with pytest.raises(TypeError, match="does not support get_num_atoms"):
        MaxAtomBatchSampler(ds, max_atoms=10)


def test_subset_compatibility():
    """Works with torch.utils.data.Subset wrapping a dataset with get_num_atoms."""
    atom_counts = [2, 4, 3, 1, 5]
    ds = _FakeDataset(atom_counts)
    subset = torch.utils.data.Subset(ds, [0, 2, 4])  # 2, 3, 5 atoms
    sampler = MaxAtomBatchSampler(subset, max_atoms=5, shuffle=False)
    all_subset_indices = sorted(sum(list(sampler), []))
    # The sampler iterates over subset indices 0,1,2 (pointing to ds[0], ds[2], ds[4])
    assert all_subset_indices == [0, 1, 2]


# ---------------------------------------------------------------------------
# MaxAtomDistributedBatchSampler tests
# ---------------------------------------------------------------------------


def test_distributed_union_covers_all_samples():
    """The union of all ranks' batches covers all samples (ignoring pad repeats)."""
    atom_counts = [3] * 21  # 21 structures → 7 batches of 3 (max_atoms=9)
    ds = _FakeDataset(atom_counts)
    world_size = 4
    all_indices = []
    for rank in range(world_size):
        sampler = MaxAtomDistributedBatchSampler(
            ds, max_atoms=9, num_replicas=world_size, rank=rank, shuffle=False
        )
        for batch in sampler:
            all_indices.extend(batch)

    # 7 real batches + 1 pad batch (to make 8, divisible by 4)
    # → 8 batches total, 2 per rank, 21 + 3 (pad) = 24 index slots
    # All original 21 indices appear at least once
    assert set(all_indices) == set(range(21))


def test_distributed_disjoint_real_batches():
    """Real (non-pad) batches are assigned to distinct ranks."""
    atom_counts = [2] * 20  # → 5 batches of 4 (max_atoms=8), world_size=5
    ds = _FakeDataset(atom_counts)
    world_size = 5

    per_rank = []
    for rank in range(world_size):
        sampler = MaxAtomDistributedBatchSampler(
            ds, max_atoms=8, num_replicas=world_size, rank=rank, shuffle=False
        )
        per_rank.append(list(sampler))

    # 5 batches / 5 ranks → 1 batch per rank, no padding needed
    all_flat = sorted(sum([sum(batches, []) for batches in per_rank], []))
    assert all_flat == list(range(20))


def test_distributed_len_consistent():
    """__len__ matches the actual number of batches yielded."""
    atom_counts = [3] * 17
    ds = _FakeDataset(atom_counts)
    for world_size in [1, 2, 3, 4]:
        for rank in range(world_size):
            sampler = MaxAtomDistributedBatchSampler(
                ds, max_atoms=9, num_replicas=world_size, rank=rank, shuffle=False
            )
            assert len(sampler) == len(list(sampler))


def test_distributed_set_epoch_sync():
    """All ranks apply the same shuffle (same epoch seed) and get complementary
    batches."""
    atom_counts = [2] * 12  # 6 batches of 2 (max_atoms=4)
    ds = _FakeDataset(atom_counts)
    world_size = 2

    sampler_r0_e0 = MaxAtomDistributedBatchSampler(
        ds, max_atoms=4, num_replicas=world_size, rank=0, shuffle=True
    )
    sampler_r1_e0 = MaxAtomDistributedBatchSampler(
        ds, max_atoms=4, num_replicas=world_size, rank=1, shuffle=True
    )
    sampler_r0_e1 = MaxAtomDistributedBatchSampler(
        ds, max_atoms=4, num_replicas=world_size, rank=0, shuffle=True
    )
    sampler_r0_e1.set_epoch(1)

    batches_r0_e0 = list(sampler_r0_e0)
    batches_r1_e0 = list(sampler_r1_e0)
    batches_r0_e1 = list(sampler_r0_e1)

    # Union of ranks at epoch 0 covers all 12 samples
    union_e0 = sorted(sum(batches_r0_e0 + batches_r1_e0, []))
    assert union_e0 == list(range(12))

    # Epoch 1 gives a different order for rank 0 than epoch 0
    assert batches_r0_e0 != batches_r0_e1


# ---------------------------------------------------------------------------
# Equivalence with fixed batch_size when atom counts are uniform
# ---------------------------------------------------------------------------


def test_equivalent_to_fixed_batch_size_uniform_atoms():
    """With uniform atom counts and no shuffle, batches match fixed batch_size."""
    n, n_atoms, batch_size = 15, 4, 3
    ds = _FakeDataset([n_atoms] * n)
    sampler = MaxAtomBatchSampler(ds, max_atoms=n_atoms * batch_size, shuffle=False)
    batches = list(sampler)

    expected = [list(range(i, i + batch_size)) for i in range(0, n, batch_size)]
    assert batches == expected
    assert len(batches) == n // batch_size  # exactly n/batch_size full batches


# ---------------------------------------------------------------------------
# Stable packing (new behaviour: packing fixed at init, only order varies)
# ---------------------------------------------------------------------------


def test_packing_stable_across_epochs():
    """The set of batches (structure groupings) is identical across epochs.

    Only the order in which they are presented changes per epoch.
    """
    atom_counts = [2] * 20
    ds = _FakeDataset(atom_counts)
    sampler = MaxAtomBatchSampler(ds, max_atoms=6, shuffle=True)

    sampler.set_epoch(0)
    batches_e0 = list(sampler)
    sampler.set_epoch(1)
    batches_e1 = list(sampler)

    # Contents are the same set of batches, just in a different order.
    assert sorted(map(sorted, batches_e0)) == sorted(map(sorted, batches_e1))
    # But the presentation order differs (with overwhelming probability).
    assert batches_e0 != batches_e1


# ---------------------------------------------------------------------------
# drop_last
# ---------------------------------------------------------------------------


def test_drop_last_single_process():
    """drop_last=True drops tail batches instead of padding."""
    # 7 batches of 3 (21 structures, 3 atoms each, max_atoms=9), world_size=4
    # Without drop_last: padded to 8 → 2 per rank.
    # With drop_last: truncated to 4 → 1 per rank.
    atom_counts = [3] * 21
    ds = _FakeDataset(atom_counts)
    world_size = 4

    samplers_drop = [
        MaxAtomDistributedBatchSampler(
            ds,
            max_atoms=9,
            num_replicas=world_size,
            rank=r,
            shuffle=False,
            drop_last=True,
        )
        for r in range(world_size)
    ]

    for s in samplers_drop:
        assert len(s) == 1  # floor(7/4) = 1

    all_batches = [b for s in samplers_drop for b in s]
    assert len(all_batches) == world_size  # 4 batches total

    # All yielded indices are valid dataset indices.
    for batch in all_batches:
        for idx in batch:
            assert 0 <= idx < len(ds)


def test_drop_last_no_remainder_unchanged():
    """drop_last has no effect when batch count is already divisible by num_replicas."""
    atom_counts = [2] * 20  # → 5 batches (max_atoms=8), world_size=5
    ds = _FakeDataset(atom_counts)
    world_size = 5

    s_pad = MaxAtomDistributedBatchSampler(
        ds, max_atoms=8, num_replicas=world_size, rank=0, shuffle=False, drop_last=False
    )
    s_drop = MaxAtomDistributedBatchSampler(
        ds, max_atoms=8, num_replicas=world_size, rank=0, shuffle=False, drop_last=True
    )
    assert len(s_pad) == len(s_drop) == 1
    assert list(s_pad) == list(s_drop)


def test_max_atom_batch_sampler_drop_last():
    """MaxAtomBatchSampler forwards drop_last correctly."""
    # 7 batches (21 structures, max_atoms=9): drop_last has no effect for
    # num_replicas=1.
    atom_counts = [3] * 21
    ds = _FakeDataset(atom_counts)
    s = MaxAtomBatchSampler(ds, max_atoms=9, shuffle=False, drop_last=True)
    assert len(list(s)) == 7  # all 7 batches; floor(7/1) == 7


# ---------------------------------------------------------------------------
# Too-few-batches error
# ---------------------------------------------------------------------------


def test_too_few_batches_raises():
    """ValueError when packed batches < num_replicas."""
    # 3 structures, 1 atom each, max_atoms=10 → 1 batch; num_replicas=4 → fails.
    ds = _FakeDataset([1] * 3)
    with pytest.raises(ValueError, match="num_replicas"):
        MaxAtomDistributedBatchSampler(ds, max_atoms=10, num_replicas=4, rank=0)


# ---------------------------------------------------------------------------
# Padding with odd batch count
# ---------------------------------------------------------------------------


def test_padding_when_n_batches_not_divisible():
    """Padding is correct when n_batches is not divisible by num_replicas."""
    # 9 structures × 3 atoms, max_atoms=9 → 3 batches; world_size=2
    # remainder=1, padding_size=1 → total_size=4, num_samples=2 per rank
    atom_counts = [3] * 9
    ds = _FakeDataset(atom_counts)
    world_size = 2
    samplers = [
        MaxAtomDistributedBatchSampler(
            ds, max_atoms=9, num_replicas=world_size, rank=r, shuffle=False
        )
        for r in range(world_size)
    ]

    for s in samplers:
        assert len(s) == 2  # ceil(3/2) = 2

    # All 9 original indices appear in the union (one batch is repeated for padding).
    all_indices = [idx for s in samplers for b in s for idx in b]
    assert set(all_indices) == set(range(9))


# ---------------------------------------------------------------------------
# Cross-validation: batch contents match a fixed batch_size DataLoader
# ---------------------------------------------------------------------------


def test_max_atom_sampler_cross_validation_batch_contents(tmp_path):
    """With uniform atom counts, MaxAtomBatchSampler yields identical batch contents
    to a fixed batch_size DataLoader (both shuffle=False) on a real MemmapDataset.

    Given that test_equivalent_to_fixed_batch_size_uniform_atoms already proves index
    equivalence, this test proves the full pipeline
    (sampler → MemmapDataset.__getitem__) is equivalent, validating that the sampler
    does not corrupt data ordering or content.
    """
    rng = np.random.default_rng(0)
    ns, atoms_per = 12, 4
    total = ns * atoms_per
    na = np.arange(0, total + 1, atoms_per, dtype=np.int64)

    np.save(tmp_path / "ns.npy", ns)
    np.save(tmp_path / "na.npy", na)
    rng.uniform(0, 3, (total, 3)).astype("float32").tofile(tmp_path / "x.bin")
    np.full(total, 6, dtype="int32").tofile(tmp_path / "a.bin")
    np.zeros((ns, 3, 3), dtype="float32").tofile(tmp_path / "c.bin")
    rng.standard_normal((ns, 1)).astype("float32").tofile(tmp_path / "e.bin")

    dataset = MemmapDataset(
        tmp_path,
        {
            "energy": {
                "key": "e",
                "sample_kind": "system",
                "num_subtargets": 1,
                "type": "scalar",
                "quantity": "energy",
                "forces": False,
                "stress": False,
                "virial": False,
            }
        },
    )

    batch_size = 3  # 3 structures × 4 atoms = 12 atoms per batch

    def collate(samples):
        positions = torch.cat([s.system.positions for s in samples])
        energies = torch.cat([s.energy.block().values for s in samples])
        return positions, energies

    loader_fixed = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, collate_fn=collate
    )
    sampler = MaxAtomBatchSampler(
        dataset, max_atoms=batch_size * atoms_per, shuffle=False
    )
    loader_maxatom = DataLoader(dataset, batch_sampler=sampler, collate_fn=collate)

    batches_fixed = list(loader_fixed)
    batches_maxatom = list(loader_maxatom)

    assert len(batches_fixed) == len(batches_maxatom) == ns // batch_size
    for (pos_fixed, e_fixed), (pos_maxatom, e_maxatom) in zip(
        batches_fixed, batches_maxatom, strict=True
    ):
        torch.testing.assert_close(pos_fixed, pos_maxatom)
        torch.testing.assert_close(e_fixed, e_maxatom)
