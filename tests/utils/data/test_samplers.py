"""Tests for MaxAtom batch samplers."""

import logging

import pytest
import torch
import torch.utils.data

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
    sampler = MaxAtomBatchSampler(ds, max_atoms=9, shuffle=False, seed=0)
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
    sampler = MaxAtomBatchSampler(ds, max_atoms=6, shuffle=True, seed=42)

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
    sampler = MaxAtomBatchSampler(ds, max_atoms=6, shuffle=True, seed=7)

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
    """All ranks apply the same shuffle (same epoch seed) and get complementary batches."""
    atom_counts = [2] * 12  # 6 batches of 2 (max_atoms=4)
    ds = _FakeDataset(atom_counts)
    world_size = 2

    sampler_r0_e0 = MaxAtomDistributedBatchSampler(
        ds, max_atoms=4, num_replicas=world_size, rank=0, shuffle=True, seed=0
    )
    sampler_r1_e0 = MaxAtomDistributedBatchSampler(
        ds, max_atoms=4, num_replicas=world_size, rank=1, shuffle=True, seed=0
    )
    sampler_r0_e1 = MaxAtomDistributedBatchSampler(
        ds, max_atoms=4, num_replicas=world_size, rank=0, shuffle=True, seed=0
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
    sampler = MaxAtomBatchSampler(
        ds, max_atoms=n_atoms * batch_size, shuffle=False, seed=0
    )
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
    sampler = MaxAtomBatchSampler(ds, max_atoms=6, shuffle=True, seed=0)

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
            ds, max_atoms=9, num_replicas=world_size, rank=r,
            shuffle=False, drop_last=True,
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

    s_pad  = MaxAtomDistributedBatchSampler(
        ds, max_atoms=8, num_replicas=world_size, rank=0, shuffle=False, drop_last=False
    )
    s_drop = MaxAtomDistributedBatchSampler(
        ds, max_atoms=8, num_replicas=world_size, rank=0, shuffle=False, drop_last=True
    )
    assert len(s_pad) == len(s_drop) == 1
    assert list(s_pad) == list(s_drop)


def test_max_atom_batch_sampler_drop_last():
    """MaxAtomBatchSampler forwards drop_last correctly."""
    # 7 batches (21 structures, max_atoms=9): drop_last has no effect for num_replicas=1.
    atom_counts = [3] * 21
    ds = _FakeDataset(atom_counts)
    s = MaxAtomBatchSampler(ds, max_atoms=9, shuffle=False, drop_last=True)
    assert len(list(s)) == 7  # all 7 batches; floor(7/1) == 7


# ---------------------------------------------------------------------------
# set_epoch_and_start_iteration (mid-epoch resumption)
# ---------------------------------------------------------------------------

def test_start_iter_skips_leading_batches():
    """set_epoch_and_start_iteration skips the first start_iter batches."""
    atom_counts = [2] * 20  # 5 batches of 4 (max_atoms=8)
    ds = _FakeDataset(atom_counts)
    sampler = MaxAtomBatchSampler(ds, max_atoms=8, shuffle=False)

    full = list(sampler)  # 5 batches

    sampler.set_epoch_and_start_iteration(0, 2)
    resumed = list(sampler)

    assert resumed == full[2:]


def test_start_iter_reset_by_set_epoch():
    """set_epoch resets start_iter to 0."""
    atom_counts = [2] * 20
    ds = _FakeDataset(atom_counts)
    sampler = MaxAtomBatchSampler(ds, max_atoms=8, shuffle=False)

    sampler.set_epoch_and_start_iteration(0, 3)
    assert sampler.start_iter == 3

    sampler.set_epoch(1)
    assert sampler.start_iter == 0
    assert len(list(sampler)) == len(sampler)


# ---------------------------------------------------------------------------
# Too-few-batches assertion
# ---------------------------------------------------------------------------

def test_too_few_batches_raises():
    """AssertionError when packed batches < num_replicas."""
    # 3 structures, 1 atom each, max_atoms=10 → 1 batch; num_replicas=4 → fails.
    ds = _FakeDataset([1] * 3)
    with pytest.raises(AssertionError, match="num_replicas"):
        MaxAtomDistributedBatchSampler(ds, max_atoms=10, num_replicas=4, rank=0)


# ---------------------------------------------------------------------------
# Padding with n_batches only slightly above num_replicas
# ---------------------------------------------------------------------------

def test_padding_when_n_batches_just_above_num_replicas():
    """Padding is correct when n_batches = num_replicas + 1."""
    # 9 structures × 3 atoms, max_atoms=9 → 3 batches; world_size=2
    # → num_samples=2, total_size=4, padding_size=1
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
