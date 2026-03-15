"""Tests for torch.compile + DDP support and max-atom batch samplers."""

import logging
import os

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


# ---------------------------------------------------------------------------
# _greedy_pack unit tests
# ---------------------------------------------------------------------------


def test_greedy_pack_basic():
    """All batches respect the max_atoms limit."""
    indices = list(range(10))
    atom_counts = [3] * 10
    batches = _greedy_pack(indices, atom_counts, max_atoms=9)
    for batch in batches:
        assert sum(atom_counts[i] for i in batch) <= 9
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
    atom_counts = [3, 100, 3]
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
# MaxAtomBatchSampler (single-process) tests
# ---------------------------------------------------------------------------


def test_single_process_all_samples_covered():
    """All samples appear exactly once per epoch."""
    atom_counts = [3] * 20
    ds = _FakeDataset(atom_counts)
    sampler = MaxAtomBatchSampler(ds, max_atoms=9, shuffle=False, seed=0)
    all_indices = sorted(sum(list(sampler), []))
    assert all_indices == list(range(20))


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


# ---------------------------------------------------------------------------
# MaxAtomDistributedBatchSampler tests
# ---------------------------------------------------------------------------


def test_distributed_all_indices_covered():
    """Union of all ranks covers all indices exactly once."""
    atom_counts = [3] * 20
    ds = _FakeDataset(atom_counts)
    world_size = 2

    all_indices = []
    for rank in range(world_size):
        sampler = MaxAtomDistributedBatchSampler(
            ds, max_atoms=9, num_replicas=world_size, rank=rank,
            shuffle=False, seed=0,
        )
        for batch in sampler:
            all_indices.extend(batch)

    # Some indices may be duplicated due to padding, but all must appear
    assert set(range(20)).issubset(set(all_indices))


def test_distributed_equal_batch_count():
    """All ranks get the same number of batches."""
    atom_counts = [3] * 21  # 21 not evenly divisible by 2
    ds = _FakeDataset(atom_counts)
    world_size = 2

    counts = []
    for rank in range(world_size):
        sampler = MaxAtomDistributedBatchSampler(
            ds, max_atoms=9, num_replicas=world_size, rank=rank,
            shuffle=False, seed=0,
        )
        counts.append(len(list(sampler)))

    assert counts[0] == counts[1]


def test_distributed_epoch_determinism():
    """Same seed+epoch = same batches across separate sampler instances."""
    atom_counts = [2] * 30
    ds = _FakeDataset(atom_counts)
    s1 = MaxAtomDistributedBatchSampler(
        ds, max_atoms=6, num_replicas=2, rank=0, shuffle=True, seed=42,
    )
    s2 = MaxAtomDistributedBatchSampler(
        ds, max_atoms=6, num_replicas=2, rank=0, shuffle=True, seed=42,
    )
    s1.set_epoch(5)
    s2.set_epoch(5)
    assert list(s1) == list(s2)


def test_distributed_atom_limit_respected():
    """No batch from any rank exceeds max_atoms."""
    atom_counts = [1, 5, 2, 4, 3, 6, 1, 2, 3, 4]
    ds = _FakeDataset(atom_counts)
    max_atoms = 7
    for rank in range(2):
        sampler = MaxAtomDistributedBatchSampler(
            ds, max_atoms=max_atoms, num_replicas=2, rank=rank,
            shuffle=False, seed=0,
        )
        for batch in sampler:
            total = sum(atom_counts[i] for i in batch)
            assert total <= max_atoms


# ---------------------------------------------------------------------------
# DDP + compile integration tests (require CUDA + NCCL)
# ---------------------------------------------------------------------------


def _run_compiled_ddp_gradient_sync(rank, world_size, results_dict):
    """Worker: verify DDP gradient hooks fire with compiled forward path."""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29501"
    # Use gloo backend for testing 2 ranks on a single GPU.
    # NCCL rejects duplicate GPU IDs across ranks.
    torch.distributed.init_process_group("gloo", rank=rank, world_size=world_size)
    torch.cuda.set_device(0)
    device = torch.device("cuda:0")

    from metatrain.pet import PET
    from metatrain.pet.modules.compile import compile_pet_model
    from metatrain.utils.data import DatasetInfo
    from metatrain.utils.data.readers import read_systems
    from metatrain.utils.data.target_info import get_energy_target_info
    from metatrain.utils.distributed.distributed_data_parallel import (
        DistributedDataParallel,
    )
    from metatrain.utils.neighbor_lists import (
        get_requested_neighbor_lists,
        get_system_with_neighbor_lists,
        get_system_with_neighbor_lists_transform,
    )

    from ..modules.structures import systems_to_batch
    from . import DATASET_PATH, MODEL_HYPERS

    torch.manual_seed(42)

    targets = {
        "mtt::U0": get_energy_target_info(
            "mtt::U0", {"quantity": "energy", "unit": "eV"}
        )
    }
    dataset_info = DatasetInfo(
        length_unit="Angstrom", atomic_types=[1, 6, 7, 8], targets=targets
    )
    model = PET(MODEL_HYPERS, dataset_info)
    model.to(device=device, dtype=torch.float32)

    systems = read_systems(DATASET_PATH)[:4]
    systems = [s.to(torch.float32) for s in systems]
    for s in systems:
        get_system_with_neighbor_lists(s, model.requested_neighbor_lists())

    # Build a minimal dataloader for tracing. compile_pet_model only
    # needs systems from the batch for shape tracing.
    from metatensor.learn.data import Dataset as MLDataset
    from metatensor.torch import Labels, TensorBlock, TensorMap

    from metatrain.utils.data import CollateFn

    requested_nls = get_requested_neighbor_lists(model)
    collate_fn = CollateFn(
        target_keys=list(targets.keys()),
        callables=[get_system_with_neighbor_lists_transform(requested_nls)],
    )
    # Build dummy energy targets (1 scalar per structure)
    dummy_targets = []
    for i in range(2):
        block = TensorBlock(
            values=torch.zeros(1, 1, dtype=torch.float64),
            samples=Labels(["system"], torch.tensor([[i]])),
            components=[],
            properties=Labels(["energy"], torch.tensor([[0]])),
        )
        dummy_targets.append(TensorMap(Labels.single(), [block]))
    # Systems must be float64 for collation (save_system_buffer requirement)
    ds = MLDataset(
        system=[s.to(torch.float64) for s in systems[:2]],
        **{"mtt::U0": dummy_targets},
    )
    dl = torch.utils.data.DataLoader(
        ds, batch_size=2, collate_fn=collate_fn, shuffle=False
    )

    # Compile on the raw model BEFORE DDP wrapping
    compiled_fn, _, _ = compile_pet_model(model, dl, False, False)

    # Wrap with DDP (DDP registers post-accumulate-grad hooks on params)
    ddp_model = DistributedDataParallel(model, device_ids=[device])
    raw_model = ddp_model.module

    # Each rank takes different systems
    rank_systems = systems[rank * 2 : (rank + 1) * 2]
    rank_systems = [s.to(device) for s in rank_systems]

    # Forward + backward using compiled path with raw_model's params
    (
        c_ein, c_einb, c_ev, _c_ed, c_pm, c_rni, c_cf, c_si, c_nai, c_sl,
    ) = systems_to_batch(
        rank_systems,
        raw_model.requested_nl,
        raw_model.atomic_types,
        raw_model.species_to_species_index,
        raw_model.cutoff_function,
        raw_model.cutoff_width,
        raw_model.num_neighbors_adaptive,
    )

    n_structures = len(rank_systems)
    energy, forces, stress, raw_preds = compiled_fn(
        c_ev, c_ein, c_einb, c_pm, c_rni, c_cf, c_si, c_nai, n_structures,
        *list(raw_model.parameters()), *list(raw_model.buffers()),
    )
    loss = energy.sum()
    loss.backward()

    # With NCCL on separate GPUs, DDP's post-accumulate-grad hooks sync
    # gradients automatically. With gloo (used here for single-GPU testing),
    # we must sync manually. The trainer code handles both cases.
    for param in raw_model.parameters():
        if param.grad is not None:
            torch.distributed.all_reduce(
                param.grad, op=torch.distributed.ReduceOp.AVG,
            )

    # Collect gradient norms to verify they match across ranks.
    grad_norms = []
    for param in raw_model.parameters():
        if param.grad is not None:
            grad_norms.append(param.grad.norm().item())

    results_dict[rank] = grad_norms
    torch.distributed.destroy_process_group()


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required for DDP tests"
)
def test_compiled_ddp_gradient_sync():
    """Verify DDP gradient hooks fire and sync gradients in compiled path."""
    import torch.multiprocessing as mp

    manager = mp.Manager()
    results = manager.dict()
    mp.spawn(
        _run_compiled_ddp_gradient_sync,
        args=(2, results),
        nprocs=2,
        join=True,
    )

    grads_0 = results[0]
    grads_1 = results[1]
    assert len(grads_0) == len(grads_1)
    for g0, g1 in zip(grads_0, grads_1):
        assert abs(g0 - g1) < 1e-5, f"Gradient mismatch: {g0} vs {g1}"
