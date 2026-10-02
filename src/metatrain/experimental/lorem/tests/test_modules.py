"""``ShortRange`` / ``LongRange`` on their own. Parity with lorem-jax is in
``test_flax_checkpoint.py``."""

import torch
from metatomic.torch import NeighborListOptions, System

from metatrain.experimental.lorem.modules.lorem import (
    LongRange,
    ShortRange,
)
from metatrain.utils.neighbor_lists import get_system_with_neighbor_lists


CUTOFF = 3.0
MAX_DEGREE = 2
MAX_DEGREE_LR = 1
NUM_FEATURES = 4
NUM_RADIAL = 2
NUM_SPHERICAL_FEATURES = 2
NUM_SPECIES = 3


def _nlo(**kwargs):
    return NeighborListOptions(cutoff=CUTOFF, full_list=True, strict=True, **kwargs)


def _build_pair(seed):
    torch.manual_seed(seed)
    nlo = _nlo()
    backbone = ShortRange(
        cutoff=CUTOFF,
        max_degree=MAX_DEGREE,
        num_features=NUM_FEATURES,
        num_radial=NUM_RADIAL,
        num_spherical_features=NUM_SPHERICAL_FEATURES,
        num_species=NUM_SPECIES,
        atomic_types=[1, 6],
        neighbor_list_options=nlo,
    ).double()
    long_range = LongRange(
        feature_dim=NUM_FEATURES,
        num_spherical_features=NUM_SPHERICAL_FEATURES,
        max_degree=MAX_DEGREE,
        max_degree_lr=MAX_DEGREE_LR,
        neighbor_list_options=nlo,
        smearing=1.0,
        kspace_resolution=1.0,
    ).double()
    return backbone, long_range


def _chain_system(pbc):
    cell = torch.eye(3) * 10.0 if pbc else torch.zeros(3, 3)
    return System(
        types=torch.tensor([1, 6, 6, 1]),
        positions=torch.tensor(
            [[0.0, 0.0, 0.0], [0.0, 0.0, 1.2], [0.0, 0.0, 2.4], [0.0, 0.0, 3.6]],
            dtype=torch.float64,
        ),
        cell=cell.to(torch.float64),
        pbc=torch.tensor([pbc, pbc, pbc]),
    )


def test_short_range_forward_shapes():
    backbone, _ = _build_pair(seed=0)
    system = get_system_with_neighbor_lists(_chain_system(pbc=False), [_nlo()])
    nodes_scalar, distances, nodes_spherical, sr_energy, _snapshots = backbone([system])
    n_lm = (MAX_DEGREE + 1) ** 2
    assert nodes_scalar.shape == (4, NUM_FEATURES)
    assert nodes_spherical.shape == (4, n_lm, NUM_SPHERICAL_FEATURES)
    assert sr_energy.shape == (4,)
    assert distances.ndim == 1
    assert torch.isfinite(sr_energy).all()


def test_long_range_has_no_exclusion_radius():
    """lorem-jax builds Ewald with ``exclusion_radius=None``."""
    _, long_range = _build_pair(seed=0)
    assert long_range.ewald_calculator.potential.exclusion_radius is None
    assert long_range.direct_calculator.potential.exclusion_radius is None


def test_long_range_periodic_forward():
    backbone, long_range = _build_pair(seed=1)
    nlo = _nlo()
    system = get_system_with_neighbor_lists(_chain_system(pbc=True), [nlo])
    nodes_scalar, distances, nodes_spherical, _, _ = backbone([system])
    lr_energy, _updates = long_range([system], nodes_scalar, distances, nodes_spherical)
    assert lr_energy.shape == (4,)
    assert torch.isfinite(lr_energy).all()


def test_long_range_nonperiodic_forward():
    backbone, long_range = _build_pair(seed=2)
    nlo = _nlo()
    system = get_system_with_neighbor_lists(_chain_system(pbc=False), [nlo])
    nodes_scalar, distances, nodes_spherical, _, _ = backbone([system])
    lr_energy, _updates = long_range([system], nodes_scalar, distances, nodes_spherical)
    assert lr_energy.shape == (4,)
    assert torch.isfinite(lr_energy).all()


def test_energy_and_forces_are_finite():
    backbone, long_range = _build_pair(seed=3)
    nlo = _nlo()
    system = _chain_system(pbc=True)
    system.positions.requires_grad_(True)
    system = get_system_with_neighbor_lists(system, [nlo])
    nodes_scalar, distances, nodes_spherical, sr_energy, _ = backbone([system])
    lr_energy, _updates = long_range([system], nodes_scalar, distances, nodes_spherical)
    energy = (sr_energy + lr_energy).sum()
    (forces,) = torch.autograd.grad(energy, system.positions)
    assert torch.isfinite(energy).all()
    assert torch.isfinite(forces).all()


def _mixed_batch(full_list):
    """Triclinic periodic cells (one tiny, so k-grids differ), molecules and a
    lone atom, like the MAD batches."""
    g = torch.Generator().manual_seed(0)
    nlo = NeighborListOptions(cutoff=CUTOFF, full_list=full_list, strict=True)
    systems = []
    for n_atoms, pbc in (
        (5, [True] * 3),
        (3, [False] * 3),
        (1, [False] * 3),
        (2, [True] * 3),
        (7, [True] * 3),
    ):
        cell = torch.eye(3, dtype=torch.float64) * (2.0 + n_atoms) + 0.3 * torch.rand(
            3, 3, generator=g, dtype=torch.float64
        )
        positions = torch.rand(n_atoms, 3, generator=g, dtype=torch.float64) @ cell
        system = System(
            types=torch.tensor([1, 6, 6, 1, 6, 1, 6][:n_atoms]),
            positions=positions.requires_grad_(True),
            cell=cell if any(pbc) else torch.zeros(3, 3, dtype=torch.float64),
            pbc=torch.tensor(pbc),
        )
        systems.append(get_system_with_neighbor_lists(system, [nlo]))
    return systems, nlo


def test_batched_potentials_match_per_system_calls():
    # 0: every periodic system one by one; 400: only the largest k-grids
    for full_list, max_batched_pairs in ((True, 1 << 20), (False, 1 << 20), (True, 400), (True, 0)):
        systems, nlo = _mixed_batch(full_list)
        long_range = LongRange(
            feature_dim=NUM_FEATURES,
            num_spherical_features=NUM_SPHERICAL_FEATURES,
            max_degree=MAX_DEGREE,
            max_degree_lr=MAX_DEGREE_LR,
            neighbor_list_options=nlo,
            smearing=0.75,
            kspace_resolution=0.375,
        ).double()
        long_range.max_batched_pairs = max_batched_pairs
        n_atoms = sum(len(s) for s in systems)
        charges = torch.randn(n_atoms, 4, dtype=torch.float64, requires_grad=True)
        distances = torch.cat(
            [s.get_neighbor_list(nlo).values.squeeze(-1).norm(dim=-1) for s in systems]
        )
        results = []
        for potentials in (long_range._potentials, long_range._potentials_per_system):
            pot = potentials(systems, charges, distances)
            energy = (pot * charges).sum()
            grads = torch.autograd.grad(
                energy, [charges] + [s.positions for s in systems], create_graph=True
            )
            (second,) = torch.autograd.grad(sum((g**2).sum() for g in grads[1:]), charges)
            results.append((pot, *grads, second))
        for batched, reference in zip(*results):
            torch.testing.assert_close(batched, reference)
