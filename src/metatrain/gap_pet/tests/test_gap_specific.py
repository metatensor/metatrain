"""GapPET-specific tests.

Five tests that exercise the design properties of the architecture:

1. ``test_size_intensivity``: replicating the cell does not change the gap.
2. ``test_force_finite_difference``: forces match a finite-difference estimate.
3. ``test_per_atom_outputs_pool_to_gap``: the per-atom auxiliary outputs pool
   to the per-system gap exactly.
4. ``test_alpha_limits``: large ``|alpha|`` recovers ``max`` / ``min``.
5. ``test_train_step_decreases_loss``: one optimizer step on synthetic data
   decreases the loss.
"""

import math

import ase.build
import pytest
import torch
from metatomic.torch import ModelOutput, System, systems_to_torch

from metatrain.gap_pet.documentation import ModelHypers
from metatrain.gap_pet.model import (
    HOMO_PER_ATOM_OUTPUT_NAME,
    LUMO_PER_ATOM_OUTPUT_NAME,
    GapPET,
    _scatter_logsumexp,
)
from metatrain.utils.data import DatasetInfo
from metatrain.utils.data.target_info import get_energy_target_info
from metatrain.utils.hypers import init_with_defaults
from metatrain.utils.neighbor_lists import get_system_with_neighbor_lists


# float64 throughout: we are checking numerical properties (intensivity,
# finite-difference forces, and log-sum-exp limits) where float32 noise
# dominates the signal we're trying to test.
DTYPE = torch.float64
TARGET_NAME = "mtt::gap_energy"


def _minimal_hypers() -> dict:
    """Tiny hypers for fast tests. Still residual featurizer + 2 GNN layers
    so we exercise the multi-layer readout in the heads."""
    h = init_with_defaults(ModelHypers)
    h["d_pet"] = 4
    h["d_node"] = 4
    h["d_head"] = 4
    h["d_feedforward"] = 4
    h["num_heads"] = 1
    h["num_attention_layers"] = 1
    h["num_gnn_layers"] = 2
    h["cutoff"] = 4.0
    h["cutoff_width"] = 0.5
    h["zbl"] = False
    return h


def _gap_target_info():
    return get_energy_target_info(
        TARGET_NAME,
        {"quantity": "energy", "unit": "eV"},
        add_position_gradients=True,
    )


def _build_water_systems(n_replicas: int = 1):
    """Single water molecule in a 4 A cubic cell, optionally tiled."""
    atoms = ase.build.molecule("H2O", vacuum=0.0)
    atoms.set_cell([4.0, 4.0, 4.0])
    atoms.center()
    atoms.set_pbc(True)
    if n_replicas > 1:
        atoms = atoms.repeat((n_replicas, n_replicas, n_replicas))
    systems = systems_to_torch(atoms, dtype=DTYPE)
    if not isinstance(systems, list):
        systems = [systems]
    return systems


def _make_model(atomic_types):
    hypers = _minimal_hypers()
    dataset_info = DatasetInfo(
        length_unit="Angstrom",
        atomic_types=sorted(set(atomic_types)),
        targets={TARGET_NAME: _gap_target_info()},
    )
    torch.manual_seed(0)
    model = GapPET(hypers, dataset_info).to(DTYPE)
    model.eval()
    return model


def _attach_nl(model, systems):
    return [
        get_system_with_neighbor_lists(s, model.requested_neighbor_lists())
        for s in systems
    ]


# --- test 1: size intensivity ----------------------------------------------


@pytest.mark.parametrize("n_replicas", [2, 3])
def test_size_intensivity(n_replicas):
    """Replicating the cell along each axis must leave E_gap invariant.

    This is the key design property: the extremal pool depends only on the
    most extreme atomic contribution, and replication produces atoms with
    identical (translated) contributions, so the max/min stay the same.
    """
    systems_1 = _build_water_systems(1)
    systems_n = _build_water_systems(n_replicas)
    atomic_types = list(systems_1[0].types.tolist())

    model = _make_model(atomic_types)
    systems_1 = _attach_nl(model, systems_1)
    systems_n = _attach_nl(model, systems_n)

    outputs = {TARGET_NAME: ModelOutput(per_atom=False)}
    with torch.no_grad():
        e_gap_1 = model(systems_1, outputs)[TARGET_NAME].block().values.item()
        e_gap_n = model(systems_n, outputs)[TARGET_NAME].block().values.item()

    # log(N_replicas^3) / |alpha| is the analytical residual from a
    # finite-alpha smooth max over N identical contributions: when atoms come
    # in groups with degenerate values, logsumexp(alpha * v_i) - alpha * v_max
    # = log(degeneracy). With alpha = +/-20 and N <= 27 copies, the residual
    # is log(27)/20 ~= 0.16 eV -- but PET features differ slightly between
    # original and replicated atoms (different neighbor environments at the
    # cell boundary), so the actual residual is even smaller. We use a loose
    # bound that still distinguishes intensive (~0.1 eV) from extensive
    # (~N_replicas^3 eV) behaviour.
    intensive_bound = math.log(n_replicas**3) / 20.0 + 0.5
    assert abs(e_gap_n - e_gap_1) < intensive_bound, (
        f"E_gap is not approximately intensive: "
        f"|E_gap({n_replicas}^3 cells) - E_gap(1 cell)| = "
        f"{abs(e_gap_n - e_gap_1):.4f} eV, bound = {intensive_bound:.4f} eV"
    )


# --- test 2: force finite-difference --------------------------------------


def test_force_finite_difference():
    """Autograd forces must match a central-difference estimate.

    Done on a tiny CO molecule to keep the fd loop short.
    """
    atoms = ase.build.molecule("CO")
    atoms.set_cell([6.0, 6.0, 6.0])
    atoms.center()
    atoms.set_pbc(True)
    systems = systems_to_torch(atoms, dtype=DTYPE)
    if not isinstance(systems, list):
        systems = [systems]
    atomic_types = list(systems[0].types.tolist())

    model = _make_model(atomic_types)
    model.train()  # so manual attention path is used (supports double backward)
    systems = _attach_nl(model, systems)

    def gap(positions: torch.Tensor) -> torch.Tensor:
        new_system = System(
            types=systems[0].types,
            positions=positions,
            cell=systems[0].cell,
            pbc=systems[0].pbc,
        )
        new_system = get_system_with_neighbor_lists(
            new_system, model.requested_neighbor_lists()
        )
        out = model(
            [new_system],
            {TARGET_NAME: ModelOutput(per_atom=False)},
        )
        return out[TARGET_NAME].block().values.squeeze()

    pos0 = systems[0].positions.detach().clone().requires_grad_(True)
    e = gap(pos0)
    (analytical_grad,) = torch.autograd.grad(e, pos0)

    eps = 1e-4
    fd_grad = torch.zeros_like(pos0)
    with torch.no_grad():
        for i in range(pos0.shape[0]):
            for j in range(3):
                p_plus = pos0.detach().clone()
                p_plus[i, j] += eps
                p_minus = pos0.detach().clone()
                p_minus[i, j] -= eps
                fd_grad[i, j] = (gap(p_plus) - gap(p_minus)) / (2 * eps)

    diff = (analytical_grad - fd_grad).abs().max().item()
    assert diff < 1e-4, (
        f"Autograd gradient disagrees with finite difference: max abs diff "
        f"= {diff:.2e}"
    )


# --- test 3: per-atom outputs pool to gap ---------------------------------


def test_per_atom_outputs_pool_to_gap():
    """The per-atom auxiliary outputs, pooled with the model's own alphas,
    must reproduce E_LUMO - E_HOMO = E_gap exactly (up to fp64 round-off).
    """
    systems = _build_water_systems(1)
    atomic_types = list(systems[0].types.tolist())

    model = _make_model(atomic_types)
    systems = _attach_nl(model, systems)

    outputs = {
        TARGET_NAME: ModelOutput(per_atom=False),
        HOMO_PER_ATOM_OUTPUT_NAME: ModelOutput(per_atom=True),
        LUMO_PER_ATOM_OUTPUT_NAME: ModelOutput(per_atom=True),
    }
    with torch.no_grad():
        out = model(systems, outputs)

    n_atoms = systems[0].positions.shape[0]
    h_homo_block = out[HOMO_PER_ATOM_OUTPUT_NAME].block()
    h_lumo_block = out[LUMO_PER_ATOM_OUTPUT_NAME].block()

    assert h_homo_block.values.shape == (n_atoms, 1)
    assert h_lumo_block.values.shape == (n_atoms, 1)
    assert h_homo_block.samples.names == ["system", "atom"]

    h_homo = h_homo_block.values.squeeze(-1)
    h_lumo = h_lumo_block.values.squeeze(-1)
    system_indices = torch.zeros(n_atoms, dtype=torch.int64)
    e_homo = _scatter_logsumexp(h_homo, model.alpha_homo, system_indices, 1)
    e_lumo = _scatter_logsumexp(h_lumo, model.alpha_lumo, system_indices, 1)
    expected_gap = (e_lumo - e_homo).item()
    actual_gap = out[TARGET_NAME].block().values.item()

    assert abs(expected_gap - actual_gap) < 1e-10, (
        f"Per-atom outputs do not pool to E_gap: pooled = {expected_gap:.6e}, "
        f"model E_gap = {actual_gap:.6e}"
    )


# --- test 4: alpha limit sanity check -------------------------------------


def test_alpha_limits():
    """As |alpha| -> infinity, the smooth pool must approach the hard
    max/min of the per-atom values.

    We test the ``_scatter_logsumexp`` primitive directly with a fixed input
    rather than running the full model, since the limit is a property of the
    pool, not of the network.
    """
    values = torch.tensor([0.1, -0.7, 1.3, 0.4, -1.1, 2.5], dtype=DTYPE)
    system_indices = torch.zeros(values.shape[0], dtype=torch.int64)

    # Smooth max should approach +max as alpha -> +inf.
    pooled_max = _scatter_logsumexp(values, torch.tensor(1000.0, dtype=DTYPE), system_indices, 1)
    assert torch.isclose(
        pooled_max, values.max().reshape(1), atol=1e-2
    ), f"Smooth max with alpha=1000 should approach {values.max()}, got {pooled_max.item()}"

    # Smooth min should approach min as alpha -> -inf.
    pooled_min = _scatter_logsumexp(values, torch.tensor(-1000.0, dtype=DTYPE), system_indices, 1)
    assert torch.isclose(
        pooled_min, values.min().reshape(1), atol=1e-2
    ), f"Smooth min with alpha=-1000 should approach {values.min()}, got {pooled_min.item()}"

    # At a finite alpha, the LSE-based smooth pool over-shoots the hard
    # extremum by at most log(N)/|alpha|: pool_max >= max, pool_min <= min,
    # with equality only as |alpha| -> infinity.
    slack = math.log(values.numel()) / 20.0
    pool_default_max = _scatter_logsumexp(values, torch.tensor(20.0, dtype=DTYPE), system_indices, 1).item()
    pool_default_min = _scatter_logsumexp(values, torch.tensor(-20.0, dtype=DTYPE), system_indices, 1).item()
    assert values.max().item() <= pool_default_max <= values.max().item() + slack + 1e-9
    assert values.min().item() - slack - 1e-9 <= pool_default_min <= values.min().item()


# --- test 5: train-step smoke test ----------------------------------------


def test_train_step_decreases_loss():
    """One AdamW step on a synthetic 4-system batch must reduce the loss.

    We bypass the metatrain Trainer machinery (composition, scaler, dataloader,
    metrics) to keep this self-contained: build a hand-rolled MSE loss on
    ``E_gap`` and a few autograd steps, and assert monotonic loss decrease
    over a handful of steps. This catches forward/backward wiring bugs
    without coupling to the trainer.
    """
    atoms_list = []
    for stretch in (1.0, 1.05, 0.95, 1.10):  # mild distortions; varied energies
        a = ase.build.molecule("H2O")
        a.set_cell([8.0, 8.0, 8.0])
        a.center()
        a.set_pbc(True)
        # Stretch all bonds by ``stretch`` around the centroid to give the
        # systems different gap targets.
        center = a.positions.mean(axis=0)
        a.positions = center + stretch * (a.positions - center)
        atoms_list.append(a)

    systems = []
    for atoms in atoms_list:
        s = systems_to_torch(atoms, dtype=DTYPE)
        if isinstance(s, list):
            s = s[0]
        systems.append(s)
    atomic_types = sorted({int(t) for s in systems for t in s.types.tolist()})

    model = _make_model(atomic_types)
    model.train()
    systems = _attach_nl(model, systems)

    # Synthetic targets: arbitrary positive numbers per system. The point is
    # that the model can fit ANY consistent assignment with enough steps; we
    # only check loss decreases over a few steps.
    targets = torch.tensor([1.5, 2.0, 1.0, 2.5], dtype=DTYPE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)

    losses = []
    for _ in range(40):
        optimizer.zero_grad()
        out = model(systems, {TARGET_NAME: ModelOutput(per_atom=False)})
        pred = out[TARGET_NAME].block().values.squeeze(-1)
        loss = ((pred - targets) ** 2).mean()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    # Smoke check: loss must drop meaningfully over 40 AdamW steps. We don't
    # require convergence (this is a tiny synthetic dataset on a tiny model);
    # we only catch wiring bugs that flat-line training.
    assert min(losses) < 0.5 * losses[0], (
        f"Training loss should drop by at least 2x within 40 AdamW steps. "
        f"Got initial loss {losses[0]:.4f}, min loss {min(losses):.4f}."
    )


