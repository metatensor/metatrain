"""``JaxParityBackbone`` / ``JaxParityLongRange`` and their checkpoint loader.

These do not require a real lorem-jax checkpoint or any JAX/flax install --
the loader test builds a synthetic flax-shaped parameter tree by dumping a
freshly-initialized model's own weights (inverting the exact transforms
``load_checkpoint`` applies) and checks that loading it into a second model
reproduces the first model's parameters exactly. For validation against a
real archive checkpoint, see
``etc/lorem-parity/jax_checkpoint_parity/`` in the
`metawork <https://github.com/EricBoittier/metawork>`_ workspace.
"""

import torch
from metatomic.torch import NeighborListOptions, System

from metatrain.experimental.lorem.modules.e3x_compat import cg_phase_correction
from metatrain.experimental.lorem.modules.jax_parity import (
    JaxParityBackbone,
    JaxParityLongRange,
)
from metatrain.experimental.lorem.modules.jax_parity_checkpoint import load_checkpoint
from metatrain.experimental.lorem.modules.tensor_dense import _build_couplings
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
    backbone = JaxParityBackbone(
        cutoff=CUTOFF,
        max_degree=MAX_DEGREE,
        num_features=NUM_FEATURES,
        num_radial=NUM_RADIAL,
        num_spherical_features=NUM_SPHERICAL_FEATURES,
        num_species=NUM_SPECIES,
        atomic_types=[1, 6],
        neighbor_list_options=nlo,
    ).double()
    long_range = JaxParityLongRange(
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


def _dump_degree_wise(module, prefix, max_degree, flax):
    for ell in range(max_degree + 1):
        parity = "+" if ell % 2 == 0 else "-"
        layer = module.layers[ell]
        flax[f"{prefix}/{ell}{parity}/kernel"] = layer.weight.detach().T.clone()
        if layer.bias is not None:
            flax[f"{prefix}/{ell}{parity}/bias"] = layer.bias.detach().clone()


def _dump_tensor_weight(module, prefix, l1_max, l2_max, out_max, flax):
    n_channels = module.tensor_weight.shape[-1]
    grid = torch.zeros(
        l1_max + 1, l2_max + 1, out_max + 1, n_channels, dtype=torch.float64
    )
    _, l1_list, l2_list, L_list = _build_couplings(
        l1_max, l2_max, out_max, include_pseudotensors=False
    )
    for index, (l1, l2, L) in enumerate(zip(l1_list, l2_list, L_list, strict=True)):
        grid[l1, l2, L] = module.tensor_weight.detach()[index] * cg_phase_correction(
            l1, l2, L
        )
    flax[f"{prefix}/kernel"] = grid


def _dump_update(module, prefix, flax):
    flax[f"{prefix}/MLP_0/Dense_0/kernel"] = module.mlp0[0].weight.detach().T.clone()
    flax[f"{prefix}/MLP_0/Dense_0/bias"] = module.mlp0[0].bias.detach().clone()
    flax[f"{prefix}/MLP_0/Dense_1/kernel"] = module.mlp0[2].weight.detach().T.clone()
    flax[f"{prefix}/MLP_0/Dense_1/bias"] = module.mlp0[2].bias.detach().clone()
    flax[f"{prefix}/LayerNorm_0/scale"] = module.norm0.weight.detach().clone()
    flax[f"{prefix}/LayerNorm_0/bias"] = module.norm0.bias.detach().clone()
    flax[f"{prefix}/MLP_1/Dense_0/kernel"] = module.mlp1[0].weight.detach().T.clone()
    flax[f"{prefix}/MLP_1/Dense_0/bias"] = module.mlp1[0].bias.detach().clone()
    flax[f"{prefix}/MLP_1/Dense_1/kernel"] = module.mlp1[2].weight.detach().T.clone()
    flax[f"{prefix}/MLP_1/Dense_1/bias"] = module.mlp1[2].bias.detach().clone()
    flax[f"{prefix}/LayerNorm_1/scale"] = module.norm1.weight.detach().clone()
    flax[f"{prefix}/LayerNorm_1/bias"] = module.norm1.bias.detach().clone()


def _dump_mlp(module, prefix, flax):
    flax[f"{prefix}/Dense_0/kernel"] = module[0].weight.detach().T.clone()
    flax[f"{prefix}/Dense_0/bias"] = module[0].bias.detach().clone()
    flax[f"{prefix}/Dense_1/kernel"] = module[2].weight.detach().T.clone()
    flax[f"{prefix}/Dense_1/bias"] = module[2].bias.detach().clone()


def _dump_mlp3(module, prefix, flax):
    _dump_mlp(module, prefix, flax)
    flax[f"{prefix}/Dense_2/kernel"] = module[4].weight.detach().T.clone()
    flax[f"{prefix}/Dense_2/bias"] = module[4].bias.detach().clone()


def _dump_to_flax(backbone, long_range):
    """Inverse of ``load_checkpoint``: a synthetic flax tree matching the
    given (already-initialized) model's own parameters exactly."""
    flax = {}
    flax["Initial_0/ChemicalEmbedding_0/Embed_0/embedding"] = (
        backbone.chemical_embedding.weight.detach().clone()
    )
    flax["Dense_0/kernel"] = backbone.dense0.weight.detach().T.clone()
    flax["Dense_0/bias"] = backbone.dense0.bias.detach().clone()
    flax["Dense_1/kernel"] = backbone.dense1.weight.detach().T.clone()

    flax["RadialCoefficients_0/MLP_0/Dense_0/kernel"] = (
        backbone.radial_coefficients[0].weight.detach().T.clone()
    )
    flax["RadialCoefficients_0/MLP_0/Dense_0/bias"] = (
        backbone.radial_coefficients[0].bias.detach().clone()
    )
    flax["RadialCoefficients_0/MLP_0/Dense_1/kernel"] = (
        backbone.radial_coefficients[2].weight.detach().T.clone()
    )
    flax["RadialCoefficients_0/MLP_0/Dense_1/bias"] = (
        backbone.radial_coefficients[2].bias.detach().clone()
    )

    _dump_update(backbone.update0, "Update_0", flax)
    flax["Dense_2/kernel"] = backbone.dense2.weight.detach().T.clone()
    _dump_degree_wise(
        backbone.tensor_dense.dense, "TensorDense_0/dense", backbone.max_degree, flax
    )
    _dump_tensor_weight(
        backbone.tensor_dense,
        "TensorDense_0/tensor",
        backbone.max_degree,
        backbone.max_degree,
        backbone.max_degree,
        flax,
    )
    _dump_update(backbone.update1, "Update_1", flax)
    _dump_mlp3(backbone.energy_mlp, "MLP_0", flax)

    _dump_mlp(long_range.scalar_charge_mlp, "MLP_1", flax)
    _dump_degree_wise(
        long_range.spherical_charge_dense.dense,
        "TensorDense_1/dense",
        backbone.max_degree,
        flax,
    )
    _dump_tensor_weight(
        long_range.spherical_charge_dense,
        "TensorDense_1/tensor",
        backbone.max_degree,
        backbone.max_degree,
        long_range.max_degree_lr,
        flax,
    )
    _dump_degree_wise(
        long_range.potential_to_features, "Dense_3", long_range.max_degree_lr, flax
    )
    _dump_tensor_weight(
        long_range.potential_product,
        "Tensor_0",
        long_range.max_degree_lr,
        backbone.max_degree,
        backbone.max_degree,
        flax,
    )
    _dump_update(long_range.update2, "Update_2", flax)
    _dump_mlp3(long_range.energy_mlp, "MLP_2", flax)
    return flax


def test_jax_parity_backbone_forward_shapes():
    backbone, _ = _build_pair(seed=0)
    system = get_system_with_neighbor_lists(_chain_system(pbc=False), [_nlo()])
    nodes_scalar, distances, nodes_spherical, sr_energy = backbone([system])
    n_lm = (MAX_DEGREE + 1) ** 2
    assert nodes_scalar.shape == (4, NUM_FEATURES)
    assert nodes_spherical.shape == (4, n_lm, NUM_SPHERICAL_FEATURES)
    assert sr_energy.shape == (4,)
    assert distances.ndim == 1
    assert torch.isfinite(sr_energy).all()


def test_jax_parity_long_range_has_no_exclusion_radius():
    """Regression test: lorem-jax's own ``Ewald()`` factory
    (``jaxpme.batched_mixed.calculators``) always builds its potential with
    ``exclusion_radius=None`` -- plain, unmodified Ewald. An earlier version
    of this module set ``exclusion_radius=neighbor_list_options.cutoff``
    here (copied from the *production* ``LoremLongRangeFeaturizer``, which
    deliberately restructures that split), and that one wrong argument was
    the entire source of a large, persistent energy/force discrepancy
    against real checkpoints that looked like a torch-pme-vs-jax-pme
    numerics gap but wasn't (see this module's docstring and
    ``etc/lorem-parity/jax_checkpoint_parity/`` in metawork for the full
    writeup). Both calculators must keep ``exclusion_radius=None``.
    """
    _, long_range = _build_pair(seed=0)
    assert long_range.ewald_calculator.potential.exclusion_radius is None
    assert long_range.direct_calculator.potential.exclusion_radius is None


def test_jax_parity_long_range_periodic_forward():
    backbone, long_range = _build_pair(seed=1)
    nlo = _nlo()
    system = get_system_with_neighbor_lists(_chain_system(pbc=True), [nlo])
    nodes_scalar, distances, nodes_spherical, _ = backbone([system])
    lr_energy = long_range([system], nodes_scalar, distances, nodes_spherical)
    assert lr_energy.shape == (4,)
    assert torch.isfinite(lr_energy).all()


def test_jax_parity_long_range_nonperiodic_forward():
    backbone, long_range = _build_pair(seed=2)
    nlo = _nlo()
    system = get_system_with_neighbor_lists(_chain_system(pbc=False), [nlo])
    nodes_scalar, distances, nodes_spherical, _ = backbone([system])
    lr_energy = long_range([system], nodes_scalar, distances, nodes_spherical)
    assert lr_energy.shape == (4,)
    assert torch.isfinite(lr_energy).all()


def test_jax_parity_energy_and_forces_are_finite():
    backbone, long_range = _build_pair(seed=3)
    nlo = _nlo()
    system = _chain_system(pbc=True)
    system.positions.requires_grad_(True)
    system = get_system_with_neighbor_lists(system, [nlo])
    nodes_scalar, distances, nodes_spherical, sr_energy = backbone([system])
    lr_energy = long_range([system], nodes_scalar, distances, nodes_spherical)
    energy = (sr_energy + lr_energy).sum()
    (forces,) = torch.autograd.grad(energy, system.positions)
    assert torch.isfinite(energy).all()
    assert torch.isfinite(forces).all()


def test_jax_parity_checkpoint_loader_round_trips():
    source_backbone, source_long_range = _build_pair(seed=42)
    flax = _dump_to_flax(source_backbone, source_long_range)

    # a second model, initialized from a *different* seed, so a no-op loader
    # (or a loader that silently skips a leaf) would leave it detectably
    # different from the source model.
    target_backbone, target_long_range = _build_pair(seed=1234)
    load_checkpoint(target_backbone, target_long_range, flax)

    for (name, source_param), (target_name, target_param) in zip(
        source_backbone.named_parameters(),
        target_backbone.named_parameters(),
        strict=True,
    ):
        assert name == target_name
        torch.testing.assert_close(target_param, source_param, msg=f"backbone.{name}")
    for (name, source_param), (target_name, target_param) in zip(
        source_long_range.named_parameters(),
        target_long_range.named_parameters(),
        strict=True,
    ):
        assert name == target_name
        torch.testing.assert_close(target_param, source_param, msg=f"long_range.{name}")

    # and the loaded model actually runs and agrees with the source model
    # numerically, not just parameter-for-parameter.
    nlo = _nlo()
    system = get_system_with_neighbor_lists(_chain_system(pbc=True), [nlo])
    src_nodes_scalar, src_distances, src_nodes_spherical, src_sr_energy = (
        source_backbone([system])
    )
    tgt_nodes_scalar, tgt_distances, tgt_nodes_spherical, tgt_sr_energy = (
        target_backbone([system])
    )
    torch.testing.assert_close(src_sr_energy, tgt_sr_energy)
    src_lr_energy = source_long_range(
        [system], src_nodes_scalar, src_distances, src_nodes_spherical
    )
    tgt_lr_energy = target_long_range(
        [system], tgt_nodes_scalar, tgt_distances, tgt_nodes_spherical
    )
    torch.testing.assert_close(src_lr_energy, tgt_lr_energy)
