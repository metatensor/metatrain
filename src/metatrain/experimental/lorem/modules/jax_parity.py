"""A ``trunk``/long-range pair that mirrors ``lorem-jax``'s ``lorem.Lorem``
forward pass module-for-module, so a shipped ``lorem-jax`` flax checkpoint
has an exact 1:1 home for every learned weight.

This is deliberately a *separate* pair of modules from
:class:`~metatrain.experimental.lorem.modules.backbone.LoremBackbone` and
:class:`~metatrain.experimental.lorem.modules.long_range.LoremLongRangeFeaturizer`.
Those two already reorganize lorem-jax's computation (a fused
``feature_mlp`` in place of lorem-jax's separate species-embedding /
``RadialCoefficients`` / ``Dense_0`` / ``Dense_1`` path, a single shared
``Linear`` in place of ``e3x.nn.Dense``'s per-degree weights, and a single
final linear readout in place of two summed energy MLPs) -- real design
choices, not bugs, and existing models/tests depend on that shape. Rather
than risk either, this module gives checkpoint parity its own home.

Traced directly against ``lorem.models.mlip.Lorem.__call__`` and
``lorem.models.backbone`` (``lorem-jax`` submodule, ``lr=True``,
``num_message_passing=0`` -- the configuration every shipped
``lorem-tmlr-archive`` LOREM checkpoint besides the message-passing variants
uses) and verified module-by-module against a real checkpoint's flax
parameter tree (``evals/AuMgO/lorem`` from
https://github.com/sirmarcel/lorem-tmlr-archive), leaf name for leaf name:

================================  ===============================================
flax module                       here
================================  ===============================================
``Initial_0/ChemicalEmbedding_0``  ``JaxParityBackbone.chemical_embedding``
``Dense_0``                       ``JaxParityBackbone.dense0``
``Dense_1``                       ``JaxParityBackbone.dense1``
``RadialCoefficients_0/MLP_0``     ``JaxParityBackbone.radial_coefficients``
``Update_0``                      ``JaxParityBackbone.update0``
``Dense_2``                       ``JaxParityBackbone.dense2``
``TensorDense_0``                  ``JaxParityBackbone.tensor_dense``
``Update_1``                      ``JaxParityBackbone.update1``
``MLP_0``                         ``JaxParityBackbone.energy_mlp`` (SR-only energy)
``MLP_1``                         ``JaxParityLongRange.scalar_charge_mlp``
``TensorDense_1``                  ``JaxParityLongRange.spherical_charge_dense``
``Dense_3``                       ``JaxParityLongRange.potential_to_features``
``Tensor_0``                      ``JaxParityLongRange.potential_product``
``Update_2``                      ``JaxParityLongRange.update2``
``MLP_2``                         ``JaxParityLongRange.energy_mlp`` (LR energy)
================================  ===============================================

Scope: ``trunk="spherical"``-equivalent short-range descriptor, ``lr=True``,
``num_message_passing=0``, Ewald (periodic systems) and the plain-pairwise
non-PBC path. lorem-jax's message passing loop and ``LoremBEC`` head are not
covered here.

See :mod:`.jax_parity_checkpoint` for the checkpoint loader, and
``etc/lorem-parity/jax_checkpoint_parity/`` in the
`metawork <https://github.com/EricBoittier/metawork>`_ workspace for a
worked example (dumping a real ``lorem-tmlr-archive`` checkpoint from JAX,
loading it here, and comparing energies/forces against the JAX reference).
Checked against two real periodic checkpoints (AuMgO, bio_dimers): total
energy and forces agree with the JAX reference to ~1e-4 relative or better
(float32 noise floor), matching the paper's own reported test-set accuracy
on both. An earlier version of this module set the Ewald calculator's
``exclusion_radius`` to the model's short-range cutoff (copied from the
*production* ``LoremLongRangeFeaturizer``, which deliberately restructures
that split); lorem-jax's own ``Ewald()`` factory
(``jaxpme.batched_mixed.calculators``) always builds its potential with
``exclusion_radius=None`` -- plain, unmodified Ewald, no short-range
exclusion zone. That one argument was the entire source of what looked
like a torch-pme-vs-jax-pme numerics gap (~0.7 meV/atom on AuMgO, much
larger on bio_dimers) but wasn't -- see the worked example's README for the
full investigation and how it was found (a from-scratch analytic Madelung
-constant check proved both libraries individually correct before the real
cause -- one wrong constructor argument -- turned up).

Ewald ``smearing``/``kspace_resolution`` are *not* read from a checkpoint's
``model.yaml`` (lorem-jax's own ``marathon.prepare()`` derives them from the
model's ``cutoff`` at data-prep time: ``smearing = cutoff / 4``,
``lr_wavelength = cutoff / 8``) -- pass those derived values, not a
``torchpme.tuning.ewald.tune_ewald`` guess, when reproducing a real
checkpoint's numbers; the worked example does this.
"""

import math
from typing import List, Tuple

import torch
from metatomic.torch import NeighborListOptions, System

from .backbone import _AnalyticSphericalHarmonics, _degree_norms, _SphericartWrapper
from .radial import bernstein_basis, binomial_row
from .spherical import to_racah
from .structures import concatenate_structures
from .tensor_dense import TensorDense, TensorProduct, _DegreeWiseLinear


def _e3x_cosine_cutoff(r: torch.Tensor, cutoff: float) -> torch.Tensor:
    """``e3x.nn.functions.cosine_cutoff``: ``0.5 * (cos(pi r / cutoff) + 1)``
    for ``r < cutoff``, else ``0``. No onset/width -- unlike
    :func:`.backbone._cosine_cutoff`, which adds a ``cutoff_width`` hyper
    lorem-jax does not have.
    """
    inside = (r < cutoff).to(r.dtype)
    return 0.5 * (torch.cos(math.pi * r / cutoff) + 1.0) * inside


def _degree_wise_repeat(x: torch.Tensor, max_degree: int) -> torch.Tensor:
    """``(n, max_degree + 1, f)`` -> ``(n, (max_degree + 1) ** 2, f)``, each
    degree's row repeated ``2l + 1`` times (``lorem.models.backbone
    .degree_wise_repeat_last_axis``, transposed to a leading repeat axis)."""
    parts = [
        x[:, ell : ell + 1, :].expand(-1, 2 * ell + 1, -1)
        for ell in range(max_degree + 1)
    ]
    return torch.cat(parts, dim=1)


class _JaxUpdate(torch.nn.Module):
    """Port of lorem-jax's ``Update``: two gated MLP + LayerNorm residuals.

    ``x`` is always the running scalar node features (width ``features``);
    ``y`` is whatever this call site is folding in, and can have a different
    width (``y_dim``) -- only the first MLP's input layer depends on it.
    """

    def __init__(self, features: int, y_dim: int) -> None:
        super().__init__()
        self.mlp0 = torch.nn.Sequential(
            torch.nn.Linear(y_dim, 2 * features),
            torch.nn.SiLU(),
            torch.nn.Linear(2 * features, features),
        )
        self.norm0 = torch.nn.LayerNorm(features)
        self.mlp1 = torch.nn.Sequential(
            torch.nn.Linear(features, 2 * features),
            torch.nn.SiLU(),
            torch.nn.Linear(2 * features, features),
        )
        self.norm1 = torch.nn.LayerNorm(features)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = x + self.mlp0(y)
        x = self.norm0(x)
        x = x + self.mlp1(x)
        x = self.norm1(x)
        return x


def _energy_mlp(features: int) -> torch.nn.Sequential:
    """lorem-jax's ``MLP(features=[d, d, 1])``: three Dense layers, SiLU
    between (not after the last) -- used for both ``MLP_0`` and ``MLP_2``."""
    return torch.nn.Sequential(
        torch.nn.Linear(features, features),
        torch.nn.SiLU(),
        torch.nn.Linear(features, features),
        torch.nn.SiLU(),
        torch.nn.Linear(features, 1),
    )


class JaxParityBackbone(torch.nn.Module):
    """Short-range descriptor, module-for-module identical to lorem-jax's
    ``Lorem`` SR path (``initialize_node_features=True``,
    ``num_message_passing=0``, ``equivariant_message_passing`` irrelevant)."""

    def __init__(
        self,
        cutoff: float,
        max_degree: int,
        num_features: int,
        num_radial: int,
        num_spherical_features: int,
        num_species: int,
        atomic_types: List[int],
        neighbor_list_options: NeighborListOptions,
    ) -> None:
        super().__init__()
        self.cutoff = float(cutoff)
        self.max_degree = int(max_degree)
        self.num_features = int(num_features)
        self.num_radial = int(num_radial)
        self.num_spherical_features = int(num_spherical_features)
        self.num_species = int(num_species)
        self.neighbor_list_options = neighbor_list_options
        del atomic_types  # species are embedded directly by atomic number

        d = self.num_features
        s = self.num_spherical_features
        c = self.num_species
        num_l = self.max_degree + 1

        self.register_buffer(
            "bernstein_coeff", binomial_row(self.num_radial).to(torch.float32)
        )

        # -- Initial_0 --
        self.chemical_embedding = torch.nn.Embedding(100, c)
        if self.max_degree > 2:
            from sphericart.torch import SphericalHarmonics

            try:
                calculator = SphericalHarmonics(l_max=self.max_degree, normalized=True)
            except TypeError:
                calculator = SphericalHarmonics(l_max=self.max_degree)
            self.spherical_harmonics = (
                calculator
                if isinstance(calculator, torch.nn.Module)
                else _SphericartWrapper(calculator)
            )
        else:
            self.spherical_harmonics = _AnalyticSphericalHarmonics(self.max_degree)

        # -- RadialCoefficients_0 --
        self.radial_coefficients = torch.nn.Sequential(
            torch.nn.Linear(2 * c, d),
            torch.nn.SiLU(),
            torch.nn.Linear(d, self.num_radial * d),
        )

        # -- Dense_0 / Dense_1 --
        self.dense0 = torch.nn.Linear(c, d, bias=True)
        self.dense1 = torch.nn.Linear(d, d, bias=False)
        self.update0 = _JaxUpdate(d, y_dim=d)

        # -- Dense_2 / TensorDense_0 --
        self.dense2 = torch.nn.Linear(d, num_l * s, bias=False)
        self.tensor_dense = TensorDense(
            in_features=s,
            out_features=s,
            in_max_degree=self.max_degree,
            out_max_degree=self.max_degree,
            include_pseudotensors=False,
        )
        self.update1 = _JaxUpdate(d, y_dim=num_l * s)

        # -- MLP_0: SR-only energy contribution --
        self.energy_mlp = _energy_mlp(d)

    def forward(
        self, systems: List[System]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        (
            positions,
            centers,
            neighbors,
            species,
            cells,
            cell_shifts,
        ) = concatenate_structures(systems, self.neighbor_list_options)

        device = positions.device
        dtype = positions.dtype
        n_atoms = positions.shape[0]

        if len(cells) == 1:
            cell_contributions = cell_shifts.to(dtype) @ cells[0]
        else:
            system_sizes = torch.tensor(
                [len(system) for system in systems], device=device
            )
            system_indices = torch.repeat_interleave(
                torch.arange(len(systems), device=device), system_sizes
            )
            cell_contributions = torch.einsum(
                "ab, abc -> ac",
                cell_shifts.to(dtype),
                cells[system_indices][centers],
            )

        vectors = positions[neighbors] - positions[centers] + cell_contributions
        distances = torch.linalg.vector_norm(vectors, dim=-1)
        cutoff_weights = _e3x_cosine_cutoff(distances, self.cutoff)

        radial = bernstein_basis(distances, self.cutoff, self.bernstein_coeff)
        radial = radial * cutoff_weights.unsqueeze(-1)

        sh = self.spherical_harmonics(vectors)
        sh = to_racah(sh, self.max_degree)
        # lorem-jax's ``Initial`` only masks ``spherical_expansion`` by the
        # (padding) ``pair_mask``, not by the smooth ``cutoffs`` envelope --
        # the cutoff only enters through ``radial_expansion`` above, and
        # reaches ``edges_spherical`` indirectly via ``Dense_2``'s
        # ``edges_scalar``-derived coefficients. Multiplying ``sh`` by the
        # envelope directly (as done here previously) double-applies it.

        species_embed = self.chemical_embedding(species)  # (n_atoms, c)
        pair_features = torch.cat(
            [species_embed[centers], species_embed[neighbors]], dim=-1
        )  # (n_pairs, 2c)

        coefficients = self.radial_coefficients(pair_features).reshape(
            distances.shape[0], self.num_radial, self.num_features
        )
        edges_scalar = torch.einsum("prf,pr->pf", coefficients, radial)

        nodes_scalar = self.dense0(species_embed)
        edge_update = self.dense1(edges_scalar)
        node_update = nodes_scalar.new_zeros((n_atoms, self.num_features))
        if edge_update.shape[0] > 0:
            node_update.index_add_(0, centers, edge_update)
        nodes_scalar = self.update0(nodes_scalar, node_update)

        num_l = self.max_degree + 1
        edge_coefficients = self.dense2(edges_scalar).reshape(
            distances.shape[0], num_l, self.num_spherical_features
        )
        edge_coefficients = _degree_wise_repeat(edge_coefficients, self.max_degree)
        edges_spherical = edge_coefficients * sh.unsqueeze(-1)

        n_lm = num_l * num_l
        nodes_spherical = nodes_scalar.new_zeros(
            (n_atoms, n_lm, self.num_spherical_features)
        )
        if edges_spherical.shape[0] > 0:
            nodes_spherical.index_add_(0, centers, edges_spherical)
        nodes_spherical = self.tensor_dense(nodes_spherical)

        norms = _degree_norms(nodes_spherical, self.max_degree)
        nodes_scalar = self.update1(nodes_scalar, norms)

        sr_energy = self.energy_mlp(nodes_scalar).squeeze(-1)

        return nodes_scalar, distances, nodes_spherical, sr_energy


class JaxParityLongRange(torch.nn.Module):
    """Long-range Coulomb block, module-for-module identical to lorem-jax's
    ``if self.lr:`` branch (Ewald only -- the shipped PBC checkpoints this
    targets, e.g. AuMgO, always train with ``use_ewald``-equivalent Ewald
    summation)."""

    def __init__(
        self,
        feature_dim: int,
        num_spherical_features: int,
        max_degree: int,
        max_degree_lr: int,
        neighbor_list_options: NeighborListOptions,
        smearing: float,
        kspace_resolution: float,
    ) -> None:
        super().__init__()
        from torchpme import Calculator, CoulombPotential, EwaldCalculator

        self.max_degree = int(max_degree)
        self.max_degree_lr = int(max_degree_lr)
        self.neighbor_list_options = neighbor_list_options

        # lorem-jax's own ``Ewald()`` factory (jaxpme.batched_mixed.calculators)
        # builds its potential with ``exclusion_radius=None`` -- plain,
        # unmodified Ewald, no short-range exclusion zone. An earlier version
        # of this module set ``exclusion_radius=neighbor_list_options.cutoff``
        # here, copied from the *production* ``LoremLongRangeFeaturizer``
        # (which deliberately restructures the split, see that class's
        # docstring) -- for this exact-parity port that was simply wrong, and
        # was the entire source of a ~0.7 meV/atom energy discrepancy against
        # a real checkpoint that looked like a torch-pme/jax-pme numerics gap
        # but wasn't: with ``exclusion_radius=None``, the two potentials
        # agree to ~1e-5 (float32 noise), not ~30% off.
        self.ewald_calculator = EwaldCalculator(
            potential=CoulombPotential(
                smearing=float(smearing),
                exclusion_radius=None,
            ),
            full_neighbor_list=neighbor_list_options.full_list,
            lr_wavelength=float(kspace_resolution),
        )
        # lorem-jax's non-PBC path: plain pairwise 1/r over *all* pairs (not
        # restricted to the short-range cutoff neighbor list), no smearing,
        # no exclusion (same reasoning as ewald_calculator above).
        self.direct_calculator = Calculator(
            potential=CoulombPotential(
                smearing=None,
                exclusion_radius=None,
            ),
            full_neighbor_list=False,
        )

        d = feature_dim
        s = num_spherical_features
        self.scalar_charge_mlp = torch.nn.Sequential(
            torch.nn.Linear(d, 2 * d),
            torch.nn.SiLU(),
            torch.nn.Linear(2 * d, 1),
        )
        self.spherical_charge_dense = TensorDense(
            in_features=s,
            out_features=1,
            in_max_degree=self.max_degree,
            out_max_degree=self.max_degree_lr,
            include_pseudotensors=False,
        )
        # e3x.nn.Dense: one weight matrix *per degree* -- unlike a shared
        # torch.nn.Linear, see _DegreeWiseLinear's docstring.
        self.potential_to_features = _DegreeWiseLinear(
            1, s, self.max_degree_lr, bias=False
        )
        self.potential_product = TensorProduct(
            left_max_degree=self.max_degree_lr,
            right_max_degree=self.max_degree,
            out_max_degree=self.max_degree,
            include_pseudotensors=False,
            n_features=s,
        )
        n_update = 1 + (self.max_degree + 1) * s
        self.update2 = _JaxUpdate(d, y_dim=n_update)
        self.energy_mlp = _energy_mlp(d)

    def _potentials(
        self,
        systems: List[System],
        charges: torch.Tensor,
        neighbor_distances: torch.Tensor,
    ) -> torch.Tensor:
        last_nodes = 0
        last_edges = 0
        potentials = []
        for system in systems:
            system_charges = charges[last_nodes : last_nodes + len(system)]
            last_nodes += len(system)
            neighbor_list = system.get_neighbor_list(self.neighbor_list_options)
            neighbor_indices = torch.stack(
                [
                    neighbor_list.samples.column("first_atom"),
                    neighbor_list.samples.column("second_atom"),
                ],
                dim=-1,
            )
            neighbor_distances_system = neighbor_distances[
                last_edges : last_edges + len(neighbor_indices)
            ]
            last_edges += len(neighbor_indices)
            if system.pbc.any():
                potentials.append(
                    self.ewald_calculator.forward(
                        charges=system_charges,
                        cell=system.cell,
                        positions=system.positions,
                        neighbor_indices=neighbor_indices,
                        neighbor_distances=neighbor_distances_system,
                        periodic=system.pbc,
                    )
                )
            else:
                # lorem-jax's non-PBC path: plain 1/r over *all* pairs, not
                # just those within the short-range cutoff.
                all_pairs = torch.combinations(
                    torch.arange(len(system), device=system.positions.device), 2
                )
                all_distances = torch.linalg.vector_norm(
                    system.positions[all_pairs[:, 1]]
                    - system.positions[all_pairs[:, 0]],
                    dim=-1,
                )
                potentials.append(
                    self.direct_calculator.forward(
                        charges=system_charges,
                        cell=system.cell,
                        positions=system.positions,
                        neighbor_indices=all_pairs,
                        neighbor_distances=all_distances,
                    )
                )
        return torch.cat(potentials, dim=0)

    def forward(
        self,
        systems: List[System],
        nodes_scalar: torch.Tensor,
        neighbor_distances: torch.Tensor,
        nodes_spherical: torch.Tensor,
    ) -> torch.Tensor:
        scalar_charges = self.scalar_charge_mlp(nodes_scalar)
        spherical_charges = self.spherical_charge_dense(nodes_spherical)[:, :, 0]
        charges = torch.cat([scalar_charges, spherical_charges], dim=-1)

        potentials = self._potentials(systems, charges, neighbor_distances)
        scalar_potential = potentials[:, 0:1]
        spherical_potential = self.potential_to_features(
            potentials[:, 1:].unsqueeze(-1)
        )
        spherical_updates = self.potential_product(spherical_potential, nodes_spherical)

        norms = _degree_norms(spherical_updates, self.max_degree)
        updates = torch.cat([scalar_potential, norms], dim=-1)
        nodes_scalar = self.update2(nodes_scalar, updates)

        return self.energy_mlp(nodes_scalar).squeeze(-1)
