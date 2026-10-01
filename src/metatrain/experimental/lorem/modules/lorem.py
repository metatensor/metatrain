"""Short-range and long-range modules of the LOREM model.

They follow lorem-jax's ``Lorem`` / ``LoremBEC`` module for module, and
attributes are named after the flax submodules they mirror (``dense1`` for a
``Dense``, ``update0`` for an ``Update``, ...), so
:func:`.flax_checkpoint.load_checkpoint` can copy lorem-jax parameters across.
Born effective charges live in :mod:`.bec`, on top of these features.

Ewald ``smearing`` and ``kspace_resolution`` are not part of the model. They
are derived from the cutoff (``cutoff / 4`` and ``cutoff / 8``), as lorem-jax
does when it prepares a structure.
"""

import math
from typing import Final, List, Tuple

import torch
from metatomic.torch import NeighborListOptions, System
from sphericart.torch import SphericalHarmonics

from .radial import bernstein_basis, binomial_row
from .spherical import _degree_norms, to_racah
from .structures import concatenate_structures
from .tensor_dense import (
    EquivariantMessagePass,
    TensorDense,
    TensorProduct,
    _DegreeWiseLinear,
)


def _e3x_cosine_cutoff(r: torch.Tensor, cutoff: float) -> torch.Tensor:
    """``e3x`` cosine cutoff: ``0.5 * (cos(pi r / cutoff) + 1)`` for
    ``r < cutoff``, else ``0``.
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


class _Update(torch.nn.Module):
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
        # flax's LayerNorm epsilon, not torch's 1e-5
        self.norm0 = torch.nn.LayerNorm(features, eps=1e-6)
        self.mlp1 = torch.nn.Sequential(
            torch.nn.Linear(features, 2 * features),
            torch.nn.SiLU(),
            torch.nn.Linear(2 * features, features),
        )
        self.norm1 = torch.nn.LayerNorm(features, eps=1e-6)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = x + self.mlp0(y)
        x = self.norm0(x)
        x = x + self.mlp1(x)
        x = self.norm1(x)
        return x


def _energy_mlp(features: int) -> torch.nn.Sequential:
    """Three linear layers, SiLU between them, one output per atom."""
    return torch.nn.Sequential(
        torch.nn.Linear(features, features),
        torch.nn.SiLU(),
        torch.nn.Linear(features, features),
        torch.nn.SiLU(),
        torch.nn.Linear(features, 1),
    )


class _MessagePassingStep(torch.nn.Module):
    """One iteration of lorem-jax's message-passing loop.

    Scalar messages always run. Equivariant messages, and the update that
    folds their norms back into the scalars, run when ``equivariant`` is
    true. Each step also adds its own energy residual.
    """

    equivariant: Final[bool]

    def __init__(
        self,
        num_features: int,
        num_radial: int,
        num_spherical_features: int,
        max_degree: int,
        equivariant: bool,
    ) -> None:
        super().__init__()
        self.num_features = int(num_features)
        self.num_radial = int(num_radial)
        self.num_spherical_features = int(num_spherical_features)
        self.max_degree = int(max_degree)
        self.equivariant = bool(equivariant)
        d = self.num_features
        s = self.num_spherical_features
        num_l = self.max_degree + 1
        self.radial_coefficients = torch.nn.Sequential(
            torch.nn.Linear(2 * d, d),
            torch.nn.SiLU(),
            torch.nn.Linear(d, self.num_radial * d),
        )
        self.edge_dense = torch.nn.Linear(d, d, bias=False)
        self.update_edges = _Update(d, y_dim=d)
        if self.equivariant:
            self.coeff_dense = torch.nn.Linear(d, num_l * s, bias=False)
            self.message_pass = EquivariantMessagePass(s, self.max_degree)
            self.update_norms = _Update(d, y_dim=num_l * s)
        self.energy_mlp = _energy_mlp(d)

    def forward(
        self,
        nodes_scalar: torch.Tensor,
        nodes_spherical: torch.Tensor,
        radial: torch.Tensor,
        sh: torch.Tensor,
        centers: torch.Tensor,
        neighbors: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        n_atoms = nodes_scalar.shape[0]
        n_edges = radial.shape[0]
        d = self.num_features
        if n_edges == 0:
            edges_scalar = nodes_scalar.new_zeros((0, d))
        else:
            pair = torch.cat([nodes_scalar[centers], nodes_scalar[neighbors]], dim=-1)
            coefficients = self.radial_coefficients(pair).reshape(
                n_edges, self.num_radial, d
            )
            edges_scalar = torch.einsum("prf,pr->pf", coefficients, radial)

        edge_update = self.edge_dense(edges_scalar)
        node_update = nodes_scalar.new_zeros((n_atoms, d))
        if edge_update.shape[0] > 0:
            node_update.index_add_(0, centers, edge_update)
        nodes_scalar = self.update_edges(nodes_scalar, node_update)

        if self.equivariant:
            num_l = self.max_degree + 1
            coeff = self.coeff_dense(edges_scalar).reshape(
                n_edges, num_l, self.num_spherical_features
            )
            coeff = _degree_wise_repeat(coeff, self.max_degree)
            edges_spherical = coeff * sh.unsqueeze(-1)
            nodes_spherical = self.message_pass(
                nodes_spherical, edges_spherical, centers, neighbors
            )
            norms = _degree_norms(nodes_spherical, self.max_degree)
            nodes_scalar = self.update_norms(nodes_scalar, norms)

        energy = self.energy_mlp(nodes_scalar).squeeze(-1)
        return nodes_scalar, nodes_spherical, energy


class ShortRange(torch.nn.Module):
    """Short-range part of lorem-jax's ``Lorem``: the initial density, then
    ``num_message_passing`` message-passing steps. Each stage adds its own
    energy residual."""

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
        num_message_passing: int = 0,
        equivariant_message_passing: bool = True,
        initialize_node_features: bool = True,
    ) -> None:
        super().__init__()
        self.cutoff = float(cutoff)
        self.max_degree = int(max_degree)
        self.num_features = int(num_features)
        self.num_radial = int(num_radial)
        self.num_spherical_features = int(num_spherical_features)
        self.num_species = int(num_species)
        self.neighbor_list_options = neighbor_list_options
        self.num_message_passing = int(num_message_passing)
        self.equivariant_message_passing = bool(equivariant_message_passing)
        # species are embedded directly by atomic number; 100 rows as in
        # lorem-jax, more only when the dataset has Z >= 100 (MAD has Fm-No)
        num_embeddings = max(100, max(atomic_types, default=0) + 1)

        d = self.num_features
        s = self.num_spherical_features
        c = self.num_species
        num_l = self.max_degree + 1

        self.register_buffer(
            "bernstein_coeff", binomial_row(self.num_radial).to(torch.float64)
        )

        # -- Initial_0 --
        self.chemical_embedding = torch.nn.Embedding(num_embeddings, c)
        self.spherical_harmonics = SphericalHarmonics(l_max=self.max_degree)

        # -- RadialCoefficients_0 --
        self.radial_coefficients = torch.nn.Sequential(
            torch.nn.Linear(2 * c, d),
            torch.nn.SiLU(),
            torch.nn.Linear(d, self.num_radial * d),
        )

        # -- Dense_0 / Dense_1 --
        # lorem-jax starts the scalars from zero unless
        # ``initialize_node_features``; an empty list keeps this TorchScript-able
        self.dense0 = torch.nn.ModuleList(
            [torch.nn.Linear(c, d, bias=True)] if initialize_node_features else []
        )
        self.dense1 = torch.nn.Linear(d, d, bias=False)
        self.update0 = _Update(d, y_dim=d)

        # -- Dense_2 / TensorDense_0 --
        self.dense2 = torch.nn.Linear(d, num_l * s, bias=False)
        self.tensor_dense = TensorDense(
            in_features=s,
            out_features=s,
            in_max_degree=self.max_degree,
            out_max_degree=self.max_degree,
        )
        self.update1 = _Update(d, y_dim=num_l * s)

        # -- MLP_0: SR-only energy contribution --
        self.energy_mlp = _energy_mlp(d)
        self.message_passing = torch.nn.ModuleList(
            [
                _MessagePassingStep(
                    d,
                    self.num_radial,
                    s,
                    self.max_degree,
                    self.equivariant_message_passing,
                )
                for _ in range(self.num_message_passing)
            ]
        )

    def forward(
        self, systems: List[System]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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
        # The cutoff enters only through the radial basis above, which also
        # scales the spherical edge features through ``dense2``. As in
        # lorem-jax, ``sh`` itself is not multiplied by the cutoff.
        sh = to_racah(sh, self.max_degree)

        species_embed = self.chemical_embedding(species)  # (n_atoms, c)
        pair_features = torch.cat(
            [species_embed[centers], species_embed[neighbors]], dim=-1
        )  # (n_pairs, 2c)

        coefficients = self.radial_coefficients(pair_features).reshape(
            distances.shape[0], self.num_radial, self.num_features
        )
        edges_scalar = torch.einsum("prf,pr->pf", coefficients, radial)

        nodes_scalar = species_embed.new_zeros((n_atoms, self.num_features))
        for layer in self.dense0:
            nodes_scalar = layer(species_embed)
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
        snapshots = nodes_spherical.unsqueeze(0)
        for step in self.message_passing:
            nodes_scalar, nodes_spherical, step_energy = step(
                nodes_scalar,
                nodes_spherical,
                radial,
                sh,
                centers,
                neighbors,
            )
            sr_energy = sr_energy + step_energy
            snapshots = torch.cat([snapshots, nodes_spherical.unsqueeze(0)], dim=0)

        return nodes_scalar, distances, nodes_spherical, sr_energy, snapshots


class LongRange(torch.nn.Module):
    """Long-range part of lorem-jax's ``Lorem``: learned scalar and
    spherical charges, their Coulomb potentials (Ewald for periodic systems,
    direct 1/r over all pairs otherwise), and the update that mixes the
    potentials back into the node features."""

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

        # lorem-jax's ``Ewald()`` factory builds the potential with
        # ``exclusion_radius=None``: plain Ewald, no short-range exclusion.
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
            n_features=s,
        )
        n_update = 1 + (self.max_degree + 1) * s
        self.update2 = _Update(d, y_dim=n_update)
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

    def charges(
        self, nodes_scalar: torch.Tensor, nodes_spherical: torch.Tensor
    ) -> torch.Tensor:
        """Scalar charge plus one channel per ``(ℓ, m)`` up to ``max_degree_lr``."""
        scalar_charges = self.scalar_charge_mlp(nodes_scalar)
        spherical_charges = self.spherical_charge_dense(nodes_spherical)[:, :, 0]
        return torch.cat([scalar_charges, spherical_charges], dim=-1)

    def forward(
        self,
        systems: List[System],
        nodes_scalar: torch.Tensor,
        neighbor_distances: torch.Tensor,
        nodes_spherical: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        charges = self.charges(nodes_scalar, nodes_spherical)
        potentials = self._potentials(systems, charges, neighbor_distances)
        scalar_potential = potentials[:, 0:1]
        spherical_potential = self.potential_to_features(
            potentials[:, 1:].unsqueeze(-1)
        )
        spherical_updates = self.potential_product(spherical_potential, nodes_spherical)

        norms = _degree_norms(spherical_updates, self.max_degree)
        updates = torch.cat([scalar_potential, norms], dim=-1)
        nodes_scalar = self.update2(nodes_scalar, updates)

        return self.energy_mlp(nodes_scalar).squeeze(-1), spherical_updates
