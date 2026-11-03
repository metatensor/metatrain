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
            pair = torch.cat(
                [
                    nodes_scalar.index_select(0, centers),
                    nodes_scalar.index_select(0, neighbors),
                ],
                dim=-1,
            )
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
                torch.arange(len(systems), device=device),
                system_sizes,
                output_size=n_atoms,
            )
            cell_contributions = torch.einsum(
                "ab, abc -> ac",
                cell_shifts.to(dtype),
                cells[system_indices][centers],
            )

        vectors = (
            positions.index_select(0, neighbors)
            - positions.index_select(0, centers)
            + cell_contributions
        )
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
            [
                species_embed.index_select(0, centers),
                species_embed.index_select(0, neighbors),
            ],
            dim=-1,
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
        from torchpme import CoulombPotential, EwaldCalculator

        # The batched real-space sum adds each directed edge once. A half
        # list would drop the reverse edges there, while the one-by-one
        # torch-pme path would still include them.
        if not neighbor_list_options.full_list:
            raise ValueError("LOREM's long-range sum requires a full neighbor list.")

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
            full_neighbor_list=True,
            lr_wavelength=float(kspace_resolution),
        )
        self.lr_wavelength = float(kspace_resolution)
        # periodic systems with more (atom, k-vector) pairs than this go
        # through torch-pme one by one: the batched k-space materializes
        # (pairs, channels) tensors, torch-pme only (pairs,), and the rare
        # huge MAD cells have millions of k-vectors
        self.max_batched_pairs = 1 << 20
        # lorem-jax's non-PBC path: plain pairwise 1/r over *all* pairs (not
        # restricted to the short-range cutoff neighbor list), no smearing,
        # no exclusion (same reasoning as ewald_calculator above).
        self.direct_potential = CoulombPotential(
            smearing=None,
            exclusion_radius=None,
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
        """Ewald (3D periodic) or direct 1/r (non-periodic) potentials for the
        whole batch at once, as ragged index operations, with a single
        device-to-host copy of the cells per batch. Fully periodic systems
        whose k-grid exceeds ``max_batched_pairs`` use one torch-pme call
        each. Anything between those two periodicities is rejected: a slab or
        a wire has a singular cell (metatomic zeros the non-periodic
        vectors) and torch-pme cannot invert it."""
        device = charges.device
        dtype = charges.dtype
        sizes = [len(system) for system in systems]
        n_systems = len(systems)
        n_atoms = charges.shape[0]

        cells = torch.stack([system.cell for system in systems])
        pbc = torch.stack([system.pbc for system in systems])
        host = (
            torch.cat([torch.linalg.norm(cells, dim=-1), pbc.to(cells.dtype)], dim=1)
            .detach()
            .cpu()
        )
        sizes_host = torch.tensor(sizes, dtype=torch.long)
        n_pbc = host[:, 3:].sum(dim=1)
        n_partial: int = ((n_pbc > 0) & (n_pbc < 3)).sum().tolist()
        if n_partial > 0:
            raise ValueError(
                "LOREM only supports fully periodic systems (pbc all true) "
                "and non-periodic systems (pbc all false). Partially "
                "periodic systems are not supported."
            )
        k_cutoff = 2 * math.pi / self.lr_wavelength
        ns_all = torch.ceil(k_cutoff * host[:, :3] / 2 / math.pi).long()
        too_large = sizes_host * ns_all.prod(dim=1) > self.max_batched_pairs
        one_by_one = (n_pbc == 3) & too_large
        periodic = (n_pbc == 3) & ~one_by_one
        non_periodic = n_pbc == 0

        offsets_host = torch.cumsum(sizes_host, 0) - sizes_host
        atom_periodic = torch.repeat_interleave(periodic, sizes_host).to(device)
        potentials = torch.zeros_like(charges)

        # real-space Ewald over the neighbor list of periodic systems
        nl_values = [
            system.get_neighbor_list(self.neighbor_list_options).samples.values
            for system in systems
        ]
        n_edges = torch.tensor([len(v) for v in nl_values], dtype=torch.long)
        edge_offsets = torch.repeat_interleave(offsets_host, n_edges).to(device)
        pairs = torch.cat(nl_values)
        centers = pairs[:, 0] + edge_offsets
        neighbors = pairs[:, 1] + edge_offsets
        ewald = self.ewald_calculator.potential
        # The model always requests a full neighbor list, so each ordered
        # pair appears once and the reverse edge is not added here.
        sr = ewald.sr_from_dist(neighbor_distances) * atom_periodic[centers].to(dtype)
        potentials.index_add_(
            0, centers, charges.index_select(0, neighbors) * sr.unsqueeze(-1)
        )

        positions = torch.cat([system.positions for system in systems])

        # direct 1/r over all ordered pairs of each non-periodic system
        np_sizes = sizes_host[non_periodic]
        if len(np_sizes) > 0:
            np_offsets = offsets_host[non_periodic]
            n_pairs = np_sizes * np_sizes
            total = int(n_pairs.sum())
            pair_system = torch.repeat_interleave(
                torch.arange(len(np_sizes)), n_pairs
            ).to(device)
            n_pairs, np_sizes, np_offsets = (
                n_pairs.to(device),
                np_sizes.to(device),
                np_offsets.to(device),
            )
            local = (
                torch.arange(total, device=device)
                - (torch.cumsum(n_pairs, 0) - n_pairs)[pair_system]
            )
            size = np_sizes[pair_system]
            first = local // size + np_offsets[pair_system]
            second = local % size + np_offsets[pair_system]
            off_diagonal = first != second
            # no zero-length vectors on the diagonal: their norm has NaN gradients
            vectors = torch.where(
                off_diagonal.unsqueeze(-1),
                positions.index_select(0, second) - positions.index_select(0, first),
                torch.ones_like(positions[first]),
            )
            direct = self.direct_potential.from_dist(
                torch.linalg.vector_norm(vectors, dim=-1)
            ) * off_diagonal.to(dtype)
            potentials.index_add_(
                0, first, charges.index_select(0, second) * direct.unsqueeze(-1)
            )

        # reciprocal-space Ewald of periodic systems, k-vectors as in torch-pme
        p_index = torch.arange(n_systems)[periodic]
        if len(p_index) > 0:
            ns = ns_all[p_index]
            n_k = ns.prod(dim=1)
            p_sizes = sizes_host[p_index]
            n_ak = p_sizes * n_k
            total_k = int(n_k.sum())
            total_ak = int(n_ak.sum())
            n_p = len(p_index)
            k_system = torch.repeat_interleave(torch.arange(n_p), n_k).to(device)
            ak_system = torch.repeat_interleave(torch.arange(n_p), n_ak).to(device)
            ns, n_k, n_ak, p_sizes = (
                ns.to(device),
                n_k.to(device),
                n_ak.to(device),
                p_sizes.to(device),
            )
            p_offsets = offsets_host[p_index].to(device)
            p_index = p_index.to(device)

            # integer k grid in fftfreq order: 0, 1, ..., -2, -1
            local_k = (
                torch.arange(total_k, device=device)
                - (torch.cumsum(n_k, 0) - n_k)[k_system]
            )
            nk = ns[k_system]
            iz = local_k % nk[:, 2]
            iy = (local_k // nk[:, 2]) % nk[:, 1]
            ix = local_k // (nk[:, 2] * nk[:, 1])
            grid = torch.stack([ix, iy, iz], dim=-1)
            grid = torch.where(grid < (nk + 1) // 2, grid, grid - nk).to(dtype)
            p_cells = cells[p_index]
            reciprocal = 2 * math.pi * torch.linalg.inv_ex(p_cells)[0].transpose(1, 2)
            kvectors = torch.einsum(
                "kd,kde->ke", grid, reciprocal.index_select(0, k_system)
            )
            volumes = torch.abs(torch.linalg.det(p_cells))
            weights = ewald.lr_from_k_sq(
                (kvectors**2).sum(dim=-1)
            ) / volumes.index_select(0, k_system)

            # (atom, k) pairs within each periodic system
            local_ak = (
                torch.arange(total_ak, device=device)
                - (torch.cumsum(n_ak, 0) - n_ak)[ak_system]
            )
            size = p_sizes[ak_system]
            ak_atom = local_ak % size + p_offsets[ak_system]
            ak_k = local_ak // size + (torch.cumsum(n_k, 0) - n_k)[ak_system]
            phase = (
                kvectors.index_select(0, ak_k) * positions.index_select(0, ak_atom)
            ).sum(dim=-1)
            cos, sin = torch.cos(phase).unsqueeze(-1), torch.sin(phase).unsqueeze(-1)
            q = charges.index_select(0, ak_atom)
            n_channels = charges.shape[1]
            s_cos = charges.new_zeros((total_k, n_channels))
            s_cos = s_cos.index_add_(0, ak_k, q * cos)
            s_sin = charges.new_zeros((total_k, n_channels))
            s_sin = s_sin.index_add_(0, ak_k, q * sin)
            kspace = charges.new_zeros((n_atoms, n_channels)).index_add_(
                0,
                ak_atom,
                weights.index_select(0, ak_k).unsqueeze(-1)
                * (
                    cos * s_cos.index_select(0, ak_k)
                    + sin * s_sin.index_select(0, ak_k)
                ),
            )

            atom_system = torch.repeat_interleave(
                torch.arange(n_systems, device=device),
                sizes_host.to(device),
                output_size=n_atoms,
            )
            charge_tot = charges.new_zeros((n_systems, n_channels)).index_add_(
                0, atom_system, charges
            )
            all_volumes = torch.ones(n_systems, dtype=dtype, device=device)
            all_volumes = all_volumes.index_put([p_index], volumes)
            kspace = kspace - charges * ewald.self_contribution()
            kspace = (
                kspace
                - 2
                * ewald.background_correction()
                * (charge_tot / all_volumes.unsqueeze(-1))[atom_system]
            )
            potentials = potentials + kspace * atom_periodic.unsqueeze(-1).to(dtype)

        potentials = potentials / 2
        edge_starts = torch.cumsum(n_edges, 0) - n_edges
        one_by_one_index: List[int] = torch.nonzero(one_by_one).flatten().tolist()
        for i in one_by_one_index:
            start, size = int(offsets_host[i]), sizes[i]
            first_edge, n_edge = int(edge_starts[i]), int(n_edges[i])
            system = systems[i]
            potentials[start : start + size] += self.ewald_calculator.forward(
                charges=charges[start : start + size],
                cell=system.cell,
                positions=system.positions,
                neighbor_indices=nl_values[i][:, :2],
                neighbor_distances=neighbor_distances[first_edge : first_edge + n_edge],
                periodic=system.pbc,
            )
        return potentials

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
