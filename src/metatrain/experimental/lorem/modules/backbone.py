import math
from typing import List, Tuple

import torch
from metatomic.torch import NeighborListOptions, System

from .radial import bernstein_basis, bessel_basis, binomial_row
from .spherical import to_racah
from .structures import concatenate_structures
from .tensor_dense import EquivariantMessagePass, TensorDense


def _degree_norms(spherical: torch.Tensor, max_degree: int) -> torch.Tensor:
    """Per-ℓ, per-feature norms with the LOREM ``(2ℓ+1)^{1/4}`` factor.

    :param spherical: ``(n_atoms, (max_degree+1)**2, n_features)``
    :return: ``(n_atoms, (max_degree+1) * n_features)``
    """
    parts: List[torch.Tensor] = []
    for ell in range(max_degree + 1):
        chunk = spherical[:, ell * ell : (ell + 1) * (ell + 1), :]
        factor = (2.0 * float(ell) + 1.0) ** 0.25
        parts.append(factor * _safe_vector_norm(chunk, dim=1))
    return torch.cat(parts, dim=-1)


def _cosine_cutoff(r: torch.Tensor, cutoff: float, width: float) -> torch.Tensor:
    """Smooth cosine cutoff: 1 below ``cutoff - width``, 0 at ``cutoff``."""
    if width <= 0.0:
        return (r < cutoff).to(r.dtype)
    onset = cutoff - width
    envelope = 0.5 * (torch.cos(math.pi * (r - onset) / width) + 1.0)
    return torch.where(
        r >= cutoff,
        torch.zeros_like(r),
        torch.where(r <= onset, torch.ones_like(r), envelope),
    )


def _bessel_basis(r: torch.Tensor, n_radial: int, cutoff: float) -> torch.Tensor:
    """Sinc Bessel radial basis of shape ``(n_edges, n_radial)``."""
    return bessel_basis(r, n_radial, cutoff)


def _safe_vector_norm(x: torch.Tensor, dim: int, eps: float = 1.0e-12) -> torch.Tensor:
    """``torch.linalg.vector_norm`` with a finite gradient at zero.

    ``vector_norm``'s backward is ``x / ||x||``, which is ``0/0 = nan`` whenever
    ``x`` is exactly zero along ``dim`` -- e.g. a vanishing ``m``-component of a
    spherical harmonic. Adding ``eps`` inside the square root keeps the value
    (and gradient) finite everywhere without materially changing it away from
    zero.
    """
    return torch.sqrt(x.pow(2).sum(dim=dim) + eps)


def _real_spherical_harmonics(vectors: torch.Tensor, l_max: int) -> torch.Tensor:
    """Real orthonormal spherical harmonics for ``l_max <= 2``.

    :param vectors: Cartesian displacements of shape ``(n_edges, 3)``.
    :param l_max: Maximum angular momentum (0, 1, or 2).
    :return: ``(n_edges, (l_max + 1) ** 2)`` real spherical harmonics.
    """
    n_lm = (l_max + 1) * (l_max + 1)
    if vectors.shape[0] == 0:
        return vectors.new_zeros((0, n_lm))
    if l_max > 2:
        raise RuntimeError(
            "Analytic spherical harmonics only support max_degree <= 2. "
            "Install sphericart-torch for higher degrees."
        )

    r = torch.linalg.vector_norm(vectors, dim=-1, keepdim=True).clamp(min=1.0e-12)
    x = vectors[:, 0:1] / r
    y = vectors[:, 1:2] / r
    z = vectors[:, 2:3] / r

    # Orthonormal real SH, m = -l ... +l within each l.
    y00 = torch.full_like(x, 0.28209479177387814)
    parts: List[torch.Tensor] = [y00]
    if l_max >= 1:
        c1 = 0.4886025119029199  # sqrt(3 / 4pi)
        parts.extend([c1 * y, c1 * z, c1 * x])
    if l_max >= 2:
        c2 = 1.0925484305920792  # sqrt(15 / 4pi)
        c20 = 0.31539156525252005  # sqrt(5 / 16pi)
        c22 = 0.5462742152960396  # sqrt(15 / 16pi)
        parts.extend(
            [
                c2 * x * y,
                c2 * y * z,
                c20 * (3.0 * z * z - 1.0),
                c2 * x * z,
                c22 * (x * x - y * y),
            ]
        )
    return torch.cat(parts, dim=-1)


class _AnalyticSphericalHarmonics(torch.nn.Module):
    """TorchScript-friendly real SH for ``l_max <= 2``."""

    def __init__(self, l_max: int) -> None:
        super().__init__()
        self.l_max = l_max

    def forward(self, vectors: torch.Tensor) -> torch.Tensor:
        return _real_spherical_harmonics(vectors, self.l_max)


class _SphericartWrapper(torch.nn.Module):
    """Wrap ``sphericart.torch.SphericalHarmonics.compute`` as ``forward``."""

    def __init__(self, calculator: torch.nn.Module) -> None:
        super().__init__()
        self.calculator = calculator

    def forward(self, vectors: torch.Tensor) -> torch.Tensor:
        return self.calculator.compute(vectors)


class LoremBackbone(torch.nn.Module):
    """Short-range spherical density + optional scalar message passing."""

    def __init__(
        self,
        hypers: dict,
        atomic_types: List[int],
        neighbor_list_options: NeighborListOptions,
    ) -> None:
        super().__init__()
        self.cutoff = float(hypers["cutoff"])
        self.cutoff_width = float(hypers["cutoff_width"])
        self.max_degree = int(hypers["max_degree"])
        self.num_radial = int(hypers["num_radial"])
        self.num_features = int(hypers["num_features"])
        self.num_spherical_features = int(hypers["num_spherical_features"])
        self.num_message_passing = int(hypers["num_message_passing"])
        self.use_bernstein = (
            str(hypers["radial_basis"]) == "basic_bernstein"
            if "radial_basis" in hypers
            else True
        )
        self.use_e3x_sh = (
            str(hypers["sh_convention"]) == "e3x" if "sh_convention" in hypers else True
        )
        self.n_lm = (self.max_degree + 1) * (self.max_degree + 1)
        self.register_buffer(
            "bernstein_coeff", binomial_row(self.num_radial).to(torch.float32)
        )
        self.n_invariants = self.num_radial * (self.max_degree + 1)
        self.neighbor_list_options = neighbor_list_options

        max_z = max(atomic_types) if len(atomic_types) > 0 else 0
        self.species_embedding = torch.nn.Embedding(max_z + 1, self.num_features)

        self.feature_mlp = torch.nn.Sequential(
            torch.nn.Linear(self.num_features + self.n_invariants, self.num_features),
            torch.nn.SiLU(),
            torch.nn.Linear(self.num_features, self.num_features),
        )

        self.mp_layers = torch.nn.ModuleList(
            [
                torch.nn.Sequential(
                    torch.nn.Linear(2 * self.num_features, self.num_features),
                    torch.nn.SiLU(),
                    torch.nn.Linear(self.num_features, self.num_features),
                )
                for _ in range(self.num_message_passing)
            ]
        )
        # lorem-jax updates the equivariant (spherical) node features at every
        # message-passing step too, not just the scalar ones -- see
        # `EquivariantMessagePass`'s docstring. `equivariant_mp_edge_coefficients`
        # builds each step's per-edge spherical "basis" (edges_scalar -> per
        # degree coefficients, broadcast onto that edge's spherical harmonics,
        # matching lorem-jax's `RadialCoefficients`-derived `edges_spherical`).
        self.equivariant_mp_edge_coefficients = torch.nn.ModuleList(
            [
                torch.nn.Linear(
                    2 * self.num_features,
                    (self.max_degree + 1) * self.num_spherical_features,
                    bias=False,
                )
                for _ in range(self.num_message_passing)
            ]
        )
        self.equivariant_mp_layers = torch.nn.ModuleList(
            [
                EquivariantMessagePass(
                    self.num_spherical_features,
                    self.max_degree,
                    include_pseudotensors=False,
                )
                for _ in range(self.num_message_passing)
            ]
        )

        # e3x.nn.TensorDense on the aggregated spherical density.
        self.tensor_dense = TensorDense(
            in_features=self.num_radial,
            out_features=self.num_spherical_features,
            in_max_degree=self.max_degree,
            out_max_degree=self.max_degree,
            include_pseudotensors=False,
        )
        n_norm = (self.max_degree + 1) * self.num_spherical_features
        self.norm_update = torch.nn.Sequential(
            torch.nn.Linear(n_norm, 2 * self.num_features),
            torch.nn.SiLU(),
            torch.nn.Linear(2 * self.num_features, self.num_features),
        )
        # lorem-jax folds each message-passing step's updated spherical
        # features back into the scalar ones via their degree-norms (a fresh
        # Update MLP per step, same shape as `norm_update` above) -- without
        # this the equivariant update has no path to the energy at all (dead
        # computation, zero gradient).
        self.equivariant_mp_norm_update = torch.nn.ModuleList(
            [
                torch.nn.Sequential(
                    torch.nn.Linear(n_norm, 2 * self.num_features),
                    torch.nn.SiLU(),
                    torch.nn.Linear(2 * self.num_features, self.num_features),
                )
                for _ in range(self.num_message_passing)
            ]
        )
        self.norm_after_density = torch.nn.LayerNorm(self.num_features)
        self.residual_after_density = torch.nn.Sequential(
            torch.nn.Linear(self.num_features, 2 * self.num_features),
            torch.nn.SiLU(),
            torch.nn.Linear(2 * self.num_features, self.num_features),
        )
        self.norm_after_residual = torch.nn.LayerNorm(self.num_features)

        if self.max_degree > 2:
            try:
                from sphericart.torch import SphericalHarmonics
            except ImportError as err:
                raise ImportError(
                    "sphericart-torch is required for LOREM with max_degree > 2. "
                    "Install it with `pip install metatrain[lorem]`."
                ) from err
            # sphericart >= 0.5 dropped ``normalized`` and is an nn.Module.
            try:
                calculator = SphericalHarmonics(l_max=self.max_degree, normalized=True)
            except TypeError:
                calculator = SphericalHarmonics(l_max=self.max_degree)
            if isinstance(calculator, torch.nn.Module):
                self.spherical_harmonics = calculator
            else:
                self.spherical_harmonics = _SphericartWrapper(calculator)
        else:
            self.spherical_harmonics = _AnalyticSphericalHarmonics(self.max_degree)

    def _invariant_density(self, density: torch.Tensor) -> torch.Tensor:
        """Contract ``(n_atoms, n_radial, n_lm)`` to rotationally invariant features."""
        parts: List[torch.Tensor] = []
        for ell in range(self.max_degree + 1):
            start = ell * ell
            end = (ell + 1) * (ell + 1)
            chunk = density[:, :, start:end]
            if ell == 0:
                parts.append(chunk.squeeze(-1))
            else:
                parts.append(_safe_vector_norm(chunk, dim=-1))
        return torch.cat(parts, dim=-1)

    def forward(
        self, systems: List[System]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return per-atom scalar features, neighbor distances, and spherical features.

        :param systems: Batch of systems with the requested neighbor list attached.
        :return: ``(features, neighbor_distances, spherical_features)`` where
            ``features`` is ``(n_atoms_total, num_features)`` and
            ``spherical_features`` is the CG-mixed neighbor density as
            ``(n_atoms_total, (max_degree + 1) ** 2, num_spherical_features)``.
        """
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
        features = self.species_embedding(species)

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
        cutoff_weights = _cosine_cutoff(distances, self.cutoff, self.cutoff_width)
        if self.use_bernstein:
            radial = bernstein_basis(distances, self.cutoff, self.bernstein_coeff)
        else:
            radial = bessel_basis(distances, self.num_radial, self.cutoff)
        radial = radial * cutoff_weights.unsqueeze(-1)

        sh = self.spherical_harmonics(vectors)
        if self.use_e3x_sh:
            sh = to_racah(sh, self.max_degree)
        edge_density = (radial.unsqueeze(-1) * sh.unsqueeze(1)).reshape(
            distances.shape[0], self.num_radial * self.n_lm
        )
        density_flat = torch.zeros(
            n_atoms, self.num_radial * self.n_lm, device=device, dtype=dtype
        )
        if edge_density.shape[0] > 0:
            density_flat.index_add_(0, centers, edge_density)
        density = density_flat.view(n_atoms, self.num_radial, self.n_lm)
        invariants = self._invariant_density(density)
        spherical_features = self.tensor_dense(density.transpose(1, 2))
        features = self.feature_mlp(torch.cat([features, invariants], dim=-1))
        features = features + self.norm_update(
            _degree_norms(spherical_features, self.max_degree)
        )
        features = self.norm_after_density(features)
        features = features + self.residual_after_density(features)
        features = self.norm_after_residual(features)

        for layer, edge_coefficients, equivariant_layer, equivariant_norm_update in zip(
            self.mp_layers,
            self.equivariant_mp_edge_coefficients,
            self.equivariant_mp_layers,
            self.equivariant_mp_norm_update,
            strict=True,
        ):
            edge_features = torch.cat([features[centers], features[neighbors]], dim=-1)
            messages = layer(edge_features)
            messages = messages * cutoff_weights.unsqueeze(-1)
            update = torch.zeros_like(features)
            if messages.shape[0] > 0:
                update.index_add_(0, centers, messages)
            features = features + update

            # Equivariant update: lorem-jax refines the spherical node
            # features at every message-passing step too (see
            # EquivariantMessagePass). Build this step's per-edge spherical
            # "basis" from a fresh scalar-edge -> per-degree-coefficient
            # projection, broadcast onto that edge's spherical harmonics.
            coefficients = edge_coefficients(edge_features).reshape(
                distances.shape[0], self.max_degree + 1, self.num_spherical_features
            )
            edge_parts: List[torch.Tensor] = []
            for ell in range(self.max_degree + 1):
                start = ell * ell
                end = (ell + 1) * (ell + 1)
                edge_parts.append(
                    coefficients[:, ell : ell + 1, :] * sh[:, start:end].unsqueeze(-1)
                )
            edges_spherical = torch.cat(edge_parts, dim=1)
            edges_spherical = edges_spherical * cutoff_weights.unsqueeze(-1).unsqueeze(
                -1
            )
            spherical_features = equivariant_layer(
                spherical_features, edges_spherical, centers, neighbors
            )
            features = features + equivariant_norm_update(
                _degree_norms(spherical_features, self.max_degree)
            )

        return features, distances, spherical_features
