from typing import List

import torch
from metatomic.torch import System

from metatrain.utils.neighbor_lists import NeighborListOptions

from ..documentation import LoremLongRangeHypers
from .tensor_dense import TensorDense, TensorProduct


def _safe_vector_norm(
    x: torch.Tensor, dim: int, keepdim: bool = False, eps: float = 1.0e-12
) -> torch.Tensor:
    """``torch.linalg.vector_norm`` with a finite gradient at zero.

    ``vector_norm``'s backward is ``x / ||x||``, which is ``0/0 = nan`` whenever
    ``x`` is exactly zero along ``dim`` -- e.g. a vanishing ``m``-component of a
    spherical harmonic. Adding ``eps`` inside the square root keeps the value
    (and gradient) finite everywhere without materially changing it away from
    zero.
    """
    return torch.sqrt(x.pow(2).sum(dim=dim, keepdim=keepdim) + eps)


def _degree_norms(spherical: torch.Tensor, max_degree: int) -> torch.Tensor:
    """Per-ℓ, per-feature norms with the LOREM :math:`(2\\ell+1)^{1/4}` factor."""
    parts: List[torch.Tensor] = []
    for ell in range(max_degree + 1):
        chunk = spherical[:, ell * ell : (ell + 1) * (ell + 1), :]
        factor = (2.0 * float(ell) + 1.0) ** 0.25
        parts.append(factor * _safe_vector_norm(chunk, dim=1))
    return torch.cat(parts, dim=-1)


class LoremLongRangeFeaturizer(torch.nn.Module):
    """Ewald / P3M / direct Coulomb features from equivariant atomic charges.

    Scalar features map to one charge channel. Spherical node features go
    through ``TensorDense`` (the ``e3x.nn.TensorDense`` self-product) down to
    ``max_degree_lr``. Those channels are evaluated with torch-pme and mixed
    back with a CG ``TensorProduct``, matching ``lorem-jax``. ``lr_scale``
    gates the residual so a warm start is a zero perturbation.
    """

    def __init__(
        self,
        hypers: LoremLongRangeHypers,
        feature_dim: int,
        num_spherical_features: int,
        max_degree: int,
        max_degree_lr: int,
        neighbor_list_options: NeighborListOptions,
    ) -> None:
        super().__init__()
        if max_degree_lr < 0:
            raise ValueError(f"max_degree_lr must be >= 0, got {max_degree_lr}")

        try:
            from torchpme import (
                Calculator,
                CoulombPotential,
                EwaldCalculator,
                P3MCalculator,
            )
        except ImportError:
            raise ImportError(
                "`torch-pme` is required for long-range models. "
                "Please install it with `pip install 'torch-pme>=0.3.2'`."
            ) from None

        self.max_degree = int(max_degree)
        self.max_degree_lr = int(max_degree_lr)
        self.n_lm_lr = (self.max_degree_lr + 1) * (self.max_degree_lr + 1)
        self.num_spherical_features = int(num_spherical_features)
        self.feature_dim = int(feature_dim)
        self.use_ewald = bool(hypers["use_ewald"])
        self.neighbor_list_options = neighbor_list_options

        self.ewald_calculator = EwaldCalculator(
            potential=CoulombPotential(
                smearing=float(hypers["smearing"]),
                exclusion_radius=neighbor_list_options.cutoff,
            ),
            full_neighbor_list=neighbor_list_options.full_list,
            lr_wavelength=float(hypers["kspace_resolution"]),
        )
        self.p3m_calculator = P3MCalculator(
            potential=CoulombPotential(
                smearing=float(hypers["smearing"]),
                exclusion_radius=neighbor_list_options.cutoff,
            ),
            interpolation_nodes=hypers["interpolation_nodes"],
            full_neighbor_list=neighbor_list_options.full_list,
            mesh_spacing=float(hypers["kspace_resolution"]),
        )
        self.direct_calculator = Calculator(
            potential=CoulombPotential(
                smearing=None,
                exclusion_radius=neighbor_list_options.cutoff,
            ),
            full_neighbor_list=False,
        )

        # lorem-jax: MLP([2 * d, 1]) on scalar features.
        self.scalar_charge_mlp = torch.nn.Sequential(
            torch.nn.Linear(feature_dim, 2 * feature_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(2 * feature_dim, 1),
        )
        self.spherical_charge_dense = TensorDense(
            in_features=num_spherical_features,
            out_features=1,
            in_max_degree=self.max_degree,
            out_max_degree=self.max_degree_lr,
            include_pseudotensors=False,
        )
        self.potential_to_features = torch.nn.Linear(
            1, num_spherical_features, bias=False
        )
        self.potential_product = TensorProduct(
            left_max_degree=self.max_degree_lr,
            right_max_degree=self.max_degree,
            out_max_degree=self.max_degree,
            include_pseudotensors=False,
            n_features=num_spherical_features,
        )

        n_update = 1 + (self.max_degree + 1) * num_spherical_features
        self.update_from_potential = torch.nn.Sequential(
            torch.nn.Linear(n_update, 2 * feature_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(2 * feature_dim, feature_dim),
        )
        extras: dict = dict(hypers)
        raw_lr_scale = extras.get("lr_scale_init", 1.0)
        lr_scale_init = float(raw_lr_scale) if raw_lr_scale is not None else 1.0
        self.lr_scale = torch.nn.Parameter(torch.tensor([lr_scale_init]))
        self.update_residual = torch.nn.Sequential(
            torch.nn.Linear(feature_dim, 2 * feature_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(2 * feature_dim, feature_dim),
        )
        self.norm_after_potential = torch.nn.LayerNorm(feature_dim)
        self.norm_after_residual = torch.nn.LayerNorm(feature_dim)

    def map_charges(
        self, features: torch.Tensor, spherical_features: torch.Tensor
    ) -> torch.Tensor:
        """Map scalar and spherical features to concatenated charge channels.

        :param features: Invariant atom features ``(n_atoms, feature_dim)``.
        :param spherical_features: CG-mixed neighbor density
            ``(n_atoms, n_lm, num_spherical_features)``.
        :return: Charges ``(n_atoms, 1 + (max_degree_lr + 1) ** 2)`` — the
            leading channel is the scalar charge, the rest are spherical
            charges ordered ``ℓ = 0..max_degree_lr``, ``m = -ℓ..ℓ``.
        """
        scalar_charges = self.scalar_charge_mlp(features)
        spherical_charges = self.spherical_charge_dense(spherical_features)[:, :, 0]
        return torch.cat([scalar_charges, spherical_charges], dim=-1)

    def _potentials_for_system(
        self,
        system: System,
        system_charges: torch.Tensor,
        neighbor_indices_system: torch.Tensor,
        neighbor_distances_system: torch.Tensor,
    ) -> torch.Tensor:
        if system.pbc.any():
            if system.pbc.sum() == 1:
                raise NotImplementedError(
                    "Long-range featurizer does not support 1D systems."
                )
            if self.use_ewald and self.training:
                return self.ewald_calculator.forward(
                    charges=system_charges,
                    cell=system.cell,
                    positions=system.positions,
                    neighbor_indices=neighbor_indices_system,
                    neighbor_distances=neighbor_distances_system,
                    periodic=system.pbc,
                )
            return self.p3m_calculator.forward(
                charges=system_charges,
                cell=system.cell,
                positions=system.positions,
                neighbor_indices=neighbor_indices_system,
                neighbor_distances=neighbor_distances_system,
                periodic=system.pbc,
            )

        neighbor_indices_system = torch.combinations(
            torch.arange(len(system), device=system.positions.device), 2
        )
        neighbor_distances_system = torch.sqrt(
            torch.sum(
                (
                    system.positions[neighbor_indices_system[:, 1]]
                    - system.positions[neighbor_indices_system[:, 0]]
                )
                ** 2,
                dim=1,
            )
        )
        return self.direct_calculator.forward(
            charges=system_charges,
            cell=system.cell,
            positions=system.positions,
            neighbor_indices=neighbor_indices_system,
            neighbor_distances=neighbor_distances_system,
        )

    def _evaluate_potentials(
        self,
        systems: List[System],
        charges: torch.Tensor,
        neighbor_distances: torch.Tensor,
    ) -> torch.Tensor:
        last_len_nodes = 0
        last_len_edges = 0
        potentials: List[torch.Tensor] = []
        for system in systems:
            system_charges = charges[last_len_nodes : last_len_nodes + len(system)]
            last_len_nodes += len(system)

            neighbor_list = system.get_neighbor_list(self.neighbor_list_options)
            neighbor_indices_system = torch.stack(
                [
                    neighbor_list.samples.column("first_atom"),
                    neighbor_list.samples.column("second_atom"),
                ],
                dim=-1,
            )
            neighbor_distances_system = neighbor_distances[
                last_len_edges : last_len_edges + len(neighbor_indices_system)
            ]
            last_len_edges += len(neighbor_indices_system)
            potentials.append(
                self._potentials_for_system(
                    system,
                    system_charges,
                    neighbor_indices_system,
                    neighbor_distances_system,
                )
            )
        return torch.cat(potentials, dim=0)

    def _invariant_updates(
        self, potentials: torch.Tensor, spherical_features: torch.Tensor
    ) -> torch.Tensor:
        """CG-mix potentials into spherical features, then take degree norms."""
        scalar_potential = potentials[:, 0:1]
        spherical_potential = self.potential_to_features(
            potentials[:, 1:].unsqueeze(-1)
        )
        mixed = self.potential_product(spherical_potential, spherical_features)
        norms = _degree_norms(mixed, self.max_degree)
        return torch.cat([scalar_potential, norms], dim=-1)

    def forward(
        self,
        systems: List[System],
        features: torch.Tensor,
        neighbor_distances: torch.Tensor,
        spherical_features: torch.Tensor,
    ) -> torch.Tensor:
        """Return invariant long-range feature updates.

        :param systems: Systems with the requested neighbor list attached.
        :param features: Short-range invariant features.
        :param neighbor_distances: Neighbor distances matching that list.
        :param spherical_features: Short-range spherical features.
        :return: Updates of shape ``(n_atoms, feature_dim)``.
        """
        original = features
        charges = self.map_charges(features, spherical_features)
        potentials = self._evaluate_potentials(systems, charges, neighbor_distances)
        updates = self._invariant_updates(potentials, spherical_features)
        features = features + self.update_from_potential(updates)
        features = self.norm_after_potential(features)
        features = features + self.update_residual(features)
        features = self.norm_after_residual(features)
        return original + self.lr_scale * (features - original)
