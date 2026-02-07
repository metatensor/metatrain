"""System-level conditioning embeddings for charge and spin."""

import torch


class SystemConditioningEmbedding(torch.nn.Module):
    """Embeds per-system charge and spin multiplicity into per-atom features.

    Each system in a batch can have a different total charge and spin multiplicity.
    These are embedded via learned lookup tables, concatenated, and projected down
    to the model's feature dimension. The resulting per-system embedding is broadcast
    to all atoms belonging to that system.

    :param d_out: Output embedding dimension (should match d_pet).
    :param max_charge: Maximum absolute charge value. Supports charges in
        the range ``[-max_charge, +max_charge]``.
    :param max_spin: Maximum spin multiplicity (2S+1). Supports values in
        the range ``[1, max_spin]``.
    """

    def __init__(self, d_out: int, max_charge: int = 10, max_spin: int = 10):
        super().__init__()
        self.max_charge = max_charge
        self.max_spin = max_spin
        self.charge_embedding = torch.nn.Embedding(2 * max_charge + 1, d_out)
        self.spin_embedding = torch.nn.Embedding(max_spin, d_out)
        self.project = torch.nn.Sequential(
            torch.nn.Linear(2 * d_out, d_out),
            torch.nn.SiLU(),
        )

    def forward(
        self,
        charge: torch.Tensor,
        spin: torch.Tensor,
        system_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Compute per-atom conditioning features from per-system charge and spin.

        :param charge: Per-system total charge, shape ``[n_systems]``, integer.
        :param spin: Per-system spin multiplicity (2S+1), shape ``[n_systems]``,
            integer >= 1.
        :param system_indices: Maps each atom to its system index,
            shape ``[n_atoms]``.
        :return: Per-atom conditioning features, shape ``[n_atoms, d_out]``.
        """
        if (charge < -self.max_charge).any() or (charge > self.max_charge).any():
            raise ValueError(
                f"charge values must be in [{-self.max_charge}, "
                f"{self.max_charge}], got min={charge.min().item()}, "
                f"max={charge.max().item()}. Increase max_charge in "
                f"model hypers to support wider charge ranges."
            )
        if (spin < 1).any() or (spin > self.max_spin).any():
            raise ValueError(
                f"spin multiplicity values must be in [1, "
                f"{self.max_spin}], got min={spin.min().item()}, "
                f"max={spin.max().item()}. Increase max_spin in "
                f"model hypers to support higher spin multiplicities."
            )
        c_emb = self.charge_embedding(charge + self.max_charge)  # [n_systems, d_out]
        s_emb = self.spin_embedding(spin - 1)  # [n_systems, d_out]
        system_emb = self.project(torch.cat([c_emb, s_emb], dim=-1))  # [n_systems, d]
        return system_emb[system_indices]  # [n_atoms, d_out]
