"""System-level conditioning embeddings for charge and spin."""

from typing import Callable, Dict, List, Tuple

import torch
from metatensor.torch import Labels, TensorBlock, TensorMap
from metatomic.torch import System


class SystemConditioningEmbedding(torch.nn.Module):
    """Embeds per-system charge and spin multiplicity into per-atom features.

    Each system in a batch can have a different total charge and spin multiplicity.
    These are embedded via learned lookup tables, concatenated, and projected down
    to the model's feature dimension. The resulting per-system embedding is broadcast
    to all atoms belonging to that system.

    The module declares :attr:`required_data_keys` so that the training pipeline
    can automatically set up the data transforms that attach the required values
    to each ``System``, without the trainer needing to know the key names.

    :param d_out: Output embedding dimension (should match d_node).
    :param max_charge: Maximum absolute charge value. Supports charges in
        the range ``[-max_charge, +max_charge]``.
    :param max_spin: Maximum spin multiplicity (2S+1). Supports values in
        the range ``[1, max_spin]``.
    """

    required_data_keys: List[str] = ["mtt::charge", "mtt::spin"]

    def __init__(self, d_out: int, max_charge: int = 10, max_spin: int = 10):
        super().__init__()
        self.max_charge = max_charge
        self.max_spin = max_spin
        d_inner = d_out
        self.charge_embedding = torch.nn.Embedding(2 * max_charge + 1, d_inner)
        self.spin_embedding = torch.nn.Embedding(max_spin, d_inner)
        gate = torch.nn.Linear(d_inner, d_out)
        torch.nn.init.zeros_(gate.weight)
        torch.nn.init.zeros_(gate.bias)
        self.project = torch.nn.Sequential(
            torch.nn.Linear(2 * d_inner, d_inner),
            torch.nn.SiLU(),
            gate,
        )

    def validate(self, charge: torch.Tensor, spin: torch.Tensor) -> None:
        """Check that charge and spin values are within the supported range.

        Call this outside of ``torch.compile`` regions to get descriptive errors.

        :param charge: Per-system total charge, shape ``[n_systems]``.
        :param spin: Per-system spin multiplicity, shape ``[n_systems]``.
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
        c_emb = self.charge_embedding(charge + self.max_charge)  # [n_systems, d_out]
        s_emb = self.spin_embedding(spin - 1)  # [n_systems, d_out]
        system_emb = self.project(torch.cat([c_emb, s_emb], dim=-1))  # [n_systems, d]
        return system_emb[system_indices]  # [n_atoms, d_out]


def get_system_conditioning_transform(
    conditioning_keys: List[str],
) -> Callable[
    [List[System], Dict[str, TensorMap], Dict[str, TensorMap]],
    Tuple[List[System], Dict[str, TensorMap], Dict[str, TensorMap]],
]:
    """Return a CollateFn callable that moves conditioning data from the extra
    dict into each System via ``add_data``.

    After ``group_and_join`` the extra dict contains batched TensorMaps keyed by
    ``"mtt::charge"`` and ``"mtt::spin"``, with samples indexed 0, 1, … matching
    the position of each system in the batch list.  This callable re-attaches
    per-system scalars to the System objects so the PET model can read them with
    ``system.get_data("mtt::charge")``.

    NaN values are treated as missing: the corresponding system will not have the
    data key attached and the model will fall back to its default (charge=0, spin=1).

    :param conditioning_keys: List of extra_data keys present in
        ``extra_data_info``, e.g. ``["mtt::charge", "mtt::spin"]``.
    :return: A three-argument callable ``(systems, targets, extra) -> (systems,
        targets, extra)``.
    """

    def transform(
        systems: List[System],
        targets: Dict[str, TensorMap],
        extra: Dict[str, TensorMap],
    ) -> Tuple[List[System], Dict[str, TensorMap], Dict[str, TensorMap]]:
        for key in conditioning_keys:
            if key not in extra:
                continue
            prop_name = key.split("::")[-1]
            block = extra[key].block()
            for row_idx in range(len(block.samples)):
                sys_idx = int(block.samples.entry(row_idx)["system"])
                val = block.values[row_idx : row_idx + 1]  # shape [1, n_props]
                if torch.isnan(val).any():
                    continue
                systems[sys_idx].add_data(
                    key,
                    TensorMap(
                        keys=Labels.single(),
                        blocks=[
                            TensorBlock(
                                values=val,
                                samples=Labels(
                                    "system",
                                    torch.tensor(
                                        [[sys_idx]],
                                        device=val.device,
                                        dtype=torch.int32,
                                    ),
                                ),
                                components=[],
                                properties=Labels.range(prop_name, val.shape[-1]),
                            )
                        ],
                    ),
                )
        return systems, targets, extra

    return transform
