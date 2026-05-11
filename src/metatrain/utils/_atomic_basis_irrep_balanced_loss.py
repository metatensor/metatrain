from __future__ import annotations

from typing import Any, Dict, List

import torch
from metatensor.torch import TensorMap

from metatrain.utils.data.target_info import TargetInfo
from metatrain.utils.scaler import Scaler


class _AtomicBasisIrrepBalancedLoss:
    """Private opt-in loss for spherical atomic-basis fair-control studies.

    Atomic-basis targets can have many sparse blocks for the same irrep, split by
    species/property metadata. This objective compares predictions and targets in
    physical sparse coefficient space, groups blocks by ``(o3_lambda, o3_sigma)``,
    normalizes each group by one RMS scale derived from the fitted scaler, and then
    averages groups equally. Metrics and model outputs remain unchanged.

    This helper is deliberately isolated so the experimental PET/E-PET hook can be
    reverted cleanly: remove this file, the ``atomic_basis_irrep_balanced_loss``
    trainer option, and the two call sites in PET/E-PET trainers.
    """

    def __init__(
        self,
        target_infos: Dict[str, TargetInfo],
        config: Dict[str, Dict[str, Any]],
        scaler: Scaler,
        scale_targets: bool,
    ):
        self.target_weights: Dict[str, float] = {}
        self.group_scales: Dict[str, Dict[tuple[int, int], torch.Tensor]] = {}

        if config and not scale_targets:
            raise ValueError(
                "atomic_basis_irrep_balanced_loss requires scale_targets=true."
            )

        for target_name, target_config in config.items():
            if target_name not in target_infos:
                raise ValueError(
                    f"atomic_basis_irrep_balanced_loss target '{target_name}' is "
                    "not present in the dataset targets."
                )
            target_info = target_infos[target_name]
            if not target_info.is_atomic_basis or not target_info.is_spherical:
                raise ValueError(
                    "atomic_basis_irrep_balanced_loss is only supported for "
                    f"per-atom spherical atomic-basis targets, got '{target_name}'."
                )
            if not target_info.per_atom:
                raise ValueError(
                    "atomic_basis_irrep_balanced_loss only supports per-atom "
                    f"targets, got '{target_name}'."
                )
            if target_info.gradients:
                raise ValueError(
                    "atomic_basis_irrep_balanced_loss does not support target "
                    f"gradients, got '{target_name}'."
                )

            scale_mode = target_config.get("scale", "per_irrep_rms")
            if scale_mode != "per_irrep_rms":
                raise ValueError(
                    "atomic_basis_irrep_balanced_loss only supports "
                    "scale='per_irrep_rms'."
                )
            if target_config.get("gradients"):
                raise ValueError(
                    "atomic_basis_irrep_balanced_loss does not support configured "
                    f"gradients, got '{target_name}'."
                )

            self.target_weights[target_name] = float(target_config.get("weight", 1.0))
            self.group_scales[target_name] = self._compute_group_scales(
                target_name, scaler
            )

    @property
    def target_names(self) -> set[str]:
        return set(self.target_weights)

    @staticmethod
    def _irrep_group_from_key(key) -> tuple[int, int]:
        return int(key["o3_lambda"]), int(key["o3_sigma"])

    @classmethod
    def _compute_group_scales(
        cls, target_name: str, scaler: Scaler
    ) -> Dict[tuple[int, int], torch.Tensor]:
        scales = scaler.model.scales[target_name]
        values_by_group: Dict[tuple[int, int], List[torch.Tensor]] = {}
        for key, block in scales.items():
            group = cls._irrep_group_from_key(key)
            values = block.values
            if "atom_type" in key.names and block.samples.names == ["atomic_type"]:
                atom_type = int(key["atom_type"])
                type_index = int(scaler.model.type_to_index[atom_type].item())
                values = values[type_index]
            flat_values = values.reshape(-1)
            valid_values = flat_values[torch.isfinite(flat_values)]
            valid_values = valid_values[valid_values > 0]
            if valid_values.numel() == 0:
                continue
            values_by_group.setdefault(group, []).append(valid_values)

        group_scales: Dict[tuple[int, int], torch.Tensor] = {}
        for group, group_values in values_by_group.items():
            values = torch.cat(group_values)
            group_scales[group] = torch.sqrt(torch.mean(values.pow(2))).clamp_min(
                torch.finfo(values.dtype).eps
            )

        if not group_scales:
            raise ValueError(
                f"Could not compute any per-irrep scaler values for '{target_name}'."
            )
        return group_scales

    def compute(
        self,
        systems,
        predictions: Dict[str, TensorMap],
        targets: Dict[str, TensorMap],
        reverse_atomic_basis_transform,
        scaler: Scaler,
    ) -> torch.Tensor:
        first_tensor_map = next(iter(predictions.values()))
        first_block = first_tensor_map.block(first_tensor_map.keys[0])
        total_loss = torch.zeros(
            (), dtype=first_block.values.dtype, device=first_block.values.device
        )

        selected_predictions = {
            name: predictions[name] for name in self.target_names if name in predictions
        }
        if not selected_predictions:
            return total_loss
        selected_targets = {name: targets[name] for name in selected_predictions}

        dense_physical_predictions = scaler(systems, selected_predictions)
        dense_physical_targets = scaler(systems, selected_targets)
        _, sparse_predictions, _ = reverse_atomic_basis_transform(
            systems, dict(dense_physical_predictions), {}
        )
        _, sparse_targets, _ = reverse_atomic_basis_transform(
            systems, dict(dense_physical_targets), {}
        )

        for target_name, weight in self.target_weights.items():
            if target_name not in sparse_predictions:
                continue
            group_residuals: Dict[tuple[int, int], List[torch.Tensor]] = {}
            for key in sparse_predictions[target_name].keys:
                group = self._irrep_group_from_key(key)
                prediction_block = sparse_predictions[target_name].block(key)
                target_block = sparse_targets[target_name].block(key)
                valid_mask = torch.isfinite(target_block.values)
                if not valid_mask.any():
                    continue
                residual = prediction_block.values[valid_mask] - target_block.values[
                    valid_mask
                ]
                group_residuals.setdefault(group, []).append(residual.pow(2))

            group_losses = []
            for group, residuals in group_residuals.items():
                scale = self.group_scales[target_name][group].to(
                    device=total_loss.device, dtype=total_loss.dtype
                )
                group_losses.append(torch.cat(residuals).mean() / scale.pow(2))

            if group_losses:
                total_loss = total_loss + weight * torch.stack(group_losses).mean()

        return total_loss
