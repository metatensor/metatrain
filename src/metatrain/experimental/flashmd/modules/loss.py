from typing import Any, Dict, List, Optional

import metatensor.torch as mts
import torch
from metatensor.torch import TensorMap
from metatomic.torch import System

from metatrain.utils.data import TargetInfo
from metatrain.utils.loss import LossAggregator


class FlashMDLoss(LossAggregator):
    """
    Documentation.
    """

    def __init__(
        self, targets: Dict[str, TargetInfo], config: Dict[str, Dict[str, Any]]
    ):
        """
        :param targets: mapping from target names to :py:class:`TargetInfo`.
        :param config: per-target configuration dict.
        """
        # Check that the keys targets and config are "positions" and "momenta"
        expected_keys = {"positions", "momenta"}
        if set(targets.keys()) != expected_keys:
            raise ValueError(
                f"Expected target keys to be {expected_keys}, got {set(targets.keys())}"
            )
        if set(config.keys()) != expected_keys:
            raise ValueError(
                f"Expected config keys to be {expected_keys}, got {set(config.keys())}"
            )

        super().__init__(targets, config)

    def compute(  # type: ignore
        self,
        systems: List[System],
        predictions: Dict[str, TensorMap],
        targets: Dict[str, TensorMap],
        extra_data: Optional[Any] = None,
    ) -> torch.Tensor:
        """
        Sum over all scheduled losses present in the predictions.
        """
        if extra_data is not None:
            raise ValueError("FlashMDLoss does not accept extra_data.")

        if "positions" not in predictions or "momenta" not in predictions:
            raise ValueError("Predictions must contain both 'positions' and 'momenta'.")
        if "positions" not in targets or "momenta" not in targets:
            raise ValueError("Targets must contain both 'positions' and 'momenta'.")

        # Scaling by the square root of the masses

        all_masses = []
        for system in systems:
            if "masses" not in system.known_data():
                raise ValueError(
                    "System is missing 'masses' data required for FlashMDLoss."
                )
            masses = system.get_data("masses").block().values.squeeze(-1)  # [n_atoms]
            all_masses.append(masses)
        all_masses = torch.concat(all_masses, dim=0)  # [total_n_atoms]

        scaled_predictions = {}
        scaled_targets = {}
        scaled_predictions["positions"] = scale_tensor_map_by_sqrt_masses(
            predictions["positions"], "multiply", all_masses
        )
        scaled_targets["positions"] = scale_tensor_map_by_sqrt_masses(
            targets["positions"], "multiply", all_masses
        )
        scaled_predictions["momenta"] = scale_tensor_map_by_sqrt_masses(
            predictions["momenta"], "divide", all_masses
        )
        scaled_targets["momenta"] = scale_tensor_map_by_sqrt_masses(
            targets["momenta"], "divide", all_masses
        )

        return super().compute(scaled_predictions, scaled_targets)


def scale_tensor_map_by_sqrt_masses(
    tensor_map: TensorMap, multiply_or_divide: str, masses: torch.Tensor
) -> TensorMap:
    """
    Scale the values in a TensorMap by the square root of the atomic masses.

    :param tensor_map: input :py:class:`TensorMap`
    :param multiply_or_divide: either "multiply" or "divide" to specify the operation.
    :param masses: 1D tensor of atomic masses, shape [n_atoms]
    :return: new :py:class:`TensorMap` with scaled values.
    """

    if multiply_or_divide not in ("multiply", "divide"):
        raise ValueError("multiply_or_divide must be either 'multiply' or 'divide'.")

    if len(tensor_map) != 1:
        raise ValueError("Expected TensorMap with a single block.")

    if len(tensor_map.block().shape) != 3:
        raise ValueError("Expected TensorMap blocks to have three dimensions.")

    return mts.TensorMap(
        keys=tensor_map.keys,
        blocks=[
            mts.TensorBlock(
                samples=tensor_map.block().samples,
                components=tensor_map.block().components,
                properties=tensor_map.block().properties,
                values=(
                    tensor_map.block().values
                    * torch.sqrt(masses[:, None, None])  # [n_atoms, 1, 1]
                    if multiply_or_divide == "multiply"
                    else tensor_map.block().values
                    / torch.sqrt(masses[:, None, None])  # [n_atoms, 1, 1]
                ),
            )
        ],
    )
