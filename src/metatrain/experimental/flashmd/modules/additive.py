from typing import Dict, List, Optional

import torch
from metatensor.torch import Labels, TensorBlock, TensorMap
from metatomic.torch import ModelOutput, NeighborListOptions, System

from metatrain.utils.data import DatasetInfo, TargetInfo
from metatrain.utils.jsonschema import validate


class PositionAdditive(torch.nn.Module):
    """
    A simple model for short-range repulsive interactions.
    """

    def __init__(self, hypers: Dict, dataset_info: DatasetInfo):
        super().__init__()

        # `hypers` should be an empty dictionary
        validate(
            instance=hypers,
            schema={"type": "object", "additionalProperties": False},
        )

        self.dataset_info = dataset_info
        self.atomic_types = sorted(dataset_info.atomic_types)

        self.outputs = {
            key: ModelOutput(
                quantity=value.quantity,
                unit=value.unit,
                per_atom=True,
            )
            for key, value in dataset_info.targets.items()
        }

    def restart(self, dataset_info: DatasetInfo) -> "PositionAdditive":
        """Restart the model with a new dataset info.

        :param dataset_info: New dataset information to be used.
        :return: The restarted model.
        """

        return self({}, self.dataset_info.union(dataset_info))

    def supported_outputs(self) -> Dict[str, ModelOutput]:
        return self.outputs

    def forward(
        self,
        systems: List[System],
        outputs: Dict[str, ModelOutput],
        selected_atoms: Optional[Labels] = None,
    ) -> Dict[str, TensorMap]:
        """Compute the energies of a system solely based on a ZBL repulsive
        potential.

        :param systems: List of systems to calculate the ZBL energy.
        :param outputs: Dictionary containing the model outputs.
        :param selected_atoms: Optional selection of atoms for which to compute the
            predictions.
        :return: A dictionary with the computed predictions for each system.

        :raises ValueError: If the `outputs` contain unsupported keys.
        """

        # TODO: SELECTED ATOMS

        all_positions = torch.concatenate([system.positions for system in systems])
        return {
            "positions": TensorMap(
                keys=Labels(
                    names=["_"],
                    values=torch.zeros(
                        (1, 1), dtype=torch.int32, device=all_positions.device
                    ),
                ),
                blocks=[
                    TensorBlock(
                        values=all_positions.unsqueeze(-1),
                        samples=Labels(
                            names=["system", "atom"],
                            values=torch.tensor(
                                [
                                    [i, j]
                                    for i, system in enumerate(systems)
                                    for j in range(len(system))
                                ],
                                dtype=torch.int32,
                                device=all_positions.device,
                            ),
                        ),
                        components=[
                            Labels(
                                names=["xyz"],
                                values=torch.arange(
                                    3, device=all_positions.device
                                ).unsqueeze(-1),
                            )
                        ],
                        properties=Labels(
                            names=["_"],
                            values=torch.zeros(
                                (1, 1), dtype=torch.int32, device=all_positions.device
                            ),
                        ),
                    )
                ],
            )
        }

    def requested_neighbor_lists(self) -> List[NeighborListOptions]:
        return []

    @staticmethod
    def is_valid_target(target_name: str, target_info: TargetInfo) -> bool:
        return True
