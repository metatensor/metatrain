from typing import Dict, List, Optional

import metatensor.torch as mts
import torch
from metatensor.torch import Labels, TensorBlock, TensorMap
from metatomic.torch import ModelOutput, NeighborListOptions, System

from metatrain.utils.data import DatasetInfo, TargetInfo
from metatrain.utils.jsonschema import validate


class PositionAdditive(torch.nn.Module):
    """
    A simple additive model that adds the positions of the system to any outputs that
    is either "positions" or one of its variants.

    Optionally, it can also do the same with momenta.
    """

    def __init__(self, hypers: Dict, dataset_info: DatasetInfo):
        super().__init__()

        validate(
            instance=hypers,
            schema={
                "type": "object",
                "properties": {
                    "also_momenta": {"type": "boolean"},
                },
                "required": ["also_momenta"],
                "additionalProperties": False,
            },
        )
        self.do_momenta = hypers["also_momenta"]

        self.dataset_info = dataset_info
        self.atomic_types = sorted(dataset_info.atomic_types)

        self.outputs = {}
        for key, value in dataset_info.targets.items():
            if (key == "momenta" or key.startswith("momenta/")) and not self.do_momenta:
                # skip momenta targets unless `also_momenta` is True
                continue
            self.outputs[key] = ModelOutput(
                quantity=value.quantity,
                unit=value.unit,
                per_atom=True,
            )

    def restart(self, dataset_info: DatasetInfo) -> "PositionAdditive":
        """Restart the model with a new dataset info.

        :param dataset_info: New dataset information to be used.
        :return: The restarted model.
        """

        self.dataset_info = self.dataset_info.union(dataset_info)
        self.outputs = {}
        for key, value in self.dataset_info.targets.items():
            if (key == "momenta" or key.startswith("momenta/")) and not self.do_momenta:
                # skip momenta targets unless `also_momenta` is True
                continue
            self.outputs[key] = ModelOutput(
                quantity=value.quantity,
                unit=value.unit,
                per_atom=True,
            )
        return self

    def supported_outputs(self) -> Dict[str, ModelOutput]:
        return self.outputs

    def forward(
        self,
        systems: List[System],
        outputs: Dict[str, ModelOutput],
        selected_atoms: Optional[Labels] = None,
    ) -> Dict[str, TensorMap]:
        """Adds positions (and optionally momenta) from the system to the appropriate
        outputs.

        :param systems: List of systems to compute the predictions for.
        :param outputs: Dictionary containing the model outputs.
        :param selected_atoms: Optional selection of atoms for which to compute the
            predictions.
        :return: A dictionary with the positions (and optionally momenta) of the
            systems as "predicted" values.
        """
        device = systems[0].positions.device

        # TODO: variants

        return_dict: Dict[str, TensorMap] = {}

        single_label = Labels(
            names=["_"],
            values=torch.zeros((1, 1), dtype=torch.int32, device=device),
        )
        system_indices = torch.concatenate(
            [
                torch.full((len(system),), i, dtype=torch.int32, device=device)
                for i, system in enumerate(systems)
            ]
        )
        atom_indices = torch.concatenate(
            [
                torch.arange(len(system), device=device, dtype=torch.int32)
                for system in systems
            ]
        )
        sample_values = torch.stack(
            [system_indices, atom_indices],
            dim=1,
        )
        samples = Labels(
            names=["system", "atom"],
            values=sample_values,
        )
        components = [
            Labels(
                names=["xyz"],
                values=torch.arange(3, device=device).unsqueeze(-1),
            )
        ]
        all_positions = torch.concatenate([system.positions for system in systems])
        position_tensor_map = TensorMap(
            keys=single_label,
            blocks=[
                TensorBlock(
                    values=all_positions.unsqueeze(-1),
                    samples=samples,
                    components=components,
                    properties=Labels(
                        names=["positions"],
                        values=torch.zeros(
                            (1, 1), dtype=torch.int32, device=all_positions.device
                        ),
                    ),
                )
            ],
        )
        return_dict["positions"] = position_tensor_map

        if self.do_momenta:
            all_momenta = torch.concatenate(
                [system.get_data("momenta").block().values for system in systems]
            )
            momenta_tensor_map = TensorMap(
                keys=single_label,
                blocks=[
                    TensorBlock(
                        values=all_momenta,
                        samples=samples,
                        components=components,
                        properties=Labels(
                            names=["momenta"],
                            values=torch.zeros(
                                (1, 1), dtype=torch.int32, device=all_positions.device
                            ),
                        ),
                    )
                ],
            )
            return_dict["momenta"] = momenta_tensor_map

        if selected_atoms is not None:
            for key in list(return_dict.keys()):
                return_dict[key] = mts.slice(
                    return_dict[key],
                    axis="samples",
                    selection=selected_atoms,
                )

        return return_dict

    def requested_neighbor_lists(self) -> List[NeighborListOptions]:
        return []

    @staticmethod
    def is_valid_target(target_name: str, target_info: TargetInfo) -> bool:
        if target_name == "positions" or target_name.startswith("positions/"):
            return True
        if target_name == "momenta" or target_name.startswith("momenta/"):
            return True
        return False
