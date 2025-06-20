import logging
from typing import Dict, List, Optional

import metatensor.torch as mts
import torch
from metatensor.torch import Labels, TensorMap
from metatensor.torch.learn.data import DataLoader
from metatomic.torch import ModelOutput, System

from ..data import DatasetInfo, TargetInfo
from ..jsonschema import validate
from ._base_composition import BaseCompositionModel


class CompositionModel(torch.nn.Module):
    """
    A simple model that calculates the per-species contributions to targets
    based on the stoichiometry in a system.

    :param model_hypers: A dictionary of model hyperparameters. The paramater is ignored
        and is only present to be consistent with the general model API.
    :param dataset_info: An object containing information about the dataset, including
        target quantities and atomic types.
    """

    def __init__(
        self,
        model_hypers: Dict,
        dataset_info: DatasetInfo,
    ) -> None:
        super().__init__()

        # `model_hypers` should be an empty dictionary
        validate(
            instance=model_hypers,
            schema={"type": "object", "additionalProperties": False},
        )

        self.dataset_info = dataset_info
        self.atomic_types = sorted(dataset_info.atomic_types)

        self.register_buffer(
            "type_to_index", torch.empty(max(self.atomic_types) + 1, dtype=torch.long)
        )
        for i, atomic_type in enumerate(self.atomic_types):
            self.type_to_index[atomic_type] = i

        for target_name, target_info in dataset_info.targets.items():
            if not self.is_valid_target(target_name, target_info):
                raise ValueError(
                    f"Composition model does not support target quantity "
                    f"{target_info.quantity}. This is an architecture bug. "
                    "Please report this issue and help us improve!"
                )

        self.new_targets = {
            target_name: target_info
            for target_name, target_info in dataset_info.targets.items()
        }

        # Initialize the composition model
        self.model = BaseCompositionModel(
            atomic_types=self.atomic_types,
            layouts={
                target_name: target.layout
                for target_name, target in self.dataset_info.targets.items()
            },
        )

        self.outputs: Dict[str, ModelOutput] = {}
        for target_name, target_info in self.dataset_info.targets.items():
            self._add_output(target_name, target_info)

        # keeps track of dtype and device of the composition model
        self.register_buffer("dummy_buffer", torch.randn(1))

    def train_model(
        self,
        dataloader: DataLoader,
        # additive_models: List[torch.nn.Module],  # TODO: support this here?
        # fixed_weights: Optional[Dict[str, Dict[int, str]]] = None,
        # # TODO: support this here?
    ) -> None:
        self.model.fit(dataloader)

    def restart(self, dataset_info: DatasetInfo) -> "CompositionModel":
        """Restart the model with a new dataset info.

        :param dataset_info: New dataset information to be used.
        """
        for target_name, target_info in dataset_info.targets.items():
            if not self.is_valid_target(target_name, target_info):
                raise ValueError(
                    f"Composition model does not support target "
                    f"{target_name}. This is an architecture bug. "
                    "Please report this issue and help us improve!"
                )

        # merge old and new dataset info
        merged_info = self.dataset_info.union(dataset_info)
        new_atomic_types = [
            at for at in merged_info.atomic_types if at not in self.atomic_types
        ]

        if len(new_atomic_types) > 0:
            raise ValueError(
                f"New atomic types found in the dataset: {new_atomic_types}. "
                "The composition model does not support adding new atomic types."
            )

        self.new_targets = {
            target_name: target_info
            for target_name, target_info in merged_info.targets.items()
            if target_name not in self.dataset_info.targets
        }

        self.dataset_info = merged_info

        # register new outputs
        for target_name, target_info in self.new_targets.items():
            self._add_output(target_name, target_info)

        return self

    def forward(
        self,
        systems: List[System],
        outputs: Dict[str, ModelOutput],
        selected_samples: Optional[Labels] = None,
    ) -> Dict[str, TensorMap]:
        """Compute the targets for each system based on the composition weights.

        :param systems: List of systems to calculate the energy.
        :param outputs: Dictionary containing the model outputs.
        :param selected_samples: Optional selection of samples for which to compute the
            predictions.
        :returns: A dictionary with the computed predictions for each system.

        :raises ValueError: If no weights have been computed or if `outputs` keys
            contain unsupported keys.
        """
        for output_name in outputs.keys():
            if output_name not in self.outputs:
                raise ValueError(
                    f"Output {output_name} is not supported by the "
                    "composition model. Supported outputs are: "
                    f"{list(self.outputs.keys())}"
                )

        return self.model.forward(
            systems,
            outputs=outputs,
            selected_samples=selected_samples,
        )

    def supported_outputs(self) -> Dict[str, ModelOutput]:
        return self.outputs

    def _add_output(self, target_name: str, target_info: TargetInfo) -> None:
        self.outputs[target_name] = ModelOutput(
            quantity=target_info.quantity,
            unit=target_info.unit,
            per_atom=True,
        )

    @staticmethod
    def is_valid_target(target_name: str, target_info: TargetInfo) -> bool:
        """Finds if a ``TargetInfo`` object is compatible with a composition model.

        :param target_info: The ``TargetInfo`` object to be checked.
        """
        # only scalars can have composition contributions
        if not target_info.is_scalar and not target_info.is_spherical:
            logging.debug(
                f"Composition model does not support target {target_name} "
                "since it is not either scalar or spherical."
            )
            return False
        if (
            target_info.is_spherical
            and len(target_info.layout.blocks({"o3_lambda": 0, "o3_sigma": 1})) == 0
        ):
            logging.debug(
                f"Composition model does not support spherical target {target_name} "
                "since it does not have any invariant blocks."
            )
            return False
        return True

    def sync_tensor_maps(self):
        # Reload the weights of the (old) targets, which are not stored in the model
        # state_dict, from the buffers
        for k in self.dataset_info.targets:
            self.model.weights[k] = mts.load_buffer(
                self.__getattr__(k + "_composition_buffer")
            )
