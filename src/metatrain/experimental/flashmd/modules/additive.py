from collections.abc import Mapping
from typing import Any, Dict, List, Literal, Optional

import metatensor.torch as mts
import torch
from metatensor.torch import Labels, TensorBlock, TensorMap
from metatomic.torch import (
    AtomisticModel,
    ModelCapabilities,
    ModelMetadata,
    ModelOutput,
    NeighborListOptions,
    System,
)
from pydantic import TypeAdapter
from typing_extensions import TypedDict

from metatrain.utils.abc import ModelInterface
from metatrain.utils.data import DatasetInfo, TargetInfo
from metatrain.utils.dtype import dtype_to_str
from metatrain.utils.metadata import merge_metadata


class PositionAdditiveHypers(TypedDict):
    also_momenta: bool


class PositionAdditive(ModelInterface):
    """
    A simple additive model that adds the positions of the system to any outputs that
    is either "position" or one of its variants.

    Optionally, it can also do the same with momenta.
    """

    __checkpoint_version__ = 1
    __supported_devices__ = ["cuda", "cpu"]
    __supported_dtypes__ = [torch.float32, torch.float64]
    __default_metadata__ = ModelMetadata()

    def __init__(self, hypers: PositionAdditiveHypers, dataset_info: DatasetInfo):
        super().__init__(hypers, dataset_info, self.__default_metadata__)

        TypeAdapter(PositionAdditiveHypers).validate_python(
            hypers, strict=True, extra="forbid"
        )
        self.do_momenta = hypers["also_momenta"]

        self.dataset_info = dataset_info
        self.atomic_types = sorted(dataset_info.atomic_types)

        self.outputs = {}
        for key, value in dataset_info.targets.items():
            if (
                key == "momentum" or key.startswith("momentum/")
            ) and not self.do_momenta:
                # skip momenta targets unless `also_momenta` is True
                continue
            self.outputs[key] = ModelOutput(
                unit=value.unit,
                sample_kind="atom",
                description=value.description,
            )

    def restart(
        self,
        dataset_info: DatasetInfo,
        model_hypers: Optional[Mapping[str, Any]] = None,
    ) -> "PositionAdditive":
        """Restart the model with a new dataset info.

        :param dataset_info: New dataset information to be used.
        :param model_hypers: Not used.
        :return: The restarted model.
        """

        self.dataset_info = self.dataset_info.union(dataset_info)
        self.outputs = {}
        for key, value in self.dataset_info.targets.items():
            if (
                key == "momentum" or key.startswith("momentum/")
            ) and not self.do_momenta:
                # skip momenta targets unless `also_momenta` is True
                continue
            self.outputs[key] = ModelOutput(
                unit=value.unit,
                sample_kind="atom",
                description=value.description,
            )
        return self

    def remove_output(self, target_name: str) -> None:
        """
        Remove a previously registered output target.

        :param target_name: Name of the target to remove.
        """
        self.outputs.pop(target_name, None)
        self.dataset_info.targets.pop(target_name, None)

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
                        names=["position"],
                        values=torch.zeros(
                            (1, 1), dtype=torch.int32, device=all_positions.device
                        ),
                    ),
                )
            ],
        )
        return_dict["position"] = position_tensor_map

        if self.do_momenta:
            all_momenta = torch.concatenate(
                [system.get_data("momentum").block().values for system in systems]
            )
            momenta_tensor_map = TensorMap(
                keys=single_label,
                blocks=[
                    TensorBlock(
                        values=all_momenta,
                        samples=samples,
                        components=components,
                        properties=Labels(
                            names=["momentum"],
                            values=torch.zeros(
                                (1, 1), dtype=torch.int32, device=all_positions.device
                            ),
                        ),
                    )
                ],
            )
            return_dict["momentum"] = momenta_tensor_map

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
        if target_name == "position" or target_name.startswith("position/"):
            return True
        if target_name == "momentum" or target_name.startswith("momentum/"):
            return True
        return False

    def get_checkpoint(
        self, best_model_state_dict: Optional[dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Get the checkpoint of the model. This should contain all the information
        needed by `load_checkpoint` to recreate the same model instance.

        :return: The model's checkpoint.
        """
        model_state_dict = self.state_dict()
        checkpoint = {
            "architecture_name": "flashmd_position_additive",
            "model_ckpt_version": self.__checkpoint_version__,
            "model_data": {
                "hypers": self.hypers,
                "dataset_info": self.dataset_info,
            },
            "model_state_dict": model_state_dict,
            "best_model_state_dict": best_model_state_dict or model_state_dict,
        }
        return checkpoint

    @classmethod
    def load_checkpoint(
        cls,
        checkpoint: Dict[str, Any],
        context: Literal["restart", "finetune", "export"],
    ) -> "PositionAdditive":
        """
        Create a model from a checkpoint (i.e. state dictionary).

        :param checkpoint: Checkpoint's state dictionary.
        :param context: Context in which to load the model. Not used.

        :return: An instance of the model.
        """
        model = cls(
            hypers=checkpoint["model_data"]["hypers"],
            dataset_info=checkpoint["model_data"]["dataset_info"],
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        return model

    @classmethod
    def upgrade_checkpoint(
        cls, checkpoint: Dict, version: Optional[int] = None
    ) -> Dict:
        """
        Upgrade the checkpoint to the current version of the model.

        :param checkpoint: Checkpoint's state dictionary.
        :param version: Version to which the checkpoint should be upgraded. If ``None``,
            the checkpoint will be upgraded to the current version of the model.

        :return: The upgraded checkpoint.
        """
        if version is None:
            version = cls.__checkpoint_version__
        if checkpoint["model_ckpt_version"] != version:
            raise NotImplementedError(
                "PositionAdditive does not support checkpoint upgrading. "
                "There is only one checkpoint version."
            )
        return checkpoint

    def export(self, metadata: Optional[ModelMetadata] = None) -> AtomisticModel:
        """
        Turn this model into an :py:class:`metatomic.torch.AtomisticModel`,
        containing the model itself, its capabilities and its metadata.

        :param metadata: Additional metadata to add to the model, as specified
            by the user.
        :return: An instance of :py:class:`metatomic.torch.AtomisticModel`.
        """
        capabilities = ModelCapabilities(
            outputs=self.outputs,
            atomic_types=self.atomic_types,
            interaction_range=0.0,
            length_unit=self.dataset_info.length_unit,
            supported_devices=self.__supported_devices__,
            dtype=dtype_to_str(torch.float64),
        )

        metadata = merge_metadata(self.metadata, metadata)

        return AtomisticModel(self.eval(), metadata, capabilities)
