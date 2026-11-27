from typing import Any, Optional

import metatensor.torch as mts
from metatensor.torch import Labels, TensorBlock, TensorMap
from metatomic.torch import ModelOutput, System

from metatrain.utils.data import DatasetInfo, TargetInfo

from ..abc import ExtenderInterface
from ..helpers import UnavailableOutputError
from .documentation import Hypers


class MeanPooling(ExtenderInterface[Hypers]):
    """Average an architecture's atomic predictions within each system.

    :param hypers: Extender options specifying the output name.
    :param dataset_info: The target metadata for the training data.
    """

    __checkpoint_version__ = 1

    def __init__(self, hypers: Hypers, dataset_info: DatasetInfo):
        super().__init__(hypers, dataset_info)
        self.output_name = hypers["outputs"]
        if self.output_name not in dataset_info.targets:
            raise UnavailableOutputError(
                f"Missing mean_pooling target {self.output_name}"
            )
        target = dataset_info.targets[self.output_name]
        if (
            not target.is_scalar
            or target.sample_kind != "system"
            or len(target.layout.block().properties) != 1
        ):
            raise ValueError(
                "mean_pooling requires a single scalar property per system"
            )
        block = target.layout.block()
        self.input_name = (
            f"mtt::extender_aux::mean_pooling::{self.output_name.removeprefix('mtt::')}"
        )
        atomic_block = TensorBlock(
            values=block.values,
            samples=Labels(["system", "atom"], block.samples.values.reshape(0, 2)),
            components=[],
            properties=block.properties,
        )
        self.atomic_input = TargetInfo(
            TensorMap(target.layout.keys, [atomic_block]),
            quantity=target.quantity,
            unit=target.unit,
        )
        available = {**dataset_info.targets, **dataset_info.extra_data}
        if (
            self.input_name in available
            and available[self.input_name] != self.atomic_input
        ):
            raise ValueError(f"Incompatible layout for atomic input {self.input_name}")
        self.target = target

    def supported_outputs(self) -> dict[str, TargetInfo]:
        return {self.output_name: self.target}

    def requested_extender_inputs(
        self, outputs: Optional[dict[str, ModelOutput]] = None
    ) -> dict[str, TargetInfo]:
        if outputs is None or self.output_name in outputs:
            return {self.input_name: self.atomic_input}
        return {}

    def forward(
        self,
        systems: list[System],
        outputs: dict[str, ModelOutput],
        inputs: dict[str, TensorMap],
        selected_atoms: Optional[Labels] = None,
    ) -> dict[str, TensorMap]:
        if self.output_name not in outputs:
            return {}
        if outputs[self.output_name].sample_kind != "system":
            raise ValueError("mean_pooling only produces system-level outputs")
        return {
            self.output_name: mts.mean_over_samples(inputs[self.input_name], "atom")
        }

    def get_checkpoint(self, best_state_dict: Optional[dict[str, Any]] = None) -> dict:
        return {
            "extender_name": "mean_pooling",
            "extender_ckpt_version": self.__checkpoint_version__,
            "extender_data": {"hypers": self.hypers, "dataset_info": self.dataset_info},
            "model_state_dict": self.state_dict(),
        }

    @classmethod
    def load_checkpoint(cls, checkpoint: dict[str, Any], context: str) -> "MeanPooling":
        if checkpoint["extender_ckpt_version"] != cls.__checkpoint_version__:
            raise ValueError("Unsupported mean_pooling checkpoint version")
        extender = cls(**checkpoint["extender_data"])
        extender.load_state_dict(checkpoint["model_state_dict"])
        return extender
