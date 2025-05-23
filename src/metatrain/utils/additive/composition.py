from typing import Dict, List

import torch
from metatensor.torch import TensorMap
from metatensor.torch.atomistic import System
from metatensor.torch.learn.data import DataLoader

from ..data import DatasetInfo
from ..jsonschema import validate
from ._base_composition import BaseCompositionModel


class MetatrainCompositionModel(torch.nn.Module):
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

        # keeps track of dtype and device of the composition model
        self.register_buffer("dummy_buffer", torch.randn(1))

        # Initialize the composition model
        self.model = BaseCompositionModel(
            atomic_types=self.atomic_types,
            layouts={
                target_name: target.layout
                for target_name, target in self.dataset_info.targets.items()
            },
        )

    def fit(self, dataloader: DataLoader) -> None:
        self.model.fit(dataloader)

    def forward(self, systems: List[System]) -> Dict[str, TensorMap]:
        return self.model.forward(systems)
