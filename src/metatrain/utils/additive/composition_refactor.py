import logging
from typing import Dict, List, Optional
import torch

from ..data import Dataset, DatasetInfo, TargetInfo, get_all_targets, get_atomic_types
from ..jsonschema import validate
from ..sum_over_atoms import sum_over_atoms
from ..transfer import systems_and_targets_to_device
from .remove import remove_additive

from ._base_composition import BaseCompositionModel


class CompositionModel(BaseCompositionModel):

     def __init__(
        self,
        model_hypers: Dict,
        dataset_info: DatasetInfo,
    ) -> None:
        
        super().__init__(
            atomic_types=self.atomic_types,
            layouts={
                target_name: target.layout
                for target_name, target in self.dataset_info.targets
            },
        )
        
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
        

    def fit():
    
        # make sure to update the weights buffer with the new weights
        self.register_buffer(
            target_name + "_composition_buffer",
            mts.save_buffer(self.weights[target_name].to("cpu", torch.float64)).to(
                device
            ),
        )