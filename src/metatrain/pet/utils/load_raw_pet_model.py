from typing import Dict, List

import numpy as np
import torch

from ..modules.hypers import Hypers
from ..modules.pet import PET, SelfContributionsWrapper
from .fine_tuning import LoRAWrapper
from .update_state_dict import update_state_dict


def load_raw_pet_model(
    state_dict: Dict,
    hypers: Dict,
    atomic_types: List,
    self_contributions: np.ndarray,
    **kwargs,
) -> "SelfContributionsWrapper":
    """Creates a raw PET model instance."""

    ARCHITECTURAL_HYPERS = Hypers(hypers)

    ARCHITECTURAL_HYPERS.D_OUTPUT = 1  # type: ignore
    ARCHITECTURAL_HYPERS.TARGET_AGGREGATION = "sum"  # type: ignore
    ARCHITECTURAL_HYPERS.TARGET_TYPE = "atomic"  # type: ignore

    raw_pet = PET(ARCHITECTURAL_HYPERS, 0.0, len(atomic_types))
    if "use_lora_peft" in kwargs and kwargs["use_lora_peft"] is True:
        lora_rank = kwargs["lora_rank"]
        lora_alpha = kwargs["lora_alpha"]
        raw_pet = LoRAWrapper(raw_pet, lora_rank, lora_alpha)

    new_state_dict = update_state_dict(state_dict)
    dtype = next(iter(new_state_dict.values())).dtype
    raw_pet.to(dtype).load_state_dict(new_state_dict)
    if isinstance(self_contributions, torch.Tensor):
        self_contributions = self_contributions.cpu().numpy()
    wrapper = SelfContributionsWrapper(raw_pet, self_contributions)

    return wrapper
