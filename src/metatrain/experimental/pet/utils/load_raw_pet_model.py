from typing import Dict, List

import numpy as np
from pet.hypers import Hypers
from pet.pet import PET, SelfContributionsWrapper

from .fine_tuning import LoRAWrapper
from .update_state_dict import update_state_dict


def load_raw_pet_model(
    state_dict: Dict,
    hypers: Dict,
    atomic_types: List,
    self_contributions: np.ndarray,
) -> "SelfContributionsWrapper":
    """Creates a raw PET model instance."""

    ARCHITECTURAL_HYPERS = Hypers(hypers)

    ARCHITECTURAL_HYPERS.D_OUTPUT = 1
    ARCHITECTURAL_HYPERS.TARGET_AGGREGATION = "sum"
    ARCHITECTURAL_HYPERS.TARGET_TYPE = "atomic"

    raw_pet = PET(ARCHITECTURAL_HYPERS, 0.0, len(atomic_types))
    if ARCHITECTURAL_HYPERS.USE_LORA_PEFT:
        lora_rank = ARCHITECTURAL_HYPERS.LORA_RANK
        lora_alpha = ARCHITECTURAL_HYPERS.LORA_ALPHA
        raw_pet = LoRAWrapper(raw_pet, lora_rank, lora_alpha)

    new_state_dict = update_state_dict(state_dict)

    dtype = next(iter(new_state_dict.values())).dtype
    raw_pet.to(dtype).load_state_dict(new_state_dict)
    wrapper = SelfContributionsWrapper(raw_pet, self_contributions)

    return wrapper
