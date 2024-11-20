from pet.pet import PET, SelfContributionsWrapper
from pet.hypers import Hypers
from .fine_tuning import LoRAWrapper
from . import update_state_dict
from typing import List, Dict
import numpy as np


def load_raw_pet_model(
    state_dict: Dict,
    hypers: Dict,
    atomic_types: List,
    self_contributions: np.ndarray,
) -> PET:
    """Creates a raw PET model instance."""

    ARCHITECTURAL_HYPERS = Hypers(hypers)
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
