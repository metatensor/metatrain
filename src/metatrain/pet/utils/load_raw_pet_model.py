from typing import Dict, List

import numpy as np
import torch

from ..modules.hypers import Hypers
from ..modules.pet import PET, SelfContributionsWrapper
from .fine_tuning import FinetuneWrapper
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
    if "use_ft" in kwargs and kwargs["use_ft"] is not None:
        raw_pet = FinetuneWrapper(raw_pet, **kwargs)

    new_state_dict = update_state_dict(state_dict)
    dtype = next(iter(new_state_dict.values())).dtype
    raw_pet.to(dtype).load_state_dict(new_state_dict)
    if isinstance(self_contributions, torch.Tensor):
        self_contributions = self_contributions.cpu().numpy()
    wrapper = SelfContributionsWrapper(raw_pet, self_contributions)

    return wrapper
