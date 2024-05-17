import torch
from typing import Dict
from ...utils.architectures import get_default_hypers, get_architecture_name

ARCHITECTURE_NAME: str = get_architecture_name(__file__)

__ARCHITECTURE_CAPABILITIES__ = {
    "supported_devices": ["cpu"],
    "supported_dtypes": [torch.float64, torch.float32],
}

DEFAULT_HYPERS: Dict = get_default_hypers(ARCHITECTURE_NAME)
DEFAULT_MODEL_HYPERS: Dict = DEFAULT_HYPERS["model"]

__authors__ = [
    # todo
]

__maintainers__ = [
    # todo
]

# load Model in train at the end to avoid circular imports
from .model import Model  # noqa
from .train import train  # noqa
