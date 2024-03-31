from .model import Model, LLPRModel, DEFAULT_HYPERS  # noqa: F401
from .train import train  # noqa: F401
import torch
from typing import Dict
from ...utils.architectures import get_default_hypers, get_architecture_name

ARCHITECTURE_NAME: str = get_architecture_name(__file__)

__ARCHITECTURE_CAPABILITIES__ = {
    "supported_devices": ["cuda", "cpu"],
    "supported_dtypes": [torch.float64, torch.float32],
}

DEFAULT_HYPERS: Dict = get_default_hypers(ARCHITECTURE_NAME)
DEFAULT_MODEL_HYPERS: Dict = DEFAULT_HYPERS["model"]

__authors__ = [
    ("Filippo Bigi <filippo.bigi@epfl.ch>", "@frostedoyster"),
    ("Sanggyu Chong <sanggyu.chong@epfl.ch>", "@SanggyuChong"),
]

__maintainers__ = [
    ("Filippo Bigi <filippo.bigi@epfl.ch>", "@frostedoyster"),
]

# load Model in train at the end to avoid circular imports
from .model import Model  # noqa
from .train import train  # noqa
