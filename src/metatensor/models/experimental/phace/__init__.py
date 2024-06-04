from .model import Model, DEFAULT_HYPERS  # noqa: F401
from .train import train  # noqa: F401
import torch

__ARCHITECTURE_CAPABILITIES__ = {
    "supported_devices": ["cuda", "cpu"],
    "supported_dtypes": [torch.float64, torch.float32],
}

__authors__ = [
    ("Filippo Bigi <filippo.bigi@epfl.ch>", "@frostedoyster"),
]

__maintainers__ = [
    ("Filippo Bigi <filippo.bigi@epfl.ch>", "@frostedoyster"),
]
