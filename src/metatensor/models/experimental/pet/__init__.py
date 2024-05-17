import torch
from typing import Dict
from ...utils.architectures import get_default_hypers, get_architecture_name


ARCHITECTURE_NAME: str = get_architecture_name(__file__)

__ARCHITECTURE_CAPABILITIES__ = {
    "supported_devices": ["cuda"],
    "supported_dtypes": [torch.float32],
}

DEFAULT_HYPERS: Dict = get_default_hypers(ARCHITECTURE_NAME)
DEFAULT_MODEL_HYPERS: Dict = DEFAULT_HYPERS["ARCHITECTURAL_HYPERS"]

# We hardcode some of the hypers to make PET work as a MLIP.
DEFAULT_MODEL_HYPERS.update(
    {"D_OUTPUT": 1, "TARGET_TYPE": "atomic", "TARGET_AGGREGATION": "sum"}
)

__authors__ = [
    ("Sergey Pozdnyakov <sergey.pozdnyakov@epfl.ch>", "@spozdn"),
    ("Arslan Mazitov <arslan.mazitov@epfl.ch>", "@abmazitov"),
    ("Filippo Bigi <filippo.bigi@epfl.ch>", "@frostedoyster"),
]

__maintainers__ = [
    ("Sergey Pozdnyakov <sergey.pozdnyakov@epfl.ch>", "@spozdn"),
    ("Arslan Mazitov <arslan.mazitov@epfl.ch>", "@abmazitov"),
]

# load Model in train at the end to avoid circular imports
from .model import Model  # noqa
from .train import train  # noqa
