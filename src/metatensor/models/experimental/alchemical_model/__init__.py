from .model import AlchemicalModel
from .trainer import Trainer

__model__ = AlchemicalModel
__trainer__ = Trainer
__capabilities__ = {
    "supported_devices": __model__.__supported_devices__,
    "supported_dtypes": __model__.__supported_dtypes__,
}

__authors__ = [
    ("Arslan Mazitov <arslan.mazitov@epfl.ch>", "@abmazitov"),
]

__maintainers__ = [
    ("Arslan Mazitov <arslan.mazitov@epfl.ch>", "@abmazitov"),
]
