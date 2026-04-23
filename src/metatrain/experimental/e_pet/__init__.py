from .model import EPET
from .trainer import Trainer


__model__ = EPET
__trainer__ = Trainer
__capabilities__ = {
    "supported_devices": __model__.__supported_devices__,
    "supported_dtypes": __model__.__supported_dtypes__,
}

__authors__ = [
    ("Michele Ceriotti Group", "@lab-cosmo"),
]

__maintainers__ = [
    ("Michele Ceriotti Group", "@lab-cosmo"),
]
