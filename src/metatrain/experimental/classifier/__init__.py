from .model import Classifier
from .trainer import Trainer


__model__ = Classifier
__trainer__ = Trainer
__capabilities__ = {
    "supported_devices": __model__.__supported_devices__,
    "supported_dtypes": __model__.__supported_dtypes__,
}

__authors__ = [
    ("GitHub Copilot", "@copilot"),
]

__maintainers__ = [
    ("Michele Ceriotti", "@frostedoyster"),
]
