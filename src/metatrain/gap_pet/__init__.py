from metatrain.pet.trainer import Trainer

from .model import GapPET


__model__ = GapPET
__trainer__ = Trainer
__capabilities__ = {
    "supported_devices": __model__.__supported_devices__,
    "supported_dtypes": __model__.__supported_dtypes__,
}

__authors__ = [
    ("metatrain contributors", "@metatensor"),
]

__maintainers__ = [
    ("metatrain contributors", "@metatensor"),
]
