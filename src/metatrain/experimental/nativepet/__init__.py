from .model import NativePET
from .trainer import Trainer


__model__ = NativePET
__trainer__ = Trainer
__capabilities__ = {
    "supported_devices": __model__.__supported_devices__,
    "supported_dtypes": __model__.__supported_dtypes__,
}

__authors__ = [
    ("Sergey Pozdnyakov <sergey.pozdnyakov@epfl.ch>", "@spozdn"),
    ("Arslan Mazitov <arslan.mazitov@epfl.ch>", "@abmazitov"),
    ("Filippo Bigi <filippo.bigi@epfl.ch>", "@frostedoyster"),
]

__maintainers__ = [
    ("Arslan Mazitov <arslan.mazitov@epfl.ch>", "@abmazitov"),
]
