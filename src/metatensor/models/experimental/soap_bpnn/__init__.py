from .model import SOAPBPNN
from .trainer import Trainer

__model__ = SOAPBPNN
__trainer__ = Trainer
__capabilities__ = {
    "supported_devices": __model__.__supported_devices__,
    "supported_dtypes": __model__.__supported_dtypes__,
}

__authors__ = [
    ("Filippo Bigi <filippo.bigi@epfl.ch>", "@frostedoyster"),
]

__maintainers__ = [
    ("Filippo Bigi <filippo.bigi@epfl.ch>", "@frostedoyster"),
]
