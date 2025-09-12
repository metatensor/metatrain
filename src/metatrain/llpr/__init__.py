from .model import LLPRUncertaintyModel
from .trainer import Trainer


__model__ = LLPRUncertaintyModel
__trainer__ = Trainer
__capabilities__ = {
    "supported_devices": __model__.__supported_devices__,
    "supported_dtypes": __model__.__supported_dtypes__,
}

__authors__ = [
    ("Filippo Bigi <filippo.bigi@epfl.ch>", "@frostedoyster"),
    ("Sanggyu Chong <sanggyu.chong@epfl.ch>", "@SanggyuChong"),
]

__maintainers__ = [
    ("Filippo Bigi <filippo.bigi@epfl.ch>", "@frostedoyster"),
    ("Sanggyu Chong <sanggyu.chong@epfl.ch>", "@SanggyuChong"),
]
