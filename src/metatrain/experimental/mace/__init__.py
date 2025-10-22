from .model import MetaMACE
from .trainer import Trainer
from .utils.e3nn import patch_e3nn


# Patch e3nn to make it torchscript compatible
patch_e3nn()

__model__ = MetaMACE
__trainer__ = Trainer

__authors__ = [
    ("Pol Febrer <pol.febrer@epfl.ch>", "@pfebrer"),
    ("Joseph W. Abbott <joseph.william.abbott@gmail.com>", "@jwa7"),
]

__maintainers__ = [
    ("Pol Febrer <pol.febrer@epfl.ch>", "@pfebrer"),
]
