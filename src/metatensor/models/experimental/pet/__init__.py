from .model import Model, DEFAULT_HYPERS  # noqa: F401
from .train import train  # noqa: F401

DEVICES = ["cuda"]

__authors__ = [
    ("Sergey Pozdnyakov <sergey.pozdnyakov@epfl.ch>", "@serfg"),
    ("Arslan Mazitov <arslan.mazitov@epfl.ch>", "@abmazitov"),
    ("Filippo Bigi <filippo.bigi@epfl.ch>", "@frostedoyster"),
]

__maintainers__ = [
    ("Sergey Pozdnyakov <sergey.pozdnyakov@epfl.ch>", "@serfg"),
    ("Arslan Mazitov <arslan.mazitov@epfl.ch>", "@abmazitov"),
]
