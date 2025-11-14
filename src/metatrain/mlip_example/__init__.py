"""Example MLIP architecture that always predicts zero energy."""

from .model import ZeroModel
from .trainer import ZeroTrainer


__model__ = ZeroModel
__trainer__ = ZeroTrainer

__authors__ = [
    ("GitHub Copilot <copilot@github.com>", "@copilot"),
]

__maintainers__ = [
    ("GitHub Copilot <copilot@github.com>", "@copilot"),
]
