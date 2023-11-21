from typing import Dict, List, Tuple

from metatensor.torch import TensorMap

from ..linear import LinearModel


class Example(LinearModel):
    def __init__(self, equivariant_selection: List[Tuple[int, int]], hypers: Dict):
        super(Example, self).__init__(equivariant_selection, hypers)

    def forward(systems: TensorMap):
        return systems

    def compute_features(systems: TensorMap):
        return systems
