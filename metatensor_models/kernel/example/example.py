from typing import Dict, List, Tuple

from metatensor.torch import TensorMap

from ..kernel import KernelModel


class Example(KernelModel):
    def __init__(self, equivariant_selection: List[Tuple[int, int]], hypers: Dict):
        super(Example, self).__init__(equivariant_selection, hypers)

    def forward(self, systems: TensorMap):
        return systems

    def compute_kernels(self, systems: TensorMap):
        return systems
