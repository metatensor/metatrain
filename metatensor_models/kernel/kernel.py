import abc
from typing import Dict, List, Tuple

import torch
from metatensor.torch import TensorMap  # , System


class KernelModel(torch.nn.Module, abc.ABC):
    def __init__(self, equivariant_selection: List[Tuple[int, int]], hypers: Dict):
        """
        The initialization function. This takes as arguments `equivariant_selection`,
        i.e. a list of (lambda, sigma) tuples that indicate the symmetry of the
        targets with respect to rotation and inversion, as well as the hypers of the
        model as a dictionary. These hypers are model-specific.
        """
        super(KernelModel, self).__init__()
        self.regression_coefficients = torch.empty(())

    @abc.abstractmethod
    def forward(self, systems: TensorMap) -> TensorMap:
        """
        The forward function of the linear model.
        """
        pass

    @abc.abstractmethod
    def compute_kernels(self, systems: TensorMap) -> TensorMap:
        """
        A function that computes the kernels of the model.
        """
        pass
