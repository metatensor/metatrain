from typing import Dict

import torch
from metatomic.torch import ModelOutput


class DistributedDataParallel(torch.nn.parallel.DistributedDataParallel):
    """
    DistributedDataParallel wrapper that inherits from
    :py:class`torch.nn.parallel.DistributedDataParallel`
    and adds a function to retrieve the supported outputs of the module.
    """

    def supported_outputs(self) -> Dict[str, ModelOutput]:
        return self.module.supported_outputs()
