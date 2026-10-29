from typing import Dict

import torch

from metatrain.utils.data import TargetInfo


class DistributedDataParallel(torch.nn.parallel.DistributedDataParallel):
    """
    DistributedDataParallel wrapper that inherits from
    :py:class`torch.nn.parallel.DistributedDataParallel`
    and adds a function to retrieve the supported outputs of the module.
    """

    def supported_outputs(self) -> Dict[str, TargetInfo]:
        return self.module.supported_outputs()
