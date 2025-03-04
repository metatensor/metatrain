from typing import Dict
import torch

import metatensor.torch as mts
from metatensor.torch import TensorMap


class L2Loss(torch.nn.Module):

    def __init__(self) -> None:
        """
        """
        super().__init__()
        return

    def forward(
        self, input: Dict[str, TensorMap], target: Dict[str, TensorMap], weights: Dict[str, float] = None
    ) -> torch.Tensor:
        """
        Computes the squared loss (reduction = sum) between the input and target TensorMaps
        """

        if weights is None:
            weights = {k: 1 for k in target}

        loss = 0
        for k in target.keys():
            assert k in input.keys()
            assert k in weights

            for key in target[k].keys:
                # Some prediction blocks might be empty, so just check metadata on
                # blocks we have target keys for. # Check that components and properties
                # are equal. For samples, check all dimensions except "system" which is
                # in general different due to minibatching.
                mts.equal_metadata_block_raise(
                    input[k][key], target[k][key],
                    check=["components", "properties"],
                )
                assert torch.all(
                    input[k][key].samples.values[:, 1:] == target[k][key].samples.values[:, 1:]
                )
                
                loss += weights[k] * torch.sum(
                    (input[k][key].values - target[k][key].values) ** 2
                )

        return loss