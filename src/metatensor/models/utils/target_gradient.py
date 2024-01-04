from typing import List, Optional

import torch


def compute_gradient(
    target: torch.Tensor, inputs: List[torch.Tensor], is_training: bool
) -> List[torch.Tensor]:
    """Calculates the gradient of a target tensor with respect to a list of input tensors.

    ``target`` must be a single torch.Tensor object. If target contains multiple values,
    the gradient will be calculated with respect to the sum of all values.
    """

    grad_outputs: Optional[List[Optional[torch.Tensor]]] = [torch.ones_like(target)]
    gradient = torch.autograd.grad(
        outputs=[target],
        inputs=inputs,
        grad_outputs=grad_outputs,
        retain_graph=is_training,
        create_graph=is_training,
    )
    if gradient is None:
        raise ValueError(
            "Unexpected None value for computed gradient. "
            "One or more operations inside the model might not have a gradient implementation."
        )
    else:
        return gradient
