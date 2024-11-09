import warnings
from typing import List, Optional

import torch


def compute_gradient(
    target: torch.Tensor, inputs: List[torch.Tensor], is_training: bool
) -> List[torch.Tensor]:
    """
    Calculates the gradient of a target tensor with respect to a list of input tensors.

    ``target`` must be a single torch.Tensor object. If target contains multiple values,
    the gradient will be calculated with respect to the sum of all values.
    """

    grad_outputs: Optional[List[Optional[torch.Tensor]]] = [torch.ones_like(target)]
    try:
        gradient = torch.autograd.grad(
            outputs=[target],
            inputs=inputs,
            grad_outputs=grad_outputs,
            retain_graph=is_training,
            create_graph=is_training,
        )
    except RuntimeError as e:
        # Torch raises an error if the target tensor does not require grad,
        # but this could just mean that the target is a constant tensor, like in
        # the case of composition models. In this case, we can safely ignore the error
        # and we raise a warning instead. The warning can be caught and silenced in the
        # appropriate places.
        if (
            "element 0 of tensors does not require grad and does not have a grad_fn"
            in str(e)
        ):
            warnings.warn(f"GRADIENT WARNING: {e}", RuntimeWarning, stacklevel=2)
            gradient = [torch.zeros_like(i) for i in inputs]
        else:
            # Re-raise the error if it's not the one above
            raise
    if gradient is None:
        raise ValueError(
            "Unexpected None value for computed gradient. "
            "One or more operations inside the model might "
            "not have a gradient implementation."
        )
    else:
        return gradient
