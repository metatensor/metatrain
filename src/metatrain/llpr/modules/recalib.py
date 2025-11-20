from typing import Any, Dict

import torch.nn as nn


def apply_ensemble_training_strategy(
    model: nn.Module, target: str, strategy: Dict[str, Any]
) -> nn.Module:
    """
    Apply the user-specified ensemble training strategy to the LLPR-wrapped model.
    This function modifies the model in place based on the provided strategy.
    The strategy can be one of the following:
    - full: all model weights are retrained during ensemble training
    - tagged-only: only the model weights specified in a dictionary under this tag
       are retrained during ensemble training (useful for head-only training)
    - ens-only: only the ensemble linear layer weights are trained
    input model should be the LLPRUncertaintyModel object.

    :param model: LLPR-wrapped model to be recalibrated.
    :param target: target property for which ensemble training is performed.
    :param strategy: dictionary specifying the ensemble training strategy.
    :return: the model with updated trainable parameters.
    """

    method = strategy.get("strategy", "ens_only").lower()

    # free-up last linear layers, freeze main model
    for module in model.llpr_ensemble_layers.values():
        for param in module.parameters():
            param.requires_grad = True
    for param in model.model.parameters():
        param.requires_grad = False

    if method == "full":
        # Full finetuning, all parameters are trainable
        for param in model.model.parameters():
            param.requires_grad = True

    elif method == "tagged_only":
        tagged_param_list = strategy["tagged_only_weights"]
        # only free up weights that contain string tags (useful for head-only)
        for name, param in model.model.named_parameters():
            if name in tagged_param_list:
                # above should be a list of valid parameter names
                param.requires_grad = True
                tagged_param_list.remove(name)
        if len(tagged_param_list) > 0:
            raise RuntimeError(
                f"Not all parameters have been matched within the wrapped model!\n"
                f"Remaining params: {tagged_param_list}"
            )

    elif method == "ens_only":
        # ll ensemble only
        pass

    return model
