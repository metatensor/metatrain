from typing import cast

import torch

from metatrain.utils.data import DatasetInfo, TargetInfo

from .global_multipole import GlobalMultipole
from .minmax_gap import MinMaxGap


KNOWN_POST_HOOKS = {
    "global_multipoles": GlobalMultipole,
    "minmax_gap": MinMaxGap,
}


# For documentation purposes
PostHooksHypers = dict[str, dict | str]


# To help models work with hooks.
def setup_post_hooks(
    post_hooks_hypers: PostHooksHypers, dataset_info: DatasetInfo
) -> tuple[
    list[torch.nn.Module],
    dict[str, TargetInfo],
]:
    """
    Setup the post-processing hooks to apply to the outputs of the model.

    :param post_hooks_hypers: The hyperparameters for the post-processing hooks.
    :param dataset_info: The dataset information.
    :return: A tuple containing the list of post-processing hooks, and a
        dictionary with the outputs that the model should produce before the
        hooks are applied.
    """
    model_outputs = dataset_info.targets.copy()
    post_hooks = []
    for hook_name, hook_hypers in post_hooks_hypers.items():
        if hook_name not in KNOWN_POST_HOOKS:
            raise ValueError(f"Unknown post-processing hook: {hook_name}")

        # Get outputs of the hook and remove them from the model outputs.
        if isinstance(hook_hypers, str):
            san_hook_hypers: dict[str, dict[str, str] | str] = {
                "outputs": hook_hypers,
                "inputs": {},
            }
        else:
            san_hook_hypers = cast(dict[str, dict[str, str] | str], hook_hypers)

        hook_outputs: str | dict = san_hook_hypers["outputs"]
        if isinstance(hook_outputs, str):
            hook_output_names = [hook_outputs]
        else:
            hook_output_names = list(hook_outputs.values())

        for output_name in hook_output_names:
            model_outputs.pop(output_name, None)

        # Get hook and add it to the list of hooks.
        hook_class = KNOWN_POST_HOOKS[hook_name]
        hook = hook_class(san_hook_hypers, dataset_info)
        post_hooks.append(hook)

        # Add the inputs that the hook requests to the model outputs.
        # TODO: Here we should not add the inputs that will be provided
        # as extra_data.
        requested = hook.requested_target_infos()
        model_outputs.update(requested)

    return post_hooks, model_outputs
