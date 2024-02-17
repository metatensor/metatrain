import warnings
from typing import Any

import torch
from metatensor.torch.atomistic import MetatensorAtomisticModel


def export(model: torch.nn.Module, output: str) -> None:
    """Export a trained model to allow it to make predictions,
    including within molecular simulation engines.

    :param model: The model to be exported
    :param output: Path to save the exported model
    """

    for model_output_name, model_output in model.capabilities.outputs.items():
        if model_output.unit == "":
            warnings.warn(
                f"No units were provided for the `{model_output_name}` output. "
                "As a result, this model output will be passed to MD engines as is.",
                stacklevel=1,
            )

    wrapper = MetatensorAtomisticModel(model.eval(), model.capabilities)
    wrapper.export(output)


def is_exported(model: Any):
    """Check if a model has been exported.

    :param model: The model to check
    :return: True if the model has been exported, False otherwise
    """
    return isinstance(model, torch.jit._script.RecursiveScriptModule)
