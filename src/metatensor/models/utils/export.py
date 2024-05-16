import warnings
from typing import Any

import torch
from metatensor.torch.atomistic import (
    MetatensorAtomisticModel,
    ModelCapabilities,
    ModelMetadata,
)


def export(
    model: torch.nn.Module, model_capabilities: ModelCapabilities
) -> MetatensorAtomisticModel:
    """Export a torch.nn.Module model to a MetatensorAtomisticModel.

    The exoort allows the model to make predictions especially in molecular simulation
    engines. Exported models can be be saved to a file with
    ``exported_model.export(path)``.

    :param model: model to be exported
    :param model_capabilities: capabilities of the model
    :returns: exprted model
    """

    if is_exported(model):
        return model

    if model_capabilities.length_unit == "":
        warnings.warn(
            "No `length_unit` was provided for the model. As a result, lengths "
            "and any derived quantities will be passed to MD engines as is.",
            stacklevel=1,
        )

    for model_output_name, model_output in model_capabilities.outputs.items():
        if model_output.unit == "":
            warnings.warn(
                f"No target units were provided for output {model_output_name!r}. "
                "As a result, this model output will be passed to MD engines as is.",
                stacklevel=1,
            )

    return MetatensorAtomisticModel(model.eval(), ModelMetadata(), model_capabilities)


def is_exported(model: Any) -> bool:
    """Check if a model has been exported to a MetatensorAtomisticModel.

    :param model: The model to check
    :return: :py:obj:`True` if the ``model`` has been exported, :py:obj:`False`
        otherwise.
    """
    # If the model is saved an loaded again it's type is ScriptModule
    if type(model) in [
        MetatensorAtomisticModel,
        torch.jit._script.RecursiveScriptModule,
    ]:
        return True
    else:
        return False
