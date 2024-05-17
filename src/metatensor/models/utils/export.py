import warnings
from typing import Any

import torch
from metatensor.torch.atomistic import (
    MetatensorAtomisticModel,
    ModelCapabilities,
    ModelMetadata,
)


def export(model: torch.nn.Module) -> MetatensorAtomisticModel:
    """Export a trained model to allow it to make predictions.

    This includes predictions within molecular simulation engines. Exported models can
    be be saved to a file with ``exported_model.export(path)``.

    :param model: model to be exported
    :returns: exprted model
    """

    if is_exported(model):
        return model

    if model.capabilities.length_unit == "":
        warnings.warn(
            "No `length_unit` was provided for the model. As a result, lengths "
            "and any derived quantities will be passed to MD engines as is.",
            stacklevel=1,
        )

    for model_output_name, model_output in model.capabilities.outputs.items():
        if model_output.unit == "":
            warnings.warn(
                f"No target units were provided for output {model_output_name!r}. "
                "As a result, this model output will be passed to MD engines as is.",
                stacklevel=1,
            )

    model_capabilities_with_devices = ModelCapabilities(
        length_unit=model.capabilities.length_unit,
        atomic_types=model.capabilities.atomic_types,
        outputs=model.capabilities.outputs,
        supported_devices=["cpu", "cuda"],
        interaction_range=model.capabilities.interaction_range,
        dtype=model.capabilities.dtype,
    )

    return MetatensorAtomisticModel(
        model.eval(), ModelMetadata(), model_capabilities_with_devices
    )


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
