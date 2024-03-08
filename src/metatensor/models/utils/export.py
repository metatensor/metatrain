import warnings
from typing import Any

import torch
from metatensor.torch.atomistic import MetatensorAtomisticModel, ModelMetadata, ModelCapabilities


def export(model: torch.nn.Module, output: str) -> None:
    """Export a trained model to allow it to make predictions,
    including within molecular simulation engines.

    :param model: The model to be exported
    :param output: Path to save the exported model
    """

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
        supported_devices=["cpu", "cuda"]
    )

    wrapper = MetatensorAtomisticModel(model.eval(), ModelMetadata(), model_capabilities_with_devices)
    wrapper.export(output)


def is_exported(model: Any):
    """Check if a model has been exported.

    :param model: The model to check
    :return: True if the model has been exported, False otherwise
    """
    return isinstance(model, torch.jit._script.RecursiveScriptModule)
