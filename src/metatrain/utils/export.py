import warnings

import torch
from metatensor.torch.atomistic import (
    MetatensorAtomisticModel,
    ModelCapabilities,
    ModelMetadata,
    is_atomistic_model,
)


# TODO: DELETE OR CHANGE THIS FUNCTION.
# EXPORT IS NOW PER-ARCHITECTURE


def export(
    model: torch.nn.Module, model_capabilities: ModelCapabilities
) -> MetatensorAtomisticModel:
    """Export a torch.nn.Module model to a MetatensorAtomisticModel.

    The exoort allows the model to make predictions especially in molecular simulation
    engines. Exported models can be be saved to a file with
    ``exported_model.save(path)``.

    :param model: model to be exported
    :param model_capabilities: capabilities of the model
    :returns: exprted model
    """

    if is_atomistic_model(model):
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
