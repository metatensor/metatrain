import argparse
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch
import yaml
from metatomic.torch import (
    AtomisticModel,
    ModelMetadata,
    ModelOutput,
    load_atomistic_model,
)

from ..utils.data import TargetInfo
from ..utils.io import is_exported_file, model_from_checkpoint, resolve_model_path


def _add_info_model_parser(subparser: argparse._SubParsersAction) -> None:
    """Add `info_model` parameters to an argparse (sub)-parser.

    :param subparser: The argparse (sub)-parser to add the parameters to
    """

    if info_model.__doc__ is not None:
        description = info_model.__doc__.split(r":param")[0]
    else:
        description = None

    parser = subparser.add_parser("info", description=description)
    parser.set_defaults(callable="info_model")

    parser.add_argument(
        "path",
        type=str,
        help="Model to inspect. Can be either a checkpoint (.ckpt) or an exported "
        "model (.pt) or a URL",
    )
    parser.add_argument(
        "-e",
        "--extensions-dir",
        type=str,
        required=False,
        dest="extensions_directory",
        default=None,
        help=(
            "path to a directory containing all extensions required by the exported "
            "model"
        ),
    )
    parser.add_argument(
        "--token",
        dest="hf_token",
        type=str,
        required=False,
        default=None,
        help="HuggingFace API token to download (private) models from HuggingFace. "
        "You can also set a environment variable `HF_TOKEN` to avoid passing it every "
        "time.",
    )


def info_model(
    path: Union[Path, str],
    extensions_directory: Optional[Union[Path, str]] = None,
    hf_token: Optional[str] = None,
) -> None:
    """Print information about a saved model.

    This prints a summary of a checkpoint (``.ckpt``) or exported model (``.pt``),
    including the targets that the model can predict, the supported atomic types, its
    number of parameters and metadata. For checkpoints, the architecture hypers and the
    training state are printed as well.

    :param path: local or remote path to the model file
    :param extensions_directory: path to a directory containing all extensions required
        by an exported model
    :param hf_token: HuggingFace API token to download (private) models from HuggingFace
    """
    if Path(path).suffix in [".yaml", ".yml"]:
        raise ValueError(
            f"path '{path}' seems to be a YAML option file and not a model"
        )

    path = resolve_model_path(path, hf_token=hf_token)

    if is_exported_file(path):
        model = load_atomistic_model(path, extensions_directory=extensions_directory)
        info = _exported_model_info(model)
    else:
        if extensions_directory is not None:
            logging.warning(
                "the `--extensions-dir` option is only used for exported models and "
                "will be ignored for checkpoints"
            )
        checkpoint = torch.load(path, weights_only=False, map_location="cpu")
        info = _checkpoint_info(checkpoint)

    print(yaml.safe_dump(info, sort_keys=False, default_flow_style=None))


def _checkpoint_info(checkpoint: Dict[str, Any]) -> Dict[str, Any]:
    model = model_from_checkpoint(checkpoint, context="export")

    architecture = {"name": checkpoint["architecture_name"], "model": model.hypers}
    if checkpoint.get("train_hypers") is not None:
        architecture["training"] = checkpoint["train_hypers"]

    state_keys = [
        "model_ckpt_version",
        "trainer_ckpt_version",
        "epoch",
        "best_epoch",
        "best_metric",
    ]
    state = {
        key: checkpoint[key] for key in state_keys if checkpoint.get(key) is not None
    }

    info: Dict[str, Any] = {
        "file_type": "checkpoint",
        "architecture": architecture,
        "checkpoint": state,
        "parameters": sum(p.numel() for p in model.parameters()),
    }

    metadata = _metadata_info(model.metadata)
    if metadata:
        info["metadata"] = metadata

    dataset_info = model.dataset_info
    info["length_unit"] = dataset_info.length_unit
    info["atomic_types"] = list(dataset_info.atomic_types)
    info["targets"] = {
        name: _target_info(target_info)
        for name, target_info in dataset_info.targets.items()
    }

    auxiliary_outputs = sorted(
        set(model.supported_outputs()) - set(dataset_info.targets)
    )
    if auxiliary_outputs:
        info["auxiliary_outputs"] = auxiliary_outputs

    return info


def _exported_model_info(model: AtomisticModel) -> Dict[str, Any]:
    capabilities = model.capabilities()

    info: Dict[str, Any] = {
        "file_type": "exported model",
        "parameters": sum(p.numel() for p in model.parameters()),
    }

    metadata = _metadata_info(model.metadata())
    if metadata:
        info["metadata"] = metadata

    info["length_unit"] = capabilities.length_unit
    info["atomic_types"] = list(capabilities.atomic_types)
    info["interaction_range"] = capabilities.interaction_range
    info["dtype"] = capabilities.dtype
    info["supported_devices"] = list(capabilities.supported_devices)
    info["outputs"] = {
        name: _output_info(output) for name, output in capabilities.outputs.items()
    }

    return info


def _metadata_info(metadata: ModelMetadata) -> Dict[str, Any]:
    info: Dict[str, Any] = {}

    if metadata.name:
        info["name"] = metadata.name
    if metadata.description:
        info["description"] = metadata.description
    if metadata.authors:
        info["authors"] = list(metadata.authors)

    references = {
        section: list(section_references)
        for section, section_references in metadata.references.items()
        if section_references
    }
    if references:
        info["references"] = references

    return info


def _target_info(target_info: TargetInfo) -> Dict[str, Any]:
    info: Dict[str, Any] = {}

    if target_info.quantity:
        info["quantity"] = target_info.quantity

    info["unit"] = target_info.unit
    info["type"] = _target_type(target_info)
    info["sample_kind"] = target_info.sample_kind

    if target_info.gradients:
        info["gradients"] = list(target_info.gradients)

    if target_info.description:
        info["description"] = target_info.description

    return info


def _output_info(output: ModelOutput) -> Dict[str, Any]:
    info: Dict[str, Any] = {}

    if output.quantity:
        info["quantity"] = output.quantity

    info["unit"] = output.unit
    info["sample_kind"] = output.sample_kind

    if output.explicit_gradients:
        info["explicit_gradients"] = list(output.explicit_gradients)

    if output.description:
        info["description"] = output.description

    return info


def _target_type(target_info: TargetInfo) -> str:
    if target_info.is_scalar:
        return "scalar"
    elif target_info.is_cartesian:
        return "cartesian"
    elif target_info.is_spherical:
        return "spherical"
    elif target_info.is_atomic_basis:
        return "atomic basis"
    else:
        return "unknown"
