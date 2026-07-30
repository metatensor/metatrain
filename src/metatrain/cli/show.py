import argparse
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
from ase.data import chemical_symbols
from metatomic.torch import (
    AtomisticModel,
    ModelMetadata,
    ModelOutput,
    load_atomistic_model,
)

from ..utils.data import TargetInfo
from ..utils.io import is_exported_file, model_from_checkpoint, resolve_model_path
from .formatter import CustomHelpFormatter


def _add_show_model_parser(subparser: argparse._SubParsersAction) -> None:
    """Add `show_model` parameters to an argparse (sub)-parser.

    :param subparser: The argparse (sub)-parser to add the parameters to.
    """

    if show_model.__doc__ is not None:
        description = show_model.__doc__.split(r":param")[0]
    else:
        description = None

    parser = subparser.add_parser(
        "show",
        description=description,
        formatter_class=CustomHelpFormatter,
    )
    parser.set_defaults(callable="show_model")

    parser.add_argument(
        "path",
        type=str,
        help="Model to show. Can be either a checkpoint (.ckpt) or an exported model "
        "(.pt) as a local file or a URL.",
    )
    parser.add_argument(
        "-e",
        "--extensions-dir",
        type=str,
        required=False,
        dest="extensions_directory",
        default=None,
        help=(
            "path to a directory containing extensions required by an exported model"
        ),
    )


def show_model(
    path: Union[Path, str],
    extensions_directory: Optional[Union[Path, str]] = None,
) -> None:
    """Show the contents of a saved model.

    This prints a summary of a checkpoint (``.ckpt``) or exported model (``.pt``),
    including architecture, targets that the model can predict, atomic types the model
    supports and the attached metadata.

    :param path: local or remote path to the model file (either a ``.ckpt`` checkpoint
        or an exported ``.pt`` model)
    :param extensions_directory: path to a directory containing all extensions required
        by an exported model
    """
    if Path(path).suffix in [".yaml", ".yml"]:
        raise ValueError(
            f"path '{path}' seems to be a YAML option file and not a model"
        )

    local_path = resolve_model_path(path)

    if is_exported_file(local_path):
        model = load_atomistic_model(
            local_path, extensions_directory=extensions_directory
        )
        lines = _describe_exported_model(model)
    else:
        if extensions_directory is not None:
            logging.warning(
                "the `--extensions-dir` option is only used for exported models and "
                "will be ignored for checkpoints"
            )
        checkpoint = torch.load(local_path, weights_only=False, map_location="cpu")
        lines = _describe_checkpoint(checkpoint)

    summary = "\n".join(lines)
    logging.info(f"Model information from {str(path)!r}\n\n{summary}")


def _describe_checkpoint(checkpoint: Dict[str, Any]) -> List[str]:
    architecture_name = checkpoint.get("architecture_name")
    model_ckpt_version = checkpoint.get("model_ckpt_version")
    trainer_ckpt_version = checkpoint.get("trainer_ckpt_version")
    epoch = checkpoint.get("epoch")
    best_epoch = checkpoint.get("best_epoch")
    best_metric = checkpoint.get("best_metric")

    model = model_from_checkpoint(checkpoint, context="export")

    lines = ["file type: checkpoint"]
    lines.append(f"architecture: {architecture_name}")

    if model_ckpt_version is not None:
        lines.append(f"model checkpoint version: {model_ckpt_version}")
    if trainer_ckpt_version is not None:
        lines.append(f"trainer checkpoint version: {trainer_ckpt_version}")
    if epoch is not None:
        lines.append(f"epoch: {epoch}")
    if best_epoch is not None:
        lines.append(f"best epoch: {best_epoch}")
    if best_metric is not None:
        lines.append(f"best validation metric: {best_metric}")

    lines += _describe_metadata(model.metadata)

    dataset_info = model.dataset_info
    lines.append("")
    lines.append(f"length unit: {dataset_info.length_unit or '(unknown)'}")
    lines.append(f"atomic types: {_format_atomic_types(dataset_info.atomic_types)}")

    lines.append("")
    lines.append("targets:")
    for name, target_info in dataset_info.targets.items():
        lines += _describe_target(name, target_info)

    auxiliary_outputs = sorted(
        set(model.supported_outputs()) - set(dataset_info.targets)
    )
    if auxiliary_outputs:
        lines.append("")
        lines.append("auxiliary outputs:")
        for name in auxiliary_outputs:
            lines.append(f"  - {name}")

    return lines


def _describe_exported_model(model: AtomisticModel) -> List[str]:
    capabilities = model.capabilities()

    lines = ["file type: exported model"]
    lines += _describe_metadata(model.metadata())

    lines.append("")
    lines.append(f"length unit: {capabilities.length_unit or '(unknown)'}")
    lines.append(f"atomic types: {_format_atomic_types(capabilities.atomic_types)}")

    interaction_range = f"interaction range: {capabilities.interaction_range}"
    if capabilities.length_unit:
        interaction_range += f" {capabilities.length_unit}"
    lines.append(interaction_range)

    lines.append(f"dtype: {capabilities.dtype}")
    lines.append(f"supported devices: {', '.join(capabilities.supported_devices)}")

    lines.append("")
    lines.append("outputs:")
    for name, output in capabilities.outputs.items():
        lines += _describe_output(name, output)

    return lines


def _describe_metadata(metadata: ModelMetadata) -> List[str]:
    lines = []

    if metadata.name:
        lines.append(f"  name: {metadata.name}")
    if metadata.description:
        lines.append(f"  description: {metadata.description}")
    if metadata.authors:
        lines.append("  authors: " + ", ".join(metadata.authors))

    references = []
    for section, section_references in metadata.references.items():
        for reference in section_references:
            references.append(f"    - ({section}) {reference}")
    if references:
        lines.append("  references:")
        lines += references

    if lines:
        return ["", "metadata:"] + lines
    else:
        return []


def _describe_target(name: str, target_info: TargetInfo) -> List[str]:
    lines = [f"  {name}:"]

    if target_info.quantity:
        lines.append(f"    quantity: {target_info.quantity}")

    lines.append(f"    unit: {target_info.unit or '(none)'}")
    lines.append(f"    type: {_target_type(target_info)}")
    lines.append(f"    sampled per: {target_info.sample_kind}")

    if target_info.gradients:
        lines.append("    gradients: " + ", ".join(target_info.gradients))

    if target_info.description:
        lines.append(f"    description: {target_info.description}")

    return lines


def _describe_output(name: str, output: ModelOutput) -> List[str]:
    lines = [f"  {name}:"]

    if output.quantity:
        lines.append(f"    quantity: {output.quantity}")

    lines.append(f"    unit: {output.unit or '(none)'}")
    lines.append(f"    sampled per: {output.sample_kind}")

    if output.explicit_gradients:
        lines.append("    explicit gradients: " + ", ".join(output.explicit_gradients))

    if output.description:
        lines.append(f"    description: {output.description}")

    return lines


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


def _format_atomic_types(atomic_types: List[int]) -> str:
    entries = []
    for atomic_type in atomic_types:
        if 0 < atomic_type < len(chemical_symbols):
            entries.append(f"{chemical_symbols[atomic_type]} ({atomic_type})")
        else:
            entries.append(str(atomic_type))

    return ", ".join(entries)
