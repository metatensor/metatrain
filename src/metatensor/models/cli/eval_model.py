import argparse
import logging
from typing import Dict, Tuple, Union

import torch
from omegaconf import DictConfig, OmegaConf

from ..utils.compute_loss import compute_model_loss
from ..utils.data import (
    Dataset,
    collate_fn,
    read_structures,
    read_targets,
    write_predictions,
)
from ..utils.extract_targets import get_outputs_dict
from ..utils.info import finalize_aggregated_info, update_aggregated_info
from ..utils.loss import TensorMapDictLoss
from ..utils.model_io import load_model
from ..utils.omegaconf import expand_dataset_config
from .formatter import CustomHelpFormatter


logger = logging.getLogger(__name__)


def _add_eval_model_parser(subparser: argparse._SubParsersAction) -> None:
    """Add the `eval_model` paramaters to an argparse (sub)-parser"""

    if eval_model.__doc__ is not None:
        description = eval_model.__doc__.split(r":param")[0]
    else:
        description = None

    parser = subparser.add_parser(
        "eval",
        description=description,
        formatter_class=CustomHelpFormatter,
    )
    parser.set_defaults(callable="eval_model")
    parser.add_argument(
        "model",
        type=load_model,
        help="Saved model to be evaluated.",
    )
    parser.add_argument(
        "options",
        type=OmegaConf.load,
        help="Eval options file to define a dataset for evaluation.",
    )
    parser.add_argument(
        "-o",
        "--output",
        dest="output",
        type=str,
        required=False,
        default="output.xyz",
        help="filename of the predictions (default: %(default)s)",
    )


def _eval_targets(model, dataset: Union[Dataset, torch.utils.data.Subset]) -> None:
    """Evaluate a model on a dataset and print the RMSEs for each target."""

    # Extract all the possible outputs and their gradients from the dataset:
    outputs_dict = get_outputs_dict([dataset])
    for output_name in outputs_dict.keys():
        if output_name not in model.capabilities.outputs:
            raise ValueError(
                f"Output {output_name} is not in the model's capabilities."
            )

    # Create the loss function:
    loss_weights_dict = {}
    for output_name, value_or_gradient_list in outputs_dict.items():
        loss_weights_dict[output_name] = {
            value_or_gradient: 0.0 for value_or_gradient in value_or_gradient_list
        }
    loss_fn = TensorMapDictLoss(loss_weights_dict)

    # Create a dataloader:
    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=4,  # Choose small value to not crash the system at evaluation
        shuffle=True,
        collate_fn=collate_fn,
    )

    # Compute the RMSEs:
    aggregated_info: Dict[str, Tuple[float, int]] = {}
    for batch in dataloader:
        structures, targets = batch
        _, info = compute_model_loss(loss_fn, model, structures, targets)
        aggregated_info = update_aggregated_info(aggregated_info, info)
    finalized_info = finalize_aggregated_info(aggregated_info)

    energy_counter = 0
    for output in model.capabilities.outputs.values():
        if output.quantity == "energy":
            energy_counter += 1
    if energy_counter == 1:
        only_one_energy = True
    else:
        only_one_energy = False

    log_output = []
    for key, value in finalized_info.items():
        new_key = key
        if key.endswith("_positions_gradients"):
            # check if this is a force
            target_name = key[: -len("_positions_gradients")]
            if model.capabilities.outputs[target_name].quantity == "energy":
                # if this is a force, replace the ugly name with "force"
                if only_one_energy:
                    new_key = "force"
                else:
                    new_key = f"force[{target_name}]"
        elif key.endswith("_displacement_gradients"):
            # check if this is a virial/stress
            target_name = key[: -len("_displacement_gradients")]
            if model.capabilities.outputs[target_name].quantity == "energy":
                # if this is a virial/stress,
                # replace the ugly name with "virial/stress"
                if only_one_energy:
                    new_key = "virial/stress"
                else:
                    new_key = f"virial/stress[{target_name}]"
        log_output.append(f"{new_key} RMSE: {value}")
    logger.info(", ".join(log_output))


def eval_model(
    model: torch.nn.Module, options: DictConfig, output: str = "output.xyz"
) -> None:
    """Evaluate a pretrained model on a given data set.

    If ``options`` contains a ``targets`` sub-section, RMSE values will be reported. If
    this sub-section is missing, only a xyz-file with containing the properties the model
    was trained against is written.

    :param model: Saved model to be evaluated.
    :param options: DictConfig to define a test dataset taken for the evaluation.
    :param output: Path to save the predicted values
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger.info("Setting up evaluation set.")

    options = expand_dataset_config(options)
    eval_structures = read_structures(
        filename=options["structures"]["read_from"],
        fileformat=options["structures"]["file_format"],
    )
    # Predict targets
    if hasattr(options, "targets"):
        eval_targets = read_targets(options["targets"])
        eval_dataset = Dataset(eval_structures, eval_targets)
        _eval_targets(model, eval_dataset)

    # Predict structures
    predictions = model(eval_structures, model.capabilities.outputs)
    write_predictions(output, predictions, eval_structures)
