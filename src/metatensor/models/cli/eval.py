import argparse
import logging
from pathlib import Path
from typing import Dict, Tuple, Union

import torch
from metatensor.learn.data.dataset import Dataset, _BaseDataset
from metatensor.torch.atomistic import ModelEvaluationOptions
from omegaconf import DictConfig, OmegaConf

from ..utils.compute_loss import compute_model_loss
from ..utils.data import collate_fn, read_systems, read_targets, write_predictions
from ..utils.errors import ArchitectureError
from ..utils.extract_targets import get_outputs_dict
from ..utils.info import finalize_aggregated_info, update_aggregated_info
from ..utils.loss import TensorMapDictLoss
from ..utils.model_io import load_exported_model
from ..utils.neighbors_lists import get_system_with_neighbors_lists
from ..utils.omegaconf import expand_dataset_config
from .formatter import CustomHelpFormatter


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")


def _add_eval_model_parser(subparser: argparse._SubParsersAction) -> None:
    """Add the `eval_model` paramaters to an argparse (sub)-parser"""

    if eval_model.__doc__ is not None:
        description = eval_model.__doc__.split(r":param")[0]
    else:
        description = None

    # If you change the synopsis of these commands or add new ones adjust the completion
    # script at `src/metatensor/models/share/metatensor-models-completion.bash`.
    parser = subparser.add_parser(
        "eval",
        description=description,
        formatter_class=CustomHelpFormatter,
    )
    parser.set_defaults(callable="eval_model")
    parser.add_argument(
        "model",
        type=load_exported_model,
        help="Saved exported model to be evaluated.",
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


def _eval_targets(model, dataset: Union[_BaseDataset, torch.utils.data.Subset]) -> None:
    """Evaluate an exported model on a dataset and print the RMSEs for each target."""
    # Attach neighbor lists to the systems:
    requested_neighbor_lists = model.requested_neighbors_lists()
    # working around https://github.com/lab-cosmo/metatensor/issues/521
    # Desired:
    # for system, _ in dataset:
    #     attach_neighbor_lists(system, requested_neighbors_lists)
    # Current:
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=1, collate_fn=collate_fn
    )
    for (system,), _ in dataloader:
        get_system_with_neighbors_lists(system, requested_neighbor_lists)

    # Extract all the possible outputs and their gradients from the dataset:
    outputs_dict = get_outputs_dict([dataset])
    for output_name in outputs_dict.keys():
        if output_name not in model.capabilities().outputs:
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
        _, info = compute_model_loss(loss_fn, model, structures, targets, [])

        aggregated_info = update_aggregated_info(aggregated_info, info)
    finalized_info = finalize_aggregated_info(aggregated_info)

    energy_counter = 0

    try:
        outputs_capabilities = model.capabilities().outputs
    except Exception as e:
        raise ArchitectureError(e)

    for output in outputs_capabilities.values():
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
            if outputs_capabilities[target_name].quantity == "energy":
                # if this is a force, replace the ugly name with "force"
                if only_one_energy:
                    new_key = "force"
                else:
                    new_key = f"force[{target_name}]"
        elif key.endswith("_displacement_gradients"):
            # check if this is a virial/stress
            target_name = key[: -len("_displacement_gradients")]
            if outputs_capabilities[target_name].quantity == "energy":
                # if this is a virial/stress, replace the ugly name with "virial/stress"
                if only_one_energy:
                    new_key = "virial/stress"
                else:
                    new_key = f"virial/stress[{target_name}]"
        log_output.append(f"{new_key} RMSE: {value}")
    logger.info(", ".join(log_output))


def eval_model(
    model: torch.nn.Module, options: DictConfig, output: Union[Path, str] = "output.xyz"
) -> None:
    """Evaluate an exported model on a given data set.

    If ``options`` contains a ``targets`` sub-section, RMSE values will be reported. If
    this sub-section is missing, only a xyz-file with containing the properties the
    model was trained against is written.

    :param model: Saved model to be evaluated.
    :param options: DictConfig to define a test dataset taken for the evaluation.
    :param output: Path to save the predicted values
    """
    if not isinstance(model, torch.jit._script.RecursiveScriptModule):
        raise ValueError(
            "The model must already be exported to be used in `eval`. "
            "If you are trying to evaluate a checkpoint, export it first "
            "with the `metatensor-models export` command."
        )
    logger.info("Setting up evaluation set.")

    if isinstance(output, str):
        output = Path(output)

    options_list = expand_dataset_config(options)
    for i, options in enumerate(options_list):
        if len(options_list) == 1:
            extra_log_message = ""
            file_index_suffix = ""
        else:
            extra_log_message = f" with index {i}"
            file_index_suffix = f"_{i}"
        logger.info(f"Evaulate dataset{extra_log_message}")

        eval_systems = read_systems(
            filename=options["systems"]["read_from"],
            fileformat=options["systems"]["file_format"],
        )

        # Predict targets
        if hasattr(options, "targets"):
            eval_targets = read_targets(options["targets"])
            eval_dataset = Dataset(system=eval_systems, energy=eval_targets["energy"])
            _eval_targets(model, eval_dataset)
        else:
            # TODO: batch this
            # TODO: add forces/stresses/virials if requested
            # Attach neighbors list to systems. This step is only required if no
            # targets are present. Otherwise, the neighbors list have been already
            # attached in `_eval_targets`.
            eval_systems = [
                get_system_with_neighbors_lists(
                    system, model.requested_neighbors_lists()
                )
                for system in eval_systems
            ]

        # Predict systems
        try:
            # `length_unit` is only required for unit conversions in MD engines and
            # superflous here.
            eval_options = ModelEvaluationOptions(
                length_unit="", outputs=model.capabilities().outputs
            )
            predictions = model(eval_systems, eval_options, check_consistency=True)
        except Exception as e:
            raise ArchitectureError(e)

        # TODO: adjust filename accordinglt
        write_predictions(
            filename=f"{output.stem}{file_index_suffix}{output.suffix}",
            predictions=predictions,
            systems=eval_systems,
        )
