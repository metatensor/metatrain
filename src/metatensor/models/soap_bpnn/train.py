import logging
from pathlib import Path
from typing import Dict, List, Tuple, Union

import torch
from metatensor.torch.atomistic import ModelCapabilities

from ..utils.composition import calculate_composition_weights
from ..utils.compute_loss import compute_model_loss
from ..utils.data import (
    Dataset,
    check_datasets,
    collate_fn,
    combine_dataloaders,
    get_all_targets,
)
from ..utils.info import finalize_aggregated_info, update_aggregated_info
from ..utils.loss import TensorMapDictLoss
from ..utils.model_io import save_model
from .model import DEFAULT_HYPERS, Model


logger = logging.getLogger(__name__)


def train(
    train_datasets: List[Union[Dataset, torch.utils.data.Subset]],
    validation_datasets: List[Union[Dataset, torch.utils.data.Subset]],
    model_capabilities: ModelCapabilities,
    hypers: Dict = DEFAULT_HYPERS,
    output_dir: str = ".",
):
    # Perform canonical checks on the datasets:
    logger.info("Checking datasets for consistency")
    check_datasets(
        train_datasets,
        validation_datasets,
        model_capabilities,
    )

    # Create the model:
    model = Model(
        capabilities=model_capabilities,
        hypers=hypers["model"],
    )

    # Calculate and set the composition weights for all targets:
    for target_name in model_capabilities.outputs.keys():
        # find the dataset that contains the target:
        train_dataset_with_target = None
        for dataset in train_datasets:
            if target_name in get_all_targets(dataset):
                train_dataset_with_target = dataset
                break
        if train_dataset_with_target is None:
            raise ValueError(
                f"Target {target_name} in the model's capabilities is not "
                "present in any of the training datasets."
            )
        composition_weights = calculate_composition_weights(
            train_dataset_with_target, target_name
        )
        model.set_composition_weights(target_name, composition_weights)

    hypers_training = hypers["training"]

    # Create dataloader for the training datasets:
    train_dataloaders = []
    for dataset in train_datasets:
        train_dataloaders.append(
            torch.utils.data.DataLoader(
                dataset=dataset,
                batch_size=hypers_training["batch_size"],
                shuffle=True,
                collate_fn=collate_fn,
            )
        )
    train_dataloader = combine_dataloaders(train_dataloaders, shuffle=True)

    # Create dataloader for the validation datasets:
    validation_dataloaders = []
    for dataset in validation_datasets:
        validation_dataloaders.append(
            torch.utils.data.DataLoader(
                dataset=dataset,
                batch_size=hypers_training["batch_size"],
                shuffle=False,
                collate_fn=collate_fn,
            )
        )
    validation_dataloader = combine_dataloaders(validation_dataloaders, shuffle=False)

    # Extract all the possible outputs and their gradients from the training set:
    outputs_dict = _get_outputs_dict(train_datasets)
    energy_counter = 0
    for output_name in outputs_dict.keys():
        if output_name not in model_capabilities.outputs:
            raise ValueError(
                f"Output {output_name} is not in the model's capabilities."
            )
        if model_capabilities.outputs[output_name].quantity == "energy":
            energy_counter += 1

    # This will be useful later for printing forces/virials/stresses:
    if energy_counter == 1:
        only_one_energy = True
    else:
        only_one_energy = False

    # Create a loss weight dict:
    loss_weights_dict = {}
    for output_name, value_or_gradient_list in outputs_dict.items():
        loss_weights_dict[output_name] = {
            value_or_gradient: 1.0 for value_or_gradient in value_or_gradient_list
        }

    # Create a loss function:
    loss_fn = TensorMapDictLoss(loss_weights_dict)

    # Create an optimizer:
    optimizer = torch.optim.Adam(
        model.parameters(), lr=hypers_training["learning_rate"]
    )

    # counters for early stopping:
    best_validation_loss = float("inf")
    epochs_without_improvement = 0

    # Train the model:
    for epoch in range(hypers_training["num_epochs"]):
        # aggregated information holders:
        aggregated_train_info: Dict[str, Tuple[float, int]] = {}
        aggregated_validation_info: Dict[str, Tuple[float, int]] = {}

        train_loss = 0.0
        for batch in train_dataloader:
            optimizer.zero_grad()
            structures, targets = batch
            loss, info = compute_model_loss(loss_fn, model, structures, targets)
            train_loss += loss.item()
            loss.backward()
            optimizer.step()
            aggregated_train_info = update_aggregated_info(aggregated_train_info, info)
        aggregated_train_info = finalize_aggregated_info(aggregated_train_info)

        validation_loss = 0.0
        for batch in validation_dataloader:
            structures, targets = batch
            # TODO: specify that the model is not training here to save some autograd
            loss, info = compute_model_loss(loss_fn, model, structures, targets)
            validation_loss += loss.item()
            aggregated_validation_info = update_aggregated_info(
                aggregated_validation_info, info
            )
        aggregated_validation_info = finalize_aggregated_info(
            aggregated_validation_info
        )

        # Now we log the information:
        if epoch % hypers_training["log_interval"] == 0:
            logging_string = (
                f"Epoch {epoch:4}, train loss: {train_loss:10.4f}, "
                f"validation loss: {validation_loss:10.4f}"
            )
            for name, information_holder in zip(
                ["train", "valid"], [aggregated_train_info, aggregated_validation_info]
            ):
                for key, value in information_holder.items():
                    if key.endswith("_positions_gradients"):
                        # check if this is a force
                        target_name = key[: -len("_positions_gradients")]
                        if model.capabilities.outputs[target_name].quantity == "energy":
                            # if this is a force, replace the ugly name with "force"
                            if only_one_energy:
                                key = "force"
                            else:
                                key = f"force[{target_name}]"
                    elif key.endswith("_displacement_gradients"):
                        # check if this is a virial/stress
                        target_name = key[: -len("_displacement_gradients")]
                        if model.capabilities.outputs[target_name].quantity == "energy":
                            # if this is a virial/stress,
                            # replace the ugly name with "virial/stress"
                            if only_one_energy:
                                key = "virial/stress"
                            else:
                                key = f"virial/stress[{target_name}]"
                    logging_string += f", {name} {key} RMSE: {value:10.4f}"
            logger.info(logging_string)

        if epoch % hypers_training["checkpoint_interval"] == 0:
            save_model(
                model,
                Path(output_dir) / f"model_{epoch}.pt",
            )

        # early stopping criterion:
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= 50:
                logger.info(
                    f"Early stopping criterion reached after {epoch} "
                    "epochs without improvement."
                )
                break

    return model


def _get_outputs_dict(datasets: List[Union[Dataset, torch.utils.data.Subset]]):
    """
    This is a helper function that extracts all the possible outputs and their gradients
    from a list of datasets.

    :param datasets: A list of Datasets or Subsets.

    :returns: A dictionary mapping output names to a list of "values" (always)
        and possible gradients.
    """

    outputs_dict = {}
    for dataset in datasets:
        sample_batch = next(iter(dataset))
        targets = sample_batch[1]  # this is a dictionary of TensorMaps
        for target_name, target_tmap in targets.items():
            if target_name not in outputs_dict:
                outputs_dict[target_name] = [
                    "values"
                ] + target_tmap.block().gradients_list()

    return outputs_dict
