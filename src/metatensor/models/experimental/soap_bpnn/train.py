import logging
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
from metatensor.learn.data import DataLoader
from metatensor.learn.data.dataset import Dataset
from metatensor.torch.atomistic import ModelCapabilities, ModelOutput

from ...utils.composition import calculate_composition_weights
from ...utils.compute_loss import compute_model_loss
from ...utils.data import (
    DatasetInfo,
    check_datasets,
    collate_fn,
    combine_dataloaders,
    get_all_species,
    get_all_targets,
)
from ...utils.extract_targets import get_outputs_dict
from ...utils.info import finalize_aggregated_info, update_aggregated_info
from ...utils.io import is_exported, load, save
from ...utils.logging import MetricLogger
from ...utils.loss import TensorMapDictLoss
from ...utils.merge_capabilities import merge_capabilities
from .model import DEFAULT_HYPERS, Model


logger = logging.getLogger(__name__)


# Filter out the second derivative and device warnings from rascaline-torch
warnings.filterwarnings("ignore", category=UserWarning, message="second derivative")
warnings.filterwarnings(
    "ignore", category=UserWarning, message="Systems data is on device"
)


def train(
    train_datasets: List[Union[Dataset, torch.utils.data.Subset]],
    validation_datasets: List[Union[Dataset, torch.utils.data.Subset]],
    dataset_info: DatasetInfo,
    devices: List[torch.device],
    hypers: Dict = DEFAULT_HYPERS,
    continue_from: Optional[str] = None,
    output_dir: str = ".",
):
    all_species = get_all_species(train_datasets + validation_datasets)
    outputs = {
        key: ModelOutput(
            quantity=value.quantity,
            unit=value.unit,
            per_atom=False,
        )
        for key, value in dataset_info.targets.items()
    }
    new_capabilities = ModelCapabilities(
        length_unit=dataset_info.length_unit,
        outputs=outputs,
        atomic_types=all_species,
        supported_devices=["cpu", "cuda"],
    )

    # Create the model:
    if continue_from is None:
        model = Model(
            capabilities=new_capabilities,
            hypers=hypers["model"],
        )
        novel_capabilities = new_capabilities
    else:
        model = load(continue_from)
        if is_exported(model):
            raise ValueError("model is already exported and can't be used for continue")

        filtered_new_dict = {k: v for k, v in hypers["model"].items() if k != "restart"}
        filtered_old_dict = {k: v for k, v in model.hypers.items() if k != "restart"}
        if filtered_new_dict != filtered_old_dict:
            logger.warning(
                "The hyperparameters of the model have changed since the last "
                "training run. The new hyperparameters will be discarded."
            )
        # merge the model's capabilities with the requested capabilities
        merged_capabilities, novel_capabilities = merge_capabilities(
            model.capabilities, new_capabilities
        )
        model.capabilities = merged_capabilities
        # make the new model capable of handling the new outputs
        for output_name in novel_capabilities.outputs.keys():
            model.add_output(output_name)

    model_capabilities = model.capabilities

    # Perform checks on the datasets:
    logger.info("Checking datasets for consistency")
    check_datasets(
        train_datasets,
        validation_datasets,
        raise_incompatibility_error=continue_from is None,
        # only error if we are not continuing
    )

    device = devices[0]  # only one device, as we don't support multi-gpu for now
    dtype = train_datasets[0][0].system.positions.dtype

    logger.info(f"training on device {device} with dtype {dtype}")
    model.to(device=device, dtype=dtype)

    hypers_training = hypers["training"]

    # Calculate and set the composition weights for all targets:
    logger.info("Calculating composition weights")
    for target_name in novel_capabilities.outputs.keys():
        # TODO: warn in the documentation that capabilities that are already
        # present in the model won't recalculate the composition weights
        # find the datasets that contain the target:

        if target_name in hypers_training["fixed_composition_weights"].keys():
            logger.info(
                f"For {target_name}, model will proceed with "
                "user-supplied composition weights"
            )
            cur_weight_dict = hypers_training["fixed_composition_weights"][target_name]
            species = []
            num_species = len(cur_weight_dict)
            fixed_weights = torch.zeros(num_species, dtype=dtype, device=device)

            for ii, (key, weight) in enumerate(cur_weight_dict.items()):
                species.append(key)
                fixed_weights[ii] = weight

            all_species = []
            for dataset in train_datasets:
                all_species += get_all_species(dataset)

            if not set(species) == set(all_species):
                raise ValueError(
                    "Values were not supplied for all "
                    "the species in present in the dataset"
                )
            model.set_composition_weights(target_name, fixed_weights, species)

        else:
            train_datasets_with_target = []
            for dataset in train_datasets:
                if target_name in get_all_targets(dataset):
                    train_datasets_with_target.append(dataset)
            if len(train_datasets_with_target) == 0:
                raise ValueError(
                    f"Target {target_name} in the model's new capabilities is not "
                    "present in any of the training datasets."
                )
            composition_weights, species = calculate_composition_weights(
                train_datasets_with_target, target_name
            )
            model.set_composition_weights(target_name, composition_weights, species)

    logger.info("Setting up data loaders")

    # Create dataloader for the training datasets:
    train_dataloaders = []
    for dataset in train_datasets:
        train_dataloaders.append(
            DataLoader(
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
            DataLoader(
                dataset=dataset,
                batch_size=hypers_training["batch_size"],
                shuffle=False,
                collate_fn=collate_fn,
            )
        )
    validation_dataloader = combine_dataloaders(validation_dataloaders, shuffle=False)

    # Extract all the possible outputs and their gradients from the training set:
    outputs_dict = get_outputs_dict(train_datasets)
    for output_name in outputs_dict.keys():
        if output_name not in model_capabilities.outputs:
            raise ValueError(
                f"Output {output_name} is not in the model's capabilities."
            )

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

    # Create a scheduler:
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=hypers_training["scheduler_factor"],
        patience=hypers_training["scheduler_patience"],
    )

    # counters for early stopping:
    best_validation_loss = float("inf")
    epochs_without_improvement = 0

    # Train the model:
    logger.info("Starting training")
    for epoch in range(hypers_training["num_epochs"]):
        # aggregated information holders:
        aggregated_train_info: Dict[str, Tuple[float, int]] = {}
        aggregated_validation_info: Dict[str, Tuple[float, int]] = {}

        train_loss = 0.0
        for batch in train_dataloader:
            optimizer.zero_grad()

            systems, targets = batch
            loss, info = compute_model_loss(
                loss_fn, model, systems, targets, hypers_training["per_atom_targets"]
            )

            train_loss += loss.item()
            loss.backward()
            optimizer.step()
            aggregated_train_info = update_aggregated_info(aggregated_train_info, info)
        finalized_train_info = finalize_aggregated_info(aggregated_train_info)

        validation_loss = 0.0
        for batch in validation_dataloader:
            systems, targets = batch
            # TODO: specify that the model is not training here to save some autograd

            loss, info = compute_model_loss(
                loss_fn, model, systems, targets, hypers_training["per_atom_targets"]
            )

            validation_loss += loss.item()
            aggregated_validation_info = update_aggregated_info(
                aggregated_validation_info, info
            )
        finalized_validation_info = finalize_aggregated_info(aggregated_validation_info)

        lr_scheduler.step(validation_loss)

        # Now we log the information:
        if epoch == 0:
            metric_logger = MetricLogger(
                model_capabilities,
                train_loss,
                validation_loss,
                finalized_train_info,
                finalized_validation_info,
            )
        if epoch % hypers_training["log_interval"] == 0:
            metric_logger.log(
                epoch,
                train_loss,
                validation_loss,
                finalized_train_info,
                finalized_validation_info,
            )

        if epoch % hypers_training["checkpoint_interval"] == 0:
            save(
                model,
                Path(output_dir) / f"model_{epoch}.ckpt",
            )

        # early stopping criterion:
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= hypers_training["early_stopping_patience"]:
                logger.info(
                    "Early stopping criterion reached after "
                    f"{hypers_training['early_stopping_patience']} epochs "
                    "without improvement."
                )
                break

    return model
