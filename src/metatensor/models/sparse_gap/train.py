import logging
import warnings
from typing import Dict, List, Union

import metatensor.torch
import numpy as np
import rascaline
import torch
from metatensor.learn.data.dataset import _BaseDataset
from metatensor.torch import TensorMap
from metatensor.torch.atomistic import ModelCapabilities

import metatensor

# TODO will be needed once we support more outputs
# from ..utils.data import get_all_targets
from ..utils.data import check_datasets
from ..utils.extract_targets import get_outputs_dict

# TODO might be important when we support mulitple capabilities
# from ..utils.merge_capabilities import merge_capabilities
from .model import DEFAULT_HYPERS, Model, torch_tensor_map_to_core


# TODO use this for composition
# from ..utils.composition import calculate_composition_weights
# PR COMMENT we do not use the loss utils since it seems more for batch-wise training
# from ..utils.compute_loss import compute_model_loss
# from ..utils.loss import TensorMapDictLoss


logger = logging.getLogger(__name__)

# disable rascaline logger
rascaline.set_logging_callback(lambda x, y: None)

# Filter out the second derivative and device warnings from rascaline-torch
warnings.filterwarnings("ignore", category=UserWarning, message="second derivative")
warnings.filterwarnings(
    "ignore", category=UserWarning, message="Systems data is on device"
)


def train(
    train_datasets: List[Union[_BaseDataset, torch.utils.data.Subset]],
    validation_datasets: List[Union[_BaseDataset, torch.utils.data.Subset]],
    requested_capabilities: ModelCapabilities,
    hypers: Dict = DEFAULT_HYPERS,
    output_dir: str = ".",
    device_str: str = "cpu",
):
    # Create the model:
    model = Model(
        capabilities=requested_capabilities,
        hypers=hypers["model"],
    )

    model_capabilities = model.capabilities

    # Perform checks on the datasets:
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

    logger.info("Training on device cpu")
    if device_str == "gpu":
        raise NotImplementedError("GPU support is not yet supported")
        device_str = "cuda"
    device = torch.device(device_str)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA is not available on this machine.")
        logger.info(
            "A cuda device was requested. The neural network will be run on GPU, "
            "but the SOAP features are calculated on CPU."
        )

    outputs_dict = get_outputs_dict(train_datasets)
    if len(outputs_dict.keys()) > 1:
        raise NotImplementedError("More than one output is not supported yet.")
    output_name = next(iter(outputs_dict.keys()))
    if len(train_datasets[0]._data[output_name][0].keys) > 1:
        raise NotImplementedError(
            "Found more than 1 key in properties. Assuming "
            "equivariant learning which is not supported yet."
        )
    # Calculate and set the composition weights for all targets:
    # TODO skipping just to make progress faster
    # logger.info("Calculating composition weights")
    # for target_name in requested_capabilities.outputs.keys():
    #    # TODO: warn in the documentation that capabilities that are already
    #    # present in the model won't recalculate the composition weights
    #    # find the datasets that contain the target:
    #    train_datasets_with_target = []
    #    for dataset in train_datasets:
    #        if target_name in get_all_targets(dataset):
    #            train_datasets_with_target.append(dataset)
    #    if len(train_datasets_with_target) == 0:
    #        raise ValueError(
    #            f"Target {target_name} in the model's new capabilities is not "
    #            "present in any of the training datasets."
    #        )
    #    composition_weights = calculate_composition_weights(
    #        train_datasets_with_target, target_name
    #    )
    #    model.set_composition_weights(target_name, composition_weights)

    logger.info("Setting up data loaders")

    # Information:
    # train_datasets: List[Union[_BaseDataset, torch.utils.data.Subset]],
    # train_datasets[datasets]._data["structure"][structures]
    # train_datasets[0:N_DATASETS]._data["structure"][0:N_STRUCTURES] # AtomicSystems
    # train_datasets[0:N_DATASETS]._data[output_name][0:N_STRUCTURES] # TensorMap

    train_y = metatensor.torch.join(
        [output for dataset in train_datasets for output in dataset._data[output_name]],
        axis="samples",
    )
    model._keys = train_y.keys
    # TODO why is there a tensor due to join?

    train_structures = [
        structure
        for dataset in train_datasets
        for structure in dataset._data["structure"]
    ]

    train_tensor = model._soap_torch_calculator.compute(
        train_structures, gradients=["positions"]
    )
    train_tensor = train_tensor.keys_to_samples("species_center")
    # TODO implement accumulate_key_names so we do not loose sparsity
    train_tensor = train_tensor.keys_to_properties(
        ["species_neighbor_1", "species_neighbor_2"]
    )
    # change backend
    train_tensor = TensorMap(train_y.keys, train_tensor.blocks())
    train_tensor = torch_tensor_map_to_core(train_tensor)
    train_y = torch_tensor_map_to_core(train_y)

    sparse_points = model._sampler.fit_transform(train_tensor)
    sparse_points = metatensor.operations.remove_gradients(sparse_points)
    model._subset_of_regressors.fit(
        train_tensor, sparse_points, train_y, alpha=hypers["training"]["regularizer"]
    )
    train_y_pred = model._subset_of_regressors.predict(train_tensor)

    # logger.info(
    #    "Train MAE:",
    #    metatensor.mean_over_samples(
    #        metatensor.abs(metatensor.subtract(train_y_pred, train_y)), "structure"
    #    )[0].values[0, 0],
    # )

    # PR COMMENT tried to use compute loss function utils but seems not working
    #            when I already aggregated the train_y before
    # loss_weights_dict = {}
    # for output_name, value_or_gradient_list in outputs_dict.items():
    #    loss_weights_dict[output_name] = {
    #        value_or_gradient: 1.0 for value_or_gradient in value_or_gradient_list
    #    }
    # loss_fn = TensorMapDictLoss(loss_weights_dict)
    # loss, info = compute_model_loss(
    #                loss_fn, model, train_structures,
    #                {output_name: train_y}
    #              )

    logger.info(
        "Train RMSE:",
        np.sqrt(np.mean((train_y_pred[0].values - train_y[0].values) ** 2)),
    )

    # ... TODO val
    # logger.info(
    #    "Validation MAE:",
    #    metatensor.mean_over_samples(
    #        metatensor.abs(metatensor.subtract(val_y_pred, val_y)), "structure"
    #    )[0].values[0, 0],
    # )

    # TODO consider these later
    # Extract all the possible outputs and their gradients from the training set:
    # outputs_dict = get_outputs_dict(train_datasets)
    # for output_name in outputs_dict.keys():
    #    if output_name not in model_capabilities.outputs:
    #        raise ValueError(
    #            f"Output {output_name} is not in the model's capabilities."
    #        )
    # Create a loss weight dict:
    # loss_weights_dict = {}
    # for output_name, value_or_gradient_list in outputs_dict.items():
    #    loss_weights_dict[output_name] = {
    #        value_or_gradient: 1.0 for value_or_gradient in value_or_gradient_list
    #    }

    # we export a torch scrip'table regressor TorchSubsetofRegressors that is used in
    # the forward path
    model._subset_of_regressors_torch = (
        model._subset_of_regressors.export_torch_script_model()
    )

    # TODO
    # model.to(device)
    return model
