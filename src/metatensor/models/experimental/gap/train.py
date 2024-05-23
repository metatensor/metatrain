import logging
import warnings
from typing import Dict, List, Optional, Union

import metatensor.torch
import rascaline
import torch
from metatensor.torch import TensorMap
from metatensor.torch.atomistic import ModelCapabilities, ModelOutput

import metatensor
from metatensor.models.utils.data import Dataset

from ...utils.composition import calculate_composition_weights
from ...utils.data import DatasetInfo, check_datasets, get_all_species
from ...utils.extract_targets import get_outputs_dict
from . import DEFAULT_HYPERS
from .model import Model, torch_tensor_map_to_core


logger = logging.getLogger(__name__)

# disable rascaline logger
rascaline.set_logging_callback(lambda x, y: None)

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
    checkpoint_dir: str = ".",
):
    # checks
    assert devices == [torch.device("cpu")]
    if continue_from is not None:
        raise ValueError("Training from a checkpoint is not supported in GAP.")
    dtype = train_datasets[0][0]["system"].positions.dtype
    if dtype != torch.float64:
        raise ValueError("GAP only supports float64")
    if len(dataset_info.targets) != 1:
        raise ValueError("GAP only supports a single target")
    target_name = next(iter(dataset_info.targets.keys()))
    if dataset_info.targets[target_name].quantity != "energy":
        raise ValueError("GAP only supports energies as target")
    if dataset_info.targets[target_name].per_atom:
        raise ValueError("GAP does not support per-atom energies")
    if len(train_datasets) != 1:
        raise ValueError("GAP only supports a single training dataset")
    if len(validation_datasets) != 1:
        raise ValueError("GAP only supports a single validation dataset")

    all_species = get_all_species(train_datasets + validation_datasets)
    outputs = {
        key: ModelOutput(
            quantity=value.quantity,
            unit=value.unit,
            per_atom=value.per_atom,
        )
        for key, value in dataset_info.targets.items()
    }
    capabilities = ModelCapabilities(
        length_unit=dataset_info.length_unit,
        outputs=outputs,
        atomic_types=all_species,
        supported_devices=["cpu", "cuda"],
        interaction_range=hypers["model"]["soap"]["cutoff"],
        dtype="float64",
    )

    # TODO: EXPLAIN THAT IT CAN ONLY TRAIN ON CPU BUT ALSO RUN ON GPU

    # Create the model:
    model = Model(
        capabilities=capabilities,
        hypers=hypers["model"],
    )

    model_capabilities = model.capabilities

    # Perform checks on the datasets:
    logger.info("Checking datasets for consistency")
    check_datasets(train_datasets, validation_datasets)

    # Create the model:
    model = Model(
        capabilities=model_capabilities,
        hypers=hypers["model"],
    )

    logger.info("Training on device cpu")

    outputs_dict = get_outputs_dict(train_datasets)
    if len(outputs_dict.keys()) > 1:
        raise NotImplementedError("More than one output is not supported yet.")
    output_name = next(iter(outputs_dict.keys()))

    # Calculate and set the composition weights:
    logger.info("Calculating composition weights")
    composition_weights, species = calculate_composition_weights(
        train_datasets, target_name
    )
    model.set_composition_weights(target_name, composition_weights, species)

    logger.info("Setting up data loaders")

    if len(train_datasets[0][0][output_name].keys) > 1:
        raise NotImplementedError(
            "Found more than 1 key in targets. Assuming "
            "equivariant learning which is not supported yet."
        )
    train_dataset = train_datasets[0]
    train_y = metatensor.torch.join(
        [sample[output_name] for sample in train_dataset],
        axis="samples",
        remove_tensor_name=True,
    )
    model._keys = train_y.keys
    train_structures = [sample["system"] for sample in train_dataset]
    composition_energies = torch.zeros(len(train_y.block().values), dtype=dtype)
    for i, structure in enumerate(train_structures):
        for j, s in enumerate(species):
            composition_energies[i] += (
                torch.sum(structure.types == s) * composition_weights[j]
            )
    train_y_values = train_y.block().values
    train_y_values = train_y_values - composition_energies.reshape(-1, 1)
    train_block = metatensor.torch.TensorBlock(
        values=train_y_values,
        samples=train_y.block().samples,
        components=train_y.block().components,
        properties=train_y.block().properties,
    )
    if len(train_y[0].gradients_list()) > 0:
        train_block.add_gradient("positions", train_y[0].gradient("positions"))

    train_y = metatensor.torch.TensorMap(
        train_y.keys,
        [train_block],
    )

    if len(train_y[0].gradients_list()) > 0:
        train_tensor = model._soap_torch_calculator.compute(
            train_structures, gradients=["positions"]
        )
    else:
        train_tensor = model._soap_torch_calculator.compute(train_structures)
    model._species_labels = train_tensor.keys
    train_tensor = train_tensor.keys_to_samples("center_type")
    # here, we move to properties to use metatensor operations to aggregate
    # later on. Perhaps we could retain the sparsity all the way to the kernels
    # of the soap features with a lot more implementation effort
    train_tensor = train_tensor.keys_to_properties(
        ["neighbor_1_type", "neighbor_2_type"]
    )
    # change backend
    train_tensor = TensorMap(train_y.keys, train_tensor.blocks())
    train_tensor = torch_tensor_map_to_core(train_tensor)
    train_y = torch_tensor_map_to_core(train_y)

    sparse_points = model._sampler.fit_transform(train_tensor)
    sparse_points = metatensor.operations.remove_gradients(sparse_points)
    alpha_energy = hypers["training"]["regularizer"]
    if hypers["training"]["regularizer_forces"] is None:
        alpha_forces = alpha_energy
    else:
        alpha_forces = hypers["training"]["regularizer_forces"]

    model._subset_of_regressors.fit(
        train_tensor,
        sparse_points,
        train_y,
        alpha=alpha_energy,
        alpha_forces=alpha_forces,
    )

    # we export a torch scriptable regressor TorchSubsetofRegressors that is used in
    # the forward path
    model._subset_of_regressors_torch = (
        model._subset_of_regressors.export_torch_script_model()
    )

    return model
