import logging
from typing import List, Union

import metatensor
import metatensor.torch
import torch
from metatensor.torch import TensorMap

from metatrain.utils.data import Dataset

from ...utils.composition import calculate_composition_weights
from ...utils.data import check_datasets
from . import GAP
from .model import torch_tensor_map_to_core


logger = logging.getLogger(__name__)


class Trainer:
    def __init__(self, train_hypers):
        self.hypers = train_hypers

    def train(
        self,
        model: GAP,
        devices: List[torch.device],
        train_datasets: List[Union[Dataset, torch.utils.data.Subset]],
        validation_datasets: List[Union[Dataset, torch.utils.data.Subset]],
        checkpoint_dir: str,
    ):
        # checks
        assert devices == [torch.device("cpu")]
        dtype = train_datasets[0][0]["system"].positions.dtype
        assert dtype == torch.float64
        target_name = next(iter(model.dataset_info.targets.keys()))
        if len(train_datasets) != 1:
            raise ValueError("GAP only supports a single training dataset")
        if len(validation_datasets) != 1:
            raise ValueError("GAP only supports a single validation dataset")
        outputs_dict = model.dataset_info.targets
        if len(outputs_dict.keys()) > 1:
            raise NotImplementedError("More than one output is not supported yet.")
        output_name = next(iter(outputs_dict.keys()))

        # Perform checks on the datasets:
        logger.info("Checking datasets for consistency")
        check_datasets(train_datasets, validation_datasets)

        logger.info(f"Training on device cpu with dtype {dtype}")

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

        logger.info("Fitting composition energies")
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

        logger.info("Calculating SOAP features")
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

        logger.info("Selecting sparse points")
        lens = len(train_tensor[0].values)
        if model._sampler._n_to_select > lens:
            raise ValueError(
                f"""number of sparse points ({model._sampler._n_to_select})
 should be smaller than the number of environments ({lens})"""
            )
        sparse_points = model._sampler.fit_transform(train_tensor)
        sparse_points = metatensor.operations.remove_gradients(sparse_points)
        alpha_energy = self.hypers["regularizer"]
        if self.hypers["regularizer_forces"] is None:
            alpha_forces = alpha_energy
        else:
            alpha_forces = self.hypers["regularizer_forces"]

        logger.info("Fitting GAP model")
        model._subset_of_regressors.fit(
            train_tensor,
            sparse_points,
            train_y,
            alpha=alpha_energy,
            alpha_forces=alpha_forces,
        )

        model._subset_of_regressors_torch = (
            model._subset_of_regressors.export_torch_script_model()
        )
