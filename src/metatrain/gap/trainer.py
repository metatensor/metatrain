import logging
from pathlib import Path
from typing import List, Union

import metatensor
import metatensor.torch
import torch
from metatensor.torch import TensorMap

from metatrain.utils.data import Dataset

from ..utils.additive import remove_additive
from ..utils.data import check_datasets
from ..utils.neighbor_lists import (
    get_requested_neighbor_lists,
    get_system_with_neighbor_lists,
)
from . import GAP
from .model import torch_tensor_map_to_core


class Trainer:
    def __init__(self, train_hypers):
        self.hypers = train_hypers

    def train(
        self,
        model: GAP,
        dtype: torch.dtype,
        devices: List[torch.device],
        train_datasets: List[Union[Dataset, torch.utils.data.Subset]],
        val_datasets: List[Union[Dataset, torch.utils.data.Subset]],
        checkpoint_dir: str,
    ):
        # checks
        assert dtype in GAP.__supported_dtypes__
        assert devices == [torch.device("cpu")]
        target_name = next(iter(model.dataset_info.targets.keys()))
        if len(train_datasets) != 1:
            raise ValueError("GAP only supports a single training dataset")
        if len(val_datasets) != 1:
            raise ValueError("GAP only supports a single validation dataset")
        outputs_dict = model.dataset_info.targets
        if len(outputs_dict.keys()) > 1:
            raise NotImplementedError("More than one output is not supported yet.")
        output_name = next(iter(outputs_dict.keys()))

        # Perform checks on the datasets:
        logging.info("Checking datasets for consistency")
        check_datasets(train_datasets, val_datasets)

        logging.info(f"Training on device cpu with dtype {dtype}")

        # Calculate and set the composition weights:
        logging.info("Calculating composition weights")
        # model.additive_models[0] is the composition model
        model.additive_models[0].train_model(train_datasets, model.additive_models[1:])

        logging.info("Setting up data loaders")
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

        logging.info("Calculating neighbor lists for the datasets")
        requested_neighbor_lists = get_requested_neighbor_lists(model)
        for dataset in train_datasets + val_datasets:
            for i in range(len(dataset)):
                system = dataset[i]["system"]
                # The following line attaches the neighbors lists to the system,
                # and doesn't require to reassign the system to the dataset:
                _ = get_system_with_neighbor_lists(system, requested_neighbor_lists)

        logging.info("Subtracting composition energies")  # and potentially ZBL
        train_targets = {target_name: train_y}
        for additive_model in model.additive_models:
            train_targets = remove_additive(
                train_structures,
                train_targets,
                additive_model,
                model.dataset_info.targets,
            )
        train_y = train_targets[target_name]

        logging.info("Calculating SOAP features")
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

        logging.info("Selecting sparse points")
        lens = len(train_tensor[0].values)
        if model._sampler._n_to_select > lens:
            raise ValueError(
                f"Number of sparse points ({model._sampler._n_to_select}) "
                f"should be smaller than the number of environments ({lens})"
            )
        sparse_points = model._sampler.fit_transform(train_tensor)
        sparse_points = metatensor.operations.remove_gradients(sparse_points)
        alpha_energy = self.hypers["regularizer"]
        if self.hypers["regularizer_forces"] is None:
            alpha_forces = alpha_energy
        else:
            alpha_forces = self.hypers["regularizer_forces"]

        logging.info("Fitting GAP model")
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

    def save_checkpoint(self, model, checkpoint_dir: str):
        # GAP won't save a checkpoint since it
        # doesn't support restarting training
        return

    @classmethod
    def load_checkpoint(cls, path: Union[str, Path], hypers_train) -> "GAP":
        raise ValueError("GAP does not allow restarting training")
