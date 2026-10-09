import logging
from pathlib import Path
from typing import Any, Dict, List, Literal, Union

import metatensor.torch as mts
import torch

from metatrain.composition import CompositionModel, train_or_load_composition_model
from metatrain.utils.abc import ModelInterface, TrainerInterface
from metatrain.utils.additive import remove_additive
from metatrain.utils.additive.zbl import ZBL
from metatrain.utils.data import Dataset, DatasetInfo, check_datasets
from metatrain.utils.neighbor_lists import (
    get_requested_neighbor_lists,
    get_system_with_neighbor_lists,
)
from metatrain.utils.wrapper import MetatrainModel

from . import GAP
from .documentation import ModelHypers, TrainerHypers


class Trainer(TrainerInterface[TrainerHypers, ModelHypers]):
    __checkpoint_version__ = 1

    def __init__(self, hypers: TrainerHypers):
        super().__init__(hypers)

    def setup(
        self, model_hypers: ModelHypers, dataset_info: DatasetInfo
    ) -> MetatrainModel[GAP]:

        model = GAP(hypers=model_hypers, dataset_info=dataset_info)

        # Set up additive models
        composition_model = CompositionModel.from_valid_targets(
            dataset_info, dataset_info.atomic_types
        )
        additive_models = [composition_model]

        # Adds the ZBL repulsion model if requested
        if model_hypers["zbl"]:
            zbl_targets = {
                target_name: target_info
                for target_name, target_info in dataset_info.targets.items()
                if ZBL.is_valid_target(target_name, target_info)
            }
            additive_models.append(
                ZBL(
                    {},
                    dataset_info=DatasetInfo(
                        length_unit=dataset_info.length_unit,
                        atomic_types=dataset_info.atomic_types,
                        targets=zbl_targets,
                    ),
                )
            )
        additive_models = torch.nn.ModuleList(additive_models)

        return MetatrainModel(
            model=model,
            additive_models=additive_models,
            scaler=None,
            dataset_info=dataset_info,
        )

    def restart(
        self,
        model: MetatrainModel[GAP],
        dataset_info: DatasetInfo,
        model_hypers: ModelHypers,
    ) -> MetatrainModel[GAP]:
        raise NotImplementedError("GAP does not allow restarting training")

    def train(
        self,
        model: MetatrainModel[GAP],
        dtype: torch.dtype,
        devices: List[torch.device],
        train_datasets: List[Union[Dataset, torch.utils.data.Subset]],
        val_datasets: List[Union[Dataset, torch.utils.data.Subset]],
        checkpoint_dir: str,
    ) -> None:
        # checks
        assert dtype in GAP.__supported_dtypes__
        assert devices == [torch.device("cpu")]
        assert isinstance(model, MetatrainModel)
        gap_model = model.model
        assert isinstance(gap_model, GAP)

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

        train_or_load_composition_model(
            composition_model=model.additive_models[0],
            atomic_baseline={},
            train_datasets=train_datasets,
            other_additive_models=list(model.additive_models[1:]),
            batch_size=1,
            is_distributed=False,
            checkpoint_dir=checkpoint_dir,
        )

        logging.info("Setting up data loaders")
        if len(train_datasets[0][0][output_name].keys) > 1:
            raise NotImplementedError(
                "Found more than 1 key in targets. Assuming "
                "equivariant learning which is not supported yet."
            )
        train_dataset = train_datasets[0]
        train_y = mts.join(
            [sample[output_name] for sample in train_dataset], axis="samples"
        )
        gap_model._keys = train_y.keys
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
            train_tensor = gap_model._soap_torch_calculator.compute(
                train_structures, gradients=["positions"]
            )
        else:
            train_tensor = gap_model._soap_torch_calculator.compute(train_structures)
        gap_model._species_labels = train_tensor.keys
        train_tensor = train_tensor.keys_to_samples("center_type")
        # here, we move to properties to use metatensor operations to aggregate
        # later on. Perhaps we could retain the sparsity all the way to the kernels
        # of the soap features with a lot more implementation effort
        train_tensor = train_tensor.keys_to_properties(
            ["neighbor_1_type", "neighbor_2_type"]
        )

        logging.info("Selecting sparse points")
        lens = len(train_tensor[0].values)
        assert gap_model._sampler._n_to_select is not None
        if gap_model._sampler._n_to_select > lens:
            raise ValueError(
                f"Number of sparse points ({gap_model._sampler._n_to_select}) "
                f"should be smaller than the number of environments ({lens})"
            )
        sparse_points = gap_model._sampler.fit_transform(train_tensor)
        sparse_points = mts.remove_gradients(sparse_points)
        alpha_energy = self.hypers["regularizer"]
        if self.hypers["regularizer_forces"] is None:
            alpha_forces = alpha_energy
        else:
            alpha_forces = self.hypers["regularizer_forces"]

        logging.info("Fitting GAP model")
        gap_model._subset_of_regressors.fit(
            train_tensor,
            sparse_points,
            train_y,
            alpha=alpha_energy,
            alpha_forces=alpha_forces,
        )

        gap_model._subset_of_regressors_torch = (
            gap_model._subset_of_regressors.export_torch_script_model()
        )

    def save_checkpoint(
        self, model: ModelInterface, checkpoint_dir: Union[str, Path]
    ) -> None:
        # GAP won't save a checkpoint since it
        # doesn't support restarting training
        return

    @classmethod
    def load_checkpoint(
        cls,
        checkpoint: Dict[str, Any],
        hypers: TrainerHypers,
        context: Literal["restart", "finetune"],
    ) -> "GAP":
        raise ValueError("GAP does not allow restarting training")

    @classmethod
    def upgrade_checkpoint(cls, checkpoint: Dict, version: int | None = None) -> Dict:
        raise NotImplementedError("checkpoint upgrade is not implemented for GAP")
