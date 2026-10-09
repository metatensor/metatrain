import copy
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Sequence, Union, cast

import metatensor.torch as mts
import torch
from torch.utils.data import DataLoader

from metatrain.utils.abc import (
    ModelInterface,
    TrainerInterface,
    common_upgrade_checkpoint,
)
from metatrain.utils.additive import remove_additive
from metatrain.utils.data import (
    CollateFn,
    CombinedDataLoader,
    Dataset,
    DatasetInfo,
    build_val_dataloaders,
    get_num_workers,
    unpack_batch,
    validate_num_workers,
)
from metatrain.utils.data.atomic_basis_helpers import (
    densify_atomic_basis_dataset_info,
    get_prepare_atomic_basis_targets_transform,
)
from metatrain.utils.distributed.slurm import (
    initialize_slurm_nccl_process_group,
    resolve_distributed,
)
from metatrain.utils.io import check_file_extension, model_from_checkpoint
from metatrain.utils.neighbor_lists import (
    get_system_with_neighbor_lists_transform,
)
from metatrain.utils.per_atom import average_by_num_atoms
from metatrain.utils.transfer import batch_to
from metatrain.utils.wrapper import MetatrainModel

from . import checkpoints
from .documentation import ModelHypers, TrainerHypers
from .model import Scaler


class Trainer(TrainerInterface[TrainerHypers, ModelHypers]):
    __checkpoint_version__ = 2

    def __init__(self, hypers: TrainerHypers):
        super().__init__(hypers)

    def setup(
        self, model_hypers: ModelHypers, dataset_info: DatasetInfo
    ) -> MetatrainModel[Scaler]:
        if self.hypers["densify_atomic_basis"]:
            model_dataset_info = densify_atomic_basis_dataset_info(dataset_info)
        return MetatrainModel(
            model=Scaler(model_hypers, model_dataset_info),
            additive_models=[],
            scaler=None,
            dataset_info=dataset_info,
        )

    def restart(
        self,
        model: MetatrainModel[Scaler],
        dataset_info: DatasetInfo,
        model_hypers: ModelHypers,
    ) -> MetatrainModel[Scaler]:
        if self.hypers["densify_atomic_basis"]:
            model_dataset_info = densify_atomic_basis_dataset_info(dataset_info)

        model.restart(dataset_info, model_dataset_info, model_hypers=model_hypers)

        return model

    def train(
        self,
        model: MetatrainModel[Scaler],
        dtype: torch.dtype,
        devices: List[torch.device],
        train_datasets: List[Union[Dataset, torch.utils.data.Subset]],
        val_datasets: List[Union[Dataset, torch.utils.data.Subset]],
        checkpoint_dir: str,
    ) -> None:

        assert isinstance(model, MetatrainModel)
        scaler = model.model
        assert isinstance(scaler, Scaler)

        model = model.to(dtype=dtype, device=devices[0])

        # Get additive models, if they are paths to checkpoints load them.
        input_additive_models = self.hypers["additive_models"]
        additive_models = []
        for additive_model in input_additive_models:
            if isinstance(additive_model, str):
                ckpt = torch.load(
                    additive_model, map_location="cpu", weights_only=False
                )
                additive_model = model_from_checkpoint(ckpt, context="export")
            if not isinstance(additive_model, torch.nn.Module):
                raise ValueError(
                    f"Additive model must be a path to a checkpoint or a "
                    f"torch.nn.Module instance, got {type(additive_model)}."
                )
            additive_models.append(additive_model)

        # Store in the model the identifiers of the additive models used
        # to train the scaler, so that the scaler can check that the
        # same additive models are used at inference time.
        scaler.training_additive_models = [
            additive_model.__class__.__name__ for additive_model in additive_models
        ]

        is_distributed = resolve_distributed(self.hypers.get("distributed"))
        fixed_weights = self.hypers["fixed_weights"]
        per_structure_targets = self.hypers["per_structure_targets"]
        batch_size = self.hypers["batch_size"]
        if batch_size is None:
            batch_size = min(len(dataset) for dataset in train_datasets)

        if not isinstance(train_datasets, list):
            train_datasets = [train_datasets]

        if per_structure_targets is None:
            per_structure_targets = []

        if len(scaler.target_infos) == 0:  # no (new) targets to fit
            return

        # When trained from within another architecture, the parent trainer has
        # already initialized the process group; standalone (`mtt train`)
        # distributed runs must do it here, and get their device from it.
        owns_process_group = False
        if is_distributed and not torch.distributed.is_initialized():
            device, _, _ = initialize_slurm_nccl_process_group(
                self.hypers["distributed_port"]
            )
            owns_process_group = True
        else:
            device = devices[0]
        if is_distributed:
            world_size = torch.distributed.get_world_size()
            logging.info(f"Training on {world_size} devices with dtype {dtype}")
        else:
            logging.info(f"Training on device {device} with dtype {dtype}")
        model.to(device=device)

        skip_accumulation = fixed_weights is not None and all(
            t in fixed_weights for t in scaler.new_outputs
        )

        # if per-property scales are required for any of the new outputs, we need to
        # accumulate even if fixed_weights are provided, as the per-property scales are
        # needed to apply the fixed_weights correctly
        require_per_property_scales = any(
            [
                target_name in scaler.model.multi_property_target_names
                for target_name in scaler.new_outputs
            ]
        )
        skip_accumulation = skip_accumulation and not require_per_property_scales

        device = scaler.dummy_buffer.device

        if not skip_accumulation:
            initial_transforms = []
            # Attach the neighbor lists needed by the additive models whose
            # contributions are removed below (e.g. ZBL): the scaler cannot
            # rely on a previous fit having attached them to the systems.
            requested_neighbor_lists = []
            for additive_model in additive_models:
                if hasattr(additive_model, "requested_neighbor_lists"):
                    requested_neighbor_lists += (
                        additive_model.requested_neighbor_lists()
                    )
            initial_transforms.append(
                get_system_with_neighbor_lists_transform(requested_neighbor_lists)
            )

            # Make sure the data is transformed to what the scaler needs.
            atomic_basis_transform, _ = get_prepare_atomic_basis_targets_transform(
                model.dataset_info,
                scaler.dataset_info,
            )
            initial_transforms.append(atomic_basis_transform)

            # Create dataloader for the training datasets
            dataloader = self._get_dataloader(
                model,
                train_datasets,
                batch_size,
                is_distributed=is_distributed,
                initial_transforms=initial_transforms,
            )

            # accumulate
            for batch in dataloader:
                systems, targets, extra_data = unpack_batch(batch)
                systems, targets, extra_data = batch_to(
                    systems, targets, extra_data, device=device
                )
                if len(targets) == 0:
                    break

                # remove additive contributions from these targets
                for additive_model in additive_models:
                    targets = remove_additive(
                        systems,
                        targets,
                        additive_model,
                        {
                            target_name: scaler.target_infos[target_name]
                            for target_name in targets
                        },
                    )
                targets = average_by_num_atoms(targets, systems, per_structure_targets)
                scaler.model.accumulate(systems, targets, extra_data)

            if is_distributed:
                torch.distributed.barrier()
                # All-reduce the accumulated TensorMaps across all processes
                for target_name in model.new_outputs:
                    for N_block, Y2_block in zip(
                        scaler.model.N[target_name],
                        scaler.model.Y2[target_name],
                        strict=True,
                    ):
                        torch.distributed.all_reduce(N_block.values)
                        torch.distributed.all_reduce(Y2_block.values)
        else:
            logging.info(
                "Skipping weight calculation: fixed_weights provided for all targets "
                "to fit."
            )

        # Compute the scales on all ranks
        scaler.model.fit(fixed_weights=fixed_weights, targets_to_fit=scaler.new_outputs)

        # update the buffer scales now they are fitted
        for target_name in scaler.model.scales.keys():
            scaler.register_buffer(
                target_name + "_scaler_buffer",
                mts.save_buffer(
                    mts.make_contiguous(
                        scaler.model.scales[target_name].to("cpu", torch.float64)
                    )
                ).to(device),
            )

        # update the buffer scales now they are fitted
        for target_name in scaler.model.scales.keys():
            scaler.register_buffer(
                target_name + "_per_target_scaler_buffer",
                mts.save_buffer(
                    mts.make_contiguous(
                        scaler.model.per_target_scales[target_name].to(
                            "cpu", torch.float64
                        )
                    )
                ).to(device),
            )

        if any(
            [
                target_name in scaler.model.multi_property_target_names
                for target_name in scaler.new_outputs
            ]
        ):
            # Now accumulate quantities for computing per-property scales
            for batch in dataloader:
                systems, targets, extra_data = unpack_batch(batch)
                systems, targets, extra_data = batch_to(
                    systems, targets, extra_data, device=device
                )
                if len(targets) == 0:
                    break

                # remove additive contributions from these targets
                for additive_model in additive_models:
                    targets = remove_additive(
                        systems,
                        targets,
                        additive_model,
                        {
                            target_name: scaler.target_infos[target_name]
                            for target_name in targets
                        },
                    )
                targets = average_by_num_atoms(targets, systems, per_structure_targets)
                scaler.model.accumulate_per_property(systems, targets, extra_data)

            if is_distributed:
                torch.distributed.barrier()
                # All-reduce the accumulated TensorMaps across all processes
                for target_name in scaler.new_outputs:
                    if target_name not in scaler.model.multi_property_target_names:
                        continue
                    for N_block, Y2_block in zip(
                        scaler.model.per_property_N[target_name],
                        scaler.model.per_property_Y2[target_name],
                        strict=True,
                    ):
                        torch.distributed.all_reduce(N_block.values)
                        torch.distributed.all_reduce(Y2_block.values)

            # Compute the scales on all ranks
            scaler.model.fit_per_property(targets_to_fit=scaler.new_outputs)

            # update the buffer scales now they have been updated with per-property
            # scales
            for target_name in scaler.model.scales.keys():
                scaler.register_buffer(
                    target_name + "_scaler_buffer",
                    mts.save_buffer(
                        mts.make_contiguous(
                            scaler.model.scales[target_name].to("cpu", torch.float64)
                        )
                    ).to(device),
                )

            # update the buffer scales now they are fitted
            for target_name in scaler.model.per_property_scales.keys():
                scaler.register_buffer(
                    target_name + "_per_property_scaler_buffer",
                    mts.save_buffer(
                        mts.make_contiguous(
                            scaler.model.per_property_scales[target_name].to(
                                "cpu", torch.float64
                            )
                        )
                    ).to(device),
                )

        if checkpoint_dir and (not is_distributed or torch.distributed.get_rank() == 0):
            ckpt_path = Path(checkpoint_dir) / "scaler.ckpt"
            self.save_checkpoint(model, ckpt_path)

        if owns_process_group:
            torch.distributed.destroy_process_group()

    def _get_dataloader(
        self,
        model: MetatrainModel,
        datasets: List[Union[Dataset, torch.utils.data.Subset]],
        batch_size: int,
        is_distributed: bool,
        initial_transforms: Sequence[Callable],
    ) -> DataLoader:
        """
        Create a DataLoader for the provided datasets. As the dataloader is only used to
        accumulate the quanitites needed for fitting the scales, there is no need to
        shuffle or drop the last non-full batch. Distributed sampling can be used or
        not, based on the `is_distributed` argument, and training with double precision
        is enforced.

        :param model: The scaler for which the dataloader is being created.
        :param datasets: List of datasets to create the dataloader from.
        :param batch_size: Batch size to use for the dataloader.
        :param is_distributed: Whether to use distributed sampling or not.
        :param initial_transforms: A list of callables to be included in
            the collate function. The callables passed here will be
            applied before the other callables set by the scaler.
        :return: The created DataLoader.
        """
        # Create the collate function
        targets_keys = list(model.dataset_info.targets.keys())
        collate_fn = CollateFn(
            target_keys=targets_keys, callables=[*initial_transforms]
        )

        dtype = datasets[0][0]["system"].positions.dtype
        if dtype != torch.float64:
            raise ValueError(
                f"The scaler only supports float64 during training. Got dtype: {dtype}."
            )

        if is_distributed:
            world_size = torch.distributed.get_world_size()
            rank = torch.distributed.get_rank()
            # Strided shards instead of DistributedSampler: its padding
            # duplicates samples to equalize shard sizes, which would bias
            # the least-squares fit. Unequal shard sizes are fine here, as
            # the only collective is the final all_reduce.
            samplers: List[Any] = [
                range(rank, len(dataset), world_size) for dataset in datasets
            ]
        else:
            samplers = [None] * len(datasets)

        if self.hypers["num_workers"] is None:
            num_workers = get_num_workers()
            logging.info(
                "Number of workers for data-loading not provided and chosen "
                f"automatically. Using {num_workers} workers."
            )
        else:
            num_workers = self.hypers["num_workers"]
            validate_num_workers(num_workers)

        # The validation dataloader builder covers every sample exactly
        # once, without shuffling or dropping the tail batch, which is what
        # the deterministic fit needs (the training builder does both).
        dataloaders = build_val_dataloaders(
            val_datasets=datasets,
            val_distributed_samplers=samplers,
            collate_fn_val=collate_fn,
            batch_size=batch_size,
            max_atoms_per_batch=None,
            num_workers=num_workers,
        )

        return CombinedDataLoader(dataloaders, shuffle=False)

    def save_checkpoint(self, model: ModelInterface, path: Union[str, Path]) -> None:
        # epoch, best_epoch and best_model_state_dict are already set by
        # model.get_checkpoint()
        checkpoint = model.get_checkpoint()

        # Remove the additive models from the trainer hypers, so that we don't save
        # torch modules in the checkpoint.
        trainer_hypers = cast(dict, self.hypers.copy())
        trainer_hypers.pop("additive_models", None)

        checkpoint.update(
            {
                "train_hypers": trainer_hypers,
                "trainer_ckpt_version": self.__checkpoint_version__,
            }
        )
        torch.save(checkpoint, check_file_extension(path, ".ckpt"))

    @classmethod
    def load_checkpoint(
        cls,
        checkpoint: Dict[str, Any],
        hypers: TrainerHypers,
        context: Literal["restart", "finetune"],
    ) -> "Trainer":
        trainer_hypers = copy.copy(checkpoint.get("train_hypers", {}))
        trainer_hypers.update(hypers)
        return cls(trainer_hypers)

    @classmethod
    def upgrade_checkpoint(
        cls, checkpoint: dict[str, Any], version: Optional[int] = None
    ) -> dict[str, Any]:
        return common_upgrade_checkpoint(
            checkpoint, version, cls.__checkpoint_version__, "trainer", checkpoints
        )
