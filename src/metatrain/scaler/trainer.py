import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Sequence, Union, cast

import metatensor.torch as mts
import torch
from torch.utils.data import DataLoader

from metatrain.utils.abc import ModelInterface, TrainerInterface
from metatrain.utils.additive import remove_additive
from metatrain.utils.data import (
    CollateFn,
    CombinedDataLoader,
    Dataset,
    build_val_dataloaders,
    get_num_workers,
    unpack_batch,
    validate_num_workers,
)
from metatrain.utils.data.atomic_basis_helpers import (
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

from .documentation import TrainerHypers


class Trainer(TrainerInterface[TrainerHypers]):
    __checkpoint_version__ = 1

    def __init__(self, hypers: TrainerHypers):
        super().__init__(hypers)

    def train(
        self,
        model: ModelInterface,
        dtype: torch.dtype,
        devices: List[torch.device],
        train_datasets: List[Union[Dataset, torch.utils.data.Subset]],
        val_datasets: List[Union[Dataset, torch.utils.data.Subset]],
        checkpoint_dir: str,
    ) -> None:
        from .model import Scaler

        assert isinstance(model, Scaler)

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
        model.training_additive_models = [
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

        if len(model.target_infos) == 0:  # no (new) targets to fit
            return

        # When trained from within another architecture, the parent trainer has
        # already initialized the process group; standalone (`mtt train`)
        # distributed runs must do it here, and get their device from it.
        owns_process_group = False
        if is_distributed and not torch.distributed.is_initialized():
            device, world_size, _ = initialize_slurm_nccl_process_group(
                self.hypers["distributed_port"]
            )
            owns_process_group = True
            logging.info(f"Training on {world_size} devices with dtype {dtype}")
        else:
            device = devices[0]
            logging.info(f"Training on device {device} with dtype {dtype}")
        model.to(device=device)

        skip_accumulation = fixed_weights is not None and all(
            t in fixed_weights for t in model.new_outputs
        )

        # if per-property scales are required for any of the new outputs, we need to
        # accumulate even if fixed_weights are provided, as the per-property scales are
        # needed to apply the fixed_weights correctly
        require_per_property_scales = any(
            [
                target_name in model.model.multi_property_target_names
                for target_name in model.new_outputs
            ]
        )
        skip_accumulation = skip_accumulation and not require_per_property_scales

        device = model.dummy_buffer.device

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

            if model.densify_atomic_basis:
                # The model fits and stores dense weights (see
                # Scaler.__init__), so incoming batches of (possibly sparse,
                # atom_type-keyed) atomic-basis targets need to be densified the same
                # way every other architecture densifies its own targets before
                # training.
                atomic_basis_transform, _ = get_prepare_atomic_basis_targets_transform(
                    model.dataset_info.targets, model.dataset_info.extra_data
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
                            target_name: model.target_infos[target_name]
                            for target_name in targets
                        },
                    )
                targets = average_by_num_atoms(targets, systems, per_structure_targets)
                model.model.accumulate(systems, targets, extra_data)

            if is_distributed:
                torch.distributed.barrier()
                # All-reduce the accumulated TensorMaps across all processes
                for target_name in model.new_outputs:
                    for N_block, Y2_block in zip(
                        model.model.N[target_name],
                        model.model.Y2[target_name],
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
        model.model.fit(fixed_weights=fixed_weights, targets_to_fit=model.new_outputs)

        # update the buffer scales now they are fitted
        for target_name in model.model.scales.keys():
            model.register_buffer(
                target_name + "_scaler_buffer",
                mts.save_buffer(
                    mts.make_contiguous(
                        model.model.scales[target_name].to("cpu", torch.float64)
                    )
                ).to(device),
            )

        # update the buffer scales now they are fitted
        for target_name in model.model.scales.keys():
            model.register_buffer(
                target_name + "_per_target_scaler_buffer",
                mts.save_buffer(
                    mts.make_contiguous(
                        model.model.per_target_scales[target_name].to(
                            "cpu", torch.float64
                        )
                    )
                ).to(device),
            )

        if any(
            [
                target_name in model.model.multi_property_target_names
                for target_name in model.new_outputs
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
                            target_name: model.target_infos[target_name]
                            for target_name in targets
                        },
                    )
                targets = average_by_num_atoms(targets, systems, per_structure_targets)
                model.model.accumulate_per_property(systems, targets, extra_data)

            if is_distributed:
                torch.distributed.barrier()
                # All-reduce the accumulated TensorMaps across all processes
                for target_name in model.new_outputs:
                    if target_name not in model.model.multi_property_target_names:
                        continue
                    for N_block, Y2_block in zip(
                        model.model.per_property_N[target_name],
                        model.model.per_property_Y2[target_name],
                        strict=True,
                    ):
                        torch.distributed.all_reduce(N_block.values)
                        torch.distributed.all_reduce(Y2_block.values)

            # Compute the scales on all ranks
            model.model.fit_per_property(targets_to_fit=model.new_outputs)

            # update the buffer scales now they have been updated with per-property
            # scales
            for target_name in model.model.scales.keys():
                model.register_buffer(
                    target_name + "_scaler_buffer",
                    mts.save_buffer(
                        mts.make_contiguous(
                            model.model.scales[target_name].to("cpu", torch.float64)
                        )
                    ).to(device),
                )

            # update the buffer scales now they are fitted
            for target_name in model.model.per_property_scales.keys():
                model.register_buffer(
                    target_name + "_per_property_scaler_buffer",
                    mts.save_buffer(
                        mts.make_contiguous(
                            model.model.per_property_scales[target_name].to(
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
        model: ModelInterface,
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
        raise ValueError(" Scaler does not allow restarting training")

    @staticmethod
    def upgrade_checkpoint(checkpoint: Dict) -> Dict:
        version = checkpoint.get("trainer_ckpt_version", 0)
        if version != Trainer.__checkpoint_version__:
            raise RuntimeError(
                f"Unable to upgrade the checkpoint: the checkpoint is using trainer "
                f"version {version}, while the current "
                f"trainer version is {Trainer.__checkpoint_version__}."
            )
        return checkpoint
