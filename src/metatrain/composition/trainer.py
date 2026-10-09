import copy
import logging
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

import metatensor.torch as mts
import torch

from metatrain.utils.abc import (
    ModelInterface,
    TrainerInterface,
    common_upgrade_checkpoint,
)
from metatrain.utils.additive.remove import get_remove_additive_transform
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
from metatrain.utils.io import check_file_extension
from metatrain.utils.neighbor_lists import get_system_with_neighbor_lists_transform
from metatrain.utils.transfer import batch_to
from metatrain.utils.wrapper import MetatrainModel

from . import checkpoints
from .documentation import ModelHypers, TrainerHypers
from .model import CompositionModel


class Trainer(TrainerInterface[TrainerHypers, ModelHypers]):
    __checkpoint_version__ = 3

    def __init__(self, hypers: TrainerHypers):
        super().__init__(hypers)

        # Other additive models (e.g. ZBL) whose contributions are subtracted
        # from the targets before fitting. Set by architectures that train the
        # composition model as an additive baseline (see
        # train_or_load_composition_model); empty for standalone training.
        self._additive_models: List[torch.nn.Module] = []

    def setup(
        self, model_hypers: ModelHypers, dataset_info: DatasetInfo
    ) -> MetatrainModel[CompositionModel]:
        if self.hypers["densify_atomic_basis"]:
            model_dataset_info = densify_atomic_basis_dataset_info(dataset_info)
        return MetatrainModel(
            model=CompositionModel(model_hypers, model_dataset_info),
            additive_models=[],
            scaler=None,
            dataset_info=dataset_info,
        )

    def restart(
        self,
        model: MetatrainModel[CompositionModel],
        dataset_info: DatasetInfo,
        model_hypers: ModelHypers,
    ) -> MetatrainModel[CompositionModel]:
        if self.hypers["densify_atomic_basis"]:
            model_dataset_info = densify_atomic_basis_dataset_info(dataset_info)

        model.restart(dataset_info, model_dataset_info, model_hypers=model_hypers)

        return model

    def train(
        self,
        model: MetatrainModel[CompositionModel],
        dtype: torch.dtype,
        devices: List[torch.device],
        train_datasets: List[Union[Dataset, torch.utils.data.Subset]],
        val_datasets: List[Union[Dataset, torch.utils.data.Subset]],
        checkpoint_dir: str,
    ) -> None:
        assert isinstance(model, MetatrainModel)
        composition_model = model.model
        assert isinstance(composition_model, CompositionModel)

        additive_models = self._additive_models
        is_distributed = resolve_distributed(self.hypers.get("distributed"))
        fixed_weights = self.hypers["atomic_baseline"]
        batch_size = self.hypers["batch_size"]
        if batch_size is None:
            batch_size = min(len(dataset) for dataset in train_datasets)

        if len(composition_model.target_infos) == 0:
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

        # Targets with fixed weights don't need data accumulation, only fit().
        targets_to_accumulate = [
            target_name
            for target_name in composition_model._new_outputs
            if target_name not in fixed_weights
        ]

        if len(targets_to_accumulate) > 0:
            requested_neighbor_lists = []
            for additive_model in additive_models:
                if hasattr(additive_model, "requested_neighbor_lists"):
                    requested_neighbor_lists += (
                        additive_model.requested_neighbor_lists()
                    )

            # Get the transform that prepares the atomic basis targets for
            # the composition model by converting them to its required format.
            atomic_basis_transform, _ = get_prepare_atomic_basis_targets_transform(
                model.dataset_info, composition_model.dataset_info
            )

            # The additive contributions are removed inside the collate
            # function: the neighbor lists they may need (e.g. ZBL) are
            # attached there, and do not survive the transfer from dataloader
            # worker processes to the training process. The transform gets CPU
            # copies of the additive models, since the collate runs in the
            # worker processes, where batches are on the CPU and CUDA cannot
            # be used after fork.
            cpu_additive_models = [
                copy.deepcopy(additive_model).to(device="cpu", dtype=torch.float64)
                for additive_model in additive_models
            ]
            callables = [
                atomic_basis_transform,
                get_system_with_neighbor_lists_transform(requested_neighbor_lists),
                get_remove_additive_transform(
                    cpu_additive_models, composition_model.target_infos
                ),
            ]

            collate_fn = CollateFn(
                target_keys=list(model.dataset_info.targets.keys()),
                callables=callables,
            )

            if train_datasets[0][0]["system"].positions.dtype != torch.float64:
                raise ValueError(
                    "The composition model only supports float64 during training. "
                    f"Got dtype: {train_datasets[0][0]['system'].positions.dtype}."
                )

            if is_distributed:
                world_size = torch.distributed.get_world_size()
                rank = torch.distributed.get_rank()
                # Strided shards instead of DistributedSampler: its padding
                # duplicates samples to equalize shard sizes, which would bias
                # the least-squares fit. Unequal shard sizes are fine here, as
                # the only collective is the final all_reduce.
                samplers: List[Any] = [
                    range(rank, len(dataset), world_size) for dataset in train_datasets
                ]
            else:
                samplers = [None] * len(train_datasets)

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
                val_datasets=train_datasets,
                val_distributed_samplers=samplers,
                collate_fn_val=collate_fn,
                batch_size=batch_size,
                max_atoms_per_batch=None,
                num_workers=num_workers,
            )
            dataloader = CombinedDataLoader(dataloaders, shuffle=False)

            for batch in dataloader:
                systems, targets, _ = unpack_batch(batch)
                systems, targets, _ = batch_to(systems, targets, device=device)
                targets = {
                    target_name: targets[target_name]
                    for target_name in targets
                    if target_name in targets_to_accumulate
                }
                # A batch can lack all targets to fit, e.g. when it comes from a
                # dataset of a CombinedDataLoader that only holds other targets.
                if len(targets) == 0:
                    continue

                composition_model.model.accumulate(systems, targets)

        if is_distributed:
            torch.distributed.barrier()
            # A rank whose shard of some dataset was empty never accumulated
            # the corresponding targets, so its XTX/XTY are still on the CPU,
            # while NCCL needs them on the GPU for the all_reduce.
            composition_model.model._sync_device_dtype(device, torch.float64)
            handles = []
            for target_name in targets_to_accumulate:
                for XTX_block, XTY_block in zip(
                    composition_model.model.XTX[target_name],
                    composition_model.model.XTY[target_name],
                    strict=True,
                ):
                    handles.append(
                        torch.distributed.all_reduce(XTX_block.values, async_op=True)
                    )
                    handles.append(
                        torch.distributed.all_reduce(XTY_block.values, async_op=True)
                    )
            for handle in handles:
                handle.wait()

        composition_model.model.fit(
            fixed_weights, targets_to_fit=composition_model._new_outputs
        )

        for target_name in composition_model.model.weights.keys():
            composition_model.register_buffer(
                target_name + "_composition_buffer",
                mts.save_buffer(
                    mts.make_contiguous(
                        composition_model.model.weights[target_name].to(
                            "cpu", torch.float64
                        )
                    )
                ).to(device),
            )

        if checkpoint_dir and (not is_distributed or torch.distributed.get_rank() == 0):
            ckpt_path = Path(checkpoint_dir) / "composition_model.ckpt"
            self.save_checkpoint(model, ckpt_path)

        if owns_process_group:
            torch.distributed.destroy_process_group()

    def save_checkpoint(self, model: ModelInterface, path: Union[str, Path]) -> None:
        # epoch, best_epoch and best_model_state_dict are already set by
        # model.get_checkpoint()
        checkpoint = model.get_checkpoint()
        checkpoint.update(
            {
                "train_hypers": self.hypers,
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
        cls, checkpoint: Dict, version: Optional[int] = None
    ) -> Dict:
        return common_upgrade_checkpoint(
            checkpoint, version, cls.__checkpoint_version__, "trainer", checkpoints
        )
