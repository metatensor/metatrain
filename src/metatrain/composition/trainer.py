import copy
import logging
from pathlib import Path
from typing import Any, Dict, List, Literal, Union

import metatensor.torch as mts
import torch

from metatrain.utils.abc import ModelInterface, TrainerInterface
from metatrain.utils.additive.remove import get_remove_additive_transform
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
from metatrain.utils.io import check_file_extension
from metatrain.utils.neighbor_lists import get_system_with_neighbor_lists_transform
from metatrain.utils.transfer import batch_to

from . import checkpoints
from .documentation import TrainerHypers


class Trainer(TrainerInterface[TrainerHypers]):
    __checkpoint_version__ = 2

    def __init__(self, hypers: TrainerHypers):
        super().__init__(hypers)

        # Other additive models (e.g. ZBL) whose contributions are subtracted
        # from the targets before fitting. Set by architectures that train the
        # composition model as an additive baseline (see
        # train_or_load_composition_model); empty for standalone training.
        self._additive_models: List[torch.nn.Module] = []

    def train(
        self,
        model: ModelInterface,
        dtype: torch.dtype,
        devices: List[torch.device],
        train_datasets: List[Union[Dataset, torch.utils.data.Subset]],
        val_datasets: List[Union[Dataset, torch.utils.data.Subset]],
        checkpoint_dir: str,
    ) -> None:
        from .model import CompositionModel

        assert isinstance(model, CompositionModel)

        additive_models = self._additive_models
        is_distributed = resolve_distributed(self.hypers.get("distributed"))
        fixed_weights = self.hypers["atomic_baseline"]
        batch_size = self.hypers["batch_size"]
        if batch_size is None:
            batch_size = min(len(dataset) for dataset in train_datasets)

        if len(model.target_infos) == 0:
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

        # Targets with fixed weights don't need data accumulation, only fit().
        targets_to_accumulate = [
            target_name
            for target_name in model._new_outputs
            if target_name not in fixed_weights
        ]

        if len(targets_to_accumulate) > 0:
            requested_neighbor_lists = []
            for additive_model in additive_models:
                if hasattr(additive_model, "requested_neighbor_lists"):
                    requested_neighbor_lists += (
                        additive_model.requested_neighbor_lists()
                    )

            # The model fits and stores dense weights (see
            # CompositionModel.__init__), so incoming batches of (possibly sparse,
            # atom_type-keyed) atomic-basis targets need to be densified the same
            # way every other architecture densifies its own targets before
            # training.
            atomic_basis_transform, _ = get_prepare_atomic_basis_targets_transform(
                model.dataset_info.targets, model.dataset_info.extra_data
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
                get_remove_additive_transform(cpu_additive_models, model.target_infos),
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

                model.model.accumulate(systems, targets)

        if is_distributed:
            torch.distributed.barrier()
            # A rank whose shard of some dataset was empty never accumulated
            # the corresponding targets, so its XTX/XTY are still on the CPU,
            # while NCCL needs them on the GPU for the all_reduce.
            model.model._sync_device_dtype(device, torch.float64)
            handles = []
            for target_name in targets_to_accumulate:
                for XTX_block, XTY_block in zip(
                    model.model.XTX[target_name],
                    model.model.XTY[target_name],
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

        model.model.fit(fixed_weights, targets_to_fit=model._new_outputs)

        for target_name in model.model.weights.keys():
            model.register_buffer(
                target_name + "_composition_buffer",
                mts.save_buffer(
                    mts.make_contiguous(
                        model.model.weights[target_name].to("cpu", torch.float64)
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
        raise ValueError("Composition model does not allow restarting training")

    @classmethod
    def upgrade_checkpoint(cls, checkpoint: Dict) -> Dict:
        for v in range(1, cls.__checkpoint_version__):
            if checkpoint["trainer_ckpt_version"] == v:
                update = getattr(checkpoints, f"trainer_update_v{v}_v{v + 1}")
                update(checkpoint)
                checkpoint["trainer_ckpt_version"] = v + 1

        if checkpoint["trainer_ckpt_version"] != cls.__checkpoint_version__:
            raise RuntimeError(
                f"Unable to upgrade the checkpoint: the checkpoint is using trainer "
                f"version {checkpoint['trainer_ckpt_version']}, while the current "
                f"trainer version is {cls.__checkpoint_version__}."
            )
        return checkpoint
