from pathlib import Path
from typing import Any, Dict, List, Literal, Union

import metatensor.torch as mts
import torch
from torch.utils.data import DataLoader, DistributedSampler

from metatrain.utils.abc import ModelInterface, TrainerInterface
from metatrain.utils.data import CollateFn, CombinedDataLoader, Dataset, unpack_batch
from metatrain.utils.neighbor_lists import get_system_with_neighbor_lists_transform
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
        from .model import CompositionModel, remove_additive

        assert isinstance(model, CompositionModel)

        additive_models = getattr(self, "_additive_models", [])
        is_distributed = getattr(self, "_is_distributed", False)
        fixed_weights = self.hypers.get("atomic_baseline", None)
        batch_size = self.hypers.get("batch_size", len(train_datasets[0]))

        if len(model.target_infos) == 0:
            return

        if fixed_weights is None:
            fixed_weights = {}

        requested_neighbor_lists = []
        for additive_model in additive_models:
            if hasattr(additive_model, "requested_neighbor_lists"):
                requested_neighbor_lists += additive_model.requested_neighbor_lists()

        callables = [get_system_with_neighbor_lists_transform(requested_neighbor_lists)]

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
            samplers = [
                DistributedSampler(
                    dataset,
                    num_replicas=world_size,
                    rank=rank,
                    shuffle=False,
                    drop_last=False,
                )
                for dataset in train_datasets
            ]
        else:
            samplers = [None] * len(train_datasets)

        dataloaders = []
        for dataset, sampler in zip(train_datasets, samplers, strict=True):
            if len(dataset) < batch_size:
                raise ValueError(
                    f"A training dataset has fewer samples "
                    f"({len(dataset)}) than the batch size "
                    f"({batch_size}). "
                    "Please reduce the batch size."
                )
            dataloaders.append(
                DataLoader(
                    dataset=dataset,
                    batch_size=batch_size,
                    sampler=sampler,
                    shuffle=None if sampler else False,
                    drop_last=False,
                    collate_fn=collate_fn,
                )
            )

        dataloader = CombinedDataLoader(dataloaders, shuffle=False)

        device = model.dummy_buffer.device

        for batch in dataloader:
            systems, targets, _ = unpack_batch(batch)
            systems, targets, _ = batch_to(systems, targets, device=device)
            targets = {
                target_name: targets[target_name]
                for target_name, target in targets.items()
                if target_name not in fixed_weights
                and target_name in model._new_outputs
            }
            if len(targets) == 0:
                break

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
            model.model.accumulate(systems, targets)

        if is_distributed:
            torch.distributed.barrier()
            for target_name in model._new_outputs:
                if target_name in fixed_weights:
                    continue
                for XTX_block, XTY_block in zip(
                    model.model.XTX[target_name],
                    model.model.XTY[target_name],
                    strict=True,
                ):
                    torch.distributed.all_reduce(XTX_block.values)
                    torch.distributed.all_reduce(XTY_block.values)

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

    def save_checkpoint(
        self, model: ModelInterface, checkpoint_dir: Union[str, Path]
    ) -> None:
        pass

    @classmethod
    def load_checkpoint(
        cls,
        checkpoint: Dict[str, Any],
        hypers: TrainerHypers,
        context: Literal["restart", "finetune"],
    ) -> "Trainer":
        raise ValueError("Composition model does not allow restarting training")

    @staticmethod
    def upgrade_checkpoint(checkpoint: Dict) -> Dict:
        if checkpoint.get("trainer_ckpt_version", 0) != Trainer.__checkpoint_version__:
            raise RuntimeError(
                f"Unable to upgrade the checkpoint: the checkpoint is using trainer "
                f"version {checkpoint['trainer_ckpt_version']}, while the current "
                f"trainer version is {Trainer.__checkpoint_version__}."
            )
        return checkpoint
