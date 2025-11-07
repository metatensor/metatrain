# mypy: disable-error-code=misc
# We ignore misc errors in this file because TypedDict
# with default values is not allowed by mypy.
import logging
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

import torch
from torch.utils.data import DataLoader
from typing_extensions import TypedDict

from metatrain.utils.abc import ModelInterface, TrainerInterface
from metatrain.utils.data import (
    CollateFn,
    CombinedDataLoader,
    Dataset,
    _is_disk_dataset,
)
from metatrain.utils.io import check_file_extension, model_from_checkpoint
from metatrain.utils.neighbor_lists import (
    get_requested_neighbor_lists,
    get_system_with_neighbor_lists,
)

from . import checkpoints
from .model import LLPRUncertaintyModel


class LLPRTrainerHypers(TypedDict):
    """Hyperparameters for the LLPR trainer."""

    batch_size: int = 12
    """This defines the batch size used in the computation of last-layer
    features, covariance matrix, etc."""

    regularizer: Optional[float] = None
    r"""This is the regularizer value :math:`\varsigma` that is used in
    applying Eq. 24 of Bigi et al :footcite:p:`bigi_mlst_2024`:

    .. math::

        \sigma^2_\star = \alpha^2 \boldsymbol{\mathrm{f}}^{\mathrm{T}}_\star
        (\boldsymbol{\mathrm{F}}^{\mathrm{T}} \boldsymbol{\mathrm{F}} + \varsigma^2
        \boldsymbol{\mathrm{I}})^{-1} \boldsymbol{\mathrm{f}}_\star

    If set to ``null``, the internal routine will determine the smallest regularizer
    value that guarantees numerical stability in matrix inversion. Having exposed the
    formula here, we also note to the user that the training routine of the LLPR
    wrapper model finds the ideal global calibration factor :math:`\alpha`."""

    model_checkpoint: Optional[str] = None
    """This should provide the checkpoint to the model for which the
    user wants to perform UQ based on the LLPR approach. Note that the model
    architecture must comply with the requirement that the last-layer features are
    exposed under the convention defined by metatrain."""


class Trainer(TrainerInterface[LLPRTrainerHypers]):
    __checkpoint_version__ = 1
    __hypers_cls__ = LLPRTrainerHypers

    def train(
        self,
        model: LLPRUncertaintyModel,
        dtype: torch.dtype,
        devices: List[torch.device],
        train_datasets: List[Union[Dataset, torch.utils.data.Subset]],
        val_datasets: List[Union[Dataset, torch.utils.data.Subset]],
        checkpoint_dir: str,
    ) -> None:
        # Load the wrapped model from checkpoint and set it as the wrapped model of the
        # LLPR model:
        if self.hypers["model_checkpoint"] is None:
            raise ValueError(
                "A model checkpoint must be provided to train the LLPR "
                "(model_checkpoint, under training, in the hypers)"
            )
        wrapped_model_checkpoint_path = self.hypers["model_checkpoint"]
        checkpoint = torch.load(
            wrapped_model_checkpoint_path, weights_only=False, map_location="cpu"
        )
        wrapped_model = model_from_checkpoint(checkpoint, "export")
        model.set_wrapped_model(wrapped_model)

        device = devices[0]  # this trainer doesn't support multi-GPU training
        # check device and dtype against wrapped model class
        if device.type not in wrapped_model.__class__.__supported_devices__:
            raise ValueError(
                f"Device {device} not supported by the wrapped model. "
                f"Supported devices are {wrapped_model.__class__.__supported_devices__}"
            )
        if dtype not in wrapped_model.__class__.__supported_dtypes__:
            raise ValueError(
                f"dtype {dtype} not supported by the wrapped model. "
                f"Supported dtypes are {wrapped_model.__class__.__supported_dtypes__}"
            )
        logging.info(f"Training on device {device} with dtype {dtype}")

        logging.info("Calculating neighbor lists for the datasets")
        # Calculate the neighbor lists in advance, if needed
        requested_neighbor_lists = get_requested_neighbor_lists(model)
        for dataset in train_datasets + val_datasets:
            # If the dataset is a disk dataset, the NLs are already attached, we will
            # just check the first system
            if _is_disk_dataset(dataset):
                system = dataset[0]["system"]
                for options in requested_neighbor_lists:
                    if options not in system.known_neighbor_lists():
                        raise ValueError(
                            "The requested neighbor lists are not attached to the "
                            f"system. Neighbor list {options} is missing from the "
                            "first system in the disk dataset. Make sure you save "
                            "the neighbor lists in the systems when saving the dataset."
                        )
            else:
                for sample in dataset:
                    system = sample["system"]
                    # The following line attaches the neighbors lists to the system,
                    # and doesn't require to reassign the system to the dataset:
                    get_system_with_neighbor_lists(system, requested_neighbor_lists)

        # Move the model to the device and dtype:
        model.to(device=device, dtype=dtype)

        logging.info("Setting up data loaders")

        # Create a collate function:
        targets_keys = list(model.dataset_info.targets.keys())
        collate_fn = CollateFn(target_keys=targets_keys)

        # Create dataloader for the training datasets:
        train_dataloaders = []
        for train_dataset in train_datasets:
            if len(train_dataset) < self.hypers["batch_size"]:
                raise ValueError(
                    f"A training dataset has fewer samples "
                    f"({len(train_dataset)}) than the batch size "
                    f"({self.hypers['batch_size']}). "
                    "Please reduce the batch size."
                )
            train_dataloaders.append(
                DataLoader(
                    dataset=train_dataset,
                    batch_size=self.hypers["batch_size"],
                    shuffle=False,
                    drop_last=False,
                    collate_fn=collate_fn,
                )
            )
        train_dataloader = CombinedDataLoader(train_dataloaders, shuffle=False)

        # Create dataloader for the validation datasets:
        val_dataloaders = []
        for val_dataset in val_datasets:
            if len(val_dataset) < self.hypers["batch_size"]:
                raise ValueError(
                    f"A validation dataset has fewer samples "
                    f"({len(val_dataset)}) than the batch size "
                    f"({self.hypers['batch_size']}). "
                    "Please reduce the batch size."
                )
            val_dataloaders.append(
                DataLoader(
                    dataset=val_dataset,
                    batch_size=self.hypers["batch_size"],
                    shuffle=False,
                    drop_last=False,
                    collate_fn=collate_fn,
                )
            )
        val_dataloader = CombinedDataLoader(val_dataloaders, shuffle=False)

        # Train the model:
        logging.info("Starting training")
        model.compute_covariance(train_dataloader)
        model.compute_inverse_covariance(self.hypers["regularizer"])
        model.calibrate(val_dataloader)
        model.generate_ensemble()

    def save_checkpoint(self, model: ModelInterface, path: Union[str, Path]) -> None:
        checkpoint = model.get_checkpoint()
        torch.save(
            checkpoint,
            check_file_extension(path, ".ckpt"),
        )

    @classmethod
    def load_checkpoint(
        cls,
        checkpoint: Dict[str, Any],
        hypers: LLPRTrainerHypers,
        context: Literal["restart", "finetune"],
    ) -> "LLPRUncertaintyModel":
        raise ValueError("LLPR does not allow restarting training")

    @classmethod
    def upgrade_checkpoint(cls, checkpoint: Dict) -> Dict:
        for v in range(1, cls.__checkpoint_version__):
            if checkpoint["trainer_ckpt_version"] == v:
                update = getattr(checkpoints, f"trainer_update_v{v}_v{v + 1}")
                update(checkpoint)
                checkpoint["trainer_ckpt_version"] = v + 1

        if checkpoint["trainer_ckpt_version"] != cls.__checkpoint_version__:
            raise RuntimeError(
                f"Unable to upgrade the checkpoint: the checkpoint is using "
                f"trainer version {checkpoint['trainer_ckpt_version']}, while the "
                f"current trainer version is {cls.__checkpoint_version__}."
            )
        return checkpoint
