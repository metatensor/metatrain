import copy
import logging
from pathlib import Path
from typing import Union

import torch
import metatensor.torch as mts
from metatensor.learn.data import IndexedDataset
from metatensor.torch.learn import DataLoader
from ...utils.data.dataset import DatasetInfo
from ...utils.data.target_info import TargetInfo
from ...utils.io import check_file_extension
from ...utils.logging import MetricLogger
from ...utils.loss import TensorMapDictLoss
from ...utils.metrics import RMSEAccumulator, get_selected_metric

from ..nanopet.modules.augmentation import RotationalAugmenter

from .utils import group_and_join_nonetypes, get_system_transformations
from .model import NanoPetOnBasis

logger = logging.getLogger(__name__)


class Trainer:
    def __init__(self, train_hypers):
        self.hypers = train_hypers
        self.optimizer_state_dict = None
        self.scheduler_state_dict = None
        self.epoch = None
        self.best_metric = None
        self.best_model_state_dict = None
        self.best_optimizer_state_dict = None

    def train(
        self,
        model: NanoPetOnBasis,
        device: torch.device,
        dtype: torch.dtype,
        train_dataset: IndexedDataset,
        val_dataset: IndexedDataset,
        checkpoint_dir: str,
        loss_fn: torch.nn.Module = None,
    ):

        is_distributed = False  # FIXME
        rank = 0  # FIXME

        logger.info(f"Training on {device} with dtype {dtype}")

        # Store "phony" dataset info
        target_info = {}
        if model.in_keys_node is not None:
            target_info.update(
                {
                    "mtt::node": TargetInfo(
                        quantity="mtt::node",
                        layout=mts.TensorMap(
                            model.in_keys_node,
                            [
                                mts.TensorBlock(
                                    samples=mts.Labels.empty(["system", "atom"]),
                                    components=[
                                        mts.Labels(
                                            "o3_mu",
                                            torch.arange(
                                                -k["o3_lambda"], k["o3_lambda"] + 1
                                            ).reshape(-1, 1),
                                        )
                                    ],
                                    properties=out_props,
                                    values=torch.empty(
                                        0,
                                        2 * k["o3_lambda"] + 1,
                                        len(out_props),
                                    ),
                                )
                                for k, out_props in zip(
                                    model.in_keys_node, model.out_properties_node
                                )
                            ],
                        ),
                    ),
                }
            )
        if model.in_keys_edge is not None:
            target_info.update(
                {
                    "mtt::edge": TargetInfo(
                        quantity="mtt::edge",
                        layout=mts.TensorMap(
                            model.in_keys_edge,
                            [
                                mts.TensorBlock(
                                    samples=mts.Labels.empty(
                                        [
                                            "system",
                                            "first_atom",
                                            "second_atom",
                                            "cell_shift_a",
                                            "cell_shift_b",
                                            "cell_shift_c",
                                        ]
                                    ),
                                    components=[
                                        mts.Labels(
                                            "o3_mu",
                                            torch.arange(
                                                -k["o3_lambda"], k["o3_lambda"] + 1
                                            ).reshape(-1, 1),
                                        )
                                    ],
                                    properties=out_props,
                                    values=torch.empty(
                                        0,
                                        2 * k["o3_lambda"] + 1,
                                        len(out_props),
                                    ),
                                )
                                for k, out_props in zip(
                                    model.in_keys_edge, model.out_properties_edge
                                )
                            ],
                        ),
                    ),
                }
            )

        self.dataset_info = DatasetInfo(
            length_unit="angstrom", atomic_types=model.atomic_types, targets=target_info
        )

        # Create dataloaders
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.hypers["batch_size"],
            shuffle=self.hypers["shuffle"],
            collate_fn=group_and_join_nonetypes,
        )

        val_dataloader = DataLoader(
            val_dataset,
            batch_size=self.hypers["batch_size"],
            shuffle=False,
            collate_fn=group_and_join_nonetypes,
        )

        # Set up rotational augmenter
        rotational_augmenter = RotationalAugmenter(target_info)

        # Define the loss function
        if loss_fn is None:
            loss_fn = TensorMapDictLoss(**self.hypers["loss"])

        # Set up optimizer and scheduler
        optimizer = torch.optim.Adam(
            model.parameters(), lr=self.hypers["learning_rate"]
        )
        if self.optimizer_state_dict is not None:
            optimizer.load_state_dict(self.optimizer_state_dict)

        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            factor=self.hypers["scheduler_factor"],
            patience=self.hypers["scheduler_patience"],
        )
        if self.scheduler_state_dict is not None:
            lr_scheduler.load_state_dict(self.scheduler_state_dict)

        # Log the initial learning rate:
        old_lr = optimizer.param_groups[0]["lr"]
        logger.info(f"Initial learning rate: {old_lr}")

        # Training loop
        start_epoch = 0 if self.epoch is None else self.epoch + 1
        if self.best_metric is None:
            self.best_metric = float("inf")
        epoch = start_epoch

        for epoch in range(start_epoch, start_epoch + self.hypers["num_epochs"]):

            train_rmse_calculator = RMSEAccumulator(self.hypers["log_separate_blocks"])
            val_rmse_calculator = RMSEAccumulator(self.hypers["log_separate_blocks"])

            model.train()
            train_loss = 0
            for batch in train_dataloader:

                systems_train, targets_train_node, targets_train_edge = (
                    batch.systems,
                    batch.targets_node,
                    batch.targets_edge,
                )

                # # Define a random transformation for each training system
                # rotations, inversions = get_system_transformations(systems_train)

                # # Apply rotational augmentation - node
                # if model.in_keys_node is not None:
                #     systems_train, targets_train_node = (
                #         rotational_augmenter.apply_augmentations(
                #             systems_train,
                #             {"mtt::node": targets_train_node},
                #             rotations,
                #             inversions,
                #         )
                #     )
                #     targets_train_node = targets_train_node["mtt::node"]
                #     assert mts.equal_metadata(batch.targets_node, targets_train_node)
                #     assert not mts.allclose(batch.targets_node, targets_train_node)

                # # Apply rotational augmentation - edge
                # if model.in_keys_edge is not None:
                #     systems_train, targets_train_edge = (
                #         rotational_augmenter.apply_augmentations(
                #             systems_train,
                #             {"mtt::edge": targets_train_edge},
                #             rotations,
                #             inversions,
                #         )
                #     )
                #     targets_train_edge = targets_train_edge["mtt::edge"]
                #     assert mts.equal_metadata(batch.targets_edge, targets_train_edge)
                #     assert not mts.allclose(batch.targets_edge, targets_train_edge)

                targets = {
                    "mtt::node": targets_train_node,
                    "mtt::edge": targets_train_edge,
                }

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward pass
                node_predictions, edge_predictions = model(
                    systems_train, batch.sample_id
                )
                predictions = {
                    "mtt::node": node_predictions,
                    "mtt::edge": edge_predictions,
                }

                train_loss_batch = loss_fn(predictions, targets)
                train_loss_batch.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                train_loss += train_loss_batch.item()

                train_rmse_calculator.update(predictions, targets)

            finalized_train_info = train_rmse_calculator.finalize(
                ["mtt::node", "mtt::edge"], device=device
            )

            val_loss = 0.0
            for batch in val_dataloader:

                targets = {
                    "mtt::node": batch.targets_node,
                    "mtt::edge": batch.targets_edge,
                }

                model.eval()
                node_predictions, edge_predictions = model(
                    batch.systems, batch.sample_id
                )
                predictions = {
                    "mtt::node": node_predictions,
                    "mtt::edge": edge_predictions,
                }

                val_loss_batch = loss_fn(predictions, targets)
                val_loss += val_loss_batch.item()
                val_rmse_calculator.update(predictions, targets)

            finalized_val_info = val_rmse_calculator.finalize(
                ["mtt::node", "mtt::edge"], device=device
            )

            finalized_train_info = {"loss": train_loss, **finalized_train_info}
            finalized_val_info = {
                "loss": val_loss,
                **finalized_val_info,
            }

            # Log metrics
            if epoch == start_epoch:
                metric_logger = MetricLogger(
                    log_obj=logger,
                    dataset_info=self.dataset_info,
                    initial_metrics=[finalized_train_info, finalized_val_info],
                    names=["training", "validation"],
                )
            if epoch % self.hypers["log_interval"] == 0:
                metric_logger.log(
                    metrics=[finalized_train_info, finalized_val_info],
                    epoch=epoch,
                    rank=0,
                )

            lr_scheduler.step(val_loss)
            new_lr = lr_scheduler.get_last_lr()[0]
            if new_lr != old_lr:
                if new_lr < 1e-7:
                    logger.info("Learning rate is too small, stopping training")
                    break
                else:
                    logger.info(f"Changing learning rate from {old_lr} to {new_lr}")
                    old_lr = new_lr
                    # load best model and optimizer state dict, re-initialize scheduler
                    (model.module if is_distributed else model).load_state_dict(
                        self.best_model_state_dict
                    )
                    optimizer.load_state_dict(self.best_optimizer_state_dict)
                    for param_group in optimizer.param_groups:
                        param_group["lr"] = new_lr
                    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                        optimizer,
                        factor=self.hypers["scheduler_factor"],
                        patience=self.hypers["scheduler_patience"],
                    )

            val_metric = get_selected_metric(
                finalized_val_info, self.hypers["best_model_metric"]
            )
            if val_metric < self.best_metric:
                self.best_metric = val_metric
                self.best_model_state_dict = copy.deepcopy(
                    (model.module if is_distributed else model).state_dict()
                )
                self.best_optimizer_state_dict = copy.deepcopy(optimizer.state_dict())

            if epoch % self.hypers["checkpoint_interval"] == 0:
                if is_distributed:
                    torch.distributed.barrier()
                self.optimizer_state_dict = optimizer.state_dict()
                self.scheduler_state_dict = lr_scheduler.state_dict()
                self.epoch = epoch
                if rank == 0:
                    self.save_checkpoint(
                        (model.module if is_distributed else model),
                        Path(checkpoint_dir) / f"model_{epoch}.ckpt",
                    )

        # prepare for the checkpoint that will be saved outside the function
        self.epoch = epoch
        self.optimizer_state_dict = optimizer.state_dict()
        self.scheduler_state_dict = lr_scheduler.state_dict()

    # def save_checkpoint(self, model, path: Union[str, Path]):
    #     checkpoint = {
    #         "model": model.state_dict(),
    #         "optimizer_state_dict": self.optimizer_state_dict,
    #         "scheduler_state_dict": self.scheduler_state_dict,
    #         "epoch": self.epoch,
    #         "best_metric": self.best_metric,
    #         "best_model_state_dict": self.best_model_state_dict,
    #         "best_optimizer_state_dict": self.best_optimizer_state_dict,
    #     }
    #     torch.save(checkpoint, path)
    def save_checkpoint(self, model, path: Union[str, Path]):
        checkpoint = {
            "architecture_name": "experimental.nanopet_on_basis",
            # "model_data": {
            # "model_hypers": model.hypers,
            # "dataset_info": model.dataset_info,
            # },
            "model_state_dict": model.state_dict(),
            "train_hypers": self.hypers,
            "epoch": self.epoch,
            "optimizer_state_dict": self.optimizer_state_dict,
            "scheduler_state_dict": self.scheduler_state_dict,
            "best_metric": self.best_metric,
            "best_model_state_dict": self.best_model_state_dict,
            "best_optimizer_state_dict": self.best_optimizer_state_dict,
        }
        torch.save(
            checkpoint,
            check_file_extension(path, ".ckpt"),
        )

    @classmethod
    def load_checkpoint(cls, path: Union[str, Path], train_hypers) -> "Trainer":
        # Load the checkpoint
        checkpoint = torch.load(path, weights_only=False)
        epoch = checkpoint["epoch"]
        optimizer_state_dict = checkpoint["optimizer_state_dict"]
        scheduler_state_dict = checkpoint["scheduler_state_dict"]
        best_metric = checkpoint["best_metric"]
        best_model_state_dict = checkpoint["best_model_state_dict"]
        best_optimizer_state_dict = checkpoint["best_optimizer_state_dict"]

        # Create the trainer
        trainer = cls(train_hypers)
        trainer.optimizer_state_dict = optimizer_state_dict
        trainer.scheduler_state_dict = scheduler_state_dict
        trainer.epoch = epoch
        trainer.best_metric = best_metric
        trainer.best_model_state_dict = best_model_state_dict
        trainer.best_optimizer_state_dict = best_optimizer_state_dict

        return trainer
