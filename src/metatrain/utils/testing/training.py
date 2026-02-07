import copy
from pathlib import Path
from typing import Any

import metatensor.torch as mts
import torch
from omegaconf import OmegaConf

from metatrain.utils.data.readers import read_systems
from metatrain.utils.hypers import init_with_defaults
from metatrain.utils.io import model_from_checkpoint
from metatrain.utils.loss import LossSpecification
from metatrain.utils.neighbor_lists import (
    get_system_with_neighbor_lists,
)

from .architectures import ArchitectureTests


class TrainingTests(ArchitectureTests):
    """Puts architectures to test in real training scenarios."""

    check_gradients: bool = True

    def test_continue(
        self,
        monkeypatch: Any,
        tmp_path: Path,
        dataset_path: str,
        dataset_targets: dict[str, dict],
        default_hypers: dict[str, Any],
        model_hypers: dict[str, Any],
    ) -> None:
        """Tests that a model can be checkpointed and loaded
        for a continuation of the training process

        :param monkeypatch: Pytest fixture to modify the current working
            directory.
        :param tmp_path: Temporary path to use for saving checkpoints.
        :param dataset_path: Path to the dataset to use for training.
        :param dataset_targets: Target hypers for the targets in the dataset.
        :param default_hypers: Default hyperparameters for the architecture.
        :param model_hypers: Hyperparameters to initialize the model.
        """
        monkeypatch.chdir(tmp_path)

        dataset, targets_info, dataset_info = self.get_dataset(
            dataset_targets, dataset_path
        )

        model = self.model_cls(model_hypers, dataset_info)

        hypers = copy.deepcopy(default_hypers)
        hypers["training"]["num_epochs"] = 0
        loss_conf = OmegaConf.create(
            {k: init_with_defaults(LossSpecification) for k in dataset_targets}
        )
        OmegaConf.resolve(loss_conf)
        hypers["training"]["loss"] = loss_conf

        trainer = self.trainer_cls(hypers["training"])
        trainer.train(
            model=model,
            dtype=torch.float32,
            devices=[torch.device("cpu")],
            train_datasets=[dataset],
            val_datasets=[dataset],
            checkpoint_dir=".",
        )

        trainer.save_checkpoint(model, "tmp.ckpt")

        checkpoint = torch.load("tmp.ckpt", weights_only=False, map_location="cpu")
        model_after = model_from_checkpoint(checkpoint, context="restart")
        assert isinstance(model_after, self.model_cls)
        model_after.restart(model.dataset_info)

        hypers["training"]["num_epochs"] = 0
        trainer = self.trainer_cls(hypers["training"])
        trainer.train(
            model=model_after,
            dtype=torch.float32,
            devices=[torch.device("cpu")],
            train_datasets=[dataset],
            val_datasets=[dataset],
            checkpoint_dir=".",
        )

        # evaluation
        systems = read_systems(dataset_path)
        systems = [system.to(torch.float32) for system in systems[:5]]
        for system in systems:
            system.positions.requires_grad_(True)
            get_system_with_neighbor_lists(system, model.requested_neighbor_lists())

        model.eval()
        model_after.eval()

        output_before = model(
            systems[:5], {k: model.outputs[k] for k in dataset_targets}
        )
        output_after = model_after(
            systems[:5], {k: model_after.outputs[k] for k in dataset_targets}
        )

        # For each target, check that outputs are the same after loading
        # from checkpoint, including gradients
        for i, target_key in enumerate(dataset_targets):
            assert mts.allclose(output_before[target_key], output_after[target_key]), (
                f"Output mismatch for {target_key}"
            )

            # We can't run a backward pass twice.
            if i > 0 or not self.check_gradients:
                continue

            target_before = output_before[target_key].block().values
            target_before.backward(torch.ones_like(target_before))

            gradients_before = [s.positions.grad for s in systems]

            for system in systems:
                system.positions.grad = None

            target_after = output_after[target_key].block().values
            target_after.backward(torch.ones_like(target_after))

            gradients_after = [s.positions.grad for s in systems]

            assert torch.allclose(
                torch.vstack(gradients_before), torch.vstack(gradients_after)
            )

    def test_continue_restart_num_epochs(
        self,
        monkeypatch: Any,
        tmp_path: Path,
        dataset_path: str,
        dataset_targets: dict[str, dict],
        default_hypers: dict[str, Any],
        model_hypers: dict[str, Any],
    ) -> None:
        """
        Tests that a training restart runs the correct
        number of epochs

        :param monkeypatch: Pytest fixture to modify the current working
            directory.
        :param tmp_path: Temporary path to use for saving checkpoints.
        :param dataset_path: Path to the dataset to use for training.
        :param dataset_targets: Target hypers for the targets in the dataset.
        :param default_hypers: Default hyperparameters for the architecture.
        :param model_hypers: Hyperparameters to initialize the model.
        """

        monkeypatch.chdir(tmp_path)

        dataset, targets_info, dataset_info = self.get_dataset(
            dataset_targets, dataset_path
        )

        model = self.model_cls(model_hypers, dataset_info)

        hypers = copy.deepcopy(default_hypers)
        hypers["training"]["num_epochs"] = 2
        loss_conf = OmegaConf.create(
            {k: init_with_defaults(LossSpecification) for k in dataset_targets}
        )
        OmegaConf.resolve(loss_conf)
        hypers["training"]["loss"] = loss_conf

        trainer = self.trainer_cls(hypers["training"])
        trainer.train(
            model=model,
            dtype=torch.float32,
            devices=[torch.device("cpu")],
            train_datasets=[dataset],
            val_datasets=[dataset],
            checkpoint_dir=".",
        )

        assert trainer.epoch == 1  # type: ignore
        trainer.save_checkpoint(model, "tmp.ckpt")

        checkpoint = torch.load("tmp.ckpt", weights_only=False, map_location="cpu")
        model_after = model_from_checkpoint(checkpoint, context="restart")
        assert isinstance(model_after, self.model_cls)
        model_after.restart(model.dataset_info)

        hypers["training"]["num_epochs"] = 4  # modify max num epochs to 4
        trainer = self.trainer_cls(hypers["training"])
        trainer.train(
            model=model_after,
            dtype=torch.float32,
            devices=[torch.device("cpu")],
            train_datasets=[dataset],
            val_datasets=[dataset],
            checkpoint_dir=".",
        )
        assert trainer.epoch == 3  # type: ignore

    def test_continue_finetune_num_epochs(
        self,
        monkeypatch: Any,
        tmp_path: Path,
        dataset_path: str,
        dataset_targets: dict[str, dict],
        default_hypers: dict[str, Any],
        model_hypers: dict[str, Any],
    ) -> None:
        """
        Tests that a training finetune runs the correct
        number of epochs

        :param monkeypatch: Pytest fixture to modify the current working
            directory.
        :param tmp_path: Temporary path to use for saving checkpoints.
        :param dataset_path: Path to the dataset to use for training.
        :param dataset_targets: Target hypers for the targets in the dataset.
        :param default_hypers: Default hyperparameters for the architecture.
        :param model_hypers: Hyperparameters to initialize the model.
        """

        monkeypatch.chdir(tmp_path)

        dataset, targets_info, dataset_info = self.get_dataset(
            dataset_targets, dataset_path
        )

        model = self.model_cls(model_hypers, dataset_info)

        hypers = copy.deepcopy(default_hypers)
        hypers["training"]["num_epochs"] = 2
        loss_conf = OmegaConf.create(
            {k: init_with_defaults(LossSpecification) for k in dataset_targets}
        )
        OmegaConf.resolve(loss_conf)
        hypers["training"]["loss"] = loss_conf

        trainer = self.trainer_cls(hypers["training"])
        trainer.train(
            model=model,
            dtype=torch.float32,
            devices=[torch.device("cpu")],
            train_datasets=[dataset],
            val_datasets=[dataset],
            checkpoint_dir=".",
        )

        assert trainer.epoch == 1  # type: ignore
        trainer.save_checkpoint(model, "tmp.ckpt")

        checkpoint = torch.load("tmp.ckpt", weights_only=False, map_location="cpu")
        model_after = model_from_checkpoint(checkpoint, context="finetune")
        assert isinstance(model_after, self.model_cls)
        model_after.restart(model.dataset_info)

        hypers["training"]["num_epochs"] = 1  # modify max num epochs to 1
        trainer = self.trainer_cls.load_checkpoint(
            checkpoint, hypers["training"], context="finetune"
        )
        trainer.train(
            model=model_after,
            dtype=torch.float32,
            devices=[torch.device("cpu")],
            train_datasets=[dataset],
            val_datasets=[dataset],
            checkpoint_dir=".",
        )

        assert trainer.epoch == 0  # type: ignore
