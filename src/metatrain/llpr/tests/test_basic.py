import copy
import tempfile

import pytest
import torch
from omegaconf import OmegaConf

from metatrain.pet import PET
from metatrain.pet import Trainer as PETTrainer
from metatrain.utils.architectures import get_default_hypers
from metatrain.utils.hypers import init_with_defaults
from metatrain.utils.loss import LossSpecification
from metatrain.utils.testing.base import ArchitectureTests
from metatrain.utils.testing.checkpoints import CheckpointTests


class LLPRTests(ArchitectureTests):
    architecture = "llpr"


class TestCheckpoints(CheckpointTests, LLPRTests):
    @pytest.fixture
    def model_trainer(
        self, dataset_targets, DATASET_PATH, model_hypers, default_hypers
    ):
        dataset, targets_info, dataset_info = self.get_dataset(
            dataset_targets, DATASET_PATH
        )

        hypers = copy.deepcopy(get_default_hypers("pet"))

        pet_model_hypers = hypers["model"]
        pet_model_hypers["d_pet"] = 1
        pet_model_hypers["d_head"] = 1
        pet_model_hypers["d_feedforward"] = 1
        pet_model_hypers["num_heads"] = 1
        pet_model_hypers["num_attention_layers"] = 1
        pet_model_hypers["num_gnn_layers"] = 1

        model = PET(pet_model_hypers, dataset_info)

        hypers = copy.deepcopy(hypers)
        hypers["training"]["num_epochs"] = 1
        loss_hypers = OmegaConf.create(
            {k: init_with_defaults(LossSpecification) for k in dataset_targets}
        )
        loss_hypers = OmegaConf.to_container(loss_hypers, resolve=True)
        hypers["training"]["loss"] = loss_hypers

        trainer = PETTrainer(hypers["training"])

        trainer.train(
            model,
            dtype=model.__supported_dtypes__[0],
            devices=[torch.device("cpu")],
            train_datasets=[dataset],
            val_datasets=[dataset],
            checkpoint_dir="",
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            trainer.save_checkpoint(model, f"{tmpdir}/pet_checkpoint.ckpt")

            # train LLPR model
            hypers = copy.deepcopy(model_hypers)
            hypers["num_ensemble_members"] = {"energy": 8}

            model = self.model_cls(hypers, dataset_info)

            hypers = copy.deepcopy(default_hypers)
            hypers["training"]["model_checkpoint"] = f"{tmpdir}/pet_checkpoint.ckpt"

            trainer = self.trainer_cls(hypers["training"])
            trainer.train(
                model,
                dtype=model.__supported_dtypes__[0],
                devices=[torch.device("cpu")],
                train_datasets=[dataset],
                val_datasets=[dataset],
                checkpoint_dir="",
            )

        return model, trainer
