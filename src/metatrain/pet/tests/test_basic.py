import copy
import glob
import gzip

import pytest
import torch

from metatrain.utils.additive import ZBL
from metatrain.utils.architectures import get_default_hypers
from metatrain.utils.io import model_from_checkpoint
from metatrain.utils.testing import (
    ArchitectureTests,
    AutogradTests,
    CheckpointTests,
    ExportedTests,
    InputTests,
    OutputTests,
    TorchscriptTests,
    TrainingTests,
)


class PETTests(ArchitectureTests):
    architecture = "pet"

    @pytest.fixture
    def minimal_model_hypers(self):
        hypers = get_default_hypers(self.architecture)["model"]
        hypers = copy.deepcopy(hypers)
        hypers["d_pet"] = 1
        hypers["d_head"] = 1
        hypers["d_node"] = 1
        hypers["d_feedforward"] = 1
        hypers["num_heads"] = 1
        hypers["num_attention_layers"] = 1
        hypers["num_gnn_layers"] = 1
        return hypers


class TestInput(InputTests, PETTests):
    def test_zbl_is_an_additive_model(
        self, minimal_model_hypers, dataset_info, default_hypers
    ):
        """Asking for ZBL attaches a ZBL additive model over the dataset types."""
        hypers = copy.deepcopy(minimal_model_hypers)
        hypers["zbl"] = True
        trainer = self.trainer_cls(default_hypers["training"])
        model = trainer.setup(hypers, dataset_info)

        zbl_models = [m for m in model.additive_models if isinstance(m, ZBL)]
        assert len(zbl_models) == 1
        assert zbl_models[0].dataset_info.atomic_types == dataset_info.atomic_types


class TestOutput(OutputTests, PETTests):
    is_equivariant_rotations = False
    is_equivariant_reflections = False

    @pytest.fixture
    def n_features(self, model_hypers):
        num_readout_layers = (
            1
            if model_hypers["featurizer_type"] == "feedforward"
            else model_hypers["num_gnn_layers"]
        )

        return (model_hypers["d_node"] + model_hypers["d_pet"]) * num_readout_layers

    @pytest.fixture
    def n_last_layer_features(self, model_hypers):
        num_readout_layers = (
            1
            if model_hypers["featurizer_type"] == "feedforward"
            else model_hypers["num_gnn_layers"]
        )

        return model_hypers["d_head"] * num_readout_layers * 2


class TestAutograd(AutogradTests, PETTests): ...


class TestTorchscript(TorchscriptTests, PETTests):
    float_hypers = ["cutoff", "cutoff_width"]


class TestExported(ExportedTests, PETTests): ...


class TestTraining(TrainingTests, PETTests): ...


class TestCheckpoints(CheckpointTests, PETTests):
    incompatible_trainer_checkpoints = [
        "checkpoints/model-v1_trainer-v1.ckpt.gz",
        "checkpoints/model-v2_trainer-v1.ckpt.gz",
        "checkpoints/model-v3_trainer-v1.ckpt.gz",
        "checkpoints/model-v3_trainer-v2.ckpt.gz",
        "checkpoints/model-v4_trainer-v2.ckpt.gz",
        "checkpoints/model-v4_trainer-v3.ckpt.gz",
        "checkpoints/model-v4_trainer-v4.ckpt.gz",
        "checkpoints/model-v5_trainer-v3.ckpt.gz",
        "checkpoints/model-v6_trainer-v3.ckpt.gz",
        "checkpoints/model-v6_trainer-v4.ckpt.gz",
    ]

    def test_old_checkpoints_keep_scaler_and_composition_model(self):
        """Old checkpoints, including those from before the ``MetatrainModel``
        wrapper, keep their scaler and composition model when loaded by the current
        model, whose hypers they lack."""
        for path in glob.glob("checkpoints/*.ckpt.gz"):
            with gzip.open(path, "rb") as fd:
                checkpoint = torch.load(fd, weights_only=False)
            model = model_from_checkpoint(checkpoint, "export")
            assert model.scaler is not None, path
            assert len(model.additive_models) == 1, path
