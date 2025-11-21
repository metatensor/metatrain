import copy

import pytest

from metatrain.utils.architectures import get_default_hypers
from metatrain.utils.testing.autograd import AutogradTests
from metatrain.utils.testing.base import ArchitectureTests
from metatrain.utils.testing.checkpoints import CheckpointTests
from metatrain.utils.testing.exported import ExportedTests
from metatrain.utils.testing.input import InputTests
from metatrain.utils.testing.output import OutputTests
from metatrain.utils.testing.torchscript import TorchscriptTests
from metatrain.utils.testing.training import TrainingTests


class PETTests(ArchitectureTests):
    architecture = "pet"

    @pytest.fixture
    def minimal_model_hypers(self):
        hypers = get_default_hypers(self.architecture)["model"]
        hypers = copy.deepcopy(hypers)
        hypers["d_pet"] = 1
        hypers["d_head"] = 1
        hypers["d_feedforward"] = 1
        hypers["num_heads"] = 1
        hypers["num_attention_layers"] = 1
        hypers["num_gnn_layers"] = 1
        return hypers


class TestInput(InputTests, PETTests): ...


class TestOutput(OutputTests, PETTests):
    is_equivariant_model = False

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
