import copy

import pytest

from metatrain.utils.architectures import get_default_hypers
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


class SoapBPNNTests(ArchitectureTests):
    architecture = "soap_bpnn"

    @pytest.fixture
    def minimal_model_hypers(self):
        hypers = get_default_hypers(self.architecture)["model"]
        hypers = copy.deepcopy(hypers)
        hypers["soap"]["max_angular"] = 1
        hypers["soap"]["max_radial"] = 1
        hypers["bpnn"]["num_neurons_per_layer"] = 1
        hypers["bpnn"]["num_hidden_layers"] = 1
        return hypers


class TestInput(InputTests, SoapBPNNTests): ...


class TestOutput(OutputTests, SoapBPNNTests):
    supports_vector_outputs = True

    @pytest.fixture
    def n_features(self):
        return 128

    @pytest.fixture
    def n_last_layer_features(self):
        return 128

    @pytest.fixture
    def single_atom_energy(self):
        return 0.0


class TestAutograd(AutogradTests, SoapBPNNTests):
    cuda_nondet_tolerance = 1e-12


class TestTorchscript(TorchscriptTests, SoapBPNNTests):
    float_hypers = ["soap.cutoff.radius", "soap.cutoff.width"]

    def test_torchscript_with_identity(self, model_hypers, dataset_info, dtype):
        hypers = copy.deepcopy(model_hypers)
        hypers["bpnn"]["layernorm"] = False
        self.test_torchscript(
            model_hypers=hypers, dataset_info=dataset_info, dtype=dtype
        )


class TestExported(ExportedTests, SoapBPNNTests): ...


class TestTraining(TrainingTests, SoapBPNNTests): ...


class TestCheckpoints(CheckpointTests, SoapBPNNTests):
    incompatible_trainer_checkpoints = [
        "checkpoints/model-v1_trainer-v1.ckpt.gz",
        "checkpoints/model-v2_trainer-v1.ckpt.gz",
        "checkpoints/model-v2_trainer-v2.ckpt.gz",
        "checkpoints/model-v3_trainer-v2.ckpt.gz",
        "checkpoints/model-v3_trainer-v3.ckpt.gz",
        "checkpoints/model-v3_trainer-v4.ckpt.gz",
        "checkpoints/model-v4_trainer-v3.ckpt.gz",
        "checkpoints/model-v4_trainer-v4.ckpt.gz",
        "checkpoints/model-v4_trainer-v5.ckpt.gz",
    ]
