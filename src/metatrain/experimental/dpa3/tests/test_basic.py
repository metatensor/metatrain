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


class DPA3Tests(ArchitectureTests):
    architecture = "experimental.dpa3"

    @pytest.fixture
    def minimal_model_hypers(self) -> dict:
        """Minimal hyperparameters for a DPA3 model for the smallest
        checkpoint possible.

        :return: Hyperparameters for the model.
        """
        hypers = copy.deepcopy(get_default_hypers(self.architecture)["model"])
        hypers["descriptor"]["repflow"]["n_dim"] = 2
        hypers["descriptor"]["repflow"]["e_dim"] = 2
        hypers["descriptor"]["repflow"]["a_dim"] = 2
        hypers["descriptor"]["repflow"]["e_sel"] = 1
        hypers["descriptor"]["repflow"]["a_sel"] = 1
        hypers["descriptor"]["repflow"]["axis_neuron"] = 1
        hypers["descriptor"]["repflow"]["nlayers"] = 1
        hypers["fitting_net"]["neuron"] = [1, 1]
        return hypers


class TestInput(InputTests, DPA3Tests): ...


class TestOutput(OutputTests, DPA3Tests):
    supports_multiscalar_outputs = False
    supports_spherical_outputs = False
    supports_vector_outputs = False
    supports_features = False
    supports_last_layer_features = False


class TestAutograd(AutogradTests, DPA3Tests):
    cuda_nondet_tolerance = 1e-12


class TestTorchscript(TorchscriptTests, DPA3Tests):
    float_hypers = ["descriptor.repflow.e_rcut", "descriptor.repflow.e_rcut_smth"]
    supports_spherical_outputs = False


class TestExported(ExportedTests, DPA3Tests): ...


class TestTraining(TrainingTests, DPA3Tests): ...


class TestCheckpoints(CheckpointTests, DPA3Tests):
    incompatible_trainer_checkpoints = []
