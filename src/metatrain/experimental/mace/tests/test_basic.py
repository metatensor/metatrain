import copy
import pytest

from e3nn import o3
import e3nn.util.jit
import torch

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

class MACETests(ArchitectureTests):
    architecture = "experimental.mace"

    @pytest.fixture
    def model_hypers(self):
        """Smaller hyperparameters than the defaults for faster testing."""
        defaults = copy.deepcopy(get_default_hypers(self.architecture)["model"])
        defaults["hidden_irreps"] = "20x0e + 20x1o + 20x2e"
        return defaults
    
    @pytest.fixture
    def minimal_model_hypers(self):
        hypers = copy.deepcopy(get_default_hypers(self.architecture)["model"])
        hypers["hidden_irreps"] = "1x0e + 1x1o"
        hypers["num_interactions"] = 1
        return hypers

class TestInput(InputTests, MACETests): ...


class TestOutput(OutputTests, MACETests):

    @pytest.fixture
    def n_features(self, model_hypers) -> list[int]:
        hidden_irreps = o3.Irreps(model_hypers["hidden_irreps"])
        num_interactions = model_hypers["num_interactions"]

        features_irreps = hidden_irreps * (num_interactions - 1) + o3.Irreps(f"{hidden_irreps.count((0,1))}x0e")

        return [ir.mul for ir in features_irreps]
    
    @pytest.fixture
    def n_last_layer_features(self, model_hypers) -> int:
        hidden_irreps = o3.Irreps(model_hypers["hidden_irreps"])
        num_interactions = model_hypers["num_interactions"]
        MLP_irreps = o3.Irreps(model_hypers["MLP_irreps"])

        return hidden_irreps.count((0,1)) * (num_interactions - 1) + MLP_irreps.count((0,1))


class TestAutograd(AutogradTests, MACETests): ...


class TestTorchscript(TorchscriptTests, MACETests):
    float_hypers = ["cutoff"]

    def jit_compile(self, model):
        return torch.jit.script(e3nn.util.jit.compile(model))


class TestExported(ExportedTests, MACETests): ...


class TestTraining(TrainingTests, MACETests): ...


class TestCheckpoints(CheckpointTests, MACETests):
    incompatible_trainer_checkpoints = []
