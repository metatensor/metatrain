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


class PhACETests(ArchitectureTests):
    architecture = "experimental.phace"

    @pytest.fixture
    def minimal_model_hypers(self):
        hypers = get_default_hypers(self.architecture)["model"]
        hypers = copy.deepcopy(hypers)
        hypers["num_element_channels"] = 8
        hypers["num_message_passing_layers"] = 1
        hypers["max_correlation_order_per_layer"] = 2
        return hypers


class TestInput(InputTests, PhACETests): ...


class TestOutput(OutputTests, PhACETests):
    is_equivariant_reflections = False

    @pytest.fixture
    def n_last_layer_features(self) -> int:
        return 256


class TestAutograd(AutogradTests, PhACETests): ...


class TestTorchscript(TorchscriptTests, PhACETests):
    float_hypers = [
        "cutoff",
        "cutoff_width",
        "nu_scaling",
        "mp_scaling",
        "overall_scaling",
        "radial_basis.max_eigenvalue",
        "radial_basis.scale",
    ]


class TestExported(ExportedTests, PhACETests): ...


class TestTraining(TrainingTests, PhACETests): ...


class TestCheckpoints(CheckpointTests, PhACETests):
    incompatible_trainer_checkpoints = []

    @pytest.fixture
    def default_hypers(self):
        hypers = get_default_hypers(self.architecture)
        hypers = copy.deepcopy(hypers)
        # Disable torch.compile for CPU testing
        hypers["training"]["compile"] = False
        return hypers
