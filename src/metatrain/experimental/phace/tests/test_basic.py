import pytest

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

    @pytest.fixture(params=[0, 1, 2])
    def o3_lambda(self, request: pytest.FixtureRequest) -> int:
        return request.param


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
