import pytest
import torch

from metatrain.utils.testing import (
    ArchitectureTests,
    CheckpointTests,
    ExportedTests,
    InputTests,
    OutputTests,
    TorchscriptTests,
    TrainingTests,
)


class ScalerTests(ArchitectureTests):
    architecture = "scaler"

    # Scaler only supports float64 (see Scaler.__supported_dtypes__),
    # so the dtype fixture is fixed to float64 instead of being parametrized over
    # (float32, float64) like the default one.
    @pytest.fixture
    def dtype(self):
        return torch.float64


class TestInput(InputTests, ScalerTests): ...


class TestOutput(OutputTests, ScalerTests):
    supports_features = False
    supports_last_layer_features = False
    is_equivariant_rotations = False
    is_equivariant_reflections = False
    monomer_equal_dimer = True


class TestTorchscript(TorchscriptTests, ScalerTests): ...


class TestExported(ExportedTests, ScalerTests):
    @pytest.fixture
    def device(self):
        return torch.device("cpu")


class TestCheckpoints(CheckpointTests, ScalerTests):
    # The composition trainer does not support restarting training, so its
    # checkpoints cannot be loaded in the "restart" context.
    incompatible_trainer_checkpoints = [
        "checkpoints/model-v1_trainer-v1.ckpt.gz",
    ]


class TestTraining(TrainingTests, ScalerTests):
    def test_continue(
        self,
        monkeypatch,
        tmp_path,
        dataset_path,
        dataset_targets,
        default_hypers,
        model_hypers,
    ):
        pytest.skip("Scaler model does not support restarting training")

    def test_continue_restart_num_epochs(
        self,
        monkeypatch,
        tmp_path,
        dataset_path,
        dataset_targets,
        default_hypers,
        model_hypers,
    ):
        pytest.skip("Scaler model does not support restarting training")

    def test_continue_finetune_num_epochs(
        self,
        monkeypatch,
        tmp_path,
        dataset_path,
        dataset_targets,
        default_hypers,
        model_hypers,
    ):
        pytest.skip("Scaler model does not support restarting training")
