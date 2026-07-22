import pytest
import torch

from metatrain.utils.data import DatasetInfo
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


class CompositionTests(ArchitectureTests):
    architecture = "composition"

    # Composition only supports float64 (see CompositionModel.__supported_dtypes__),
    # so the dtype fixture is fixed to float64 instead of being parametrized over
    # (float32, float64) like the default one.
    @pytest.fixture
    def dtype(self):
        return torch.float64

    @pytest.fixture(params=[0])
    def o3_lambda(self, request: pytest.FixtureRequest) -> int:
        """Composition model only supports scalars.

        :param request: The pytest request fixture.
        :return: The O(3) lambda value.
        """
        return request.param

    @pytest.fixture(params=[1])
    def o3_sigma(self, request: pytest.FixtureRequest) -> int:
        """Composition model only supports scalars.

        :param request: The pytest request fixture.
        :return: The O(3) sigma value.
        """
        return request.param


class TestInput(InputTests, CompositionTests): ...


class TestOutput(OutputTests, CompositionTests):
    supports_vector_outputs: bool = False
    supports_selected_atoms: bool = False
    supports_features: bool = False
    supports_last_layer_features: bool = False

    def test_output_multispherical(
        self,
        model_hypers: dict,
        dataset_info_multispherical: DatasetInfo,
        sample_kind: str,
    ) -> None:
        pytest.skip("Composition only supports invariant spherical blocks")


@pytest.mark.skip("Composition model output does not depend on positions or cell")
class TestAutograd(AutogradTests, CompositionTests): ...


class TestTorchscript(TorchscriptTests, CompositionTests): ...


class TestExported(ExportedTests, CompositionTests): ...


class TestCheckpoints(CheckpointTests, CompositionTests):
    # The composition trainer does not support restarting training, so its
    # checkpoints cannot be loaded in the "restart" context.
    incompatible_trainer_checkpoints = [
        "checkpoints/model-v1_trainer-v1.ckpt.gz",
    ]


class TestTraining(TrainingTests, CompositionTests):
    def test_continue(
        self,
        monkeypatch,
        tmp_path,
        dataset_path,
        dataset_targets,
        default_hypers,
        model_hypers,
    ):
        pytest.skip("Composition model does not support restarting training")

    def test_continue_restart_num_epochs(
        self,
        monkeypatch,
        tmp_path,
        dataset_path,
        dataset_targets,
        default_hypers,
        model_hypers,
    ):
        pytest.skip("Composition model does not support restarting training")

    def test_continue_finetune_num_epochs(
        self,
        monkeypatch,
        tmp_path,
        dataset_path,
        dataset_targets,
        default_hypers,
        model_hypers,
    ):
        pytest.skip("Composition model does not support restarting training")
