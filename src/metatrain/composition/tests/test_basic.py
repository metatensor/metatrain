import pytest
import torch

from metatrain.utils.data import DatasetInfo
from metatrain.utils.data.target_info import (
    get_energy_target_info,
    get_generic_target_info,
)
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

    @pytest.fixture
    def dataset_info(self) -> DatasetInfo:
        return DatasetInfo(
            length_unit="Angstrom",
            atomic_types=[1, 6, 7, 8],
            targets={
                "energy": get_energy_target_info(
                    "energy", {"quantity": "energy", "unit": "eV"}
                )
            },
        )

    @pytest.fixture
    def dataset_info_spherical(self, sample_kind: str) -> DatasetInfo:
        return DatasetInfo(
            length_unit="Angstrom",
            atomic_types=[1, 6, 7, 8],
            targets={
                "spherical_target": get_generic_target_info(
                    "spherical_target",
                    {
                        "quantity": "",
                        "unit": "",
                        "type": {
                            "spherical": {"irreps": [{"o3_lambda": 0, "o3_sigma": 1}]}
                        },
                        "num_subtargets": 5,
                        "sample_kind": sample_kind,
                    },
                )
            },
        )


class TestInput(InputTests, CompositionTests): ...


class TestOutput(OutputTests, CompositionTests):
    supports_scalar_outputs: bool = True
    supports_multiscalar_outputs: bool = False
    supports_vector_outputs: bool = False
    supports_spherical_outputs: bool = True
    supports_spherical_rank2_outputs: bool = False
    supports_spherical_atomic_basis_outputs: bool = True
    supports_selected_atoms: bool = False
    supports_features: bool = False
    supports_last_layer_features: bool = False
    is_equivariant_rotations: bool = False
    is_equivariant_reflections: bool = False

    def test_single_atom(
        self,
        model_hypers: dict,
        dataset_info: DatasetInfo,
        single_atom_energy: float | None,
    ) -> None:
        pytest.skip("Composition needs training data to produce outputs")

    def test_prediction_energy_subset_elements(
        self, model_hypers: dict, dataset_info: DatasetInfo
    ) -> None:
        pytest.skip("Composition needs training data to produce outputs")

    def test_output_multispherical(
        self,
        model_hypers: dict,
        dataset_info_multispherical: DatasetInfo,
        sample_kind: str,
    ) -> None:
        pytest.skip("Composition only supports invariant spherical blocks")


class TestAutograd(AutogradTests, CompositionTests):
    @pytest.fixture
    def device(self):
        return torch.device("cpu")

    def test_autograd_positions(
        self, device: torch.device, model_hypers: dict, dataset_info: DatasetInfo
    ) -> None:
        pytest.skip("Composition model output does not depend on positions")

    def test_autograd_cell(
        self, device: torch.device, model_hypers: dict, dataset_info: DatasetInfo
    ) -> None:
        pytest.skip("Composition model output does not depend on cell")


class TestTorchscript(TorchscriptTests, CompositionTests):
    # Composition only supports float64 (see CompositionModel.__supported_dtypes__),
    # so the dtype fixture is fixed to float64 instead of being parametrized over
    # (float32, float64) like the default one.
    @pytest.fixture
    def dtype(self):
        return torch.float64

    def test_torchscript_spherical(self, model_hypers, dataset_info_spherical):
        # Overridden because the base implementation hardcodes dtype=torch.float32.
        self.test_torchscript(
            model_hypers=model_hypers,
            dataset_info=dataset_info_spherical,
            dtype=torch.float64,
        )

    def test_torchscript_integers(self, model_hypers, dataset_info):
        # Overridden because the base implementation hardcodes dtype=torch.float32.
        self.test_torchscript(
            model_hypers=model_hypers, dataset_info=dataset_info, dtype=torch.float64
        )


class TestExported(ExportedTests, CompositionTests):
    @pytest.fixture
    def device(self):
        return torch.device("cpu")

    # Composition only supports float64 (see CompositionModel.__supported_dtypes__)
    @pytest.fixture
    def dtype(self):
        return torch.float64


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
