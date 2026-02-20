# mypy: disable-error-code="override"
import copy
from pathlib import Path
from typing import Any

import e3nn.util.jit
import pytest
import torch
from e3nn import o3

from metatrain.experimental.mace.model import MetaMACE
from metatrain.utils.abc import ModelInterface, TrainerInterface
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
    def mace_model_path(self) -> Path:
        """Path to a small MACE model for testing.

        :return: Path to the MACE model.
        """
        return Path(__file__).parent / "mace_small.model"

    @pytest.fixture(params=["from_hypers", "from_file"])
    def mace_init_mode(self, request: pytest.FixtureRequest) -> str:
        """Type of MACE model to use: loaded from file or built from hypers.

        :param request: Pytest fixture request.

        :return: Whether to load the MACE model from file.
        """
        return request.param

    @pytest.fixture
    def model_hypers(self, mace_init_mode: str, mace_model_path: Path) -> dict:
        """Smaller hyperparameters than the defaults for faster testing.

        :param mace_init_mode: How to initialize the MACE model.
        :return: Hyperparameters for the model.
        """
        defaults = copy.deepcopy(get_default_hypers(self.architecture)["model"])
        if mace_init_mode == "from_hypers":
            defaults["hidden_irreps"] = "20x0e + 20x1o + 20x2e"
        else:
            defaults["mace_model"] = mace_model_path
        return defaults

    @pytest.fixture
    def minimal_model_hypers(self, mace_init_mode: str) -> dict:
        """Minimal hyperparameters for the MACE model for fastest testing.

        :param mace_init_mode: How to initialize the MACE model.
        :return: Hyperparameters for the model.
        """
        hypers = copy.deepcopy(get_default_hypers(self.architecture)["model"])
        if mace_init_mode == "from_hypers":
            hypers["hidden_irreps"] = "1x0e + 1x1o"
            hypers["num_interactions"] = 1
            hypers["max_ell"] = 1
            hypers["correlation"] = 1
            hypers["radial_MLP"] = [1, 1, 1]
        else:
            hypers["mace_model"] = Path(__file__).parent / "mace_small.model"
        return hypers


class TestInput(InputTests, MACETests): ...


class TestOutput(OutputTests, MACETests):
    supports_features = False

    @pytest.fixture
    def n_features(self, model_hypers: dict) -> list[int]:
        """Features output was renamed to mtt::aux:mace_features so
        for now this is not used."""
        hidden_irreps = o3.Irreps(model_hypers["hidden_irreps"])
        num_interactions = model_hypers["num_interactions"]

        features_irreps = hidden_irreps * (num_interactions - 1) + o3.Irreps(
            f"{hidden_irreps.count((0, 1))}x0e"
        )

        return [ir.mul for ir in features_irreps]

    @pytest.fixture
    def n_last_layer_features(self, model_hypers: dict) -> int:
        hidden_irreps = o3.Irreps(model_hypers["hidden_irreps"])
        num_interactions = model_hypers["num_interactions"]
        MLP_irreps = o3.Irreps(model_hypers["MLP_irreps"])

        return hidden_irreps.count((0, 1)) * (num_interactions - 1) + MLP_irreps.count(
            (0, 1)
        )


class TestAutograd(AutogradTests, MACETests):
    cuda_nondet_tolerance = 1e-12


class TestTorchscript(TorchscriptTests, MACETests):
    float_hypers = ["r_max"]

    def jit_compile(self, model: MetaMACE) -> torch.jit.ScriptModule:
        return torch.jit.script(e3nn.util.jit.compile(model))


class TestExported(ExportedTests, MACETests): ...


class TestTraining(TrainingTests, MACETests): ...


class TestCheckpoints(CheckpointTests, MACETests):
    incompatible_trainer_checkpoints = []

    def test_checkpoint_did_not_change(
        self,
        monkeypatch: Any,
        tmp_path: str,
        model_trainer: tuple[ModelInterface, TrainerInterface],
        mace_init_mode: str,
    ) -> None:
        if mace_init_mode == "from_file":
            pytest.skip(
                "Skipping checkpoint equality test when loading MACE model from file."
            )

        super().test_checkpoint_did_not_change(monkeypatch, tmp_path, model_trainer)
