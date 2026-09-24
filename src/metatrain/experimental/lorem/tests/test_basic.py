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


def _minimal_hypers(arch: str) -> dict:
    hypers = copy.deepcopy(get_default_hypers(arch)["model"])
    hypers["cutoff"] = 3.0
    hypers["cutoff_width"] = 0.5
    hypers["max_degree"] = 1
    hypers["max_degree_lr"] = 0
    hypers["num_features"] = 4
    hypers["num_spherical_features"] = 1
    hypers["num_radial"] = 2
    hypers["num_message_passing"] = 0
    return hypers


class LoremTests(ArchitectureTests):
    architecture = "experimental.lorem"

    @pytest.fixture
    def model_hypers(self) -> dict:
        return _minimal_hypers(self.architecture)

    @pytest.fixture
    def minimal_model_hypers(self) -> dict:
        return _minimal_hypers(self.architecture)


class TestInput(InputTests, LoremTests): ...


class TestOutput(OutputTests, LoremTests):
    supports_multiscalar_outputs = False
    supports_spherical_outputs = False
    supports_spherical_rank2_outputs = False
    supports_spherical_atomic_basis_outputs = False
    supports_vector_outputs = True
    supports_features = False
    supports_last_layer_features = False

    def test_prediction_energy_subset_atoms(self, *args, **kwargs) -> None:
        """LOREM's long-range interaction range is infinite (see
        ``LOREM.export``), so this test's assumption -- that a selected
        subset of atoms predicts the same energy whether or not a second,
        spatially-distant monomer is present in the same system -- does not
        hold exactly: the two monomers still interact through their charges'
        Coulomb tail. ``selected_atoms`` itself is fully supported; only
        this locality assumption does not apply to a genuinely long-range
        architecture."""
        pytest.skip(
            "LOREM has an infinite interaction range (long-range Coulomb is "
            "always on), so subset-atom energies are not exactly local."
        )


class TestAutograd(AutogradTests, LoremTests):
    cuda_nondet_tolerance = 1e-12


class TestTorchscript(TorchscriptTests, LoremTests):
    float_hypers = ["cutoff", "cutoff_width"]
    supports_spherical_outputs = False


class TestExported(ExportedTests, LoremTests): ...


class TestTraining(TrainingTests, LoremTests):
    supports_atomic_basis = False


class TestCheckpoints(CheckpointTests, LoremTests):
    incompatible_trainer_checkpoints = []
