import collections
import copy
import json
from pathlib import Path

import pytest
import torch

from metatrain.experimental.dpa3 import DPA3
from metatrain.experimental.dpa3.documentation import ModelHypers
from metatrain.utils.architectures import get_default_hypers
from metatrain.utils.data import DatasetInfo
from metatrain.utils.data.target_info import get_energy_target_info
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
    hypers["descriptor"]["repflow"]["n_dim"] = 2
    hypers["descriptor"]["repflow"]["e_dim"] = 2
    hypers["descriptor"]["repflow"]["a_dim"] = 2
    hypers["descriptor"]["repflow"]["e_sel"] = 1
    hypers["descriptor"]["repflow"]["a_sel"] = 1
    hypers["descriptor"]["repflow"]["axis_neuron"] = 1
    hypers["descriptor"]["repflow"]["nlayers"] = 1
    hypers["fitting_net"]["neuron"] = [1, 1]
    return hypers


class DPA3Tests(ArchitectureTests):
    architecture = "experimental.dpa3"

    @pytest.fixture(params=[True, False], ids=["multitask", "single_task"])
    def dpa3_multitask(self, request: pytest.FixtureRequest):
        return request.param

    @pytest.fixture(params=["from_hypers", "from_file"])
    def source(self, request: pytest.FixtureRequest):
        return request.param

    def _make_dataset_info(self) -> DatasetInfo:
        targets = {
            "mtt::U0": get_energy_target_info(
                "mtt::U0", {"quantity": "energy", "unit": "eV"}
            )
        }
        return DatasetInfo(
            length_unit="Angstrom", atomic_types=[1, 6, 7, 8], targets=targets
        )

    def _to_model_hypers(self, hypers: dict) -> ModelHypers:
        return copy.deepcopy(hypers)

    @pytest.fixture(params=("cpu",))
    def device(self, request):
        """DPA3 model construction (get_standard_model) is expensive.
        Restrict parametrized tests to CPU to stay within CSCS CI time
        limits.  CUDA coverage is provided by test_pretrained.py and
        test_regression.py which test the device-handling code paths
        directly."""
        return torch.device(request.param)

    @pytest.fixture
    def model_hypers(
        self,
        tmp_path: Path,
        source: str,
        dpa3_multitask: bool,
    ) -> dict:
        hypers = copy.deepcopy(get_default_hypers(self.architecture)["model"])
        if dpa3_multitask:
            dpa3_model_path = tmp_path / "multi_task.pt"
            branches = ["Alpha", "Beta"]
            hypers["dpa3_model_branch"] = branches[0]
        else:
            dpa3_model_path = tmp_path / "single_task.pt"
            branches = None
            hypers["dpa3_model_branch"] = None

        if source == "from_file" and not dpa3_model_path.exists():
            hypers["dpa3_model"] = str(dpa3_model_path)
            hypers_for_file = self._to_model_hypers(
                _minimal_hypers("experimental.dpa3")
            )
            hypers_for_file["descriptor"]["precision"] = 32
            base = DPA3(hypers_for_file, self._make_dataset_info())
            config = json.loads(base.model.get_model_def_script())

            state = collections.OrderedDict()
            for prefix in [f"model.{task}." for task in branches or ["Default"]]:
                for key, value in base.model.state_dict().items():
                    if prefix == "model.Beta.":
                        state[prefix + key] = (
                            value + 1.0
                        )  # make Beta different from Alpha
                    else:
                        state[prefix + key] = value

            state["_extra_state"] = {
                "model_params": (
                    {"model_dict": {branch: config for branch in branches}}
                    if branches
                    else config
                ),
                "train_infos": {"lr": 0, "step": 0},
            }
            torch.save({"model": state}, dpa3_model_path)
        return hypers

    @pytest.fixture
    def minimal_model_hypers(self) -> dict:
        """Minimal hyperparameters for a DPA3 model for the smallest
        checkpoint possible.

        :return: Hyperparameters for the model.
        """
        return _minimal_hypers(self.architecture)


class TestInput(InputTests, DPA3Tests): ...


class TestOutput(OutputTests, DPA3Tests):
    supports_multiscalar_outputs = False
    supports_spherical_outputs = False
    supports_vector_outputs = False
    supports_features = False
    supports_last_layer_features = False

    def test_prediction_energy_subset_atoms(self, model_hypers, dataset_info, source):
        # deepmd-kit precision is a construction-time setting.  This test sets
        # torch.set_default_dtype(float64) internally, but deepmd-kit's linear
        # layers use self.prec (set at construction).  Build with float64
        # precision to avoid numerical noise from neighbor list construction
        # across different system sizes.
        if source == "from_file":
            pytest.skip(
                "the precision of a pretrained model comes from its checkpoint, "
                "so the float64 model this test requires cannot be built from a file"
            )
        model_hypers = copy.deepcopy(model_hypers)
        model_hypers["descriptor"]["precision"] = 64
        model_hypers["fitting_net"]["precision"] = 64
        super().test_prediction_energy_subset_atoms(model_hypers, dataset_info)


class TestAutograd(AutogradTests, DPA3Tests):
    cuda_nondet_tolerance = 1e-12

    @pytest.fixture
    def model_hypers(self) -> dict:
        # Autograd tests require float64 for numerical gradcheck stability.
        # Deepmd-kit precision must be set at construction time, so we build
        # the model in float64 instead of relying on .to(float64).
        hypers = _minimal_hypers(self.architecture)
        hypers["descriptor"]["precision"] = 64
        hypers["fitting_net"]["precision"] = 64
        return hypers


class TestTorchscript(TorchscriptTests, DPA3Tests):
    float_hypers = ["descriptor.repflow.e_rcut", "descriptor.repflow.e_rcut_smth"]
    supports_spherical_outputs = False

    @pytest.fixture
    def model_hypers(self, dtype: torch.dtype) -> dict:
        # Deepmd-kit precision must match test dtype at construction time.
        hypers = _minimal_hypers(self.architecture)
        prec = 64 if dtype == torch.float64 else 32
        hypers["descriptor"]["precision"] = prec
        hypers["fitting_net"]["precision"] = prec
        return hypers

    def test_torchscript_integers(self, model_hypers, dataset_info):
        # test_torchscript_integers internally uses dtype=float32, but our
        # model_hypers fixture may set float64 precision when parameterized
        # with dtype1.  deepmd-kit precision cannot be changed via .to(),
        # so force float32 precision here.
        model_hypers = copy.deepcopy(model_hypers)
        model_hypers["descriptor"]["precision"] = 32
        model_hypers["fitting_net"]["precision"] = 32
        super().test_torchscript_integers(model_hypers, dataset_info)


class TestExported(ExportedTests, DPA3Tests):
    @pytest.fixture
    def model_hypers(self, dtype: torch.dtype) -> dict:
        # Deepmd-kit precision is a construction-time setting, so match the
        # descriptor/fitting_net precision to the test dtype.
        hypers = _minimal_hypers(self.architecture)
        prec = 64 if dtype == torch.float64 else 32
        hypers["descriptor"]["precision"] = prec
        hypers["fitting_net"]["precision"] = prec
        return hypers


class TestTraining(TrainingTests, DPA3Tests):
    supports_atomic_basis = False

    @pytest.mark.skip(reason="DPA3: multi-epoch training too slow for CI time limits")
    def test_continue_restart_num_epochs(self, *a, **kw):
        pass

    @pytest.mark.skip(reason="DPA3: multi-epoch training too slow for CI time limits")
    def test_continue_finetune_num_epochs(self, *a, **kw):
        pass


class TestCheckpoints(CheckpointTests, DPA3Tests):
    incompatible_trainer_checkpoints: list[str] = []
