import collections
import copy
import json
from pathlib import Path

import pytest
import torch
from metatomic.torch import ModelOutput, System

from metatrain.experimental.se_e2_a import SeE2A
from metatrain.utils.architectures import get_default_hypers
from metatrain.utils.data import DatasetInfo
from metatrain.utils.data.target_info import get_energy_target_info
from metatrain.utils.neighbor_lists import get_system_with_neighbor_lists
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
    hypers["descriptor"]["sel"] = 2
    hypers["descriptor"]["rcut"] = 4.0
    hypers["descriptor"]["rcut_smth"] = 3.0
    hypers["descriptor"]["neuron"] = [4, 4]
    hypers["descriptor"]["axis_neuron"] = 2
    hypers["fitting_net"]["neuron"] = [2, 2]
    return hypers


class SeE2ATests(ArchitectureTests):
    architecture = "experimental.se_e2_a"

    @pytest.fixture(params=[True, False], ids=["multitask", "single_task"])
    def se_e2_a_multitask(self, request: pytest.FixtureRequest):
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

    @pytest.fixture(params=("cpu",))
    def device(self, request):
        """deepmd-kit builds the module on CPU. Keep this suite there."""
        return torch.device(request.param)

    def _make_model_hypers(
        self,
        tmp_path: Path,
        source: str,
        se_e2_a_multitask: bool,
        precision: int = 32,
    ) -> dict:
        hypers = _minimal_hypers(self.architecture)
        hypers["descriptor"]["precision"] = precision
        hypers["fitting_net"]["precision"] = precision
        if se_e2_a_multitask:
            checkpoint_path = tmp_path / "multi_task.pt"
            branches = ["Alpha", "Beta"]
            hypers["se_e2_a_model_branch"] = branches[0]
        else:
            checkpoint_path = tmp_path / "single_task.pt"
            branches = None
            hypers["se_e2_a_model_branch"] = None

        if source == "from_file" and not checkpoint_path.exists():
            hypers["se_e2_a_model"] = str(checkpoint_path)
            hypers_for_file = _minimal_hypers(self.architecture)
            hypers_for_file["descriptor"]["precision"] = precision
            hypers_for_file["fitting_net"]["precision"] = precision
            base = SeE2A(
                hypers_for_file,  # type: ignore[arg-type]
                self._make_dataset_info(),
            )
            config = json.loads(base.model.get_model_def_script())
            state = collections.OrderedDict()
            for prefix in [f"model.{task}." for task in branches or ["Default"]]:
                for key, value in base.model.state_dict().items():
                    if prefix == "model.Beta.":
                        state[prefix + key] = value + 1.0
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
            torch.save({"model": state}, checkpoint_path)
        return hypers

    @pytest.fixture
    def model_hypers(
        self,
        tmp_path: Path,
        source: str,
        se_e2_a_multitask: bool,
    ) -> dict:
        return self._make_model_hypers(tmp_path, source, se_e2_a_multitask)

    @pytest.fixture
    def minimal_model_hypers(self) -> dict:
        """Smallest se_e2_a used when a checkpoint has to stay under 50 kB."""
        return _minimal_hypers(self.architecture)


class TestInput(InputTests, SeE2ATests): ...


class TestOutput(OutputTests, SeE2ATests):
    supports_multiscalar_outputs = False
    supports_spherical_outputs = False
    supports_vector_outputs = False
    supports_features = False
    # The generic check fits one line through float32 features. The energy
    # forward and the embedding forward are separate deepmd-kit calls, and in
    # float32 they can disagree by about the 1e-3 tolerance of that check.
    # test_llpr.py checks the same relation in float64.
    last_layer_features_are_affine = False

    @pytest.fixture
    def model_hypers(
        self,
        tmp_path: Path,
        source: str,
        se_e2_a_multitask: bool,
    ) -> dict:
        hypers = self._make_model_hypers(tmp_path, source, se_e2_a_multitask)
        if source == "from_hypers":
            # Two hidden units and four elements is 4 * (2 + 1) = 12 features,
            # fewer than the rows in the generic affine check.
            hypers["fitting_net"]["neuron"] = [2, 2]
        return hypers

    def test_prediction_energy_subset_atoms(self, model_hypers, dataset_info, source):
        # deepmd-kit precision is fixed at construction. This test compares
        # float64 values, so the module has to be built in float64.
        if source == "from_file":
            pytest.skip(
                "the precision of a pretrained model comes from its checkpoint, "
                "so the float64 model this test requires cannot be built from a file"
            )
        model_hypers = copy.deepcopy(model_hypers)
        model_hypers["descriptor"]["precision"] = 64
        model_hypers["fitting_net"]["precision"] = 64
        super().test_prediction_energy_subset_atoms(model_hypers, dataset_info)


class TestAutograd(AutogradTests, SeE2ATests):
    cuda_nondet_tolerance = 1e-12

    @pytest.fixture
    def model_hypers(self) -> dict:
        # The autograd system is periodic with a cell of 2. A short cutoff
        # drops neighbors as positions move by the finite-difference step.
        # Keep every image inside the flat part of the cutoff.
        hypers = _minimal_hypers(self.architecture)
        hypers["descriptor"]["precision"] = 64
        hypers["fitting_net"]["precision"] = 64
        hypers["descriptor"]["sel"] = 40
        hypers["descriptor"]["rcut"] = 6.0
        hypers["descriptor"]["rcut_smth"] = 5.0
        return hypers

    def test_autograd_positions(self, device, model_hypers, dataset_info):
        # deepmd-kit backpropagates inside the forward to build forces, and
        # that backward is not itself differentiable, so gradgradcheck does
        # not apply. First derivatives do.
        self._check_first_derivative(
            device, model_hypers, dataset_info, which="positions"
        )

    def test_autograd_cell(self, device, model_hypers, dataset_info):
        self._check_first_derivative(device, model_hypers, dataset_info, which="cell")

    def _check_first_derivative(self, device, model_hypers, dataset_info, which: str):
        import random

        import numpy as np

        random.seed(0)
        np.random.seed(0)
        torch.manual_seed(0)
        device = torch.device(device)
        model = self.model_cls(model_hypers, dataset_info).to(
            dtype=torch.float64, device=device
        )

        def compute(tensor: torch.Tensor) -> torch.Tensor:
            if which == "positions":
                positions, cell = (
                    tensor,
                    torch.eye(3, dtype=torch.float64, device=device) * self.cell_param,
                )
            else:
                positions = torch.tensor(
                    self.positions, dtype=torch.float64, device=device
                )
                cell = tensor
            system = get_system_with_neighbor_lists(
                System(
                    types=torch.tensor([6, 6], device=device),
                    positions=positions,
                    cell=cell,
                    pbc=torch.tensor([True, True, True], device=device),
                ),
                model.requested_neighbor_lists(),
            )
            return (
                model([system], {"energy": ModelOutput(sample_kind="system")})["energy"]
                .block()
                .values.sum()
            )

        if which == "positions":
            argument = torch.tensor(
                self.positions, dtype=torch.float64, requires_grad=True, device=device
            )
        else:
            argument = (
                torch.eye(3, dtype=torch.float64, device=device) * self.cell_param
            ).requires_grad_()
        assert torch.autograd.gradcheck(compute, argument, fast_mode=True)


class TestTorchscript(TorchscriptTests, SeE2ATests):
    float_hypers = ["descriptor.rcut", "descriptor.rcut_smth"]
    supports_spherical_outputs = False

    @pytest.fixture
    def model_hypers(self, dtype: torch.dtype) -> dict:
        hypers = _minimal_hypers(self.architecture)
        prec = 64 if dtype == torch.float64 else 32
        hypers["descriptor"]["precision"] = prec
        hypers["fitting_net"]["precision"] = prec
        return hypers

    def test_torchscript_integers(self, model_hypers, dataset_info):
        model_hypers = copy.deepcopy(model_hypers)
        model_hypers["descriptor"]["precision"] = 32
        model_hypers["fitting_net"]["precision"] = 32
        super().test_torchscript_integers(model_hypers, dataset_info)


class TestExported(ExportedTests, SeE2ATests):
    @pytest.fixture
    def model_hypers(self, dtype: torch.dtype) -> dict:
        hypers = _minimal_hypers(self.architecture)
        prec = 64 if dtype == torch.float64 else 32
        hypers["descriptor"]["precision"] = prec
        hypers["fitting_net"]["precision"] = prec
        return hypers


class TestTraining(TrainingTests, SeE2ATests):
    supports_atomic_basis = False

    @pytest.mark.skip(reason="se_e2_a: multi-epoch training is too slow for CI")
    def test_continue_restart_num_epochs(self, *a, **kw):
        pass

    @pytest.mark.skip(reason="se_e2_a: multi-epoch training is too slow for CI")
    def test_continue_finetune_num_epochs(self, *a, **kw):
        pass


class TestCheckpoints(CheckpointTests, SeE2ATests):
    incompatible_trainer_checkpoints: list[str] = []


def test_sel_list_must_match_the_type_map():
    """A per-element neighbor list has to name every element once."""
    hypers = _minimal_hypers("experimental.se_e2_a")
    hypers["descriptor"]["sel"] = [2, 2]
    dataset_info = DatasetInfo(
        length_unit="Angstrom",
        atomic_types=[1, 6, 8],
        targets={
            "energy": get_energy_target_info(
                "energy", {"quantity": "energy", "unit": "eV"}
            )
        },
    )
    with pytest.raises(ValueError, match="3 atomic types"):
        SeE2A(hypers, dataset_info)


def test_axis_neuron_cannot_exceed_the_embedding():
    """deepmd-kit rejects an axis neuron wider than the embedding."""
    hypers = _minimal_hypers("experimental.se_e2_a")
    hypers["descriptor"]["axis_neuron"] = 8
    dataset_info = DatasetInfo(
        length_unit="Angstrom",
        atomic_types=[1],
        targets={
            "energy": get_energy_target_info(
                "energy", {"quantity": "energy", "unit": "eV"}
            )
        },
    )
    with pytest.raises(ValueError, match="axis_neuron"):
        SeE2A(hypers, dataset_info)
