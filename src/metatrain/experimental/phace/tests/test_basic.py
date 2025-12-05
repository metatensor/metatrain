import pytest

from metatrain.utils.testing import (
    ArchitectureTests,
    CheckpointTests,
    OutputTests,
    TorchscriptTests,
)

import torch
import copy
from metatomic.torch import System

from metatrain.utils.data import DatasetInfo
from metatrain.utils.neighbor_lists import get_system_with_neighbor_lists

from typing import Any



class PhACETests(ArchitectureTests):
    architecture = "experimental.phace"

    @pytest.fixture(params=[0, 1, 2])
    def o3_lambda(self, request: pytest.FixtureRequest) -> int:
        return request.param


class TestOutput(OutputTests, PhACETests):
    is_equivariant_reflections = False

    @pytest.fixture
    def n_last_layer_features(self) -> int:
        return 256


class TestTorchscript(TorchscriptTests, PhACETests):
    float_hypers = [
        "cutoff",
        "cutoff_width",
        "nu_scaling",
        "mp_scaling",
        "overall_scaling",
        "radial_basis.max_eigenvalue",
    ]

    def test_torchscript(self, model_hypers: dict, dataset_info: DatasetInfo) -> None:

        model = self.model_cls(model_hypers, dataset_info)
        system = System(
            types=torch.tensor([6, 1, 8, 7]),
            positions=torch.tensor(
                [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 2.0], [0.0, 0.0, 3.0]]
            ),
            cell=torch.zeros(3, 3),
            pbc=torch.tensor([False, False, False]),
        )
        system = get_system_with_neighbor_lists(
            system, model.requested_neighbor_lists()
        )

        model.module = model.fake_gradient_model
        del model.gradient_model
        del model.fake_gradient_model

        model = torch.jit.script(model)
        model(
            [system],
            model.outputs,
        )

    def test_torchscript_save_load(
        self, tmpdir: Any, model_hypers: dict, dataset_info: DatasetInfo
    ) -> None:
        """Tests that the model can be jitted, saved and loaded.

        :param tmpdir: Temporary directory where to save the
          model.
        :param model_hypers: Hyperparameters to initialize the model.
        :param dataset_info: Dataset to initialize the model.
        """

        model = self.model_cls(model_hypers, dataset_info)

        model.module = model.fake_gradient_model
        del model.gradient_model
        del model.fake_gradient_model

        with tmpdir.as_cwd():
            torch.jit.save(torch.jit.script(model), "model.pt")
            torch.jit.load("model.pt")

    def test_torchscript_integers(
        self, model_hypers: dict, dataset_info: DatasetInfo
    ) -> None:
        """Tests that the model can be jitted when some float
        parameters are instead supplied as integers.

        :param model_hypers: Hyperparameters to initialize the model.
        :param dataset_info: Dataset to initialize the model.
        """

        new_hypers = copy.deepcopy(model_hypers)
        for hyper in self.float_hypers:
            nested_key = hyper.split(".")
            sub_dict = new_hypers
            for key in nested_key[:-1]:
                sub_dict = sub_dict[key]
            sub_dict[nested_key[-1]] = int(sub_dict[nested_key[-1]])

        model = self.model_cls(new_hypers, dataset_info)

        system = System(
            types=torch.tensor([6, 1, 8, 7]),
            positions=torch.tensor(
                [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 2.0], [0.0, 0.0, 3.0]]
            ),
            cell=torch.zeros(3, 3),
            pbc=torch.tensor([False, False, False]),
        )
        system = get_system_with_neighbor_lists(
            system, model.requested_neighbor_lists()
        )

        model.module = model.fake_gradient_model
        del model.gradient_model
        del model.fake_gradient_model

        model = torch.jit.script(model)
        model(
            [system],
            model.outputs,
        )


class TestCheckpoints(CheckpointTests, PhACETests):
    incompatible_trainer_checkpoints = []
