import copy
from typing import Any

import torch
from metatomic.torch import System

from metatrain.utils.data import DatasetInfo
from metatrain.utils.neighbor_lists import get_system_with_neighbor_lists

from .architectures import ArchitectureTests


class TorchscriptTests(ArchitectureTests):
    """Test suite to check that architectures can be jit compiled with
    TorchScript."""

    float_hypers: list[str] = []
    """List of hyperparameter keys (dot-separated for nested keys)
    that are floats. A test will set these to integers to test that
    TorchScript compilation works in that case."""

    def test_torchscript(self, model_hypers: dict, dataset_info: DatasetInfo) -> None:
        """Tests that the model can be jitted.

        If this test fails it probably means that there is some
        code in the model that is not compatible with TorchScript.
        The exception raised by the test should indicate where
        the problem is.

        :param model_hypers: Hyperparameters to initialize the model.
        :param dataset_info: Dataset to initialize the model.
        """

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

        model = torch.jit.script(model)
        model(
            [system],
            model.outputs,
        )

    def test_torchscript_spherical(
        self, model_hypers: dict, dataset_info_spherical: DatasetInfo
    ) -> None:
        """Tests that there is no problem with jitting with spherical targets.

        :param model_hypers: Hyperparameters to initialize the model.
        :param dataset_info_spherical: Dataset to initialize the model
        (containing spherical targets).
        """

        self.test_torchscript(
            model_hypers=model_hypers, dataset_info=dataset_info_spherical
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

        model = torch.jit.script(model)
        model(
            [system],
            model.outputs,
        )
