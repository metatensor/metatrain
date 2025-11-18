import copy
from turtle import mode

import torch
from metatomic.torch import System

from metatrain.utils.neighbor_lists import get_system_with_neighbor_lists

from .base import ArchitectureTests

class TorchscriptTests(ArchitectureTests):

    float_hypers = []

    def test_torchscript(self, model_hypers, dataset_info):
        """Tests that the model can be jitted."""

        model = self.model_cls(model_hypers, dataset_info)
        system = System(
            types=torch.tensor([6, 1, 8, 7]),
            positions=torch.tensor(
                [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 2.0], [0.0, 0.0, 3.0]]
            ),
            cell=torch.zeros(3, 3),
            pbc=torch.tensor([False, False, False]),
        )
        system = get_system_with_neighbor_lists(system, model.requested_neighbor_lists())

        model = torch.jit.script(model)
        model(
            [system],
            model.outputs,
        )

    def test_torchscript_spherical(self, model_hypers, dataset_info_spherical):
        """Tests that there is no problem with jitting with spherical targets."""

        self.test_torchscript(
            model_hypers=model_hypers, dataset_info=dataset_info_spherical
        )


    def test_torchscript_save_load(self,tmpdir, model_hypers, dataset_info):
        """Tests that the model can be jitted and saved."""

        model = self.model_cls(model_hypers, dataset_info)

        with tmpdir.as_cwd():
            torch.jit.save(torch.jit.script(model), "model.pt")
            torch.jit.load("model.pt")


    def test_torchscript_integers(self, model_hypers, dataset_info):
        """Tests that the model can be jitted when some float
        parameters are instead supplied as integers."""

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
        system = get_system_with_neighbor_lists(system, model.requested_neighbor_lists())

        model = torch.jit.script(model)
        model(
            [system],
            model.outputs,
        )
