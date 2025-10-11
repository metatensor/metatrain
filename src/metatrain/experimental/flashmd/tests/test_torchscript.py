import pytest
import torch
from metatensor.torch import Labels, TensorBlock, TensorMap
from metatomic.torch import System
from omegaconf import OmegaConf

from metatrain.experimental.flashmd import FlashMD
from metatrain.utils.data import DatasetInfo
from metatrain.utils.data.target_info import TargetInfo
from metatrain.utils.neighbor_lists import (
    get_requested_neighbor_lists,
    get_system_with_neighbor_lists,
)

from . import MODEL_HYPERS


@pytest.mark.filterwarnings("ignore:custom data:UserWarning")
def test_torchscript():
    """Tests that the model can be jitted."""

    # load default hyper parameters for FlashMD
    full_hypers = OmegaConf.load("../default-hypers.yaml")
    model_hypers = dict(full_hypers)["architecture"]["model"]

    # define dataset (especially the targets)
    dataset_info = DatasetInfo(
        length_unit="angstrom",
        atomic_types=[1, 6],
        targets={
            name: TargetInfo(
                layout=TensorMap(
                    keys=Labels.single(),
                    blocks=[
                        TensorBlock(
                            values=torch.empty((0, 3, 1), dtype=torch.float64),
                            samples=Labels(
                                names=["system", "atom"],
                                values=torch.empty((0, 2), dtype=int),
                            ),
                            components=[
                                Labels.range("xyz", 3),
                            ],
                            properties=Labels.range("length", 1),
                        )
                    ],
                ),
                quantity="length",
                unit="angstrom",
            )
            for name in ["positions", "momenta"]
        },
    )

    # create a FlashMD model and attach a (random) raw PET model
    model = FlashMD(model_hypers, dataset_info)

    # define example systems
    dtype = torch.float32
    systems = [
        System(
            types=torch.tensor([1, 6, 1]),
            positions=torch.randn(3, 3, dtype=dtype),
            cell=torch.eye(3, dtype=dtype),
            pbc=torch.tensor([True] * 3),
        )
    ]

    # add random momenta to the systems
    for system in systems:
        num_atoms = len(system)
        tmap = TensorMap(
            keys=Labels.single(),
            blocks=[
                TensorBlock(
                    # TODO: get the momenta from the system if available!
                    values=torch.randn(num_atoms, 3, dtype=dtype),
                    samples=Labels(
                        names=["system"],
                        values=torch.arange(num_atoms, dtype=int).unsqueeze(-1),
                    ),
                    components=[],
                    properties=Labels.range("length", 3),
                ),
            ],
        )
        system.add_data("momenta", tmap)

    requested_neighbor_lists = get_requested_neighbor_lists(model)
    model = torch.jit.script(model)

    systems = [
        get_system_with_neighbor_lists(system, requested_neighbor_lists)
        for system in systems
    ]
    model(
        systems,
        {"positions": model.outputs["positions"], "momenta": model.outputs["momenta"]},
    )


def test_torchscript_save_load(tmpdir):
    """Tests that the model can be jitted and saved."""

    dataset_info = DatasetInfo(
        length_unit="angstrom",
        atomic_types=[1, 6],
        targets={
            name: TargetInfo(
                layout=TensorMap(
                    keys=Labels.single(),
                    blocks=[
                        TensorBlock(
                            values=torch.empty((0, 3, 1), dtype=torch.float64),
                            samples=Labels(
                                names=["system", "atom"],
                                values=torch.empty((0, 2), dtype=int),
                            ),
                            components=[
                                Labels.range("xyz", 3),
                            ],
                            properties=Labels.range("length", 1),
                        )
                    ],
                ),
                quantity="length",
                unit="angstrom",
            )
            for name in ["positions", "momenta"]
        },
    )
    model = FlashMD(MODEL_HYPERS, dataset_info)
    model.to(torch.float64)
    model.additive_models[0].weights_to(device="cpu", dtype=torch.float64)
    model.scaler.scales_to(device="cpu", dtype=torch.float64)

    with tmpdir.as_cwd():
        torch.jit.save(torch.jit.script(model), "model.pt")
        torch.jit.load("model.pt")
