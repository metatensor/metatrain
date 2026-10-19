import pytest
import torch
from metatensor.torch import Labels, TensorBlock, TensorMap
from metatomic.torch import ModelOutput, System

from metatrain.experimental.flashmd.model import FlashMD
from metatrain.utils.architectures import get_default_hypers
from metatrain.utils.data import DatasetInfo
from metatrain.utils.data.target_info import TargetInfo
from metatrain.utils.neighbor_lists import (
    get_requested_neighbor_lists,
    get_system_with_neighbor_lists,
)


@pytest.mark.filterwarnings("ignore:custom data:UserWarning")
def test_forward():
    "Run a forward pass of FlashMD on two small systems and verify the output shapes."

    # load default hyper parameters for FlashMD
    architecture_hypers = get_default_hypers("experimental.flashmd")
    model_hypers = architecture_hypers["model"]

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
            for name in ["position", "momentum"]
        },
    )

    # create a FlashMD model and attach a (random) raw PET model
    model = FlashMD(model_hypers, dataset_info)

    # define example systems
    dtype = torch.float32
    systems = []
    systems.append(
        # system 0: 3 atoms
        System(
            types=torch.tensor([1, 6, 1]),
            positions=torch.randn(3, 3, dtype=dtype),
            cell=torch.eye(3, dtype=dtype),
            pbc=torch.tensor([True] * 3),
        )
    )
    systems.append(
        # system 1: 2 atoms
        System(
            types=torch.tensor([6, 1]),
            positions=torch.randn(2, 3, dtype=dtype),
            cell=torch.eye(3, dtype=dtype),
            pbc=torch.tensor([True] * 3),
        )
    )

    # attach neighbor lists to the systems
    nbr_lists = get_requested_neighbor_lists(model)
    systems = [get_system_with_neighbor_lists(system, nbr_lists) for system in systems]

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
        system.add_data("momentum", tmap)

    outputs = {
        "position": ModelOutput(unit="angstrom", sample_kind="atom"),
        "momentum": ModelOutput(unit="angstrom", sample_kind="atom"),
    }
    result_dict = model(systems, outputs)

    assert set(result_dict.keys()) == set(outputs.keys())

    # 2+3=5 atoms in total, both outputs are 3D vectors per atom
    assert result_dict["position"][0].values.shape == (5, 3, 1)
    assert result_dict["momentum"][0].values.shape == (5, 3, 1)


@pytest.mark.parametrize("predict_momenta_as_difference", [True, False])
@pytest.mark.filterwarnings("ignore:custom data:UserWarning")
def test_eval_subset_of_outputs(predict_momenta_as_difference):
    """At evaluation, any subset of the outputs can be requested, and gives the
    same values as when all of them are requested."""
    from metatrain.experimental.flashmd import Trainer

    from . import MODEL_HYPERS

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
                            components=[Labels.range("xyz", 3)],
                            properties=Labels.range(name, 1),
                        )
                    ],
                ),
                quantity=quantity,
                unit=unit,
            )
            for name, quantity, unit in [
                ("position", "length", "angstrom"),
                ("momentum", "momentum", "(eV*u)^(1/2)"),
            ]
        },
    )

    # small model, to keep the test fast
    model_hypers = {
        **MODEL_HYPERS,
        "d_pet": 1,
        "d_head": 1,
        "d_node": 1,
        "d_feedforward": 1,
        "num_heads": 1,
        "num_attention_layers": 1,
        "num_gnn_layers": 1,
        "predict_momenta_as_difference": predict_momenta_as_difference,
    }
    trainer = Trainer(get_default_hypers("experimental.flashmd")["training"])
    model = trainer.setup(model_hypers, dataset_info)
    model.core.set_masses({1: 1.008, 6: 12.011})

    dtype = torch.float32
    model = model.to(dtype=dtype)
    system = System(
        types=torch.tensor([1, 6, 1]),
        positions=torch.randn(3, 3, dtype=dtype),
        cell=torch.eye(3, dtype=dtype),
        pbc=torch.tensor([True] * 3),
    )
    system = get_system_with_neighbor_lists(system, get_requested_neighbor_lists(model))
    system.add_data(
        "momentum",
        TensorMap(
            keys=Labels.single(),
            blocks=[
                TensorBlock(
                    values=torch.randn(3, 3, 1, dtype=dtype),
                    samples=Labels(
                        names=["system", "atom"],
                        values=torch.tensor([[0, 0], [0, 1], [0, 2]]),
                    ),
                    components=[Labels.range("xyz", 3)],
                    properties=Labels.range("momentum", 1),
                ),
            ],
        ),
    )

    model.eval()
    all_outputs = [
        "position",
        "momentum",
        "feature",
        "mtt::aux::position_last_layer_features",
    ]
    reference = model(
        [system], {name: ModelOutput(sample_kind="atom") for name in all_outputs}
    )

    # each output requested alone
    for name in all_outputs:
        result = model([system], {name: ModelOutput(sample_kind="atom")})
        assert list(result.keys()) == [name]
        torch.testing.assert_close(
            result[name].block().values, reference[name].block().values
        )


def test_exported_timestep(tmpdir):
    """The timestep can be read from the exported model's module, where the
    ``flashmd`` integrators and LAMMPS' ``fix metatomic`` look for it."""
    from metatomic.torch import load_atomistic_model

    from metatrain.experimental.flashmd import Trainer

    from . import MODEL_HYPERS

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
                            components=[Labels.range("xyz", 3)],
                            properties=Labels.range(name, 1),
                        )
                    ],
                ),
                quantity=quantity,
                unit=unit,
            )
            for name, quantity, unit in [
                ("position", "length", "angstrom"),
                ("momentum", "momentum", "(eV*u)^(1/2)"),
            ]
        },
    )

    trainer = Trainer(get_default_hypers("experimental.flashmd")["training"])
    model = trainer.setup(MODEL_HYPERS, dataset_info)
    model.core.set_timestep(16.0)

    with tmpdir.as_cwd():
        model.export().save("model.pt")
        exported = load_atomistic_model("model.pt")

    assert float(exported.module.timestep) == 16.0
