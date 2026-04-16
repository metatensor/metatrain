from pathlib import Path

import metatensor.torch as mts
import pytest
import torch
from metatensor.torch import Labels, TensorBlock, TensorMap
from metatomic.torch import ModelOutput, System
from omegaconf import OmegaConf

from metatrain.utils.additive import (
    ZBL,
    CompositionModel,
    remove_additive,
)
from metatrain.utils.data import Dataset, DatasetInfo
from metatrain.utils.data.atomic_basis_helpers import (
    densify_atomic_basis_dataset_info,
    get_prepare_atomic_basis_targets_transform,
)
from metatrain.utils.data.readers import read_systems, read_targets
from metatrain.utils.data.target_info import (
    get_energy_target_info,
    get_generic_target_info,
)
from metatrain.utils.neighbor_lists import (
    get_requested_neighbor_lists,
    get_system_with_neighbor_lists,
)
from metatrain.utils.data.readers.metatensor import _empty_tensor_map_like



RESOURCES_PATH = Path(__file__).parents[1] / "resources"


def test_composition_model_float32_error():
    systems = [
        System(
            positions=torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float64),
            types=torch.tensor([8]),
            cell=torch.eye(3, dtype=torch.float64),
            pbc=torch.tensor([True, True, True]),
        ).to(torch.float32),
    ]
    energies = [1.0]
    dataset = Dataset.from_dict({"system": systems, "energy": energies})

    composition_model = CompositionModel(
        hypers={},
        dataset_info=DatasetInfo(
            length_unit="angstrom",
            atomic_types=[1, 8],
            targets={"energy": get_energy_target_info("energy", {"unit": "eV"})},
        ),
    )
    with pytest.raises(
        ValueError,
        match=(
            "The composition model only supports float64 "
            "during training. Got dtype: torch.float32."
        ),
    ):
        # This should raise an error because the systems are in float32
        composition_model.train_model(dataset, [], batch_size=1, is_distributed=False)


@pytest.mark.parametrize("fixed_weights", [True, False])
def test_composition_model_train(fixed_weights):
    """Test the calculation of composition weights for a per-structure scalar."""

    # Here we use three synthetic structures:
    # - O atom, with an energy of 1.0
    # - H2O molecule, with an energy of 5.0
    # - H4O2 molecule, with an energy of 10.0
    # The expected composition weights are 2.0 for H and 1.0 for O.

    systems = [
        System(
            positions=torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float64),
            types=torch.tensor([8]),
            cell=torch.eye(3, dtype=torch.float64),
            pbc=torch.tensor([True, True, True]),
        ),
        System(
            positions=torch.tensor(
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=torch.float64
            ),
            types=torch.tensor([1, 1, 8]),
            cell=torch.eye(3, dtype=torch.float64),
            pbc=torch.tensor([True, True, True]),
        ),
        System(
            positions=torch.tensor(
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                    [1.0, 0.0, 1.0],
                    [0.0, 1.0, 1.0],
                ],
                dtype=torch.float64,
            ),
            types=torch.tensor([1, 1, 8, 1, 1, 8]),
            cell=torch.eye(3, dtype=torch.float64),
            pbc=torch.tensor([True, True, True]),
        ),
    ]
    energies = [1.0, 5.0, 10.0]
    energies = [
        TensorMap(
            keys=Labels(names=["_"], values=torch.tensor([[0]])),
            blocks=[
                TensorBlock(
                    values=torch.tensor([[e]], dtype=torch.float64),
                    samples=Labels(names=["system"], values=torch.tensor([[i]])),
                    components=[],
                    properties=Labels(names=["energy"], values=torch.tensor([[0]])),
                )
            ],
        )
        for i, e in enumerate(energies)
    ]
    dataset = Dataset.from_dict({"system": systems, "energy": energies})

    composition_model = CompositionModel(
        hypers={},
        dataset_info=DatasetInfo(
            length_unit="angstrom",
            atomic_types=[1, 8],
            targets={"energy": get_energy_target_info("energy", {"unit": "eV"})},
        ),
    )

    system_H = System(
        positions=torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float64),
        types=torch.tensor([1]),
        cell=torch.eye(3, dtype=torch.float64),
        pbc=torch.tensor([True, True, True]),
    )
    system_O = System(
        positions=torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float64),
        types=torch.tensor([8]),
        cell=torch.eye(3, dtype=torch.float64),
        pbc=torch.tensor([True, True, True]),
    )

    if fixed_weights:
        fixed_weights = {"energy": {1: 2.0, 8: 1.0}}
    else:
        fixed_weights = None

    composition_model.train_model(
        dataset, [], batch_size=1, is_distributed=False, fixed_weights=fixed_weights
    )
    assert composition_model.atomic_types == [1, 8]
    output_H = composition_model(
        [system_H], {"energy": ModelOutput(quantity="energy", unit="", per_atom=False)}
    )
    torch.testing.assert_close(
        output_H["energy"].block().values, torch.tensor([[2.0]], dtype=torch.float64)
    )
    output_O = composition_model(
        [system_O], {"energy": ModelOutput(quantity="energy", unit="", per_atom=False)}
    )
    torch.testing.assert_close(
        output_O["energy"].block().values, torch.tensor([[1.0]], dtype=torch.float64)
    )

    composition_model.train_model([dataset], [], batch_size=1, is_distributed=False)
    output_H = composition_model(
        [system_H], {"energy": ModelOutput(quantity="energy", unit="", per_atom=False)}
    )
    torch.testing.assert_close(
        output_H["energy"].block().values, torch.tensor([[2.0]], dtype=torch.float64)
    )
    output_O = composition_model(
        [system_O], {"energy": ModelOutput(quantity="energy", unit="", per_atom=False)}
    )
    torch.testing.assert_close(
        output_O["energy"].block().values, torch.tensor([[1.0]], dtype=torch.float64)
    )

    composition_model.train_model(
        [dataset, dataset, dataset], [], batch_size=1, is_distributed=False
    )
    assert composition_model.atomic_types == [1, 8]
    output_H = composition_model(
        [system_H], {"energy": ModelOutput(quantity="energy", unit="", per_atom=False)}
    )
    torch.testing.assert_close(
        output_H["energy"].block().values, torch.tensor([[2.0]], dtype=torch.float64)
    )
    output_O = composition_model(
        [system_O], {"energy": ModelOutput(quantity="energy", unit="", per_atom=False)}
    )
    torch.testing.assert_close(
        output_O["energy"].block().values, torch.tensor([[1.0]], dtype=torch.float64)
    )


def test_composition_model_float_fixed_weight():
    """Test that passing a single weight for all types works.

    In particular, we test that passing 0.0 as the weight for
    a target effectively disables the composition model.
    """

    systems = [
        System(
            positions=torch.tensor(
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=torch.float64
            ),
            types=torch.tensor([1, 1, 8]),
            cell=torch.eye(3, dtype=torch.float64),
            pbc=torch.tensor([True, True, True]),
        ),
    ]
    energies = [2.0]
    energies = [
        TensorMap(
            keys=Labels(names=["_"], values=torch.tensor([[0]])),
            blocks=[
                TensorBlock(
                    values=torch.tensor([[e]], dtype=torch.float64),
                    samples=Labels(names=["system"], values=torch.tensor([[i]])),
                    components=[],
                    properties=Labels(names=["energy"], values=torch.tensor([[0]])),
                )
            ],
        )
        for i, e in enumerate(energies)
    ]
    dataset = Dataset.from_dict({"system": systems, "energy": energies})

    composition_model = CompositionModel(
        hypers={},
        dataset_info=DatasetInfo(
            length_unit="angstrom",
            atomic_types=[1, 8],
            targets={"energy": get_energy_target_info("energy", {"unit": "eV"})},
        ),
    )

    system_H = System(
        positions=torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float64),
        types=torch.tensor([1]),
        cell=torch.eye(3, dtype=torch.float64),
        pbc=torch.tensor([True, True, True]),
    )
    system_O = System(
        positions=torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float64),
        types=torch.tensor([8]),
        cell=torch.eye(3, dtype=torch.float64),
        pbc=torch.tensor([True, True, True]),
    )

    fixed_weights = {"energy": 0.0}

    composition_model.train_model(
        dataset, [], batch_size=1, is_distributed=False, fixed_weights=fixed_weights
    )
    assert composition_model.atomic_types == [1, 8]
    output_H = composition_model(
        [system_H], {"energy": ModelOutput(quantity="energy", unit="", per_atom=False)}
    )
    torch.testing.assert_close(
        output_H["energy"].block().values, torch.tensor([[0.0]], dtype=torch.float64)
    )
    output_O = composition_model(
        [system_O], {"energy": ModelOutput(quantity="energy", unit="", per_atom=False)}
    )
    torch.testing.assert_close(
        output_O["energy"].block().values, torch.tensor([[0.0]], dtype=torch.float64)
    )


def test_fixed_weights_missing_types():
    """
    Tests that a meaningful error is raised when the provided fixed
    weights are missing some atomic types.
    """

    systems = [
        System(
            positions=torch.tensor(
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=torch.float64
            ),
            types=torch.tensor([1, 1, 8]),
            cell=torch.eye(3, dtype=torch.float64),
            pbc=torch.tensor([True, True, True]),
        ),
    ]
    energies = [2.0]
    energies = [
        TensorMap(
            keys=Labels(names=["_"], values=torch.tensor([[0]])),
            blocks=[
                TensorBlock(
                    values=torch.tensor([[e]], dtype=torch.float64),
                    samples=Labels(names=["system"], values=torch.tensor([[i]])),
                    components=[],
                    properties=Labels(names=["energy"], values=torch.tensor([[0]])),
                )
            ],
        )
        for i, e in enumerate(energies)
    ]
    dataset = Dataset.from_dict({"system": systems, "energy": energies})

    composition_model = CompositionModel(
        hypers={},
        dataset_info=DatasetInfo(
            length_unit="angstrom",
            atomic_types=[1, 8],
            targets={"energy": get_energy_target_info("energy", {"unit": "eV"})},
        ),
    )

    fixed_weights = {"energy": {1: 0.0}}  # Missing O weight

    error_msg = (
        r"Fixed weights for target 'energy' are missing"
        r" the following atomic types: \{8\}"
    )

    with pytest.raises(ValueError, match=error_msg):
        composition_model.train_model(
            dataset, [], batch_size=1, is_distributed=False, fixed_weights=fixed_weights
        )


@pytest.mark.parametrize("device", ("cpu", "cuda"))
def test_composition_model_predict(device):
    """Test the prediction of composition energies."""

    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    dataset_path = RESOURCES_PATH / "qm9_reduced_100.xyz"
    systems = read_systems(dataset_path)

    conf = {
        "mtt::U0": {
            "quantity": "energy",
            "read_from": dataset_path,
            "file_format": ".xyz",
            "reader": "ase",
            "key": "U0",
            "unit": "eV",
            "type": "scalar",
            "per_atom": False,
            "num_subtargets": 1,
            "forces": False,
            "stress": False,
            "virial": False,
        }
    }
    targets, target_info = read_targets(OmegaConf.create(conf))
    dataset = Dataset.from_dict({"system": systems, "mtt::U0": targets["mtt::U0"]})

    composition_model = CompositionModel(
        hypers={},
        dataset_info=DatasetInfo(
            length_unit="angstrom",
            atomic_types=[1, 6, 7, 8],
            targets=target_info,
        ),
    )
    composition_model.train_model(
        [dataset], additive_models=[], batch_size=1, is_distributed=False
    )

    systems_to_predict = [system.to(device=device) for system in systems[:5]]

    # per_atom = False
    output = composition_model(
        systems_to_predict,
        {"mtt::U0": ModelOutput(quantity="energy", unit="", per_atom=False)},
    )
    assert "mtt::U0" in output
    assert output["mtt::U0"].block().samples.names == ["system"]
    assert output["mtt::U0"].block().values.shape == (5, 1)
    assert output["mtt::U0"].block().values.device.type == device

    # per_atom = True
    output = composition_model(
        systems_to_predict,
        {"mtt::U0": ModelOutput(quantity="energy", unit="", per_atom=True)},
    )
    assert "mtt::U0" in output
    assert output["mtt::U0"].block().samples.names == ["system", "atom"]
    assert output["mtt::U0"].block().values.shape != (5, 1)
    assert output["mtt::U0"].block().values.device.type == device

    # with selected_atoms
    selected_atoms = mts.Labels(
        names=["system", "atom"],
        values=torch.tensor([[0, 0]]),
    ).to(device=device)

    output = composition_model(
        systems_to_predict,
        {"mtt::U0": ModelOutput(quantity="energy", unit="", per_atom=True)},
        selected_atoms=selected_atoms,
    )
    assert "mtt::U0" in output
    assert output["mtt::U0"].block().samples.names == ["system", "atom"]
    assert output["mtt::U0"].block().values.shape == (1, 1)
    assert output["mtt::U0"].block().values.device.type == device

    output = composition_model(
        systems_to_predict,
        {"mtt::U0": ModelOutput(quantity="energy", unit="", per_atom=False)},
        selected_atoms=selected_atoms,
    )
    assert "mtt::U0" in output
    assert output["mtt::U0"].block().samples.names == ["system"]
    assert output["mtt::U0"].block().values.shape == (1, 1)
    assert output["mtt::U0"].block().values.device.type == device


def test_composition_model_torchscript(tmpdir):
    """Test the torchscripting, saving and loading of the composition model."""
    system = System(
        positions=torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float64),
        types=torch.tensor([8]),
        cell=torch.eye(3, dtype=torch.float64),
        pbc=torch.tensor([True, True, True]),
    )

    composition_model = CompositionModel(
        hypers={},
        dataset_info=DatasetInfo(
            length_unit="angstrom",
            atomic_types=[1, 8],
            targets={"energy": get_energy_target_info("energy", {"unit": "eV"})},
        ),
    )
    composition_model = torch.jit.script(composition_model)
    composition_model(
        [system], {"energy": ModelOutput(quantity="energy", unit="", per_atom=False)}
    )

    with tmpdir.as_cwd():
        torch.jit.save(composition_model, "composition_model.pt")
        composition_model = torch.jit.load("composition_model.pt")

    composition_model(
        [system], {"energy": ModelOutput(quantity="energy", unit="", per_atom=False)}
    )


def test_remove_additive():
    """Tests the remove_additive function."""

    dataset_path = RESOURCES_PATH / "qm9_reduced_100.xyz"
    systems = read_systems(dataset_path)

    conf = {
        "mtt::U0": {
            "quantity": "energy",
            "read_from": dataset_path,
            "file_format": ".xyz",
            "reader": "ase",
            "key": "U0",
            "unit": "eV",
            "type": "scalar",
            "per_atom": False,
            "num_subtargets": 1,
            "forces": False,
            "stress": False,
            "virial": False,
        }
    }
    targets, target_info = read_targets(OmegaConf.create(conf))
    dataset = Dataset.from_dict({"system": systems, "mtt::U0": targets["mtt::U0"]})

    composition_model = CompositionModel(
        hypers={},
        dataset_info=DatasetInfo(
            length_unit="angstrom",
            atomic_types=[1, 6, 7, 8],
            targets=target_info,
        ),
    )
    composition_model.train_model(
        [dataset], additive_models=[], batch_size=1, is_distributed=False
    )

    # concatenate all targets
    targets["mtt::U0"] = mts.join(targets["mtt::U0"], axis="samples")

    std_before = targets["mtt::U0"].block().values.std().item()
    remove_additive(systems, targets, composition_model, target_info)
    std_after = targets["mtt::U0"].block().values.std().item()

    # In QM9 the composition contribution is very large: the standard deviation
    # of the energies is reduced by a factor of over 100 upon removing the composition
    assert std_after < 100.0 * std_before


def test_composition_model_missing_types(caplog):
    """
    Test the error when there are too many types in the dataset
    compared to those declared at initialization.
    """

    # Here we use three synthetic structures:
    # - O atom, with an energy of 1.0
    # - H2O molecule, with an energy of 5.0
    # - H4O2 molecule, with an energy of 10.0
    # The expected composition weights are 2.0 for H and 1.0 for O.

    systems = [
        System(
            positions=torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float64),
            types=torch.tensor([8]),
            cell=torch.eye(3, dtype=torch.float64),
            pbc=torch.tensor([True, True, True]),
        ),
        System(
            positions=torch.tensor(
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=torch.float64
            ),
            types=torch.tensor([1, 1, 8]),
            cell=torch.eye(3, dtype=torch.float64),
            pbc=torch.tensor([True, True, True]),
        ),
        System(
            positions=torch.tensor(
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                    [1.0, 0.0, 1.0],
                    [0.0, 1.0, 1.0],
                ],
                dtype=torch.float64,
            ),
            types=torch.tensor([1, 1, 8, 1, 1, 8]),
            cell=torch.eye(3, dtype=torch.float64),
            pbc=torch.tensor([True, True, True]),
        ),
    ]
    energies = [1.0, 5.0, 10.0]
    energies = [
        TensorMap(
            keys=Labels(names=["_"], values=torch.tensor([[0]])),
            blocks=[
                TensorBlock(
                    values=torch.tensor([[e]], dtype=torch.float64),
                    samples=Labels(names=["system"], values=torch.tensor([[i]])),
                    components=[],
                    properties=Labels(names=["energy"], values=torch.tensor([[0]])),
                )
            ],
        )
        for i, e in enumerate(energies)
    ]
    dataset = Dataset.from_dict({"system": systems, "energy": energies})

    composition_model = CompositionModel(
        hypers={},
        dataset_info=DatasetInfo(
            length_unit="angstrom",
            atomic_types=[1],
            targets={"energy": get_energy_target_info("energy", {"unit": "eV"})},
        ),
    )
    with pytest.raises(
        ValueError,
        match="unexpected atom types",
    ):
        composition_model.train_model([dataset], [], batch_size=1, is_distributed=False)


def test_composition_model_wrong_target():
    """
    Test the error when a non-scalar is fed to the composition model.
    """
    with pytest.raises(ValueError, match="does not support target quantity force"):
        CompositionModel(
            hypers={},
            dataset_info=DatasetInfo(
                length_unit="angstrom",
                atomic_types=[1],
                targets={
                    "force": get_generic_target_info(
                        "force",
                        {
                            "quantity": "force",
                            "unit": "",
                            "type": {"cartesian": {"rank": 1}},
                            "num_subtargets": 1,
                            "per_atom": True,
                        },
                    )
                },
            ),
        )


def test_zbl():
    """Test the ZBL model."""

    dataset_path = RESOURCES_PATH / "qm9_reduced_100.xyz"

    systems = read_systems(dataset_path)[:5]

    conf = {
        "mtt::U0": {
            "quantity": "energy",
            "read_from": dataset_path,
            "file_format": ".xyz",
            "reader": "ase",
            "key": "U0",
            "unit": "eV",
            "type": "scalar",
            "per_atom": False,
            "num_subtargets": 1,
            "forces": False,
            "stress": False,
            "virial": False,
        }
    }
    _, target_info = read_targets(OmegaConf.create(conf))

    zbl = ZBL(
        hypers={},
        dataset_info=DatasetInfo(
            length_unit="angstrom",
            atomic_types=[1, 6, 7, 8],
            targets=target_info,
        ),
    )

    requested_neighbor_lists = get_requested_neighbor_lists(zbl)
    for system in systems:
        get_system_with_neighbor_lists(system, requested_neighbor_lists)

    output = zbl(
        systems,
        {"mtt::U0": ModelOutput(quantity="energy", unit="", per_atom=True)},
    )
    assert "mtt::U0" in output
    assert output["mtt::U0"].block().samples.names == ["system", "atom"]
    assert output["mtt::U0"].block().values.shape != (5, 1)

    # with selected_atoms
    selected_atoms = mts.Labels(
        names=["system", "atom"],
        values=torch.tensor([[0, 0]]),
    )

    output = zbl(
        systems,
        {"mtt::U0": ModelOutput(quantity="energy", unit="", per_atom=True)},
        selected_atoms=selected_atoms,
    )
    assert "mtt::U0" in output
    assert output["mtt::U0"].block().samples.names == ["system", "atom"]
    assert output["mtt::U0"].block().values.shape == (1, 1)

    # per_atom = False
    output = zbl(
        systems,
        {"mtt::U0": ModelOutput(quantity="energy", unit="", per_atom=False)},
    )
    assert "mtt::U0" in output
    assert output["mtt::U0"].block().samples.names == ["system"]
    assert output["mtt::U0"].block().values.shape == (5, 1)

    # check that the result is the same without batching
    expected = output["mtt::U0"].block().values[3]
    system = systems[3]
    output = zbl(
        [system],
        {"mtt::U0": ModelOutput(quantity="energy", unit="", per_atom=False)},
    )
    assert torch.allclose(output["mtt::U0"].block().values[0], expected)


@pytest.mark.parametrize("where_is_center_type", ["samples", "nowhere"])
def test_composition_model_train_per_atom(where_is_center_type):
    """Test the calculation of composition weights for a per-atom scalar."""

    # Here we use two synthetic structures:
    # - O atom, with an energy of 1.0
    # - H2O molecule, with energies of 1.0, 1.5, 2.0
    # The expected composition weights are 2.0 for H and 1.0 for O.

    systems = [
        System(
            positions=torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float64),
            types=torch.tensor([8]),
            cell=torch.eye(3, dtype=torch.float64),
            pbc=torch.tensor([True, True, True]),
        ),
        System(
            positions=torch.tensor(
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=torch.float64
            ),
            types=torch.tensor([1, 1, 8]),
            cell=torch.eye(3, dtype=torch.float64),
            pbc=torch.tensor([True, True, True]),
        ),
    ]
    tensor_map_1 = TensorMap(
        keys=Labels(names=["center_type"], values=torch.tensor([[8]])),
        blocks=[
            TensorBlock(
                values=torch.tensor([[1.0]], dtype=torch.float64),
                samples=Labels(names=["system", "atom"], values=torch.tensor([[0, 0]])),
                components=[],
                properties=Labels(names=["energy"], values=torch.tensor([[0]])),
            ),
        ],
    )
    tensor_map_2 = TensorMap(
        keys=Labels(names=["center_type"], values=torch.tensor([[1], [8]])),
        blocks=[
            TensorBlock(
                values=torch.tensor([[1.0], [1.5]], dtype=torch.float64),
                samples=Labels(
                    names=["system", "atom"],
                    values=torch.tensor([[1, 0], [1, 1]]),
                ),
                components=[],
                properties=Labels(names=["energy"], values=torch.tensor([[0]])),
            ),
            TensorBlock(
                values=torch.tensor([[2.0]], dtype=torch.float64),
                samples=Labels(
                    names=["system", "atom"],
                    values=torch.tensor([[1, 2]]),
                ),
                components=[],
                properties=Labels(names=["energy"], values=torch.tensor([[0]])),
            ),
        ],
    )
    if where_is_center_type in ["samples", "nowhere"]:
        tensor_map_1 = tensor_map_1.keys_to_samples("center_type")
        tensor_map_2 = tensor_map_2.keys_to_samples("center_type")
    if where_is_center_type == "nowhere":
        tensor_map_1 = mts.remove_dimension(tensor_map_1, "samples", "center_type")
        tensor_map_2 = mts.remove_dimension(tensor_map_2, "samples", "center_type")

    energies = [tensor_map_1, tensor_map_2]
    dataset = Dataset.from_dict({"system": systems, "energy": energies})

    composition_model = CompositionModel(
        hypers={},
        dataset_info=DatasetInfo(
            length_unit="angstrom",
            atomic_types=[1, 8],
            targets={
                "energy": get_generic_target_info(
                    "energy",
                    {
                        "quantity": "energy",
                        "unit": "",
                        "type": "scalar",
                        "num_subtargets": 1,
                        "per_atom": True,
                    },
                )
            },
        ),
    )

    system_H = System(
        positions=torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float64),
        types=torch.tensor([1]),
        cell=torch.eye(3, dtype=torch.float64),
        pbc=torch.tensor([True, True, True]),
    )
    system_O = System(
        positions=torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float64),
        types=torch.tensor([8]),
        cell=torch.eye(3, dtype=torch.float64),
        pbc=torch.tensor([True, True, True]),
    )

    composition_model.train_model([dataset], [], batch_size=1, is_distributed=False)
    assert composition_model.atomic_types == [1, 8]
    output_H = composition_model(
        [system_H], {"energy": ModelOutput(quantity="energy", unit="", per_atom=False)}
    )
    torch.testing.assert_close(
        output_H["energy"].block().values, torch.tensor([[1.25]], dtype=torch.float64)
    )
    output_O = composition_model(
        [system_O], {"energy": ModelOutput(quantity="energy", unit="", per_atom=False)}
    )
    torch.testing.assert_close(
        output_O["energy"].block().values, torch.tensor([[1.5]], dtype=torch.float64)
    )


def test_composition_many_subtargets():
    """Test the calculation of composition weights for a per-structure scalar."""

    # Here we use three synthetic structures, each with 2 energies:
    # - O atom, with energies of 1.0, 0.0
    # - H2O molecule, with energies of 5.0, 1.0
    # - H4O2 molecule, with energies of 10.0, 2.0
    # The expected composition weights are 2.0, 0.5 for H and 1.0, 0.0 for O.

    systems = [
        System(
            positions=torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float64),
            types=torch.tensor([8]),
            cell=torch.eye(3, dtype=torch.float64),
            pbc=torch.tensor([True, True, True]),
        ),
        System(
            positions=torch.tensor(
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=torch.float64
            ),
            types=torch.tensor([1, 1, 8]),
            cell=torch.eye(3, dtype=torch.float64),
            pbc=torch.tensor([True, True, True]),
        ),
        System(
            positions=torch.tensor(
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                    [1.0, 0.0, 1.0],
                    [0.0, 1.0, 1.0],
                ],
                dtype=torch.float64,
            ),
            types=torch.tensor([1, 1, 8, 1, 1, 8]),
            cell=torch.eye(3, dtype=torch.float64),
            pbc=torch.tensor([True, True, True]),
        ),
    ]
    energies = [[1.0, 0.0], [5.0, 1.0], [10.0, 2.0]]
    energies = [
        TensorMap(
            keys=Labels(names=["_"], values=torch.tensor([[0]])),
            blocks=[
                TensorBlock(
                    values=torch.tensor([e], dtype=torch.float64),
                    samples=Labels(names=["system"], values=torch.tensor([[i]])),
                    components=[],
                    properties=Labels(
                        names=["energy"], values=torch.tensor([[0], [1]])
                    ),
                )
            ],
        )
        for i, e in enumerate(energies)
    ]
    dataset = Dataset.from_dict({"system": systems, "energy": energies})

    composition_model = CompositionModel(
        hypers={},
        dataset_info=DatasetInfo(
            length_unit="angstrom",
            atomic_types=[1, 8],
            targets={
                "energy": get_generic_target_info(
                    "energy",
                    {
                        "quantity": "energy",
                        "unit": "",
                        "type": "scalar",
                        "num_subtargets": 2,
                        "per_atom": False,
                    },
                )
            },
        ),
    )

    system_H = System(
        positions=torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float64),
        types=torch.tensor([1]),
        cell=torch.eye(3, dtype=torch.float64),
        pbc=torch.tensor([True, True, True]),
    )
    system_O = System(
        positions=torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float64),
        types=torch.tensor([8]),
        cell=torch.eye(3, dtype=torch.float64),
        pbc=torch.tensor([True, True, True]),
    )

    composition_model.train_model([dataset], [], batch_size=1, is_distributed=False)
    assert composition_model.atomic_types == [1, 8]
    output_H = composition_model(
        [system_H], {"energy": ModelOutput(quantity="energy", unit="", per_atom=False)}
    )
    torch.testing.assert_close(
        output_H["energy"].block().values,
        torch.tensor([[2.0, 0.5]], dtype=torch.float64),
    )
    output_O = composition_model(
        [system_O], {"energy": ModelOutput(quantity="energy", unit="", per_atom=False)}
    )
    torch.testing.assert_close(
        output_O["energy"].block().values,
        torch.tensor([[1.0, 0.0]], dtype=torch.float64),
    )


def test_composition_spherical():
    """
    Test the calculation of composition weights for a per-structure spherical target.
    """

    # Here we use three synthetic structures, each with an invariant target and random
    # spherical targets with L=1:
    # - O atom, with an invariant target of 1.0
    # - H2O molecule, with an invariant target of 5.0
    # - H4O2 molecule, with an invariant target of 10.0
    # The expected composition weights are 2.0 for H and 1.0 for O.

    systems = [
        System(
            positions=torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float64),
            types=torch.tensor([8]),
            cell=torch.eye(3, dtype=torch.float64),
            pbc=torch.tensor([True, True, True]),
        ),
        System(
            positions=torch.tensor(
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=torch.float64
            ),
            types=torch.tensor([1, 1, 8]),
            cell=torch.eye(3, dtype=torch.float64),
            pbc=torch.tensor([True, True, True]),
        ),
        System(
            positions=torch.tensor(
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                    [1.0, 0.0, 1.0],
                    [0.0, 1.0, 1.0],
                ],
                dtype=torch.float64,
            ),
            types=torch.tensor([1, 1, 8, 1, 1, 8]),
            cell=torch.eye(3, dtype=torch.float64),
            pbc=torch.tensor([True, True, True]),
        ),
    ]
    energies = [1.0, 5.0, 10.0]
    energies = [
        TensorMap(
            keys=Labels(
                names=["o3_lambda", "o3_sigma"], values=torch.tensor([[0, 1], [1, 1]])
            ),
            blocks=[
                TensorBlock(
                    values=torch.tensor([[[e]]], dtype=torch.float64),
                    samples=Labels(names=["system"], values=torch.tensor([[i]])),
                    components=[
                        Labels(
                            names=["o3_mu"],
                            values=torch.arange(0, 1).reshape(-1, 1),
                        )
                    ],
                    properties=Labels(names=["energy"], values=torch.tensor([[0]])),
                ),
                TensorBlock(
                    values=torch.randn((1, 3, 1), dtype=torch.float64),
                    samples=Labels(names=["system"], values=torch.tensor([[i]])),
                    components=[
                        Labels(
                            names=["o3_mu"],
                            values=torch.arange(-1, 2).reshape(-1, 1),
                        )
                    ],
                    properties=Labels(names=["energy"], values=torch.tensor([[0]])),
                ),
            ],
        )
        for i, e in enumerate(energies)
    ]
    dataset = Dataset.from_dict({"system": systems, "energy": energies})

    composition_model = CompositionModel(
        hypers={},
        dataset_info=DatasetInfo(
            length_unit="angstrom",
            atomic_types=[1, 8],
            targets={
                "energy": get_generic_target_info(
                    "energy",
                    {
                        "quantity": "energy",
                        "unit": "",
                        "type": {
                            "spherical": {
                                "irreps": [
                                    {"o3_lambda": 0, "o3_sigma": 1},
                                    {"o3_lambda": 1, "o3_sigma": 1},
                                ]
                            }
                        },
                        "num_subtargets": 1,
                        "per_atom": False,
                    },
                )
            },
        ),
    )

    system_H = System(
        positions=torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float64),
        types=torch.tensor([1]),
        cell=torch.eye(3, dtype=torch.float64),
        pbc=torch.tensor([True, True, True]),
    )
    system_O = System(
        positions=torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float64),
        types=torch.tensor([8]),
        cell=torch.eye(3, dtype=torch.float64),
        pbc=torch.tensor([True, True, True]),
    )

    composition_model.train_model([dataset], [], batch_size=1, is_distributed=False)
    assert composition_model.atomic_types == [1, 8]
    output_H = composition_model(
        [system_H], {"energy": ModelOutput(quantity="energy", unit="", per_atom=False)}
    )
    torch.testing.assert_close(
        output_H["energy"].block({"o3_lambda": 0}).values,
        torch.tensor([[[2.0]]], dtype=torch.float64),
    )
    output_O = composition_model(
        [system_O], {"energy": ModelOutput(quantity="energy", unit="", per_atom=False)}
    )
    torch.testing.assert_close(
        output_O["energy"].block({"o3_lambda": 0}).values,
        torch.tensor([[[1.0]]], dtype=torch.float64),
    )


@pytest.mark.parametrize("missing_type", [False, True])
def test_composition_spherical_atomic_basis(missing_type):
    """Test the calculation of composition weights for a spherical
    target that is on an atomic basis (and per atom).

    :param missing_type: whether to include set up `DatasetInfo` with an atomic type
       that is not present in the dataset, to test that it is correctly ignored
       and does not cause an error.
    """

    # Here we use two synthetic structures:
    # - O atom, with a scalar of 1.0
    # - H2O molecule, with scalars of 1.0, 1.5, 2.0
    # The expected composition weights are 1.25 for H and 1.5 for O.

    systems = [
        System(
            positions=torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float64),
            types=torch.tensor([8]),
            cell=torch.eye(3, dtype=torch.float64),
            pbc=torch.tensor([True, True, True]),
        ),
        System(
            positions=torch.tensor(
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=torch.float64
            ),
            types=torch.tensor([1, 1, 8]),
            cell=torch.eye(3, dtype=torch.float64),
            pbc=torch.tensor([True, True, True]),
        ),
    ]
    tensor_map_1 = TensorMap(
        keys=Labels(
            names=["o3_lambda", "o3_sigma", "atom_type"],
            values=torch.tensor([[0, 1, 8], [1, 1, 8]]),
        ),
        blocks=[
            TensorBlock(
                values=torch.tensor([[1.0]], dtype=torch.float64).reshape(-1, 1, 1),
                samples=Labels(names=["system", "atom"], values=torch.tensor([[0, 0]])),
                components=[
                    Labels(
                        names=["o3_mu"],
                        values=torch.tensor([[0]]),
                    )
                ],
                properties=Labels(names=["_"], values=torch.tensor([[0]])),
            ),
            TensorBlock(
                values=torch.randn((1, 3, 1), dtype=torch.float64),
                samples=Labels(names=["system", "atom"], values=torch.tensor([[0, 0]])),
                components=[
                    Labels(
                        names=["o3_mu"],
                        values=torch.arange(-1, 2).reshape(-1, 1),
                    )
                ],
                properties=Labels(names=["_"], values=torch.tensor([[0]])),
            ),
        ],
    )
    tensor_map_2 = TensorMap(
        keys=Labels(
            names=["o3_lambda", "o3_sigma", "atom_type"],
            values=torch.tensor([[0, 1, 1], [0, 1, 8], [1, 1, 8]]),
        ),
        blocks=[
            TensorBlock(
                values=torch.tensor([[1.0], [1.5]], dtype=torch.float64).reshape(
                    -1, 1, 1
                ),
                samples=Labels(
                    names=["system", "atom"],
                    values=torch.tensor([[1, 0], [1, 1]]),
                ),
                components=[
                    Labels(
                        names=["o3_mu"],
                        values=torch.tensor([[0]]),
                    )
                ],
                properties=Labels(names=["_"], values=torch.tensor([[0]])),
            ),
            TensorBlock(
                values=torch.tensor([[2.0]], dtype=torch.float64).reshape(-1, 1, 1),
                samples=Labels(
                    names=["system", "atom"],
                    values=torch.tensor([[1, 2]]),
                ),
                components=[
                    Labels(
                        names=["o3_mu"],
                        values=torch.tensor([[0]]),
                    )
                ],
                properties=Labels(names=["_"], values=torch.tensor([[0]])),
            ),
            TensorBlock(
                values=torch.randn((1, 3, 1), dtype=torch.float64),
                samples=Labels(
                    names=["system", "atom"],
                    values=torch.tensor([[1, 2]]),
                ),
                components=[
                    Labels(
                        names=["o3_mu"],
                        values=torch.arange(-1, 2).reshape(-1, 1),
                    )
                ],
                properties=Labels(names=["_"], values=torch.tensor([[0]])),
            ),
        ],
    )

    dataset = Dataset.from_dict(
        {"system": systems, "spherical_atomic_basis": [tensor_map_1, tensor_map_2]}
    )

    # Set up basis.
    atomic_types = [1, 8]
    irreps = {
        1: [
            {"o3_lambda": 0, "o3_sigma": 1},
        ],
        8: [
            {"o3_lambda": 0, "o3_sigma": 1},
            {"o3_lambda": 1, "o3_sigma": 1},
        ],
    }
    # Add missing type to the basis.
    if missing_type:
        atomic_types.append(9)
        irreps[9] = [
            {"o3_lambda": 0, "o3_sigma": 1},
        ]

    composition_model = CompositionModel(
        hypers={},
        dataset_info=DatasetInfo(
            length_unit="angstrom",
            atomic_types=atomic_types,
            targets={
                "spherical_atomic_basis": get_generic_target_info(
                    "spherical_atomic_basis",
                    {
                        "quantity": "",
                        "unit": "",
                        "type": {"spherical": {"irreps": irreps}},
                        "num_subtargets": 1,
                        "per_atom": True,
                    },
                )
            },
        ),
    )

    composition_model.train_model([dataset], [], batch_size=1, is_distributed=False)
    assert composition_model.atomic_types == atomic_types
    output = composition_model(
        [systems[1]], {"spherical_atomic_basis": ModelOutput(per_atom=True)}
    )

    H_block = output["spherical_atomic_basis"].block({"atom_type": 1})
    O_block = output["spherical_atomic_basis"].block({"atom_type": 8})

    # Check that the composition weights are correct for both H and O.
    torch.testing.assert_close(
        H_block.values,
        torch.tensor([1.25, 1.25], dtype=torch.float64).reshape(-1, 1, 1),
    )
    torch.testing.assert_close(
        O_block.values, torch.tensor([1.5], dtype=torch.float64).reshape(-1, 1, 1)
    )

    if missing_type:
        # Check that if we pass a system with the missing type, we get a zero
        # contribution from the composition model.
        system_F = System(
            positions=torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float64),
            types=torch.tensor([9]),
            cell=torch.eye(3, dtype=torch.float64),
            pbc=torch.tensor([True, True, True]),
        )
        output_F = composition_model(
            [system_F], {"spherical_atomic_basis": ModelOutput(per_atom=True)}
        )
        F_block = output_F["spherical_atomic_basis"].block({"atom_type": 9})
        torch.testing.assert_close(
            F_block.values, torch.tensor([0.0], dtype=torch.float64).reshape(-1, 1, 1)
        )


def test_composition_spherical_atomic_basis_dense():
    """
    Test the composition weights for an atomic-basis spherical target in the
    dense representation are correct.

    With two H atoms (invariant values 2.0 and 4.0 on H's single basis function) and two
    O atoms (invariant values (3.0, 5.0) and (7.0, 9.0) on O's two basis functions), the
    expected composition weights are:

        W[H, n=0] = mean(2.0, 4.0) = 3.0  (H data, H's single basis)
        W[H, n=0] = NaN
        W[O, n=0] = mean(3.0, 7.0) = 5.0  (O data, O's first basis)
        W[O, n=1] = mean(5.0, 9.0) = 7.0  (O data, O's second basis)
    """
    atomic_types = [1, 8]

    # Build targets in the sparse representation with ``atom_type`` as a key
    # dimension.
    components = [Labels(names=["o3_mu"], values=torch.tensor([[0]]))]

    tensor_map_1 = TensorMap(
        keys=Labels(
            names=["o3_lambda", "o3_sigma", "atom_type"],
            values=torch.tensor([[0, 1, 1]]),
        ),
        blocks=[
            TensorBlock(
                values=torch.tensor(
                    [[[2.0]], [[4.0]]], dtype=torch.float64
                ),  # two H atoms, single invariant basis
                samples=Labels(
                    names=["system", "atom"],
                    values=torch.tensor([[0, 0], [0, 1]]),
                ),
                components=components,
                properties=Labels(names=["n"], values=torch.tensor([[0]])),
            ),
        ],
    )
    tensor_map_2 = TensorMap(
        keys=Labels(
            names=["o3_lambda", "o3_sigma", "atom_type"],
            values=torch.tensor([[0, 1, 8]]),
        ),
        blocks=[
            TensorBlock(
                values=torch.tensor(
                    [[[3.0, 5.0]], [[7.0, 9.0]]], dtype=torch.float64
                ),  # two O atoms, two invariant bases each
                samples=Labels(
                    names=["system", "atom"],
                    values=torch.tensor([[1, 0], [1, 1]]),
                ),
                components=components,
                properties=Labels(names=["n"], values=torch.tensor([[0], [1]])),
            ),
        ],
    )

    systems = [
        System(
            positions=torch.zeros((2, 3), dtype=torch.float64),
            types=torch.tensor([1, 1]),
            cell=torch.eye(3, dtype=torch.float64),
            pbc=torch.tensor([True, True, True]),
        ),
        System(
            positions=torch.zeros((2, 3), dtype=torch.float64),
            types=torch.tensor([8, 8]),
            cell=torch.eye(3, dtype=torch.float64),
            pbc=torch.tensor([True, True, True]),
        ),
    ]

    # ``mtt::aux::system_index`` is auto-added by DiskDataset but not by
    # ``Dataset.from_dict``; the atomic-basis densification transform in the
    # collate function asserts on its presence, so we add it explicitly here.
    system_indices = [
        TensorMap(
            keys=Labels(names=["_"], values=torch.tensor([[0]])),
            blocks=[
                TensorBlock(
                    values=torch.tensor([[i]], dtype=torch.float64),
                    samples=Labels(names=["system"], values=torch.tensor([[i]])),
                    components=[],
                    properties=Labels(names=["_"], values=torch.tensor([[0]])),
                )
            ],
        )
        for i in range(len(systems))
    ]
    dataset = Dataset.from_dict(
        {
            "system": systems,
            "spherical_atomic_basis": [tensor_map_1, tensor_map_2],
            "mtt::aux::system_index": system_indices,
        }
    )

    # H has 1 invariant basis function; O has 2 invariant basis functions.
    irreps = {
        1: [{"o3_lambda": 0, "o3_sigma": 1, "num": 1}],
        8: [{"o3_lambda": 0, "o3_sigma": 1, "num": 2}],
    }

    dataset_info = DatasetInfo(
        length_unit="angstrom",
        atomic_types=atomic_types,
        targets={
            "spherical_atomic_basis": get_generic_target_info(
                "spherical_atomic_basis",
                {
                    "quantity": "",
                    "unit": "",
                    "type": {"spherical": {"irreps": irreps}},
                    "num_subtargets": 1,
                    "per_atom": True,
                },
            )
        },
    )

    # Densify targets to the dense representation
    atomic_basis_transform, _ = get_prepare_atomic_basis_targets_transform(
        dataset_info.targets, {}
    )
    dataset_info_dense = densify_atomic_basis_dataset_info(dataset_info)

    composition_model = CompositionModel(
        hypers={},
        dataset_info=dataset_info_dense,
    )
    composition_model.train_model(
        [dataset],
        [],
        batch_size=2,
        is_distributed=False,
        initial_transforms=[atomic_basis_transform],
    )

    # Check weights
    weight_block = composition_model.model.weights["spherical_atomic_basis"].block(
        {"o3_lambda": 0, "o3_sigma": 1}
    )
    W = weight_block.values
    nan = float("nan")
    expected = torch.tensor(
        [
            [[3.0, nan]],
            [[5.0, 7.0]],
        ],
        dtype=torch.float64,
    )
    print("Composition weights:\n", W)
    print("Expected weights:\n", expected)
    torch.testing.assert_close(W, expected, rtol=1e-10, atol=1e-10, equal_nan=True)

    # Check predictions
    predictions = composition_model(
        systems,
        outputs={"spherical_atomic_basis": ModelOutput(per_atom=True)},
    )
    pred_values = (
        predictions["spherical_atomic_basis"]
        .block({"o3_lambda": 0, "o3_sigma": 1})
        .values
    )
    expected_preds = torch.tensor(
        [
            [[3.0, nan]],
            [[3.0, nan]],
            [[5.0, 7.0]],
            [[5.0, 7.0]],
        ],
        dtype=torch.float64,
    )
    print("Predicted values:\n", pred_values)
    print("Expected values:\n", expected_preds)
    torch.testing.assert_close(
        pred_values, expected_preds, rtol=1e-10, atol=1e-10, equal_nan=True
    )


def test_composition_atomic_basis_sparse_dense_consistency():
    """
    Fit composition model in both the sparse and dense representations for an atomic
    basis target, and check for consistency.
    """
    atomic_types = [1, 8]
    n_atoms_per_type = 3

    h_values = torch.tensor([[2.0], [4.0], [6.0]], dtype=torch.float64)
    o_values = torch.tensor([[3.0, 5.0], [7.0, 9.0], [11.0, 13.0]], dtype=torch.float64)

    systems = [
        System(
            positions=torch.zeros((2 * n_atoms_per_type, 3), dtype=torch.float64),
            types=torch.tensor([1, 1, 1, 8, 8, 8]),
            cell=torch.eye(3, dtype=torch.float64),
            pbc=torch.tensor([True, True, True]),
        )
    ]
    components = [Labels(names=["o3_mu"], values=torch.tensor([[0]]))]

    sparse_keys = Labels(
        names=["o3_lambda", "o3_sigma", "atom_type"],
        values=torch.tensor([[0, 1, 1], [0, 1, 8]]),
    )
    target = TensorMap(
        sparse_keys,
        [
            TensorBlock(
                values=h_values.reshape(-1, 1, 1),
                samples=Labels(
                    names=["system", "atom"],
                    values=torch.tensor([[0, 0], [0, 1], [0, 2]]),
                ),
                components=components,
                properties=Labels(names=["n"], values=torch.tensor([[0]])),
            ),
            TensorBlock(
                values=o_values.reshape(-1, 1, 2),
                samples=Labels(
                    names=["system", "atom"],
                    values=torch.tensor([[0, 3], [0, 4], [0, 5]]),
                ),
                components=components,
                properties=Labels(names=["n"], values=torch.tensor([[0], [1]])),
            ),
        ],
    )
    # ``mtt::aux::system_index`` is needed for densifying atomic basis targets.
    system_indices = [
        TensorMap(
            keys=Labels(names=["_"], values=torch.tensor([[0]])),
            blocks=[
                TensorBlock(
                    values=torch.tensor([[i]], dtype=torch.float64),
                    samples=Labels(names=["system"], values=torch.tensor([[i]])),
                    components=[],
                    properties=Labels(names=["_"], values=torch.tensor([[0]])),
                )
            ],
        )
        for i in range(len(systems))
    ]
    dataset = Dataset.from_dict(
        {
            "system": systems,
            "target": [target],
            "mtt::aux::system_index": system_indices,
        }
    )

    # Target-info: H with a single invariant basis, O with two invariant bases.
    irreps = {
        1: [{"o3_lambda": 0, "o3_sigma": 1, "num": 1}],
        8: [{"o3_lambda": 0, "o3_sigma": 1, "num": 2}],
    }
    dataset_info = DatasetInfo(
        length_unit="angstrom",
        atomic_types=atomic_types,
        targets={
            "target": get_generic_target_info(
                "target",
                {
                    "quantity": "",
                    "unit": "",
                    "type": {"spherical": {"irreps": irreps}},
                    "num_subtargets": 1,
                    "per_atom": True,
                },
            )
        },
    )

    # Fit sparse and dense CompositionModels on the same dataset.
    sparse_model = CompositionModel(
        hypers={},
        dataset_info=dataset_info,
    )
    sparse_model.train_model(
        [dataset], [], batch_size=1, is_distributed=False, initial_transforms=[]
    )

    # Densify targets to the dense representation
    atomic_basis_transform, _ = get_prepare_atomic_basis_targets_transform(
        dataset_info.targets, {}
    )
    dataset_info_dense = densify_atomic_basis_dataset_info(dataset_info)

    dense_model = CompositionModel(
        hypers={},
        dataset_info=dataset_info_dense,
    )
    dense_model.train_model(
        [dataset],
        [],
        batch_size=1,
        is_distributed=False,
        initial_transforms=[atomic_basis_transform],
    )

    # Get the weights
    w_sparse_H = (
        sparse_model.model.weights["target"]
        .block({"o3_lambda": 0, "o3_sigma": 1, "atom_type": 1})
        .values
    )
    w_sparse_O = (
        sparse_model.model.weights["target"]
        .block({"o3_lambda": 0, "o3_sigma": 1, "atom_type": 8})
        .values
    )
    w_dense = (
        dense_model.model.weights["target"]
        .block({"o3_lambda": 0, "o3_sigma": 1})
        .values
    )

    h_index = 0
    o_index = 1

    # Consistency checks
    torch.testing.assert_close(
        w_dense[h_index, 0, 0],
        w_sparse_H[h_index, 0, 0],
        rtol=1e-10,
        atol=1e-10,
    )
    torch.testing.assert_close(
        w_dense[o_index, 0, :],
        w_sparse_O[o_index, 0, :],
        rtol=1e-10,
        atol=1e-10,
    )
    assert torch.isnan(w_dense[h_index, 0, 1])

    # Finally check the actual values of the sparse weights
    torch.testing.assert_close(
        w_sparse_H[h_index, 0, 0],
        h_values.mean(),
        rtol=1e-10,
        atol=1e-10,
    )
    torch.testing.assert_close(
        w_sparse_O[o_index, 0, :],
        o_values.mean(dim=0),
        rtol=1e-10,
        atol=1e-10,
    )


def test_composition_spherical_atomic_basis_dense_nan_weights():
    """
    Test that when fitting an atomic-basis spherical target in the *dense*
    representation, produces ``NaN`` composition weights for exactly the properties
    whose target values were NaN-padded for the given atom types during densification.
    """
    atomic_types = [1, 8]
    components = [Labels(names=["o3_mu"], values=torch.tensor([[0]]))]

    # Structure 1: one O atom with its two-basis invariant values.
    tensor_map_1 = TensorMap(
        keys=Labels(
            names=["o3_lambda", "o3_sigma", "atom_type"],
            values=torch.tensor([[0, 1, 8]]),
        ),
        blocks=[
            TensorBlock(
                values=torch.tensor([[[1.0, 2.0]]], dtype=torch.float64),
                samples=Labels(names=["system", "atom"], values=torch.tensor([[0, 0]])),
                components=components,
                properties=Labels(names=["n"], values=torch.tensor([[0], [1]])),
            ),
        ],
    )

    # Structure 2: H2O molecule — two H and one O, each on their own bases.
    tensor_map_2 = TensorMap(
        keys=Labels(
            names=["o3_lambda", "o3_sigma", "atom_type"],
            values=torch.tensor([[0, 1, 1], [0, 1, 8]]),
        ),
        blocks=[
            TensorBlock(
                values=torch.tensor(
                    [[[1.0]], [[1.5]]], dtype=torch.float64
                ),  # two H atoms, single invariant basis
                samples=Labels(
                    names=["system", "atom"],
                    values=torch.tensor([[1, 0], [1, 1]]),
                ),
                components=components,
                properties=Labels(names=["n"], values=torch.tensor([[0]])),
            ),
            TensorBlock(
                values=torch.tensor(
                    [[[3.0, 4.0]]], dtype=torch.float64
                ),  # one O atom, two invariant bases
                samples=Labels(names=["system", "atom"], values=torch.tensor([[1, 2]])),
                components=components,
                properties=Labels(names=["n"], values=torch.tensor([[0], [1]])),
            ),
        ],
    )

    systems = [
        System(
            positions=torch.zeros((1, 3), dtype=torch.float64),
            types=torch.tensor([8]),
            cell=torch.eye(3, dtype=torch.float64),
            pbc=torch.tensor([True, True, True]),
        ),
        System(
            positions=torch.zeros((3, 3), dtype=torch.float64),
            types=torch.tensor([1, 1, 8]),
            cell=torch.eye(3, dtype=torch.float64),
            pbc=torch.tensor([True, True, True]),
        ),
    ]
    # ``mtt::aux::system_index`` is needed for densifying atomic basis targets.
    system_indices = [
        TensorMap(
            keys=Labels(names=["_"], values=torch.tensor([[0]])),
            blocks=[
                TensorBlock(
                    values=torch.tensor([[i]], dtype=torch.float64),
                    samples=Labels(names=["system"], values=torch.tensor([[i]])),
                    components=[],
                    properties=Labels(names=["_"], values=torch.tensor([[0]])),
                )
            ],
        )
        for i in range(len(systems))
    ]
    dataset = Dataset.from_dict(
        {
            "system": systems,
            "spherical_atomic_basis": [tensor_map_1, tensor_map_2],
            "mtt::aux::system_index": system_indices,
        }
    )

    # H has 1 invariant basis; O has 2 invariant bases.
    irreps = {
        1: [{"o3_lambda": 0, "o3_sigma": 1, "num": 1}],
        8: [{"o3_lambda": 0, "o3_sigma": 1, "num": 2}],
    }

    dataset_info = DatasetInfo(
        length_unit="angstrom",
        atomic_types=atomic_types,
        targets={
            "spherical_atomic_basis": get_generic_target_info(
                "spherical_atomic_basis",
                {
                    "quantity": "",
                    "unit": "",
                    "type": {"spherical": {"irreps": irreps}},
                    "num_subtargets": 1,
                    "per_atom": True,
                },
            )
        },
    )
    # Densify targets to the dense representation
    atomic_basis_transform, _ = get_prepare_atomic_basis_targets_transform(
        dataset_info.targets, {}
    )
    dataset_info_dense = densify_atomic_basis_dataset_info(dataset_info)

    composition_model = CompositionModel(
        hypers={},
        dataset_info=dataset_info_dense,
    )
    composition_model.train_model(
        [dataset],
        [],
        batch_size=1,
        is_distributed=False,
        initial_transforms=[atomic_basis_transform],
    )

    # Get weights
    W = (
        composition_model.model.weights["spherical_atomic_basis"]
        .block({"o3_lambda": 0, "o3_sigma": 1})
        .values
    )
    h_index, o_index = 0, 1

    nan_mask_expected = torch.tensor(
        [
            [[False, True]],  # H: n=0 is finite, n=1 is NaN
            [[False, False]],  # O: both positions finite
        ]
    )
    assert torch.equal(torch.isnan(W), nan_mask_expected)

    # Check non-NaN weights
    torch.testing.assert_close(
        W[h_index, 0, 0],
        torch.tensor((1.0 + 1.5) / 2, dtype=torch.float64),
        rtol=1e-10,
        atol=1e-10,
    )
    torch.testing.assert_close(
        W[o_index, 0, :],
        torch.tensor([(1.0 + 3.0) / 2, (2.0 + 4.0) / 2], dtype=torch.float64),
        rtol=1e-10,
        atol=1e-10,
    )

    # Get predictions
    predictions = composition_model(
        [systems[1]],
        outputs={"spherical_atomic_basis": ModelOutput(per_atom=True)},
    )
    pred_values = (
        predictions["spherical_atomic_basis"]
        .block({"o3_lambda": 0, "o3_sigma": 1})
        .values
    )
    pred_nan_mask_expected = torch.tensor(
        [
            [[False, True]],  # H atom 0
            [[False, True]],  # H atom 1
            [[False, False]],  # O atom
        ]
    )
    assert torch.equal(torch.isnan(pred_values), pred_nan_mask_expected)

    # Check predictions
    torch.testing.assert_close(
        pred_values[0, 0, 0],
        torch.tensor(1.25, dtype=torch.float64),
        rtol=1e-10,
        atol=1e-10,
    )
    torch.testing.assert_close(
        pred_values[1, 0, 0],
        torch.tensor(1.25, dtype=torch.float64),
        rtol=1e-10,
        atol=1e-10,
    )
    torch.testing.assert_close(
        pred_values[2, 0, :],
        torch.tensor([2.0, 3.0], dtype=torch.float64),
        rtol=1e-10,
        atol=1e-10,
    )



def test_composition_spherical_per_atom_rank_2():
    """
    Test the calculation of composition weights for a spherical per-atom rank 2 target
    (keys: o3_lambda_1, o3_lambda_2, o3_sigma_1, o3_sigma_2) is correct.

    All atoms contribute to the same blocks,
    so the composition model fits a single weight per atomic type.
    """

    systems = [
        System(
            positions=torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float64),
            types=torch.tensor([8]),
            cell=torch.eye(3, dtype=torch.float64),
            pbc=torch.tensor([True, True, True]),
        ),
        System(
            positions=torch.tensor(
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                dtype=torch.float64,
            ),
            types=torch.tensor([1, 1, 8]),
            cell=torch.eye(3, dtype=torch.float64),
            pbc=torch.tensor([True, True, True]),
        ),
    ]

    pp_full_sys1 = torch.zeros(1, 3, 3, 1, dtype=torch.float64)
    pp_full_sys1[0, torch.arange(3), torch.arange(3), 0] = torch.tensor(
        [1.0, 2.0, 3.0], dtype=torch.float64
    )
    pp_full_sys2 = torch.zeros(3, 3, 3, 1, dtype=torch.float64)
    pp_full_sys2[2, torch.arange(3), torch.arange(3), 0] = torch.tensor(
        [3.0, 4.0, 5.0], dtype=torch.float64
    )

    tensor_map_1 = TensorMap(
        keys=Labels(
            names=["o3_lambda_1", "o3_lambda_2", "o3_sigma_1", "o3_sigma_2"],
            values=torch.tensor([[0, 0, 1, 1], [1, 1, 1, 1]]),
        ),
        blocks=[
            TensorBlock(
                values=torch.tensor([[[[1.0]]]], dtype=torch.float64),
                samples=Labels(names=["system", "atom"], values=torch.tensor([[0, 0]])),
                components=[
                    Labels(names=["o3_mu_1"], values=torch.tensor([[0]])),
                    Labels(names=["o3_mu_2"], values=torch.tensor([[0]])),
                ],
                properties=Labels(names=["_"], values=torch.tensor([[0]])),
            ),
            TensorBlock(
                values=pp_full_sys1,
                samples=Labels(names=["system", "atom"], values=torch.tensor([[0, 0]])),
                components=[
                    Labels(names=["o3_mu_1"], values=torch.arange(-1, 2).reshape(-1, 1)),
                    Labels(names=["o3_mu_2"], values=torch.arange(-1, 2).reshape(-1, 1)),
                ],
                properties=Labels(names=["_"], values=torch.tensor([[0]])),
            ),
        ],
    )

    ss_vals_sys2 = torch.tensor([[[[1.0]]], [[[1.5]]], [[[2.0]]]], dtype=torch.float64)
    tensor_map_2 = TensorMap(
        keys=Labels(
            names=["o3_lambda_1", "o3_lambda_2", "o3_sigma_1", "o3_sigma_2"],
            values=torch.tensor([[0, 0, 1, 1], [1, 1, 1, 1]]),
        ),
        blocks=[
            TensorBlock(
                values=ss_vals_sys2,
                samples=Labels(
                    names=["system", "atom"],
                    values=torch.tensor([[1, 0], [1, 1], [1, 2]]),
                ),
                components=[
                    Labels(names=["o3_mu_1"], values=torch.tensor([[0]])),
                    Labels(names=["o3_mu_2"], values=torch.tensor([[0]])),
                ],
                properties=Labels(names=["_"], values=torch.tensor([[0]])),
            ),
            TensorBlock(
                values=pp_full_sys2,
                samples=Labels(
                    names=["system", "atom"],
                    values=torch.tensor([[1, 0], [1, 1], [1, 2]]),
                ),
                components=[
                    Labels(names=["o3_mu_1"], values=torch.arange(-1, 2).reshape(-1, 1)),
                    Labels(names=["o3_mu_2"], values=torch.arange(-1, 2).reshape(-1, 1)),
                ],
                properties=Labels(names=["_"], values=torch.tensor([[0]])),
            ),
        ],
    )

    dataset = Dataset.from_dict(
        {"system": systems, "rank_2_target": [tensor_map_1, tensor_map_2]}
    )

    target_info = get_generic_target_info(
        "rank_2_target",
        {
            "quantity": "",
            "unit": "",
            "type": {
                "spherical": {
                    "irreps": {
                        1: [{"o3_lambda": 0, "o3_sigma": 1}],
                        8: [
                            {"o3_lambda": 0, "o3_sigma": 1},
                            {"o3_lambda": 1, "o3_sigma": 1},
                        ],
                    },
                    "product": "cartesian",
                }
            },
            "num_subtargets": 1,
            "per_atom": True,
        },
    )
    target_info.layout = _empty_tensor_map_like(tensor_map_1)

    composition_model = CompositionModel(
        hypers={},
        dataset_info=DatasetInfo(
            length_unit="angstrom",
            atomic_types=[1, 8],
            targets={"rank_2_target": target_info},
        ),
    )

    composition_model.train_model([dataset], [], batch_size=1, is_distributed=False)

    output = composition_model(
        [systems[1]], {"rank_2_target": ModelOutput(per_atom=True)}
    )

    ss_key = {"o3_lambda_1": 0, "o3_lambda_2": 0, "o3_sigma_1": 1, "o3_sigma_2": 1}
    pp_key = {"o3_lambda_1": 1, "o3_lambda_2": 1, "o3_sigma_1": 1, "o3_sigma_2": 1}

    ss_block = output["rank_2_target"].block(ss_key)
    torch.testing.assert_close(
        ss_block.values,
        torch.tensor(
            [1.25, 1.25, 1.5], dtype=torch.float64
        ).reshape(-1, 1, 1, 1),
    )

    pp_block = output["rank_2_target"].block(pp_key)
    expected_pp = torch.zeros(3, 3, 3, 1, dtype=torch.float64)
    expected_pp[2, torch.arange(3), torch.arange(3), 0] = torch.tensor(
        [3.0, 3.0, 3.0], dtype=torch.float64
    )
    torch.testing.assert_close(pp_block.values, expected_pp)

def test_composition_spherical_per_atom_rank_2_rotation_invariance():
    """
    Test the calculation of composition weights for a spherical per-atom rank 2 target
    (keys: o3_lambda_1, o3_lambda_2, o3_sigma_1, o3_sigma_2) is invariant under fitting
    on a rotated version of the dataset.
    """

    def Rz(theta):
        c, s = torch.cos(torch.tensor(theta)), torch.sin(torch.tensor(theta))
        return torch.tensor(
            [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float64
        )

    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=torch.float64
    )
    R = Rz(torch.pi / 2)
    positions_rotated = positions @ R.T

    system = System(
        positions=positions,
        types=torch.tensor([8, 1, 1]),
        cell=torch.eye(3, dtype=torch.float64),
        pbc=torch.tensor([True, True, True]),
    )
    system_rotated = System(
        positions=positions_rotated,
        types=torch.tensor([8, 1, 1]),
        cell=torch.eye(3, dtype=torch.float64),
        pbc=torch.tensor([True, True, True]),
    )

    ss_vals = torch.tensor([[[[1.0]]], [[[1.5]]], [[[2.0]]]], dtype=torch.float64)

    pp_full = torch.zeros(3, 3, 3, 1, dtype=torch.float64)
    pp_full[torch.arange(3), torch.arange(3), torch.arange(3), 0] = torch.tensor(
        [1.0, 2.0, 3.0], dtype=torch.float64
    )
    pp_vals = pp_full[..., 0]  
    pp_vals_rotated = torch.einsum("ac,bd,icd->iab", R, R, pp_vals).unsqueeze(-1)

    def make_tensor_map(ss_values, pp_values, system_idx):
        samples_3 = Labels(
            names=["system", "atom"],
            values=torch.tensor(
                [[system_idx, 0], [system_idx, 1], [system_idx, 2]]
            ),
        )
        return TensorMap(
            keys=Labels(
                names=["o3_lambda_1", "o3_lambda_2", "o3_sigma_1", "o3_sigma_2"],
                values=torch.tensor([[0, 0, 1, 1], [1, 1, 1, 1]]),
            ),
            blocks=[
                TensorBlock(
                    values=ss_values,
                    samples=samples_3,
                    components=[
                        Labels(names=["o3_mu_1"], values=torch.tensor([[0]])),
                        Labels(names=["o3_mu_2"], values=torch.tensor([[0]])),
                    ],
                    properties=Labels(names=["_"], values=torch.tensor([[0]])),
                ),
                TensorBlock(
                    values=pp_values,
                    samples=samples_3,
                    components=[
                        Labels(
                            names=["o3_mu_1"],
                            values=torch.arange(-1, 2).reshape(-1, 1),
                        ),
                        Labels(
                            names=["o3_mu_2"],
                            values=torch.arange(-1, 2).reshape(-1, 1),
                        ),
                    ],
                    properties=Labels(names=["_"], values=torch.tensor([[0]])),
                ),
            ],
        )

    target_info = get_generic_target_info(
        "rank_2_target",
        {
            "quantity": "",
            "unit": "",
            "type": {
                "spherical": {
                    "irreps": {
                        1: [{"o3_lambda": 0, "o3_sigma": 1}],
                        8: [
                            {"o3_lambda": 0, "o3_sigma": 1},
                            {"o3_lambda": 1, "o3_sigma": 1},
                        ],
                    },
                    "product": "cartesian",
                }
            },
            "num_subtargets": 1,
            "per_atom": True,
        },
    )
    target_info.layout = _empty_tensor_map_like(make_tensor_map(ss_vals, pp_full, 0))

    dataset_info = DatasetInfo(
        length_unit="angstrom",
        atomic_types=[1, 8],
        targets={"rank_2_target": target_info},
    )

    dataset_orig = Dataset.from_dict(
        {
            "system": [system],
            "rank_2_target": [make_tensor_map(ss_vals, pp_full, 0)],
        }
    )
    model_orig = CompositionModel(hypers={}, dataset_info=dataset_info)
    model_orig.train_model([dataset_orig], [], batch_size=1, is_distributed=False)

    dataset_rot = Dataset.from_dict(
        {
            "system": [system_rotated],
            "rank_2_target": [make_tensor_map(ss_vals, pp_vals_rotated, 0)],
        }
    )
    model_rot = CompositionModel(hypers={}, dataset_info=dataset_info)
    model_rot.train_model([dataset_rot], [], batch_size=1, is_distributed=False)

    weights_orig = model_orig.model.weights["rank_2_target"]
    weights_rot = model_rot.model.weights["rank_2_target"]

    ss_key = {"o3_lambda_1": 0, "o3_lambda_2": 0, "o3_sigma_1": 1, "o3_sigma_2": 1}
    pp_key = {"o3_lambda_1": 1, "o3_lambda_2": 1, "o3_sigma_1": 1, "o3_sigma_2": 1}

    torch.testing.assert_close(
        weights_orig.block(ss_key).values,
        weights_rot.block(ss_key).values,
    )
    torch.testing.assert_close(
        weights_orig.block(pp_key).values,
        weights_rot.block(pp_key).values,
    )

@pytest.mark.parametrize("missing_type", [False, True])
def test_composition_spherical_atomic_basis_rank_2(missing_type):
    """
    Test the calculation of composition weights for a spherical per-atom rank 2 atomic
    basis target (keys: o3_lambda_1, o3_sigma_1, o3_lambda_2, o3_sigma_2, atom_type) is
    correct.
    """

    systems = [
        System(
            positions=torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float64),
            types=torch.tensor([8]),
            cell=torch.eye(3, dtype=torch.float64),
            pbc=torch.tensor([True, True, True]),
        ),
        System(
            positions=torch.tensor(
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                dtype=torch.float64,
            ),
            types=torch.tensor([1, 1, 8]),
            cell=torch.eye(3, dtype=torch.float64),
            pbc=torch.tensor([True, True, True]),
        ),
    ]

    pp_full_sys1 = torch.zeros(1, 3, 3, 1, dtype=torch.float64)
    pp_full_sys1[0, torch.arange(3), torch.arange(3), 0] = torch.tensor(
        [1.0, 2.0, 3.0], dtype=torch.float64
    )
    pp_full_sys2 = torch.zeros(1, 3, 3, 1, dtype=torch.float64)
    pp_full_sys2[0, torch.arange(3), torch.arange(3), 0] = torch.tensor(
        [3.0, 4.0, 5.0], dtype=torch.float64
    )

    tensor_map_1 = TensorMap(
        keys=Labels(
            names=[
                "o3_lambda_1",
                "o3_lambda_2",
                "o3_sigma_1",
                "o3_sigma_2",
                "atom_type",
            ],
            values=torch.tensor([[0, 0, 1, 1, 8], [1, 1, 1, 1, 8]]),
        ),
        blocks=[
            TensorBlock(
                values=torch.tensor([[[[1.0]]]], dtype=torch.float64),
                samples=Labels(names=["system", "atom"], values=torch.tensor([[0, 0]])),
                components=[
                    Labels(names=["o3_mu_1"], values=torch.tensor([[0]])),
                    Labels(names=["o3_mu_2"], values=torch.tensor([[0]])),
                ],
                properties=Labels(names=["_"], values=torch.tensor([[0]])),
            ),
            TensorBlock(
                values=pp_full_sys1,
                samples=Labels(names=["system", "atom"], values=torch.tensor([[0, 0]])),
                components=[
                    Labels(
                        names=["o3_mu_1"], values=torch.arange(-1, 2).reshape(-1, 1)
                    ),
                    Labels(
                        names=["o3_mu_2"], values=torch.arange(-1, 2).reshape(-1, 1)
                    ),
                ],
                properties=Labels(names=["_"], values=torch.tensor([[0]])),
            ),
        ],
    )

    tensor_map_2 = TensorMap(
        keys=Labels(
            names=[
                "o3_lambda_1",
                "o3_lambda_2",
                "o3_sigma_1",
                "o3_sigma_2",
                "atom_type",
            ],
            values=torch.tensor([[0, 0, 1, 1, 1], [0, 0, 1, 1, 8], [1, 1, 1, 1, 8]]),
        ),
        blocks=[
            TensorBlock(
                values=torch.tensor([[[[1.0]]], [[[1.5]]]], dtype=torch.float64),
                samples=Labels(
                    names=["system", "atom"],
                    values=torch.tensor([[1, 0], [1, 1]]),
                ),
                components=[
                    Labels(names=["o3_mu_1"], values=torch.tensor([[0]])),
                    Labels(names=["o3_mu_2"], values=torch.tensor([[0]])),
                ],
                properties=Labels(names=["_"], values=torch.tensor([[0]])),
            ),
            TensorBlock(
                values=torch.tensor([[[[2.0]]]], dtype=torch.float64),
                samples=Labels(
                    names=["system", "atom"],
                    values=torch.tensor([[1, 2]]),
                ),
                components=[
                    Labels(names=["o3_mu_1"], values=torch.tensor([[0]])),
                    Labels(names=["o3_mu_2"], values=torch.tensor([[0]])),
                ],
                properties=Labels(names=["_"], values=torch.tensor([[0]])),
            ),
            TensorBlock(
                values=pp_full_sys2,
                samples=Labels(
                    names=["system", "atom"],
                    values=torch.tensor([[1, 2]]),
                ),
                components=[
                    Labels(
                        names=["o3_mu_1"], values=torch.arange(-1, 2).reshape(-1, 1)
                    ),
                    Labels(
                        names=["o3_mu_2"], values=torch.arange(-1, 2).reshape(-1, 1)
                    ),
                ],
                properties=Labels(names=["_"], values=torch.tensor([[0]])),
            ),
        ],
    )

    dataset = Dataset.from_dict(
        {
            "system": systems,
            "uncoupled_hamiltonian": [tensor_map_1, tensor_map_2],
        }
    )

    atomic_types = [1, 8]
    irreps = {
        1: [{"o3_lambda": 0, "o3_sigma": 1}],
        8: [
            {"o3_lambda": 0, "o3_sigma": 1},
            {"o3_lambda": 1, "o3_sigma": 1},
        ],
    }
    if missing_type:
        atomic_types.append(9)
        irreps[9] = [{"o3_lambda": 0, "o3_sigma": 1}]

    composition_model = CompositionModel(
        hypers={},
        dataset_info=DatasetInfo(
            length_unit="angstrom",
            atomic_types=atomic_types,
            targets={
                "uncoupled_hamiltonian": get_generic_target_info(
                    "uncoupled_hamiltonian",
                    {
                        "quantity": "",
                        "unit": "",
                        "type": {
                            "spherical": {"irreps": irreps, "product": "cartesian"}
                        },
                        "num_subtargets": 1,
                        "per_atom": True,
                    },
                )
            },
        ),
    )

    composition_model.train_model([dataset], [], batch_size=1, is_distributed=False)
    assert composition_model.atomic_types == atomic_types

    output = composition_model(
        [systems[1]], {"uncoupled_hamiltonian": ModelOutput(per_atom=True)}
    )

    ss_key = {"o3_lambda_1": 0, "o3_lambda_2": 0, "o3_sigma_1": 1, "o3_sigma_2": 1}
    pp_key = {"o3_lambda_1": 1, "o3_lambda_2": 1, "o3_sigma_1": 1, "o3_sigma_2": 1}

    H_ss_block = output["uncoupled_hamiltonian"].block({**ss_key, "atom_type": 1})
    torch.testing.assert_close(
        H_ss_block.values,
        torch.tensor([1.25, 1.25], dtype=torch.float64).reshape(-1, 1, 1, 1),
    )

    O_ss_block = output["uncoupled_hamiltonian"].block({**ss_key, "atom_type": 8})
    torch.testing.assert_close(
        O_ss_block.values,
        torch.tensor([1.5], dtype=torch.float64).reshape(-1, 1, 1, 1),
    )

    O_pp_block = output["uncoupled_hamiltonian"].block({**pp_key, "atom_type": 8})
    expected_pp = torch.zeros(1, 3, 3, 1, dtype=torch.float64)
    expected_pp[0, torch.arange(3), torch.arange(3), 0] = torch.tensor(
        [3.0, 3.0, 3.0], dtype=torch.float64
    )
    torch.testing.assert_close(O_pp_block.values, expected_pp)

    if missing_type:
        system_F = System(
            positions=torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float64),
            types=torch.tensor([9]),
            cell=torch.eye(3, dtype=torch.float64),
            pbc=torch.tensor([True, True, True]),
        )
        output_F = composition_model(
            [system_F], {"uncoupled_hamiltonian": ModelOutput(per_atom=True)}
        )
        F_ss_block = output_F["uncoupled_hamiltonian"].block({**ss_key, "atom_type": 9})
        torch.testing.assert_close(
            F_ss_block.values,
            torch.tensor([0.0], dtype=torch.float64).reshape(-1, 1, 1, 1),
        )

@pytest.mark.parametrize("missing_type", [False, True])
def test_composition_spherical_atomic_basis_rank_2_rotation_invariance(missing_type):
    """
    Test the calculation of composition weights for a spherical per-atom rank 2 atomic
    basis target (keys: o3_lambda_1, o3_lambda_2, o3_sigma_1, o3_sigma_2, atom_type) is
    invariant under fitting on a rotated version of the dataset.
    """

    def Rz(theta):
        c, s = torch.cos(torch.tensor(theta)), torch.sin(torch.tensor(theta))
        return torch.tensor(
            [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float64
        )

    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=torch.float64
    )
    R = Rz(torch.pi / 2)
    positions_rotated = positions @ R.T

    system = System(
        positions=positions,
        types=torch.tensor([8, 1, 1]),
        cell=torch.eye(3, dtype=torch.float64),
        pbc=torch.tensor([True, True, True]),
    )
    system_rotated = System(
        positions=positions_rotated,
        types=torch.tensor([8, 1, 1]),
        cell=torch.eye(3, dtype=torch.float64),
        pbc=torch.tensor([True, True, True]),
    )

    pp_O = torch.zeros(1, 3, 3, 1, dtype=torch.float64)
    pp_O[0, torch.arange(3), torch.arange(3), 0] = torch.tensor(
        [1.0, 2.0, 3.0], dtype=torch.float64
    )
    pp_O_rotated = torch.einsum(
        "ac,bd,icd->iab", R, R, pp_O[..., 0]
    ).unsqueeze(-1)

    def make_tensor_map(system_idx, pp_O_vals):
        return TensorMap(
            keys=Labels(
                names=[
                    "o3_lambda_1",
                    "o3_lambda_2",
                    "o3_sigma_1",
                    "o3_sigma_2",
                    "atom_type",
                ],
                values=torch.tensor([[0, 0, 1, 1, 1], [0, 0, 1, 1, 8], [1, 1, 1, 1, 8]]),
            ),
            blocks=[
                TensorBlock(
                    values=torch.tensor([[[[1.0]]], [[[1.5]]]], dtype=torch.float64),
                    samples=Labels(
                        names=["system", "atom"],
                        values=torch.tensor([[system_idx, 1], [system_idx, 2]]),
                    ),
                    components=[
                        Labels(names=["o3_mu_1"], values=torch.tensor([[0]])),
                        Labels(names=["o3_mu_2"], values=torch.tensor([[0]])),
                    ],
                    properties=Labels(names=["_"], values=torch.tensor([[0]])),
                ),
                TensorBlock(
                    values=torch.tensor([[[[2.0]]]], dtype=torch.float64),
                    samples=Labels(
                        names=["system", "atom"],
                        values=torch.tensor([[system_idx, 0]]),
                    ),
                    components=[
                        Labels(names=["o3_mu_1"], values=torch.tensor([[0]])),
                        Labels(names=["o3_mu_2"], values=torch.tensor([[0]])),
                    ],
                    properties=Labels(names=["_"], values=torch.tensor([[0]])),
                ),
                TensorBlock(
                    values=pp_O_vals,
                    samples=Labels(
                        names=["system", "atom"],
                        values=torch.tensor([[system_idx, 0]]),
                    ),
                    components=[
                        Labels(
                            names=["o3_mu_1"],
                            values=torch.arange(-1, 2).reshape(-1, 1),
                        ),
                        Labels(
                            names=["o3_mu_2"],
                            values=torch.arange(-1, 2).reshape(-1, 1),
                        ),
                    ],
                    properties=Labels(names=["_"], values=torch.tensor([[0]])),
                ),
            ],
        )

    atomic_types = [1, 8]
    irreps = {
        1: [{"o3_lambda": 0, "o3_sigma": 1}],
        8: [
            {"o3_lambda": 0, "o3_sigma": 1},
            {"o3_lambda": 1, "o3_sigma": 1},
        ],
    }
    if missing_type:
        atomic_types.append(9)
        irreps[9] = [{"o3_lambda": 0, "o3_sigma": 1}]

    target_info = get_generic_target_info(
        "uncoupled_hamiltonian",
        {
            "quantity": "",
            "unit": "",
            "type": {
                "spherical": {"irreps": irreps, "product": "cartesian"}
            },
            "num_subtargets": 1,
            "per_atom": True,
        },
    )
    dataset_info = DatasetInfo(
        length_unit="angstrom",
        atomic_types=atomic_types,
        targets={"uncoupled_hamiltonian": target_info},
    )

    dataset_orig = Dataset.from_dict(
        {
            "system": [system],
            "uncoupled_hamiltonian": [make_tensor_map(0, pp_O)],
        }
    )
    model_orig = CompositionModel(hypers={}, dataset_info=dataset_info)
    model_orig.train_model([dataset_orig], [], batch_size=1, is_distributed=False)

    dataset_rot = Dataset.from_dict(
        {
            "system": [system_rotated],
            "uncoupled_hamiltonian": [make_tensor_map(0, pp_O_rotated)],
        }
    )
    model_rot = CompositionModel(hypers={}, dataset_info=dataset_info)
    model_rot.train_model([dataset_rot], [], batch_size=1, is_distributed=False)

    weights_orig = model_orig.model.weights["uncoupled_hamiltonian"]
    weights_rot = model_rot.model.weights["uncoupled_hamiltonian"]

    ss_key = {"o3_lambda_1": 0, "o3_lambda_2": 0, "o3_sigma_1": 1, "o3_sigma_2": 1}
    pp_key = {"o3_lambda_1": 1, "o3_lambda_2": 1, "o3_sigma_1": 1, "o3_sigma_2": 1}

    torch.testing.assert_close(
        weights_orig.block({**ss_key, "atom_type": 1}).values,
        weights_rot.block({**ss_key, "atom_type": 1}).values,
    )
    torch.testing.assert_close(
        weights_orig.block({**ss_key, "atom_type": 8}).values,
        weights_rot.block({**ss_key, "atom_type": 8}).values,
    )

    torch.testing.assert_close(
        weights_orig.block({**pp_key, "atom_type": 8}).values,
        weights_rot.block({**pp_key, "atom_type": 8}).values,
    )