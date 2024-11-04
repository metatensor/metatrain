from pathlib import Path

import metatensor.torch
import pytest
import torch
from metatensor.torch import Labels, TensorBlock, TensorMap
from metatensor.torch.atomistic import ModelOutput, System
from omegaconf import OmegaConf

from metatrain.utils.additive import ZBL, CompositionModel, remove_additive
from metatrain.utils.data import Dataset, DatasetInfo, TargetInfo, TargetInfoDict
from metatrain.utils.data.readers import read_systems, read_targets
from metatrain.utils.neighbor_lists import (
    get_requested_neighbor_lists,
    get_system_with_neighbor_lists,
)


RESOURCES_PATH = Path(__file__).parents[1] / "resources"


def test_composition_model_train():
    """Test the calculation of composition weights."""

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
        model_hypers={},
        dataset_info=DatasetInfo(
            length_unit="angstrom",
            atomic_types=[1, 8],
            targets=TargetInfoDict(
                {
                    "energy": TargetInfo(
                        quantity="energy",
                        per_atom=False,
                    )
                }
            ),
        ),
    )

    composition_model.train_model(dataset)
    assert composition_model.weights.shape[0] == 1
    assert composition_model.weights.shape[1] == 2
    assert composition_model.output_to_output_index == {"energy": 0}
    assert composition_model.atomic_types == [1, 8]
    torch.testing.assert_close(
        composition_model.weights, torch.tensor([[2.0, 1.0]], dtype=torch.float64)
    )

    composition_model.train_model([dataset])
    assert composition_model.weights.shape[0] == 1
    assert composition_model.weights.shape[1] == 2
    assert composition_model.output_to_output_index == {"energy": 0}
    assert composition_model.atomic_types == [1, 8]
    torch.testing.assert_close(
        composition_model.weights, torch.tensor([[2.0, 1.0]], dtype=torch.float64)
    )

    composition_model.train_model([dataset, dataset, dataset])
    assert composition_model.weights.shape[0] == 1
    assert composition_model.weights.shape[1] == 2
    assert composition_model.output_to_output_index == {"energy": 0}
    assert composition_model.atomic_types == [1, 8]
    torch.testing.assert_close(
        composition_model.weights, torch.tensor([[2.0, 1.0]], dtype=torch.float64)
    )


def test_composition_model_predict():
    """Test the prediction of composition energies."""

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
            "forces": False,
            "stress": False,
            "virial": False,
        }
    }
    targets, target_info = read_targets(OmegaConf.create(conf))
    dataset = Dataset.from_dict({"system": systems, "mtt::U0": targets["mtt::U0"]})

    composition_model = CompositionModel(
        model_hypers={},
        dataset_info=DatasetInfo(
            length_unit="angstrom",
            atomic_types=[1, 6, 7, 8],
            targets=target_info,
        ),
    )

    composition_model.train_model(dataset)

    # per_atom = False
    output = composition_model(
        systems[:5],
        {"mtt::U0": ModelOutput(quantity="energy", unit="", per_atom=False)},
    )
    assert "mtt::U0" in output
    assert output["mtt::U0"].block().samples.names == ["system"]
    assert output["mtt::U0"].block().values.shape == (5, 1)

    # per_atom = True
    output = composition_model(
        systems[:5],
        {"mtt::U0": ModelOutput(quantity="energy", unit="", per_atom=True)},
    )
    assert "mtt::U0" in output
    assert output["mtt::U0"].block().samples.names == ["system", "atom"]
    assert output["mtt::U0"].block().values.shape != (5, 1)

    # with selected_atoms
    selected_atoms = metatensor.torch.Labels(
        names=["system", "atom"],
        values=torch.tensor([[0, 0]]),
    )

    output = composition_model(
        systems[:5],
        {"mtt::U0": ModelOutput(quantity="energy", unit="", per_atom=True)},
        selected_atoms=selected_atoms,
    )
    assert "mtt::U0" in output
    assert output["mtt::U0"].block().samples.names == ["system", "atom"]
    assert output["mtt::U0"].block().values.shape == (1, 1)

    output = composition_model(
        systems[:5],
        {"mtt::U0": ModelOutput(quantity="energy", unit="", per_atom=False)},
        selected_atoms=selected_atoms,
    )
    assert "mtt::U0" in output
    assert output["mtt::U0"].block().samples.names == ["system"]
    assert output["mtt::U0"].block().values.shape == (1, 1)


def test_composition_model_torchscript(tmpdir):
    """Test the torchscripting, saving and loading of the composition model."""
    system = System(
        positions=torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float64),
        types=torch.tensor([8]),
        cell=torch.eye(3, dtype=torch.float64),
        pbc=torch.tensor([True, True, True]),
    )

    composition_model = CompositionModel(
        model_hypers={},
        dataset_info=DatasetInfo(
            length_unit="angstrom",
            atomic_types=[1, 8],
            targets=TargetInfoDict(
                {
                    "energy": TargetInfo(
                        quantity="energy",
                        per_atom=False,
                    )
                }
            ),
        ),
    )
    composition_model = torch.jit.script(composition_model)
    composition_model(
        [system], {"energy": ModelOutput(quantity="energy", unit="", per_atom=False)}
    )
    torch.jit.save(composition_model, tmpdir / "composition_model.pt")
    composition_model = torch.jit.load(tmpdir / "composition_model.pt")
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
            "forces": False,
            "stress": False,
            "virial": False,
        }
    }
    targets, target_info = read_targets(OmegaConf.create(conf))
    dataset = Dataset.from_dict({"system": systems, "mtt::U0": targets["mtt::U0"]})

    composition_model = CompositionModel(
        model_hypers={},
        dataset_info=DatasetInfo(
            length_unit="angstrom",
            atomic_types=[1, 6, 7, 8],
            targets=target_info,
        ),
    )
    composition_model.train_model(dataset)

    # concatenate all targets
    targets["mtt::U0"] = metatensor.torch.join(targets["mtt::U0"], axis="samples")

    std_before = targets["mtt::U0"].block().values.std().item()
    remove_additive(systems, targets, composition_model, target_info)
    std_after = targets["mtt::U0"].block().values.std().item()

    # In QM9 the composition contribution is very large: the standard deviation
    # of the energies is reduced by a factor of over 100 upon removing the composition
    assert std_after < 100.0 * std_before


def test_composition_model_missing_types():
    """
    Test the error when there are too many or too types in the dataset
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
        model_hypers={},
        dataset_info=DatasetInfo(
            length_unit="angstrom",
            atomic_types=[1],
            targets=TargetInfoDict(
                {
                    "energy": TargetInfo(
                        quantity="energy",
                        per_atom=False,
                    )
                }
            ),
        ),
    )
    with pytest.raises(
        ValueError,
        match="unknown atomic types",
    ):
        composition_model.train_model(dataset)

    composition_model = CompositionModel(
        model_hypers={},
        dataset_info=DatasetInfo(
            length_unit="angstrom",
            atomic_types=[1, 8, 100],
            targets=TargetInfoDict(
                {
                    "energy": TargetInfo(
                        quantity="energy",
                        per_atom=False,
                    )
                }
            ),
        ),
    )
    with pytest.warns(
        UserWarning,
        match="do not contain atomic types",
    ):
        composition_model.train_model(dataset)


def test_composition_model_wrong_target():
    """
    Test the error when a non-energy is fed to the composition model.
    """

    with pytest.raises(
        ValueError,
        match="only supports energy-like outputs",
    ):
        CompositionModel(
            model_hypers={},
            dataset_info=DatasetInfo(
                length_unit="angstrom",
                atomic_types=[1],
                targets=TargetInfoDict(
                    {
                        "energy": TargetInfo(
                            quantity="FOO",
                            per_atom=False,
                        )
                    }
                ),
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
            "forces": False,
            "stress": False,
            "virial": False,
        }
    }
    _, target_info = read_targets(OmegaConf.create(conf))

    zbl = ZBL(
        model_hypers={},
        dataset_info=DatasetInfo(
            length_unit="angstrom",
            atomic_types=[1, 6, 7, 8],
            targets=target_info,
        ),
    )

    requested_neighbor_lists = get_requested_neighbor_lists(zbl)
    for system in systems:
        get_system_with_neighbor_lists(system, requested_neighbor_lists)

    # per_atom = True
    output = zbl(
        systems,
        {"mtt::U0": ModelOutput(quantity="energy", unit="", per_atom=True)},
    )
    assert "mtt::U0" in output
    assert output["mtt::U0"].block().samples.names == ["system", "atom"]
    assert output["mtt::U0"].block().values.shape != (5, 1)

    # with selected_atoms
    selected_atoms = metatensor.torch.Labels(
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
