import logging
from pathlib import Path

import metatensor.torch
import pytest
import torch
from metatensor.torch import Labels, TensorBlock, TensorMap
from metatomic.torch import ModelOutput, System
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from metatrain.utils.additive import (
    ZBL,
    CompositionModel,
    OldCompositionModel,
    remove_additive,
)
from metatrain.utils.data import CollateFn, Dataset, DatasetInfo
from metatrain.utils.data.readers import read_systems, read_targets
from metatrain.utils.data.target_info import (
    get_energy_target_info,
    get_generic_target_info,
)
from metatrain.utils.neighbor_lists import (
    get_requested_neighbor_lists,
    get_system_with_neighbor_lists,
)


RESOURCES_PATH = Path(__file__).parents[1] / "resources"


def test_old_composition_model_train():
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

    composition_model = OldCompositionModel(
        hypers={},
        dataset_info=DatasetInfo(
            length_unit="angstrom",
            atomic_types=[1, 8],
            targets={"energy": get_energy_target_info({"unit": "eV"})},
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

    composition_model.train_model(dataset, [])
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

    composition_model.train_model([dataset], [])
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

    composition_model.train_model([dataset, dataset, dataset], [])
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


def test_composition_model_train():
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
    collate_fn = CollateFn(target_keys=["energy"])
    dataloader = DataLoader(dataset, batch_size=1, collate_fn=collate_fn)

    composition_model = CompositionModel(
        hypers={},
        dataset_info=DatasetInfo(
            length_unit="angstrom",
            atomic_types=[1, 8],
            targets={"energy": get_energy_target_info({"unit": "eV"})},
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

    composition_model.train_model(dataloader, additive_models=[])
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


def test_old_composition_model_predict():
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

    composition_model = OldCompositionModel(
        hypers={},
        dataset_info=DatasetInfo(
            length_unit="angstrom",
            atomic_types=[1, 6, 7, 8],
            targets=target_info,
        ),
    )

    composition_model.train_model(dataset, [])

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


@pytest.mark.parametrize("device", ("cpu", "cuda"))
def test_composition_model_predict(device):
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
    collate_fn = CollateFn(target_keys=["mtt::U0"])
    dataloader = DataLoader(dataset, batch_size=1, collate_fn=collate_fn)

    composition_model = CompositionModel(
        hypers={},
        dataset_info=DatasetInfo(
            length_unit="angstrom",
            atomic_types=[1, 6, 7, 8],
            targets=target_info,
        ),
    )

    composition_model.train_model(dataloader, additive_models=[])

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
    selected_atoms = metatensor.torch.Labels(
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

    # with selected_atoms
    selected_atoms = metatensor.torch.Labels(
        names=["system"],
        values=torch.tensor([[0]]),
    ).to(device=device)

    output = composition_model(
        systems_to_predict,
        {"mtt::U0": ModelOutput(quantity="energy", unit="", per_atom=False)},
        selected_atoms=selected_atoms,
    )
    assert "mtt::U0" in output
    assert output["mtt::U0"].block().samples.names == ["system"]
    assert output["mtt::U0"].block().values.shape == (1, 1)
    assert output["mtt::U0"].block().values.device.type == device


def test_old_composition_model_torchscript(tmpdir):
    """Test the torchscripting, saving and loading of the composition model."""
    system = System(
        positions=torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float64),
        types=torch.tensor([8]),
        cell=torch.eye(3, dtype=torch.float64),
        pbc=torch.tensor([True, True, True]),
    )

    composition_model = OldCompositionModel(
        hypers={},
        dataset_info=DatasetInfo(
            length_unit="angstrom",
            atomic_types=[1, 8],
            targets={"energy": get_energy_target_info({"unit": "eV"})},
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
            targets={"energy": get_energy_target_info({"unit": "eV"})},
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


def test_old_remove_additive():
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

    composition_model = OldCompositionModel(
        hypers={},
        dataset_info=DatasetInfo(
            length_unit="angstrom",
            atomic_types=[1, 6, 7, 8],
            targets=target_info,
        ),
    )
    composition_model.train_model(dataset, [])

    # concatenate all targets
    targets["mtt::U0"] = metatensor.torch.join(targets["mtt::U0"], axis="samples")

    std_before = targets["mtt::U0"].block().values.std().item()
    remove_additive(systems, targets, composition_model, target_info)
    std_after = targets["mtt::U0"].block().values.std().item()

    # In QM9 the composition contribution is very large: the standard deviation
    # of the energies is reduced by a factor of over 100 upon removing the composition
    assert std_after < 100.0 * std_before


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
    collate_fn = CollateFn(target_keys=["mtt::U0"])
    dataloader = DataLoader(dataset, batch_size=1, collate_fn=collate_fn)

    composition_model = CompositionModel(
        hypers={},
        dataset_info=DatasetInfo(
            length_unit="angstrom",
            atomic_types=[1, 6, 7, 8],
            targets=target_info,
        ),
    )
    composition_model.train_model(dataloader, additive_models=[])

    # concatenate all targets
    targets["mtt::U0"] = metatensor.torch.join(targets["mtt::U0"], axis="samples")

    std_before = targets["mtt::U0"].block().values.std().item()
    remove_additive(systems, targets, composition_model, target_info)
    std_after = targets["mtt::U0"].block().values.std().item()

    # In QM9 the composition contribution is very large: the standard deviation
    # of the energies is reduced by a factor of over 100 upon removing the composition
    assert std_after < 100.0 * std_before


def test_old_composition_model_missing_types(caplog):
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

    composition_model = OldCompositionModel(
        hypers={},
        dataset_info=DatasetInfo(
            length_unit="angstrom",
            atomic_types=[1],
            targets={"energy": get_energy_target_info({"unit": "eV"})},
        ),
    )
    with pytest.raises(
        ValueError,
        match="unknown atomic types",
    ):
        composition_model.train_model(dataset, [])

    composition_model = OldCompositionModel(
        hypers={},
        dataset_info=DatasetInfo(
            length_unit="angstrom",
            atomic_types=[1, 8, 100],
            targets={"energy": get_energy_target_info({"unit": "eV"})},
        ),
    )
    # need to capture the warning from the logger
    with caplog.at_level(logging.WARNING):
        composition_model.train_model(dataset, [])
    assert "do not contain atomic types" in caplog.text


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
    collate_fn = CollateFn(target_keys=["energy"])
    dataloader = DataLoader(dataset, batch_size=1, collate_fn=collate_fn)

    composition_model = CompositionModel(
        hypers={},
        dataset_info=DatasetInfo(
            length_unit="angstrom",
            atomic_types=[1],
            targets={"energy": get_energy_target_info({"unit": "eV"})},
        ),
    )
    with pytest.raises(
        ValueError,
        match="unexpected atom types",
    ):
        composition_model.train_model(dataloader, [])


def test_old_composition_model_wrong_target():
    """
    Test the error when a non-scalar is fed to the composition model.
    """
    with pytest.raises(ValueError, match="does not support target quantity force"):
        OldCompositionModel(
            hypers={},
            dataset_info=DatasetInfo(
                length_unit="angstrom",
                atomic_types=[1],
                targets={
                    "force": get_generic_target_info(
                        {
                            "quantity": "force",
                            "unit": "",
                            "type": {"cartesian": {"rank": 1}},
                            "num_subtargets": 1,
                            "per_atom": True,
                        }
                    )
                },
            ),
        )


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
                        {
                            "quantity": "force",
                            "unit": "",
                            "type": {"cartesian": {"rank": 1}},
                            "num_subtargets": 1,
                            "per_atom": True,
                        }
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


@pytest.mark.parametrize("where_is_center_type", ["keys", "samples", "nowhere"])
def test_old_composition_model_train_per_atom(where_is_center_type):
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
        tensor_map_1 = metatensor.torch.remove_dimension(
            tensor_map_1, "samples", "center_type"
        )
        tensor_map_2 = metatensor.torch.remove_dimension(
            tensor_map_2, "samples", "center_type"
        )

    energies = [tensor_map_1, tensor_map_2]
    dataset = Dataset.from_dict({"system": systems, "energy": energies})

    composition_model = OldCompositionModel(
        hypers={},
        dataset_info=DatasetInfo(
            length_unit="angstrom",
            atomic_types=[1, 8],
            targets={
                "energy": get_generic_target_info(
                    {
                        "quantity": "energy",
                        "unit": "",
                        "type": "scalar",
                        "num_subtargets": 1,
                        "per_atom": True,
                    }
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

    composition_model.train_model(dataset, [])
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
        tensor_map_1 = metatensor.torch.remove_dimension(
            tensor_map_1, "samples", "center_type"
        )
        tensor_map_2 = metatensor.torch.remove_dimension(
            tensor_map_2, "samples", "center_type"
        )

    energies = [tensor_map_1, tensor_map_2]
    dataset = Dataset.from_dict({"system": systems, "energy": energies})
    collate_fn = CollateFn(target_keys=["energy"])
    dataloader = DataLoader(dataset, batch_size=1, collate_fn=collate_fn)

    composition_model = CompositionModel(
        hypers={},
        dataset_info=DatasetInfo(
            length_unit="angstrom",
            atomic_types=[1, 8],
            targets={
                "energy": get_generic_target_info(
                    {
                        "quantity": "energy",
                        "unit": "",
                        "type": "scalar",
                        "num_subtargets": 1,
                        "per_atom": True,
                    }
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

    composition_model.train_model(dataloader, [])
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


def test_old_composition_many_subtargets():
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

    composition_model = OldCompositionModel(
        hypers={},
        dataset_info=DatasetInfo(
            length_unit="angstrom",
            atomic_types=[1, 8],
            targets={
                "energy": get_generic_target_info(
                    {
                        "quantity": "energy",
                        "unit": "",
                        "type": "scalar",
                        "num_subtargets": 2,
                        "per_atom": False,
                    }
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

    composition_model.train_model(dataset, [])
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
    collate_fn = CollateFn(target_keys=["energy"])
    dataloader = DataLoader(dataset, batch_size=1, collate_fn=collate_fn)

    composition_model = CompositionModel(
        hypers={},
        dataset_info=DatasetInfo(
            length_unit="angstrom",
            atomic_types=[1, 8],
            targets={
                "energy": get_generic_target_info(
                    {
                        "quantity": "energy",
                        "unit": "",
                        "type": "scalar",
                        "num_subtargets": 2,
                        "per_atom": False,
                    }
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

    composition_model.train_model(dataloader, [])
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


def test_old_composition_spherical():
    """Test the calculation of composition weights for a per-structure scalar."""

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

    composition_model = OldCompositionModel(
        hypers={},
        dataset_info=DatasetInfo(
            length_unit="angstrom",
            atomic_types=[1, 8],
            targets={
                "energy": get_generic_target_info(
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
                    }
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

    composition_model.train_model(dataset, [])
    assert composition_model.atomic_types == [1, 8]
    output_H = composition_model(
        [system_H], {"energy": ModelOutput(quantity="energy", unit="", per_atom=False)}
    )
    torch.testing.assert_close(
        output_H["energy"].block({"o3_lambda": 0}).values,
        torch.tensor([[[2.0]]], dtype=torch.float64),
    )
    torch.testing.assert_close(
        output_H["energy"].block({"o3_lambda": 1}).values,
        torch.zeros_like(output_H["energy"].block({"o3_lambda": 1}).values),
    )
    output_O = composition_model(
        [system_O], {"energy": ModelOutput(quantity="energy", unit="", per_atom=False)}
    )
    torch.testing.assert_close(
        output_O["energy"].block({"o3_lambda": 0}).values,
        torch.tensor([[[1.0]]], dtype=torch.float64),
    )
    torch.testing.assert_close(
        output_O["energy"].block({"o3_lambda": 1}).values,
        torch.zeros_like(output_O["energy"].block({"o3_lambda": 1}).values),
    )


def test_composition_spherical():
    """Test the calculation of composition weights for a per-structure scalar."""

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
    collate_fn = CollateFn(target_keys=["energy"])
    dataloader = DataLoader(dataset, batch_size=1, collate_fn=collate_fn)

    composition_model = CompositionModel(
        hypers={},
        dataset_info=DatasetInfo(
            length_unit="angstrom",
            atomic_types=[1, 8],
            targets={
                "energy": get_generic_target_info(
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
                    }
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

    composition_model.train_model(dataloader, [])
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
