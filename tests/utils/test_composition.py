from pathlib import Path

import metatensor.torch
import torch
from metatensor.torch import Labels, TensorBlock, TensorMap
from metatensor.torch.atomistic import ModelOutput, System
from omegaconf import OmegaConf

from metatrain.utils.composition import CompositionModel, remove_composition
from metatrain.utils.data import Dataset, DatasetInfo, TargetInfo, TargetInfoDict
from metatrain.utils.data.readers import read_systems, read_targets


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
        ),
        System(
            positions=torch.tensor(
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=torch.float64
            ),
            types=torch.tensor([1, 1, 8]),
            cell=torch.eye(3, dtype=torch.float64),
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
    dataset = Dataset({"system": systems, "energy": energies})

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
    dataset = Dataset({"system": systems, "mtt::U0": targets["mtt::U0"]})

    composition_model = CompositionModel(
        model_hypers={},
        dataset_info=DatasetInfo(
            length_unit="angstrom",
            atomic_types=[1, 6, 7, 8],
            targets=target_info,
        ),
    )

    composition_model.train_model(dataset)

    output = composition_model(
        systems[:5],
        {"mtt::U0": ModelOutput(quantity="energy", unit="", per_atom=False)},
    )
    assert "mtt::U0" in output
    assert output["mtt::U0"].block().samples.names == ["system"]
    assert output["mtt::U0"].block().values.shape == (5, 1)

    output = composition_model(
        systems[:5],
        {"mtt::U0": ModelOutput(quantity="energy", unit="", per_atom=True)},
    )
    assert "mtt::U0" in output
    assert output["mtt::U0"].block().samples.names == ["system", "atom"]
    assert output["mtt::U0"].block().values.shape != (5, 1)


def test_composition_model_torchscript(tmpdir):
    """Test the torchscripting, saving and loading of the composition model."""
    system = System(
        positions=torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float64),
        types=torch.tensor([8]),
        cell=torch.eye(3, dtype=torch.float64),
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


def test_remove_composition():
    """Tests the remove_composition function."""

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
    dataset = Dataset({"system": systems, "mtt::U0": targets["mtt::U0"]})

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
    remove_composition(systems, targets, composition_model)
    std_after = targets["mtt::U0"].block().values.std().item()

    # In QM9 the composition contribution is very large: the standard deviation
    # of the energies is reduced by a factor of over 100 upon removing the composition
    assert std_after < 100.0 * std_before
