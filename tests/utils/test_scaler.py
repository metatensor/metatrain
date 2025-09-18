from pathlib import Path

import torch
from metatensor.torch import Labels, TensorBlock, TensorMap
from metatomic.torch import System
from omegaconf import OmegaConf

from metatrain.utils.data import Dataset, DatasetInfo
from metatrain.utils.data.readers import read_systems, read_targets
from metatrain.utils.data.target_info import get_energy_target_info
from metatrain.utils.old_scaler import OldScaler, remove_scale


RESOURCES_PATH = Path(__file__).parents[1] / "resources"


def test_scaler_train():
    """Test the calculation of scaling weights."""

    # Here we use three synthetic structures:
    # - O atom, with an energy of 3.0
    # - H2O molecule, with an energy of 4.0 * 3
    # - H4O2 molecule, with an energy of 12.0 * 6
    # The expected standard deviation is 13/sqrt(3).

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
    energies = [3.0, 4.0 * 3, 12.0 * 6]
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

    scaler = OldScaler(
        hypers={},
        dataset_info=DatasetInfo(
            length_unit="angstrom",
            atomic_types=[1, 8],
            targets={"energy": get_energy_target_info({"unit": "eV"})},
        ),
    )

    scaler.train_model(dataset, additive_models=[], treat_as_additive=True)
    assert scaler.scales.shape == (1,)
    assert scaler.output_name_to_output_index == {"energy": 0}
    torch.testing.assert_close(
        scaler.scales, torch.tensor([13.0 / 3**0.5], dtype=torch.float64)
    )

    scaler.train_model([dataset], additive_models=[], treat_as_additive=True)
    assert scaler.scales.shape == (1,)
    assert scaler.output_name_to_output_index == {"energy": 0}
    torch.testing.assert_close(
        scaler.scales, torch.tensor([13.0 / 3**0.5], dtype=torch.float64)
    )

    scaler.train_model(
        [dataset, dataset, dataset], additive_models=[], treat_as_additive=True
    )
    assert scaler.scales.shape == (1,)
    assert scaler.output_name_to_output_index == {"energy": 0}
    torch.testing.assert_close(
        scaler.scales, torch.tensor([13.0 / 3**0.5], dtype=torch.float64)
    )


def test_scale():
    """Test the scaling of the scale, both at training and prediction
    time."""

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

    scaler = OldScaler(
        hypers={},
        dataset_info=DatasetInfo(
            length_unit="angstrom",
            atomic_types=[1, 6, 7, 8],
            targets=target_info,
        ),
    )

    scaler.train_model(dataset, additive_models=[], treat_as_additive=True)
    scale = scaler.scales[0].item()

    fake_output_or_target = TensorMap(
        keys=Labels.single(),
        blocks=[
            TensorBlock(
                values=torch.tensor([[1.0], [2.0], [3.0]], dtype=torch.float64),
                samples=Labels(
                    names=["system"],
                    values=torch.tensor([[0], [1], [2]]),
                ),
                components=[],
                properties=Labels.single(),
            )
        ],
    )
    fake_output_or_target = {"mtt::U0": fake_output_or_target}

    scaled_output = scaler(fake_output_or_target)
    assert "mtt::U0" in scaled_output
    torch.testing.assert_close(
        scaled_output["mtt::U0"].block().values,
        torch.tensor([[1.0], [2.0], [3.0]], dtype=torch.float64) * scale,
    )

    # Test the remove_scale function
    scaled_output = remove_scale(fake_output_or_target, scaler)
    assert "mtt::U0" in fake_output_or_target
    torch.testing.assert_close(
        scaled_output["mtt::U0"].block().values,
        torch.tensor([[1.0], [2.0], [3.0]], dtype=torch.float64) / scale,
    )


def test_scaler_torchscript(tmpdir):
    """Test the torchscripting, saving and loading of a scaler model."""

    scaler = OldScaler(
        hypers={},
        dataset_info=DatasetInfo(
            length_unit="angstrom",
            atomic_types=[1, 8],
            targets={"energy": get_energy_target_info({"unit": "eV"})},
        ),
    )

    fake_output = TensorMap(
        keys=Labels.single(),
        blocks=[
            TensorBlock(
                values=torch.tensor([[1.0], [2.0], [3.0]], dtype=torch.float64),
                samples=Labels(
                    names=["system"],
                    values=torch.tensor([[0], [1], [2]]),
                ),
                components=[],
                properties=Labels.single(),
            )
        ],
    )
    fake_output = {"energy": fake_output}

    scaler = torch.jit.script(scaler)
    scaler(fake_output)

    with tmpdir.as_cwd():
        torch.jit.save(scaler, tmpdir / "scaler.pt")
        scaler = torch.jit.load(tmpdir / "scaler.pt")

    scaler(fake_output)
