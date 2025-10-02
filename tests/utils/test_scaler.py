import copy
from pathlib import Path

import pytest
import torch
from metatensor.torch import Labels, TensorBlock, TensorMap
from metatomic.torch import System

from metatrain.utils.data import Dataset, DatasetInfo
from metatrain.utils.data.readers import read_systems
from metatrain.utils.data.target_info import (
    get_energy_target_info,
    get_generic_target_info,
)
from metatrain.utils.scaler import Scaler, remove_scale


RESOURCES_PATH = Path(__file__).parents[1] / "resources"


@pytest.mark.parametrize("batch_size", [1, 2])
def test_scaler_scalar_single_property(batch_size):
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

    scaler = Scaler(
        hypers={},
        dataset_info=DatasetInfo(
            length_unit="angstrom",
            atomic_types=[1, 8],
            targets={"energy": get_energy_target_info("energy", {"unit": "eV"})},
        ),
    ).to(torch.float64)
    scaler2 = copy.deepcopy(scaler)
    scaler3 = copy.deepcopy(scaler)

    # fake output to test how the scaler acts on it
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
                properties=Labels.range("energy", 1),
            )
        ],
    )
    fake_output = {"energy": fake_output}

    expected_scales = torch.tensor(
        [[13.0 / 3**0.5], [13.0 / 3**0.5], [13.0 / 3**0.5]], dtype=torch.float64
    )

    scaler.train_model(
        dataset, additive_models=[], batch_size=batch_size, is_distributed=False
    )
    fake_output_after_scaling = scaler(systems, fake_output)
    scales = (
        fake_output_after_scaling["energy"].block().values
        / fake_output["energy"].block().values
    )
    torch.testing.assert_close(scales, expected_scales)

    scaler2.train_model(
        [dataset], additive_models=[], batch_size=batch_size, is_distributed=False
    )
    fake_output_after_scaling = scaler(systems, fake_output)
    scales = (
        fake_output_after_scaling["energy"].block().values
        / fake_output["energy"].block().values
    )
    torch.testing.assert_close(scales, expected_scales)

    scaler3.train_model(
        [dataset, dataset, dataset],
        additive_models=[],
        batch_size=batch_size,
        is_distributed=False,
    )
    fake_output_after_scaling = scaler(systems, fake_output)
    scales = (
        fake_output_after_scaling["energy"].block().values
        / fake_output["energy"].block().values
    )
    torch.testing.assert_close(scales, expected_scales)

    # Test the remove_scale function
    removed_output = remove_scale(systems, fake_output, scaler)
    torch.testing.assert_close(
        removed_output["energy"].block().values,
        fake_output["energy"].block().values / expected_scales,
    )


@pytest.mark.parametrize("batch_size", [1, 2])
def test_scaler_scalar_multiple_properties(batch_size):
    """Test the calculation of scaling weights."""

    # Here we use three synthetic structures and two properties.
    #
    # The first property is the same as in the single-property test:
    # - O atom, with an energy of 3.0
    # - H2O molecule, with an energy of 4.0 * 3
    # - H4O2 molecule, with an energy of 12.0 * 6
    # The expected standard deviation is 13/sqrt(3).
    #
    # The second property is just twice the first one, so the expected standard
    # deviation is 26/sqrt(3).

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
                    values=torch.tensor([[e, 2 * e]], dtype=torch.float64),
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

    scaler = Scaler(
        hypers={},
        dataset_info=DatasetInfo(
            length_unit="angstrom",
            atomic_types=[1, 8],
            targets={
                "energy": get_generic_target_info(
                    "energy",
                    {
                        "quantity": "energy",
                        "unit": "eV",
                        "num_subtargets": 2,
                        "type": "scalar",
                        "per_atom": False,
                    },
                )
            },
        ),
    ).to(torch.float64)
    scaler2 = copy.deepcopy(scaler)
    scaler3 = copy.deepcopy(scaler)

    # fake output to test how the scaler acts on it
    fake_output = TensorMap(
        keys=Labels.single(),
        blocks=[
            TensorBlock(
                values=torch.tensor(
                    [[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]], dtype=torch.float64
                ),
                samples=Labels(
                    names=["system"],
                    values=torch.tensor([[0], [1], [2]]),
                ),
                components=[],
                properties=Labels.range("energy", 2),
            )
        ],
    )
    fake_output = {"energy": fake_output}

    expected_scales = torch.tensor(
        [
            [13.0 / 3**0.5, 26.0 / 3**0.5],
            [13.0 / 3**0.5, 26.0 / 3**0.5],
            [13.0 / 3**0.5, 26.0 / 3**0.5],
        ],
        dtype=torch.float64,
    )

    scaler.train_model(
        dataset, additive_models=[], batch_size=batch_size, is_distributed=False
    )
    fake_output_after_scaling = scaler(systems, fake_output)
    scales = (
        fake_output_after_scaling["energy"].block().values
        / fake_output["energy"].block().values
    )
    torch.testing.assert_close(scales, expected_scales)

    scaler2.train_model(
        [dataset], additive_models=[], batch_size=batch_size, is_distributed=False
    )
    fake_output_after_scaling = scaler(systems, fake_output)
    scales = (
        fake_output_after_scaling["energy"].block().values
        / fake_output["energy"].block().values
    )
    torch.testing.assert_close(scales, expected_scales)

    scaler3.train_model(
        [dataset, dataset, dataset],
        additive_models=[],
        batch_size=batch_size,
        is_distributed=False,
    )
    fake_output_after_scaling = scaler(systems, fake_output)
    scales = (
        fake_output_after_scaling["energy"].block().values
        / fake_output["energy"].block().values
    )
    torch.testing.assert_close(scales, expected_scales)

    # Test the remove_scale function
    removed_output = remove_scale(systems, fake_output, scaler)
    torch.testing.assert_close(
        removed_output["energy"].block().values,
        fake_output["energy"].block().values / expected_scales,
    )


@pytest.mark.parametrize("batch_size", [1, 2])
def test_scaler_cartesian_per_atom(batch_size):
    """Test the calculation of scaling weights."""

    # Here we use two synthetic structures.
    # - O atom, with a force of [1.0, 1.0, 1.0]
    # - H2O molecule, with forces of
    #   - [2.0, 2.0, 2.0] (H)
    #   - [3.0, 3.0, 3.0] (H)
    #   - [4.0, 4.0, 4.0] (O)
    # The expected standard deviations are sqrt(17/2) for O and sqrt(13/2) for H.

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
    forces = [
        [
            [1.0, 1.0, 1.0],  # O
        ],
        [
            [2.0, 2.0, 2.0],  # H
            [3.0, 3.0, 3.0],  # H
            [4.0, 4.0, 4.0],  # O
        ],
    ]
    forces = [
        TensorMap(
            keys=Labels(names=["_"], values=torch.tensor([[0]])),
            blocks=[
                TensorBlock(
                    values=torch.tensor(f, dtype=torch.float64).unsqueeze(-1),
                    samples=Labels(
                        names=["system", "atom"],
                        values=torch.tensor([[i, j] for j in range(len(f))]),
                    ),
                    components=[Labels.range("xyz", 3)],
                    properties=Labels(names=["forces"], values=torch.tensor([[0]])),
                )
            ],
        )
        for i, f in enumerate(forces)
    ]
    dataset = Dataset.from_dict({"system": systems, "forces": forces})

    scaler = Scaler(
        hypers={},
        dataset_info=DatasetInfo(
            length_unit="angstrom",
            atomic_types=[1, 8],
            targets={
                "forces": get_generic_target_info(
                    "forces",
                    {
                        "quantity": "forces",
                        "unit": "eV/A",
                        "type": {"cartesian": {"rank": 1}},
                        "per_atom": True,
                        "num_subtargets": 1,
                    },
                )
            },
        ),
    ).to(torch.float64)
    scaler2 = copy.deepcopy(scaler)
    scaler3 = copy.deepcopy(scaler)

    # fake output to test how the scaler acts on it
    fake_output = TensorMap(
        keys=Labels.single(),
        blocks=[
            TensorBlock(
                values=torch.tensor(
                    [
                        [1.0, 1.0, 1.0],
                        [1.0, 1.0, 1.0],
                        [1.0, 1.0, 1.0],
                        [1.0, 1.0, 1.0],
                    ],
                    dtype=torch.float64,
                ).unsqueeze(-1),
                samples=Labels(
                    names=["system", "atom"],
                    values=torch.tensor([[0, 0], [1, 0], [1, 1], [1, 2]]),
                ),
                components=[Labels.range("xyz", 3)],
                properties=Labels.range("forces", 1),
            )
        ],
    )
    fake_output = {"forces": fake_output}

    expected_scales = torch.tensor(
        [
            [(17.0 / 2) ** 0.5, (17.0 / 2) ** 0.5, (17.0 / 2) ** 0.5],  # O in system 0
            [(13.0 / 2) ** 0.5, (13.0 / 2) ** 0.5, (13.0 / 2) ** 0.5],  # H in system 1
            [(13.0 / 2) ** 0.5, (13.0 / 2) ** 0.5, (13.0 / 2) ** 0.5],  # H in system 1
            [(17.0 / 2) ** 0.5, (17.0 / 2) ** 0.5, (17.0 / 2) ** 0.5],  # O in system 1
        ],
        dtype=torch.float64,
    ).unsqueeze(-1)

    scaler.train_model(
        dataset, additive_models=[], batch_size=batch_size, is_distributed=False
    )
    fake_output_after_scaling = scaler(systems, fake_output)
    scales = (
        fake_output_after_scaling["forces"].block().values
        / fake_output["forces"].block().values
    )
    torch.testing.assert_close(scales, expected_scales)

    scaler2.train_model(
        [dataset], additive_models=[], batch_size=batch_size, is_distributed=False
    )
    fake_output_after_scaling = scaler(systems, fake_output)
    scales = (
        fake_output_after_scaling["forces"].block().values
        / fake_output["forces"].block().values
    )
    torch.testing.assert_close(scales, expected_scales)

    scaler3.train_model(
        [dataset, dataset, dataset],
        additive_models=[],
        batch_size=batch_size,
        is_distributed=False,
    )
    fake_output_after_scaling = scaler(systems, fake_output)
    scales = (
        fake_output_after_scaling["forces"].block().values
        / fake_output["forces"].block().values
    )
    torch.testing.assert_close(scales, expected_scales)

    # Test the remove_scale function
    removed_output = remove_scale(systems, fake_output, scaler)
    torch.testing.assert_close(
        removed_output["forces"].block().values,
        fake_output["forces"].block().values / expected_scales,
    )


def test_scaler_torchscript(tmpdir):
    """Test the torchscripting, saving and loading of a scaler model."""

    dataset_path = RESOURCES_PATH / "qm9_reduced_100.xyz"
    systems = read_systems(dataset_path)

    scaler = Scaler(
        hypers={},
        dataset_info=DatasetInfo(
            length_unit="angstrom",
            atomic_types=[1, 8],
            targets={"energy": get_energy_target_info("energy", {"unit": "eV"})},
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
                properties=Labels.range("energy", 1),
            )
        ],
    )
    fake_output = {"energy": fake_output}

    scaler = torch.jit.script(scaler)
    scaler(systems, fake_output)

    with tmpdir.as_cwd():
        torch.jit.save(scaler, tmpdir / "scaler.pt")
        scaler = torch.jit.load(tmpdir / "scaler.pt")

    scaler(systems, fake_output)
