import copy
from pathlib import Path

import metatensor.torch as mts
import pytest
import torch
from metatensor.torch import Labels, TensorBlock, TensorMap
from metatomic.torch import System

from metatrain.utils.augmentation import RotationalAugmenter
from metatrain.utils.data import Dataset, DatasetInfo
from metatrain.utils.data.readers import read_systems
from metatrain.utils.data.target_info import (
    get_energy_target_info,
    get_generic_target_info,
)
from metatrain.utils.scaler import Scaler, remove_scale


RESOURCES_PATH = Path(__file__).parents[1] / "resources"


@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("fixed_scaling_weights", [False, True])
def test_scaler_scalar_single_property(batch_size, fixed_scaling_weights):
    """Test the calculation of scaling weights for a single scalar property."""

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

    if fixed_scaling_weights:
        expected_scales = torch.tensor([[0.1], [0.1], [0.1]], dtype=torch.float64)
    else:
        expected_scales = torch.tensor(
            [[13.0 / 3**0.5], [13.0 / 3**0.5], [13.0 / 3**0.5]], dtype=torch.float64
        )

    scaler.train_model(
        dataset,
        additive_models=[],
        batch_size=batch_size,
        is_distributed=False,
        fixed_weights=({"energy": 0.1} if fixed_scaling_weights else None),
    )
    fake_output_after_scaling = scaler(systems, fake_output)
    scales = (
        fake_output_after_scaling["energy"].block().values
        / fake_output["energy"].block().values
    )
    torch.testing.assert_close(scales, expected_scales)

    scaler2.train_model(
        [dataset, dataset],
        additive_models=[],
        batch_size=batch_size,
        is_distributed=False,
        fixed_weights=({"energy": 0.1} if fixed_scaling_weights else None),
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
        fixed_weights=({"energy": 0.1} if fixed_scaling_weights else None),
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
    """Test the calculation of scaling weights for multiple scalar properties."""

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
        [dataset, dataset],
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
@pytest.mark.parametrize("fixed_scaling_weights", [False, True])
def test_scaler_cartesian_per_atom(batch_size, fixed_scaling_weights):
    """Test the calculation of scaling weights for a cartesian per-atom property."""

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

    if fixed_scaling_weights:
        expected_scales = torch.tensor(
            [[0.1, 0.1, 0.1], [0.2, 0.2, 0.2], [0.2, 0.2, 0.2], [0.1, 0.1, 0.1]],
            dtype=torch.float64,
        ).unsqueeze(-1)
    else:
        expected_scales = torch.tensor(
            [
                [(17.0 / 2) ** 0.5, (17.0 / 2) ** 0.5, (17.0 / 2) ** 0.5],  # O in sys 0
                [(13.0 / 2) ** 0.5, (13.0 / 2) ** 0.5, (13.0 / 2) ** 0.5],  # H in sys 1
                [(13.0 / 2) ** 0.5, (13.0 / 2) ** 0.5, (13.0 / 2) ** 0.5],  # H in sys 1
                [(17.0 / 2) ** 0.5, (17.0 / 2) ** 0.5, (17.0 / 2) ** 0.5],  # O in sys 1
            ],
            dtype=torch.float64,
        ).unsqueeze(-1)

    scaler.train_model(
        dataset,
        additive_models=[],
        batch_size=batch_size,
        is_distributed=False,
        fixed_weights={"forces": {1: 0.2, 8: 0.1}} if fixed_scaling_weights else None,
    )
    fake_output_after_scaling = scaler(systems, fake_output)
    scales = (
        fake_output_after_scaling["forces"].block().values
        / fake_output["forces"].block().values
    )
    torch.testing.assert_close(scales, expected_scales)

    scaler2.train_model(
        [dataset, dataset],
        additive_models=[],
        batch_size=batch_size,
        is_distributed=False,
        fixed_weights={"forces": {1: 0.2, 8: 0.1}} if fixed_scaling_weights else None,
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
        fixed_weights={"forces": {1: 0.2, 8: 0.1}} if fixed_scaling_weights else None,
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


@pytest.mark.parametrize("batch_size", [1, 2])
def test_scaler_spherical(batch_size):
    """Test the calculation of scaling weights for a multi-block spherical property."""

    # Here we use two synthetic structures, each with a scalar and a rank-2 spherical
    # tensor in the same target.
    # - O atom, with a scalar of 3.0 and a rank-2 of [1.0, 2.0, 3.0, 4.0, 5.0]
    # - H2O molecule, with a scalar of 9.0 and a rank-2 of 3*[6.0, 7.0, 8.0, 9.0, 10.0]
    # The expected standard deviations are 3/sqrt(2) for the scalar and sqrt(77/2) for
    # the rank-2 spherical tensor.

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
    spherical = [
        [[3.0], [1.0, 2.0, 3.0, 4.0, 5.0]],
        [[9.0], [18.0, 21.0, 24.0, 27.0, 30.0]],
    ]
    spherical = [
        TensorMap(
            keys=Labels(
                names=["o3_lambda", "o3_sigma"], values=torch.tensor([[0, 1], [2, 1]])
            ),
            blocks=[
                TensorBlock(
                    values=torch.tensor([sc], dtype=torch.float64).unsqueeze(-1),
                    samples=Labels(names=["system"], values=torch.tensor([[i]])),
                    components=[Labels.range("o3_mu", 1)],
                    properties=Labels(names=["spherical"], values=torch.tensor([[0]])),
                ),
                TensorBlock(
                    values=torch.tensor([sph], dtype=torch.float64).unsqueeze(-1),
                    samples=Labels(names=["system"], values=torch.tensor([[i]])),
                    components=[Labels.range("o3_mu", 5)],
                    properties=Labels(names=["spherical"], values=torch.tensor([[1]])),
                ),
            ],
        )
        for i, (sc, sph) in enumerate(spherical)
    ]
    dataset = Dataset.from_dict({"system": systems, "spherical": spherical})

    scaler = Scaler(
        hypers={},
        dataset_info=DatasetInfo(
            length_unit="angstrom",
            atomic_types=[1, 8],
            targets={
                "spherical": get_generic_target_info(
                    "spherical",
                    {
                        "quantity": "spherical",
                        "unit": "",
                        "type": {
                            "spherical": {
                                "irreps": [
                                    {"o3_lambda": 0, "o3_sigma": 1},
                                    {"o3_lambda": 2, "o3_sigma": 1},
                                ]
                            }
                        },
                        "per_atom": False,
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
        keys=Labels(
            names=["o3_lambda", "o3_sigma"],
            values=torch.tensor([[0, 1], [2, 1]]),
        ),
        blocks=[
            TensorBlock(
                values=torch.tensor([[1.0], [1.0]], dtype=torch.float64).unsqueeze(-1),
                samples=Labels(
                    names=["system"],
                    values=torch.tensor([[0], [1]]),
                ),
                components=[Labels.range("o3_mu", 1)],
                properties=Labels.range("spherical", 1),
            ),
            TensorBlock(
                values=torch.tensor(
                    [
                        [1.0, 1.0, 1.0, 1.0, 1.0],
                        [1.0, 1.0, 1.0, 1.0, 1.0],
                    ],
                    dtype=torch.float64,
                ).unsqueeze(-1),
                samples=Labels(
                    names=["system"],
                    values=torch.tensor([[0], [1]]),
                ),
                components=[Labels.range("o3_mu", 5)],
                properties=Labels.range("spherical", 1),
            ),
        ],
    )
    fake_output = {"spherical": fake_output}

    expected_scales_scalar = torch.tensor(
        [[3.0], [3.0]], dtype=torch.float64
    ).unsqueeze(-1)
    expected_scales_spherical = torch.tensor(
        [
            [
                (77.0 / 2) ** 0.5,
                (77.0 / 2) ** 0.5,
                (77.0 / 2) ** 0.5,
                (77.0 / 2) ** 0.5,
                (77.0 / 2) ** 0.5,
            ],
            [
                (77.0 / 2) ** 0.5,
                (77.0 / 2) ** 0.5,
                (77.0 / 2) ** 0.5,
                (77.0 / 2) ** 0.5,
                (77.0 / 2) ** 0.5,
            ],
        ],
        dtype=torch.float64,
    ).unsqueeze(-1)

    scaler.train_model(
        dataset,
        additive_models=[],
        batch_size=batch_size,
        is_distributed=False,
    )
    fake_output_after_scaling = scaler(systems, fake_output)
    scales_scalar = (
        fake_output_after_scaling["spherical"].block({"o3_lambda": 0}).values
        / fake_output["spherical"].block({"o3_lambda": 0}).values
    )
    scales_spherical = (
        fake_output_after_scaling["spherical"].block({"o3_lambda": 2}).values
        / fake_output["spherical"].block({"o3_lambda": 2}).values
    )
    torch.testing.assert_close(scales_scalar, expected_scales_scalar)
    torch.testing.assert_close(scales_spherical, expected_scales_spherical)

    scaler2.train_model(
        [dataset, dataset],
        additive_models=[],
        batch_size=batch_size,
        is_distributed=False,
    )
    fake_output_after_scaling = scaler(systems, fake_output)
    scales_scalar = (
        fake_output_after_scaling["spherical"].block({"o3_lambda": 0}).values
        / fake_output["spherical"].block({"o3_lambda": 0}).values
    )
    scales_spherical = (
        fake_output_after_scaling["spherical"].block({"o3_lambda": 2}).values
        / fake_output["spherical"].block({"o3_lambda": 2}).values
    )
    torch.testing.assert_close(scales_scalar, expected_scales_scalar)
    torch.testing.assert_close(scales_spherical, expected_scales_spherical)

    scaler3.train_model(
        [dataset, dataset, dataset],
        additive_models=[],
        batch_size=batch_size,
        is_distributed=False,
    )
    fake_output_after_scaling = scaler(systems, fake_output)
    scales_scalar = (
        fake_output_after_scaling["spherical"].block({"o3_lambda": 0}).values
        / fake_output["spherical"].block({"o3_lambda": 0}).values
    )
    scales_spherical = (
        fake_output_after_scaling["spherical"].block({"o3_lambda": 2}).values
        / fake_output["spherical"].block({"o3_lambda": 2}).values
    )
    torch.testing.assert_close(scales_scalar, expected_scales_scalar)
    torch.testing.assert_close(scales_spherical, expected_scales_spherical)

    # Test the remove_scale function
    removed_output = remove_scale(systems, fake_output, scaler)
    torch.testing.assert_close(
        removed_output["spherical"].block({"o3_lambda": 0}).values,
        fake_output["spherical"].block({"o3_lambda": 0}).values
        / expected_scales_scalar,
    )
    torch.testing.assert_close(
        removed_output["spherical"].block({"o3_lambda": 2}).values,
        fake_output["spherical"].block({"o3_lambda": 2}).values
        / expected_scales_spherical,
    )


def test_scaler_rotation_invariance():
    """Test the calculation of scaling weights for a multi-block spherical property."""

    # Here we use two synthetic structures, each with a scalar and a rank-2 spherical
    # tensor in the same target.
    # - O atom, with a scalar of 3.0 and a rank-2 of [1.0, 2.0, 3.0, 4.0, 5.0]
    # - H2O molecule, with a scalar of 9.0 and a rank-2 of 3*[6.0, 7.0, 8.0, 9.0, 10.0]
    #
    # The test checks that computing the scales with the dataset and a rotated version
    # gives the same result.

    num_checks = 5

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
    spherical = [
        [[3.0], [1.0, 2.0, 3.0, 4.0, 5.0]],
        [[9.0], [18.0, 21.0, 24.0, 27.0, 30.0]],
    ]
    spherical = [
        TensorMap(
            keys=Labels(
                names=["o3_lambda", "o3_sigma"], values=torch.tensor([[0, 1], [2, 1]])
            ),
            blocks=[
                TensorBlock(
                    values=torch.tensor([sc], dtype=torch.float64).unsqueeze(-1),
                    samples=Labels(names=["system"], values=torch.tensor([[i]])),
                    components=[Labels(["o3_mu"], torch.arange(1).reshape(-1, 1))],
                    properties=Labels(names=["spherical"], values=torch.tensor([[0]])),
                ),
                TensorBlock(
                    values=torch.tensor([sph], dtype=torch.float64).unsqueeze(-1),
                    samples=Labels(names=["system"], values=torch.tensor([[i]])),
                    components=[Labels(["o3_mu"], torch.arange(-2, 3).reshape(-1, 1))],
                    properties=Labels(names=["spherical"], values=torch.tensor([[1]])),
                ),
            ],
        )
        for i, (sc, sph) in enumerate(spherical)
    ]

    dataset_info = DatasetInfo(
        length_unit="angstrom",
        atomic_types=[1, 8],
        targets={
            "spherical": get_generic_target_info(
                "spherical",
                {
                    "quantity": "spherical",
                    "unit": "",
                    "type": {
                        "spherical": {
                            "irreps": [
                                {"o3_lambda": 0, "o3_sigma": 1},
                                {"o3_lambda": 2, "o3_sigma": 1},
                            ]
                        }
                    },
                    "per_atom": False,
                    "num_subtargets": 1,
                },
            )
        },
    )

    # Create the dataset for the unrotated systems and train the scaler
    dataset = Dataset.from_dict({"system": systems, "spherical": spherical})
    scaler = Scaler(hypers={}, dataset_info=dataset_info).to(torch.float64)
    scaler.train_model(
        dataset,
        additive_models=[],
        batch_size=1,
        is_distributed=False,
    )

    for _ in range(num_checks):
        # Create the dataset for the rotated systems and train the scaler
        rotational_augmenter = RotationalAugmenter(
            dataset_info.targets, extra_data_info_dict={}
        )
        systems_rotated = []
        spherical_rotated = []
        for system_, spherical_ in zip(systems, spherical, strict=False):
            system_rotated_, spherical_rotated_, _ = (
                rotational_augmenter.apply_random_augmentations(
                    [system_],
                    {
                        "spherical": spherical_,
                    },
                    extra_data={},
                )
            )
            systems_rotated.append(system_rotated_[0])
            spherical_rotated.append(spherical_rotated_["spherical"])
        dataset_rotated = Dataset.from_dict(
            {"system": systems_rotated, "spherical": spherical_rotated}
        )
        scaler_rotated = Scaler(hypers={}, dataset_info=dataset_info).to(torch.float64)
        scaler_rotated.train_model(
            dataset_rotated,
            additive_models=[],
            batch_size=1,
            is_distributed=False,
        )

        # Check that the scales are the same
        mts.allclose_raise(
            scaler.model.scales["spherical"], scaler_rotated.model.scales["spherical"]
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
