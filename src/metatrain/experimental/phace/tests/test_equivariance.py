import copy

import ase.io
import numpy as np
import pytest
import torch
from metatensor.torch.atomistic import System, systems_to_torch

from metatrain.experimental.phace import PhACE
from metatrain.utils.data import DatasetInfo
from metatrain.utils.data.target_info import (
    get_energy_target_info,
    get_generic_target_info,
)
from metatrain.utils.neighbor_lists import get_system_with_neighbor_lists
from metatrain.utils.testing.equivariance import (
    get_random_rotation,
    rotate_spherical_tensor,
    rotate_system,
)

from . import DATASET_PATH, DATASET_PATH_PERIODIC, MODEL_HYPERS


def test_rotational_invariance():
    """Tests that the model is rotationally invariant for a scalar target."""
    torch.set_default_dtype(torch.float64)  # make sure we have enough precision

    dataset_info = DatasetInfo(
        length_unit="Angstrom",
        atomic_types=[1, 6, 7, 8],
        targets={"energy": get_energy_target_info({"unit": "eV"})},
    )
    model = torch.jit.script(PhACE(MODEL_HYPERS, dataset_info))

    system = ase.io.read(DATASET_PATH)
    original_system = copy.deepcopy(system)
    system.rotate(48, "y")

    original_output = model(
        [
            get_system_with_neighbor_lists(
                systems_to_torch(original_system), model.requested_neighbor_lists()
            )
        ],
        {"energy": model.outputs["energy"]},
    )
    rotated_output = model(
        [
            get_system_with_neighbor_lists(
                systems_to_torch(system), model.requested_neighbor_lists()
            )
        ],
        {"energy": model.outputs["energy"]},
    )

    torch.testing.assert_close(
        original_output["energy"].block().values,
        rotated_output["energy"].block().values,
        atol=1e-5,
        rtol=1e-5,
    )

    torch.set_default_dtype(torch.float32)  # change back


@pytest.mark.parametrize("o3_lambda", [0, 1, 2, 3])
@pytest.mark.parametrize("o3_sigma", [1])
def test_equivariance_rotations(o3_lambda, o3_sigma):
    """Tests that the model is rotationally equivariant when predicting
    spherical tensors."""
    torch.set_default_dtype(torch.float64)  # make sure we have enough precision

    dataset_info = DatasetInfo(
        length_unit="Angstrom",
        atomic_types=[1, 6, 7, 8],
        targets={
            "spherical_target": get_generic_target_info(
                {
                    "quantity": "",
                    "unit": "",
                    "type": {
                        "spherical": {
                            "irreps": [{"o3_lambda": o3_lambda, "o3_sigma": o3_sigma}]
                        }
                    },
                    "num_subtargets": 100,
                    "per_atom": False,
                }
            )
        },
    )
    model = torch.jit.script(PhACE(MODEL_HYPERS, dataset_info))

    system = ase.io.read(DATASET_PATH)
    original_system = systems_to_torch(system)
    rotation = get_random_rotation()
    rotated_system = rotate_system(original_system, rotation)

    original_output = model(
        [
            get_system_with_neighbor_lists(
                original_system, model.requested_neighbor_lists()
            )
        ],
        {"spherical_target": model.outputs["spherical_target"]},
    )
    rotated_output = model(
        [
            get_system_with_neighbor_lists(
                rotated_system, model.requested_neighbor_lists()
            )
        ],
        {"spherical_target": model.outputs["spherical_target"]},
    )

    np.testing.assert_allclose(
        rotate_spherical_tensor(
            original_output["spherical_target"].block().values.detach().numpy(),
            rotation,
        ),
        rotated_output["spherical_target"].block().values.detach().numpy(),
        atol=1e-8,
        rtol=1e-8,
    )

    torch.set_default_dtype(torch.float32)  # change back


@pytest.mark.parametrize("dataset_path", [DATASET_PATH, DATASET_PATH_PERIODIC])
@pytest.mark.parametrize("o3_lambda", [0, 1, 2, 3])
@pytest.mark.parametrize("o3_sigma", [1])
def test_equivariance_inversion(dataset_path, o3_lambda, o3_sigma):
    """Tests that the model is equivariant with respect to inversions."""

    torch.set_default_dtype(torch.float64)  # make sure we have enough precision

    dataset_info = DatasetInfo(
        length_unit="Angstrom",
        atomic_types=[1, 6, 7, 8],
        targets={
            "spherical_target": get_generic_target_info(
                {
                    "quantity": "",
                    "unit": "",
                    "type": {
                        "spherical": {
                            "irreps": [{"o3_lambda": o3_lambda, "o3_sigma": o3_sigma}]
                        }
                    },
                    "num_subtargets": 100,
                    "per_atom": False,
                }
            )
        },
    )
    model = torch.jit.script(PhACE(MODEL_HYPERS, dataset_info))

    system = ase.io.read(dataset_path)
    original_system = systems_to_torch(system)
    inverted_system = System(
        positions=original_system.positions * (-1),
        cell=original_system.cell * (-1),
        types=original_system.types,
        pbc=original_system.pbc,
    )

    original_output = model(
        [
            get_system_with_neighbor_lists(
                original_system, model.requested_neighbor_lists()
            )
        ],
        {"spherical_target": model.outputs["spherical_target"]},
    )
    inverted_output = model(
        [
            get_system_with_neighbor_lists(
                inverted_system, model.requested_neighbor_lists()
            )
        ],
        {"spherical_target": model.outputs["spherical_target"]},
    )

    print(
        original_output["spherical_target"].block().values
        - inverted_output["spherical_target"].block().values
    )

    torch.testing.assert_close(
        original_output["spherical_target"].block().values
        * (-1) ** o3_lambda
        * (-1 if o3_sigma == -1 else 1),
        inverted_output["spherical_target"].block().values,
        atol=1e-8,
        rtol=1e-8,
    )

    torch.set_default_dtype(torch.float32)  # change back
