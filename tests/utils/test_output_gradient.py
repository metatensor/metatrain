import metatensor.torch
import pytest
import torch
from metatomic.torch import System

from metatrain.soap_bpnn import __model__
from metatrain.utils.data import DatasetInfo, read_systems
from metatrain.utils.data.target_info import get_energy_target_info
from metatrain.utils.neighbor_lists import (
    get_requested_neighbor_lists,
    get_system_with_neighbor_lists,
)
from metatrain.utils.output_gradient import compute_gradient

from . import MODEL_HYPERS, RESOURCES_PATH


@pytest.mark.parametrize("is_training", [True, False])
def test_forces(is_training):
    """Test that the forces are calculated correctly"""

    dataset_info = DatasetInfo(
        length_unit="angstrom",
        atomic_types={1, 6, 7, 8},
        targets={
            "energy": get_energy_target_info(
                {"unit": "eV"}, add_position_gradients=True
            )
        },
    )
    model = __model__(model_hypers=MODEL_HYPERS, dataset_info=dataset_info)
    model.to(dtype=torch.float64)

    systems = read_systems(RESOURCES_PATH / "qm9_reduced_100.xyz")[:5]
    systems = [
        System(
            positions=system.positions.requires_grad_(True),
            cell=system.cell,
            types=system.types,
            pbc=system.pbc,
        )
        for system in systems
    ]
    requested_neighbor_lists = get_requested_neighbor_lists(model)
    systems = [
        get_system_with_neighbor_lists(system, requested_neighbor_lists)
        for system in systems
    ]
    output = model(systems, {"energy": model.outputs["energy"]})
    position_gradients = compute_gradient(
        output["energy"].block().values,
        [system.positions for system in systems],
        is_training=is_training,
    )
    forces = [-position_gradient for position_gradient in position_gradients]

    jitted_model = torch.jit.script(model)
    systems = read_systems(RESOURCES_PATH / "qm9_reduced_100.xyz")[:5]
    systems = [
        System(
            positions=system.positions.requires_grad_(True),
            cell=system.cell,
            types=system.types,
            pbc=system.pbc,
        )
        for system in systems
    ]
    systems = [
        get_system_with_neighbor_lists(system, requested_neighbor_lists)
        for system in systems
    ]
    output = jitted_model(systems, {"energy": model.outputs["energy"]})
    jitted_position_gradients = compute_gradient(
        output["energy"].block().values,
        [system.positions for system in systems],
        is_training=is_training,
    )
    jitted_forces = [
        -position_gradient for position_gradient in jitted_position_gradients
    ]

    for f, jf in zip(forces, jitted_forces):
        torch.testing.assert_close(f, jf)


@pytest.mark.parametrize("is_training", [True, False])
def test_virial(is_training):
    """Test that the virial is calculated correctly"""

    dataset_info = DatasetInfo(
        length_unit="angstrom",
        atomic_types={6},
        targets={
            "energy": get_energy_target_info({"unit": "eV"}, add_strain_gradients=True)
        },
    )
    model = __model__(model_hypers=MODEL_HYPERS, dataset_info=dataset_info)
    model.to(dtype=torch.float64)

    systems = read_systems(RESOURCES_PATH / "carbon_reduced_100.xyz")[:2]

    strains = [
        torch.eye(
            3, requires_grad=True, dtype=system.cell.dtype, device=system.cell.device
        )
        for system in systems
    ]
    systems = [
        metatomic.torch.System(
            positions=system.positions @ strain,
            cell=system.cell @ strain,
            types=system.types,
            pbc=system.pbc,
        )
        for system, strain in zip(systems, strains)
    ]

    requested_neighbor_lists = get_requested_neighbor_lists(model)
    systems = [
        get_system_with_neighbor_lists(system, requested_neighbor_lists)
        for system in systems
    ]
    output = model(systems, {"energy": model.outputs["energy"]})
    strain_gradients = compute_gradient(
        output["energy"].block().values,
        strains,
        is_training=is_training,
    )
    virial = [-cell_gradient for cell_gradient in strain_gradients]

    jitted_model = torch.jit.script(model)

    strains = [
        torch.eye(
            3, requires_grad=True, dtype=system.cell.dtype, device=system.cell.device
        )
        for system in systems
    ]
    systems = [
        metatomic.torch.System(
            positions=system.positions @ strain,
            cell=system.cell @ strain,
            types=system.types,
            pbc=system.pbc,
        )
        for system, strain in zip(systems, strains)
    ]

    systems = [
        get_system_with_neighbor_lists(system, requested_neighbor_lists)
        for system in systems
    ]
    output = jitted_model(systems, {"energy": model.outputs["energy"]})
    jitted_strain_gradients = compute_gradient(
        output["energy"].block().values,
        strains,
        is_training=is_training,
    )
    jitted_virial = [-cell_gradient for cell_gradient in jitted_strain_gradients]

    for v, jv in zip(virial, jitted_virial):
        torch.testing.assert_close(v, jv)


@pytest.mark.parametrize("is_training", [True, False])
def test_both(is_training):
    """Test that the forces and virial are calculated correctly together"""
    dataset_info = DatasetInfo(
        length_unit="angstrom",
        atomic_types={6},
        targets={
            "energy": get_energy_target_info(
                {"unit": "eV"},
                add_position_gradients=True,
                add_strain_gradients=True,
            )
        },
    )
    model = __model__(model_hypers=MODEL_HYPERS, dataset_info=dataset_info)
    model.to(dtype=torch.float64)

    systems = read_systems(RESOURCES_PATH / "carbon_reduced_100.xyz")[:2]

    # Here we re-create strains and systems, otherwise torch
    # complains that the graph has already beeen freed in the last grad call
    strains = [
        torch.eye(
            3, requires_grad=True, dtype=system.cell.dtype, device=system.cell.device
        )
        for system in systems
    ]
    systems = [
        metatomic.torch.System(
            positions=system.positions @ strain,
            cell=system.cell @ strain,
            types=system.types,
            pbc=system.pbc,
        )
        for system, strain in zip(systems, strains)
    ]

    requested_neighbor_lists = get_requested_neighbor_lists(model)
    systems = [
        get_system_with_neighbor_lists(system, requested_neighbor_lists)
        for system in systems
    ]
    output = model(systems, {"energy": model.outputs["energy"]})
    gradients = compute_gradient(
        output["energy"].block().values,
        [system.positions for system in systems] + strains,
        is_training=is_training,
    )
    f_and_v = [-gradient for gradient in gradients]

    strains = [
        torch.eye(
            3, requires_grad=True, dtype=system.cell.dtype, device=system.cell.device
        )
        for system in systems
    ]
    systems = [
        metatomic.torch.System(
            positions=system.positions @ strain,
            cell=system.cell @ strain,
            types=system.types,
            pbc=system.pbc,
        )
        for system, strain in zip(systems, strains)
    ]

    jitted_model = torch.jit.script(model)
    systems = [
        get_system_with_neighbor_lists(system, requested_neighbor_lists)
        for system in systems
    ]
    output = jitted_model(systems, {"energy": model.outputs["energy"]})
    jitted_gradients = compute_gradient(
        output["energy"].block().values,
        [system.positions for system in systems] + strains,
        is_training=is_training,
    )
    jitted_f_and_v = [-jitted_gradient for jitted_gradient in jitted_gradients]

    for fv, jfv in zip(f_and_v, jitted_f_and_v):
        torch.testing.assert_close(fv, jfv)
