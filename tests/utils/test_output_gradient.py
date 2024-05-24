import metatensor.torch
import pytest
import torch
from metatensor.torch.atomistic import System

from metatensor.models.experimental.soap_bpnn import __model__
from metatensor.models.utils.data import DatasetInfo, TargetInfo, read_systems
from metatensor.models.utils.output_gradient import compute_gradient

from . import MODEL_HYPERS, RESOURCES_PATH


@pytest.mark.parametrize("is_training", [True, False])
def test_forces(is_training):
    """Test that the forces are calculated correctly"""

    dataset_info = DatasetInfo(
        length_unit="angstrom",
        atomic_types=[1, 6, 7, 8],
        targets={
            "energy": TargetInfo(
                quantity="energy", unit="eV", per_atom=False, gradients=["positions"]
            )
        },
    )
    model = __model__(model_hypers=MODEL_HYPERS, dataset_info=dataset_info)

    systems = read_systems(RESOURCES_PATH / "qm9_reduced_100.xyz")[:5]
    systems = [
        System(
            positions=system.positions.requires_grad_(True),
            cell=system.cell,
            types=system.types,
        )
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
        )
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
        atomic_types=[
            21,
            23,
            24,
            26,
            27,
            29,
            30,
            39,
            40,
            41,
            44,
            45,
            46,
            47,
            72,
            74,
            77,
            78,
        ],
        targets={
            "energy": TargetInfo(
                quantity="energy", unit="eV", per_atom=False, gradients=["strain"]
            )
        },
    )
    model = __model__(model_hypers=MODEL_HYPERS, dataset_info=dataset_info)

    systems = read_systems(RESOURCES_PATH / "alchemical_reduced_10.xyz")[:2]

    strains = [
        torch.eye(
            3, requires_grad=True, dtype=system.cell.dtype, device=system.cell.device
        )
        for system in systems
    ]
    systems = [
        metatensor.torch.atomistic.System(
            positions=system.positions @ strain,
            cell=system.cell @ strain,
            types=system.types,
        )
        for system, strain in zip(systems, strains)
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
        metatensor.torch.atomistic.System(
            positions=system.positions @ strain,
            cell=system.cell @ strain,
            types=system.types,
        )
        for system, strain in zip(systems, strains)
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
        atomic_types=[
            21,
            23,
            24,
            26,
            27,
            29,
            30,
            39,
            40,
            41,
            44,
            45,
            46,
            47,
            72,
            74,
            77,
            78,
        ],
        targets={
            "energy": TargetInfo(
                quantity="energy",
                unit="eV",
                per_atom=False,
                gradients=["positions", "strain"],
            )
        },
    )
    model = __model__(model_hypers=MODEL_HYPERS, dataset_info=dataset_info)
    systems = read_systems(RESOURCES_PATH / "alchemical_reduced_10.xyz")[:2]

    # Here we re-create strains and systems, otherwise torch
    # complains that the graph has already beeen freed in the last grad call
    strains = [
        torch.eye(
            3, requires_grad=True, dtype=system.cell.dtype, device=system.cell.device
        )
        for system in systems
    ]
    systems = [
        metatensor.torch.atomistic.System(
            positions=system.positions @ strain,
            cell=system.cell @ strain,
            types=system.types,
        )
        for system, strain in zip(systems, strains)
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
        metatensor.torch.atomistic.System(
            positions=system.positions @ strain,
            cell=system.cell @ strain,
            types=system.types,
        )
        for system, strain in zip(systems, strains)
    ]

    jitted_model = torch.jit.script(model)
    output = jitted_model(systems, {"energy": model.outputs["energy"]})
    jitted_gradients = compute_gradient(
        output["energy"].block().values,
        [system.positions for system in systems] + strains,
        is_training=is_training,
    )
    jitted_f_and_v = [-jitted_gradient for jitted_gradient in jitted_gradients]

    for fv, jfv in zip(f_and_v, jitted_f_and_v):
        torch.testing.assert_close(fv, jfv)
