from pathlib import Path

import metatensor.torch
import pytest
import rascaline.torch
import torch
from metatensor.torch.atomistic import ModelCapabilities, ModelOutput

from metatensor.models import soap_bpnn
from metatensor.models.utils.data import read_structures
from metatensor.models.utils.output_gradient import compute_gradient


RESOURCES_PATH = Path(__file__).parent.resolve() / ".." / "resources"


@pytest.mark.parametrize("is_training", [True, False])
def test_forces(is_training):
    """Test that the forces are calculated correctly"""

    capabilities = ModelCapabilities(
        length_unit="Angstrom",
        species=[1, 6, 7, 8],
        outputs={
            "energy": ModelOutput(
                quantity="energy",
                unit="eV",
            )
        },
    )

    model = soap_bpnn.Model(capabilities)
    structures = read_structures(RESOURCES_PATH / "qm9_reduced_100.xyz")[:5]
    structures = rascaline.torch.systems_to_torch(
        structures, positions_requires_grad=True
    )
    output = model(structures)
    position_gradients = compute_gradient(
        output["energy"].block().values,
        [structure.positions for structure in structures],
        is_training=is_training,
    )
    forces = [-position_gradient for position_gradient in position_gradients]

    jitted_model = torch.jit.script(model)
    structures = rascaline.torch.systems_to_torch(
        structures, positions_requires_grad=True
    )
    output = jitted_model(structures)
    jitted_position_gradients = compute_gradient(
        output["energy"].block().values,
        [structure.positions for structure in structures],
        is_training=is_training,
    )
    jitted_forces = [
        -position_gradient for position_gradient in jitted_position_gradients
    ]

    for f, jf in zip(forces, jitted_forces):
        assert torch.allclose(f, jf)


@pytest.mark.parametrize("is_training", [True, False])
def test_virial(is_training):
    """Test that the virial is calculated correctly"""

    capabilities = ModelCapabilities(
        length_unit="Angstrom",
        species=[21, 23, 24, 27, 29, 39, 40, 41, 72, 74, 78],
        outputs={
            "energy": ModelOutput(
                quantity="energy",
                unit="eV",
            )
        },
    )

    model = soap_bpnn.Model(capabilities)
    structures = read_structures(RESOURCES_PATH / "alchemical_reduced_10.xyz")[:2]

    displacements = [
        torch.eye(
            3, requires_grad=True, dtype=system.cell.dtype, device=system.cell.device
        )
        for system in structures
    ]
    systems = [
        metatensor.torch.atomistic.System(
            positions=system.positions @ displacement,
            cell=system.cell @ displacement,
            species=system.species,
        )
        for system, displacement in zip(structures, displacements)
    ]

    output = model(systems)
    displacement_gradients = compute_gradient(
        output["energy"].block().values,
        displacements,
        is_training=is_training,
    )
    virial = [-cell_gradient for cell_gradient in displacement_gradients]

    jitted_model = torch.jit.script(model)

    displacements = [
        torch.eye(
            3, requires_grad=True, dtype=system.cell.dtype, device=system.cell.device
        )
        for system in structures
    ]
    systems = [
        metatensor.torch.atomistic.System(
            positions=system.positions @ displacement,
            cell=system.cell @ displacement,
            species=system.species,
        )
        for system, displacement in zip(structures, displacements)
    ]

    output = jitted_model(systems)
    jitted_displacement_gradients = compute_gradient(
        output["energy"].block().values,
        displacements,
        is_training=is_training,
    )
    jitted_virial = [-cell_gradient for cell_gradient in jitted_displacement_gradients]

    for v, jv in zip(virial, jitted_virial):
        assert torch.allclose(v, jv)


@pytest.mark.parametrize("is_training", [True, False])
def test_both(is_training):
    """Test that the forces and virial are calculated correctly together"""

    capabilities = ModelCapabilities(
        length_unit="Angstrom",
        species=[21, 23, 24, 27, 29, 39, 40, 41, 72, 74, 78],
        outputs={
            "energy": ModelOutput(
                quantity="energy",
                unit="eV",
            )
        },
    )

    model = soap_bpnn.Model(capabilities)
    structures = read_structures(RESOURCES_PATH / "alchemical_reduced_10.xyz")[:2]

    # Here we re-create displacements and systems, otherwise torch
    # complains that the graph has already beeen freed in the last grad call
    displacements = [
        torch.eye(
            3, requires_grad=True, dtype=system.cell.dtype, device=system.cell.device
        )
        for system in structures
    ]
    systems = [
        metatensor.torch.atomistic.System(
            positions=system.positions @ displacement,
            cell=system.cell @ displacement,
            species=system.species,
        )
        for system, displacement in zip(structures, displacements)
    ]

    output = model(systems)
    print(output["energy"].block().values.requires_grad)
    gradients = compute_gradient(
        output["energy"].block().values,
        [system.positions for system in systems] + displacements,
        is_training=is_training,
    )
    f_and_v = [-gradient for gradient in gradients]

    displacements = [
        torch.eye(
            3, requires_grad=True, dtype=system.cell.dtype, device=system.cell.device
        )
        for system in structures
    ]
    systems = [
        metatensor.torch.atomistic.System(
            positions=system.positions @ displacement,
            cell=system.cell @ displacement,
            species=system.species,
        )
        for system, displacement in zip(structures, displacements)
    ]

    jitted_model = torch.jit.script(model)
    output = jitted_model(systems)
    print(output["energy"].block().values.requires_grad)
    jitted_gradients = compute_gradient(
        output["energy"].block().values,
        [system.positions for system in systems] + displacements,
        is_training=is_training,
    )
    jitted_f_and_v = [-jitted_gradient for jitted_gradient in jitted_gradients]

    for fv, jfv in zip(f_and_v, jitted_f_and_v):
        assert torch.allclose(fv, jfv)
