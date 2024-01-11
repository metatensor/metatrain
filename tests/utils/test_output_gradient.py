from pathlib import Path

import pytest
import rascaline.torch
import torch

from metatensor.models import soap_bpnn
from metatensor.models.utils.data import read_structures
from metatensor.models.utils.output_gradient import compute_gradient


RESOURCES_PATH = Path(__file__).parent.resolve() / ".." / "resources"


@pytest.mark.parametrize("is_training", [True, False])
def test_forces(is_training):
    """Test that the forces are calculated correctly"""

    model = soap_bpnn.Model(all_species=[1, 6, 7, 8])
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

    model = soap_bpnn.Model(all_species=[21, 23, 24, 27, 29, 39, 40, 41, 72, 74, 78])
    structures = read_structures(RESOURCES_PATH / "alchemical_reduced_10.xyz")[:2]
    output = model(
        rascaline.torch.systems_to_torch(structures, cell_requires_grad=True)
    )
    cell_gradients = compute_gradient(
        output["energy"].block().values,
        [structure.cell for structure in structures],
        is_training=is_training,
    )
    virial = [-cell_gradient for cell_gradient in cell_gradients]

    jitted_model = torch.jit.script(model)
    output = jitted_model(
        rascaline.torch.systems_to_torch(structures, cell_requires_grad=True)
    )
    jitted_cell_gradients = compute_gradient(
        output["energy"].block().values,
        [structure.cell for structure in structures],
        is_training=is_training,
    )
    jitted_virial = [-cell_gradient for cell_gradient in jitted_cell_gradients]

    for v, jv in zip(virial, jitted_virial):
        assert torch.allclose(v, jv)


@pytest.mark.parametrize("is_training", [True, False])
def test_both(is_training):
    """Test that the forces and virial are calculated correctly together"""

    model = soap_bpnn.Model(all_species=[21, 23, 24, 27, 29, 39, 40, 41, 72, 74, 78])
    structures = read_structures(RESOURCES_PATH / "alchemical_reduced_10.xyz")[:2]
    output = model(
        rascaline.torch.systems_to_torch(
            structures, positions_requires_grad=True, cell_requires_grad=True
        )
    )
    gradients = compute_gradient(
        output["energy"].block().values,
        [structure.positions for structure in structures]
        + [structure.cell for structure in structures],
        is_training=is_training,
    )
    f_and_v = [-gradient for gradient in gradients]

    jitted_model = torch.jit.script(model)
    output = jitted_model(
        rascaline.torch.systems_to_torch(
            structures, positions_requires_grad=True, cell_requires_grad=True
        )
    )
    jitted_gradients = compute_gradient(
        output["energy"].block().values,
        [structure.positions for structure in structures]
        + [structure.cell for structure in structures],
        is_training=is_training,
    )
    jitted_f_and_v = [-jitted_gradient for jitted_gradient in jitted_gradients]

    for fv, jfv in zip(f_and_v, jitted_f_and_v):
        assert torch.allclose(fv, jfv)
