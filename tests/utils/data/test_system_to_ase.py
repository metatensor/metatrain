import torch
from metatensor.torch.atomistic import System

from metatensor.models.utils.data import system_to_ase


def test_system_to_ase():
    """Tests the conversion of a System to an ASE atoms object."""
    # Create a system
    system = System(
        positions=torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]),
        types=torch.tensor([1, 8]),
        cell=torch.tensor([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]]),
    )

    # Convert the system to an ASE atoms object
    atoms = system_to_ase(system)

    # Check the positions
    assert atoms.positions.tolist() == system.positions.tolist()

    # Check the species
    assert atoms.numbers.tolist() == system.types.tolist()

    # Check the cell
    assert atoms.cell.tolist() == system.cell.tolist()
    assert atoms.pbc.tolist() == [True, True, True]
