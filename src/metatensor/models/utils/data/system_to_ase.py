import ase
from metatensor.torch.atomistic import System


def system_to_ase(system: System) -> ase.Atoms:
    """Converts a ``metatensor.torch.atomistic.System`` to an ``ase.Atoms`` object."""

    # Convert the system to an ASE atoms object
    positions = system.positions.detach().cpu().numpy()
    numbers = system.species.detach().cpu().numpy()
    cell = system.cell.detach().cpu().numpy()
    pbc = list(cell.any(axis=1))
    atoms = ase.Atoms(
        numbers=numbers,
        positions=positions,
        cell=cell,
        pbc=pbc,
    )

    return atoms
