import warnings

import ase
import ase.build
import ase.io
import numpy as np
from ase.calculators.emt import EMT


calculator = EMT()

structures = []

# Create multiple bulk structures with valid stress
for _ in range(5):
    bulk = ase.build.bulk("Cu", "fcc", a=3.6, cubic=True)
    bulk.rattle(0.01)  # Small perturbation to make structures different
    bulk.calc = calculator
    bulk.info["energy"] = bulk.get_potential_energy()
    bulk.arrays["forces"] = bulk.get_forces()
    bulk.info["stress"] = bulk.get_stress(voigt=False)
    bulk.calc = None
    structures.append(bulk)

# Create multiple molecules with NaN stress (stress not defined for molecules)
for i in range(5):
    molecule = ase.Atoms("Cu2", positions=[[0, 0, 0], [2.5 + 0.1 * i, 2.5, 2.5]])
    molecule.calc = calculator
    molecule.info["energy"] = molecule.get_potential_energy()
    molecule.arrays["forces"] = molecule.get_forces()
    molecule.info["stress"] = np.full((3, 3), np.nan)
    molecule.calc = None
    structures.append(molecule)

# Create multiple slabs with NaN stress (stress not defined for slabs)
for _ in range(5):
    slab = ase.build.fcc111("Cu", size=(2, 2, 4), vacuum=10.0)
    slab.pbc = (True, True, False)
    slab.rattle(0.01)  # Small perturbation
    slab.calc = calculator
    slab.info["energy"] = slab.get_potential_energy()
    slab.arrays["forces"] = slab.get_forces()
    slab.info["stress"] = np.full((3, 3), np.nan)
    slab.calc = None
    structures.append(slab)

# Write structures to file
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    ase.io.write("structures.xyz", structures)
