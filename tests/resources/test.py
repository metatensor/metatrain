# y z x

import ase.io


structures = ase.io.read("ethanol_reduced_100.xyz", index=":")

for structure in structures:
    structure.arrays["forces_spherical"] = structure.arrays["forces"][:, [1, 2, 0]]

ase.io.write("ethanol_reduced_100_spherical.xyz", structures)
