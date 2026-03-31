import ase.io
import numpy as np


all_structures = ase.io.read("mad-val-metatrain-filtered.xyz", index=":")
np.random.shuffle(all_structures)
ase.io.write("small.xyz", all_structures[:1000])
