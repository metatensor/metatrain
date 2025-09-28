"""
Saving a memory-mapped dataset
==============================

Large datasets may not fit into memory. In such cases, it is useful to save the
dataset to disk and load it on the fly during training. This example demonstrates
how to save a ``MemmapDataset`` for this purpose. Metatrain will then be able to
load ``MemmapDataset`` objects saved in this way to execute on-the-fly data loading.

Compared to ``DiskDataset``, the ``MemmapDataset`` stores the data in a format that
is more efficient to read from disk on parallel filesystems. However, it does not
support spherical targets or storing neighbor lists.
"""

# %%
#

import subprocess
from pathlib import Path

import ase.io
import numpy as np


# %%
#
# As an example, we will use 100 structures from a dataset of carbon structures.

structures = ase.io.read("carbon_reduced_100.xyz", index=":")

root = Path("carbon_reduced_100_mmap/")
root.mkdir()

N_path = root / "N.npy"
n_path = root / "n.npy"
a_path = root / "a.bin"
x_path = root / "x.bin"
c_path = root / "c.bin"
e_path = root / "e.bin"
f_path = root / "f.bin"
s_path = root / "s.bin"

N = len(structures)
n = np.cumsum(np.array([0] + [len(s) for s in structures], dtype=np.int64))
np.save(N_path, N)
np.save(n_path, n)

a_mm = np.memmap(a_path, dtype="int32", mode="w+", shape=(n[-1],))
x_mm = np.memmap(x_path, dtype="float32", mode="w+", shape=(n[-1], 3))
c_mm = np.memmap(c_path, dtype="float32", mode="w+", shape=(N, 3, 3))
e_mm = np.memmap(e_path, dtype="float32", mode="w+", shape=(N, 1))
f_mm = np.memmap(f_path, dtype="float32", mode="w+", shape=(n[-1], 3))
s_mm = np.memmap(s_path, dtype="float32", mode="w+", shape=(N, 3, 3))

for i, s in enumerate(structures):
    a_mm[n[i] : n[i + 1]] = s.numbers
    x_mm[n[i] : n[i + 1]] = s.get_positions()
    c_mm[i] = s.get_cell()[:]
    e_mm[i] = s.get_potential_energy()
    f_mm[n[i] : n[i + 1]] = s.arrays["force"]
    s_mm[i] = -s.info["virial"] / s.get_volume()

a_mm.flush()
x_mm.flush()
c_mm.flush()
e_mm.flush()
f_mm.flush()
s_mm.flush()

# %%
#
# The dataset is saved to disk. You can now provide it to ``metatrain`` as a
# dataset to train from, simply by specifying the newly created
# directory as the path from which to read the systems
# (e.g. ``read_from: carbon_reduced_100_mmap/``).
#
# For example, you can use the following options file:
#  .. literalinclude:: options.yaml
#   :language: yaml

subprocess.run(["mtt", "train", "options.yaml"])
