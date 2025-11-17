"""
How to prepare data for training
================================

This tutorial shows you how to organize your atomic structures and properties
for training machine learning models with metatrain.

.. attention::

    **Do you already have an XYZ file with your structures and properties?** If yes, you
    can probably skip this tutorial and go straight to training! This tutorial is for
    users who need to:

    - Combine data from multiple files
    - Work with very large datasets (>10,000 structures)
    - Pre-process data for faster training

.. _ASE database: https://ase-lib.org/ase/db/db.html

When to use different formats
------------------------------

**Small datasets (< 10,000 structures)**: Use simple XYZ files. Metatrain will load
everything into memory automatically. This is the easiest option for most users.

**Large datasets (10,000 - 1,000,000 structures)**: Use DiskDataset. This pre-processes
your data and loads it on-the-fly during training, avoiding memory issues.

**Very large datasets (> 1,000,000 structures) on HPC**: Use MemmapDataset. This is
optimized for parallel filesystems and avoids I/O bottlenecks.

This tutorial covers all three formats. Choose the one that fits your needs.

Getting Started
---------------

We'll demonstrate with a small example dataset. First, let's import the necessary Python
packages:
"""

# %%
#
import subprocess
from pathlib import Path

import ase.io
import numpy as np
import torch
from metatensor.torch import Labels, TensorBlock, TensorMap
from metatomic.torch import NeighborListOptions, systems_to_torch

from metatrain.utils.data.writers import DiskDatasetWriter
from metatrain.utils.neighbor_lists import get_system_with_neighbor_lists


# %%
# Option 1: Create a simple XYZ file (recommended for beginners)
# ---------------------------------------------------------------
#
# This is the easiest approach and works great for most datasets. An XYZ file stores
# atomic structures with their properties in a simple text format that ASE can read.
#
# **What you need:**
#
# - Atomic positions and elements
# - Target properties (energies, forces, etc.)
#
# **When to use:** Datasets with < 10,000 structures (works up to ~1 million on modern
# systems)
#
# Setting up the data
# ^^^^^^^^^^^^^^^^^^^
#
# For this example, we'll read structures from an existing file. In practice, your data
# might come from:
#
# - Quantum chemistry calculations (Gaussian, ORCA, CP2K, etc.)
# - Ab initio MD trajectories
# - Database files
#
# ASE_ can read many formats. Check the `ASE I/O documentation`_ for supported formats.
#
# .. _ASE: https://ase-lib.org/
# .. _ASE I/O documentation: https://wiki.fysik.dtu.dk/ase/ase/io/io.html

# In this example, all structures are in one file for simplicity.
# In reality, you might have multiple files from different calculations.
filelist = 100 * ["qm9_reduced_100.xyz"]

# %%
#
# Reading and adding properties
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Now we read structures and add properties. ASE stores:
#
# - **Scalar properties** (like energy) in the ``atoms.info`` dictionary
# - **Per-atom properties** (like forces) in the ``atoms.arrays`` dictionary
#
# .. tip::
#
#   If ASE doesn't automatically read a property from your file format, you can add it
#   manually:
#
#   - Scalars: ``atoms.info["energy"] = -100.0``
#   - Vectors: ``atoms.arrays["forces"] = force_array``  (shape: n_atoms × 3)
#   - Tensors: Flatten first, e.g., ``stress.reshape(-1, 9)`` for 3×3 stress tensors

frames = []
for i, fname in enumerate(filelist):
    # Read one structure from the file
    atoms = ase.io.read(fname, index=i)

    n_atoms = len(atoms)

    # Add energy (scalar property)
    atoms.info["U0"] = -100.0

    # Add forces (per-atom vector property)
    atoms.arrays["forces"] = np.zeros((n_atoms, 3))

    # Add a custom tensor property (e.g., stress or polarizability)
    # Tensors must be flattened: 3×3 becomes a vector of length 9
    atoms.arrays["my_tensor"] = np.zeros((n_atoms, 3, 3)).reshape(n_atoms, 9)

    frames.append(atoms)

# Write all structures to a single XYZ file
ase.io.write("data.xyz", frames)

# %%
#
# **That's it!** You now have a ``data.xyz`` file that metatrain can use
# directly. In your ``options.yaml``, reference the properties by their names:
#
# .. code-block:: yaml
#
#     training_set:
#         systems: "data.xyz"
#         targets:
#             energy:
#                 key: "U0"        # Must match the key we used above
#                 unit: "eV"
#                 forces: on       # Will look for "forces" in arrays
#
# Create a ``DiskDataset`` (large datasets)
# -----------------------------------------
#
# In addition to the systems and targets (as above), we also save the neighbor
# lists that the model will use during training. We first create the writer object that
# will write the data to a zip file.

disk_dataset_writer = DiskDatasetWriter("qm9_reduced_100.zip")

# %%
#
# Then we loop over all structures, convert them to the internal torch format using
# :func:`metatomic.torch.systems_to_torch`, compute the neighbor lists using
# :func:`metatrain.utils.neighbor_lists.get_system_with_neighbor_lists` and write
# everything to disk using the writer's ``write()`` method.

for i, fname in enumerate(filelist):
    atoms = ase.io.read(fname, index=i)

    system = systems_to_torch(atoms, dtype=torch.float64)
    system = get_system_with_neighbor_lists(
        system,
        [NeighborListOptions(cutoff=5.0, full_list=True, strict=True)],
    )
    energy = TensorMap(
        keys=Labels.single(),
        blocks=[
            TensorBlock(
                values=torch.tensor([[atoms.info["U0"]]], dtype=torch.float64),
                samples=Labels(
                    names=["system"],
                    values=torch.tensor([[i]]),
                ),
                components=[],
                properties=Labels("energy", torch.tensor([[0]])),
            )
        ],
    )
    disk_dataset_writer.write([system], {"energy": energy})

disk_dataset_writer.finish()

# %%
#
# Alternatively, you can also write the whole dataset at once, which might be more
# efficient (but also potentially run into memory issues). We use the same ``frames``
# that we created above.

disk_dataset_writer = DiskDatasetWriter("qm9_reduced_100_all_at_once.zip")

systems = systems_to_torch(frames, dtype=torch.float64)
systems = [
    get_system_with_neighbor_lists(
        system,
        [NeighborListOptions(cutoff=5.0, full_list=True, strict=True)],
    )
    for system in systems
]
energy = TensorMap(
    keys=Labels.single(),
    blocks=[
        TensorBlock(
            values=torch.tensor(
                [frame.info["U0"] for frame in frames], dtype=torch.float64
            ).reshape(-1, 1),
            samples=Labels.range("system", len(frames)),
            components=[],
            properties=Labels("energy", torch.tensor([[0]])),
        )
    ],
)

disk_dataset_writer.write(systems, {"energy": energy})
disk_dataset_writer.finish()

# %%
#
# The dataset is saved to disk. You can now provide it to ``metatrain`` as a
# dataset to train from, simply by replacing your ``.xyz`` file with the newly created
# zip file (e.g. ``read_from: qm9_reduced_100.zip``).
#
# Create a ``MemmapDataset`` (large datasets, parallel filesystems)
# -----------------------------------------------------------------
#
# If your dataset is large and you are using a parallel filesystem (e.g. on an HPC
# cluster), it is recommended to use a ``MemmapDataset`` instead of a ``DiskDataset``.
# The ``MemmapDataset`` stores the data inside memory-mapped numpy arrays instead of a
# zip file. Reading from this format avoids I/O bottlenecks, but it does not support
# spherical targets or storing neighbor lists.
#
# As an example, we will use 100 structures from a dataset of carbon structures. The
# numpy arrays must be saved inside a directory, using the following format.

structures = ase.io.read("carbon_reduced_100.xyz", index=":")

root = Path("carbon_reduced_100_memmap/")
root.mkdir(exist_ok=True)

ns_path = root / "ns.npy"
na_path = root / "na.npy"
a_path = root / "a.bin"
x_path = root / "x.bin"
c_path = root / "c.bin"
e_path = root / "e.bin"
f_path = root / "f.bin"
s_path = root / "s.bin"

ns = len(structures)
na = np.cumsum(np.array([0] + [len(s) for s in structures], dtype=np.int64))
np.save(ns_path, ns)
np.save(na_path, na)

a_mm = np.memmap(a_path, dtype="int32", mode="w+", shape=(na[-1],))
x_mm = np.memmap(x_path, dtype="float32", mode="w+", shape=(na[-1], 3))
c_mm = np.memmap(c_path, dtype="float32", mode="w+", shape=(ns, 3, 3))
e_mm = np.memmap(e_path, dtype="float32", mode="w+", shape=(ns, 1))
f_mm = np.memmap(f_path, dtype="float32", mode="w+", shape=(na[-1], 3))
s_mm = np.memmap(s_path, dtype="float32", mode="w+", shape=(ns, 3, 3))

for i, s in enumerate(structures):
    a_mm[na[i] : na[i + 1]] = s.numbers
    x_mm[na[i] : na[i + 1]] = s.get_positions()
    c_mm[i] = s.get_cell()[:]
    e_mm[i] = s.get_potential_energy()
    f_mm[na[i] : na[i + 1]] = s.arrays["force"]
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
# (e.g. ``read_from: carbon_reduced_100_memmap/``).
#
# For example, you can use the following options file:
#
# .. literalinclude:: options-memmap.yaml
#    :language: yaml

# Here, we run training as a subprocess, in reality you would run this from the command
# line as ``mtt train options-memmap.yaml``.
subprocess.run(["mtt", "train", "options-memmap.yaml"], check=True)
