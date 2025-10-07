"""
How to prepare data for training
================================

.. attention::

    This tutorial is only relevant for users who need to prepare their data from scratch
    from several files or for big datasets. If you already have your data in a common
    file format (like XYZ or ASE database), you can skip this tutorial and directly
    start training.

``metatrain`` can read data from various sources, including common file formats like
XYZ, ASE databases, and also from metrain's
:class:`metatrain.utils.data.dataset.DiskDataset <DiskDataset>` file.

For the small datasets (<10k structures), you can simply provide an XYZ file or an ASE
database to ``metatrain``, and it will handle the data loading for you. Large datasets
(>10k structures) may not fit into the GPU memory. In such cases, it is useful to
pre-process the dataset, save it to disk and load it on the fly during training.

We start by importing the necessary packages.
"""

# %%
#
import ase.io
import numpy as np
import torch
from metatensor.torch import Labels, TensorBlock, TensorMap
from metatomic.torch import NeighborListOptions, systems_to_torch

from metatrain.utils.data.writers import DiskDatasetWriter
from metatrain.utils.neighbor_lists import get_system_with_neighbor_lists


# %%
# Create a XYZ training file (small datasets)
# -------------------------------------------
#
# First, we will show how to create a XYZ file with fields of the target properties. As
# an example, we will use 100 structures from a file read by ASE. Since files from
# reference calculations may be located at different directories we first create a
# list of all path that we want to read from. Here, for simplicity, we assume that all
# files are located in the same directory.

filelist = 100 * ["qm9_reduced_100.xyz"]

# %%
#
# We will now read the structures using the ASE package. Check the ase documentation for
# more details on how to read different file formats. Instead of creating the ``atoms``
# object by reading from disk, you can also create an
# :class:`ase.Atoms` object containing the chemical ``symbols``, ``positions``, the
# ``cell`` and the periodic boundary conditions (``pbc``) by hand using its constructor.
#
# .. hint::
#
#   If a property is not read by the :func:`ase.io.read` function, you can add custom
#   scalar properties to the ``info`` dictionary. Vector properties (e.g. forces) can be
#   added to the ``arrays`` dictionary. Tensor properties (e.g. stress) must
#   be flattened before adding them to the ``arrays`` dictionary.

frames = []
for i, fname in enumerate(filelist):
    atoms = ase.io.read(fname, index=i)

    n_atoms = len(atoms)
    # scalar
    atoms.info["U0"] = -100.0
    # vector
    atoms.arrays["forces"] = np.zeros((n_atoms, 3))
    # tensor
    atoms.arrays["my_tensor"] = np.zeros((n_atoms, 3, 3)).reshape(n_atoms, 9)

    frames.append(atoms)

ase.io.write("data.xyz", frames)

# %%
#
# .. note::
#
#   The names of the added properties (like, ``U0``, etc.) must be referenced correctly
#   in the ``options.yaml`` file.
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
# efficient. Note that we use the ``frames`` that we created above.

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
