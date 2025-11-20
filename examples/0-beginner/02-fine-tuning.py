r"""
.. _label_fine_tuning_concept:

Finetuning example
==================

.. warning::

  Finetuning is currently only available for the PET architecture.


This is a simple example for fine-tuning PET-MAD (or a general PET model), that
can be used as a template for general fine-tuning with metatrain.
Fine-tuning a pretrained model allows you to obtain a model better suited for
your specific system. You need to provide a dataset of structures that have
been evaluated at a higher reference level of theory, usually DFT. Fine-tuning
a universal model such as PET-MAD allows for reasonable model performance even if little training
data is available.
It requires using a pre-trained model checkpoint with the ``mtt train`` command and setting the
new targets corresponding to the new level of theory in the ``options.yaml`` file.


In order to obtain a pretrained model, you can use a PET-MAD checkpoint from huggingface

.. code-block:: bash

  wget https://huggingface.co/lab-cosmo/pet-mad/resolve/v1.1.0/models/pet-mad-v1.1.0.ckpt

Next, we set up the ``options.yaml`` file. We can specify the fine-tuning method
in the ``finetune`` block in the ``training`` options of the ``architecture``.
Here, the basic ``full`` option is chosen, which finetunes all weights of the model.
All available fine-tuning methods are found in the concepts page
:ref:`Fine-tuning <fine-tuning>`. This section discusses implementation details,
options and recommended use cases. Other fine-tuning options can be simply substituted in this script,
by changing the ``finetune`` block. 

.. note::
Since our dataset has energies and forces obtained from reference calculations, different from
the reference of the pretrained model, it is recommended to create a new energy head.
Using this so-called energy variant can be simply invoked by requesting a new target in the options
file. Follow the nomenclature energy/{yourname}


Furthermore, you need to specify the checkpoint, that you want to fine-tune in
the ``read_from`` option.

A simple ``options-ft.yaml`` file for this task could look like this:


.. code-block:: yaml

  architecture:
    name: pet
    training:
      num_epochs: 1000
      learning_rate: 1e-5
      finetune:
        method: full
        read_from: path/to/checkpoint.ckpt
  training_set:
    systems:
        read_from: dataset.xyz
        reader: ase
        length_unit: angstrom
    targets:
        energy/ft:
            quantity: energy
            read_from: dataset.xyz
            reader: ase
            key: energy
            unit: eV
            forces:
                read_from: dataset.xyz
                reader: ase
                key: forces
            stress:
                read_from: dataset.xyz
                reader: ase
                key: stress

  test_set: 0.1
  validation_set: 0.1

In this example, we specified generic but reasonable ``num_epochs`` and ``learning_rate``
parameters. The ``learning_rate`` is chosen to be relatively low to stabilise
training.

.. warning::

  Note that in ``targets`` we use the PET-MAD ``energy`` head. This means, that there won't be a new head
  for the new reference energies provided in your dataset. This can lead to bad performance, if the reference
  energies differ from the ones used in pretraining (different levels of theory, or different electronic structure
  software used). In future it is recommended to create a new ``energy`` target for the new level of theory.
  Find more about this in :ref:`Transfer-Learning <transfer-learning>`



We assumed that the pre-trained model is trained on the dataset ``dataset.xyz`` in which
energies are written in the ``energy`` key of the ``info`` dictionary of the
energies. Additionally, forces and stresses should be provided with corresponding keys
which you can specify in the ``options.yaml`` file under ``targets``.
Further information on specifying targets can be found in the :ref:`data section of the Training YAML Reference
<data-section>`.

.. note::

  It is important that the ``length_unit`` is set to ``angstrom`` and the ``energy`` ``unit`` is ``eV`` in order
  to match the units of your reference data.


After setting up your ``options-ft.yaml`` file, you can then simply run:

.. code-block:: bash
  mtt train options-ft.yaml

You can check finetuning training curves by parsing the ``train.csv`` that is written by ``mtt train``
"""

# %%
#
import matplotlib.pyplot as plt
import numpy as np

# %%
#
csv_path = 'outputs/2025-11-19/18-36-04/train.csv'
with open(csv_path, "r") as f:
    header = f.readline().strip().split(",")
    f.readline()  # skip units row

# Build dtype
dtype = [(h, float) for h in header]

# Load data as plain float array
data = np.loadtxt(csv_path, delimiter=",", skiprows=2)

# Convert to structured
structured = np.zeros(data.shape[0], dtype=dtype)
for i, h in enumerate(header):
    structured[h] = data[:, i]

r"""
Now, let's plot the learning curves.
"""
# %%
#
training_energy_RMSE = structured["training energy/ft RMSE (per atom)"]
training_forces_MAE = structured["training forces[energy/ft] MAE"]
validation_energy_RMSE = structured["validation energy/ft RMSE (per atom)"]
validation_forces_MAE = structured["validation forces[energy/ft] MAE"]

fig, axs = plt.subplots(1, 2, figsize=(12, 5))

axs[0].plot(training_energy_RMSE, label="training energy/ft RMSE (per atom)")
axs[0].plot(validation_energy_RMSE, label="validation energy/ft RMSE (per atom)")
axs[0].set_xlabel("Epochs")
axs[0].set_ylabel("energy / meV")
axs[0].legend()
axs[1].plot(training_forces_MAE, label="training forces[energy/ft] MAE")
axs[1].plot(validation_forces_MAE, label="validation forces[energy/ft] MAE")
axs[1].set_ylabel("force / meV/A")
axs[1].set_xlabel("Epochs")
axs[1].legend()
plt.tight_layout()
plt.show()

r"""
You can see that the validation loss still decreases, however, for the sake of brevity of this exercise 
we only finetuned for 25 epochs. As further check for how well your fine-tuned model performs on a dataset
of choice, we can check the parity plots for energy and force (see :ref:`Parity plots <parity-plot>`)
For evaluation, we can compare performance of our fine-tuned model and the base model PET-MAD.
Using ``mtt eval`` we can simply run:

.. code-block:: bash
  mtt eval model.pt options-eval.yaml -o output-ft.xyz 

and reader the energy in the xyz header. Another possibility is to load your fine-tuned model ``model.pt``
as metatomic model and evaluate energies and forces with ase.
"""

# %%
#
from metatomic.torch.ase_calculator import MetatomicCalculator
targets = ase.io.read("/Users/markusfasching/EPFL/Work/metatrain/tests/resources/ethanol_reduced_100.xyz", format='extxyz', index=':')
calc_ft = MetatomicCalculator('model.pt', variants={"energy": "ft"})

e_targets = np.array(
    [frame.get_total_energy() / len(frame) for frame in targets]
)  # target energies
f_targets = np.array(
    [frame.get_forces().flatten() for frame in targets]
).flatten()  # target forces

for frame in targets:
    frame.set_calculator(calc_ft)

e_predictions = np.array(
    [frame.get_total_energy() / len(frame) for frame in targets]
)  # predicted energies
f_predictions = np.array(
    [frame.get_forces().flatten() for frame in targets]
).flatten()  # predicted forces

# %%
#
fig, axs = plt.subplots(1, 2, figsize=(12, 5))

# Parity plot for energies
axs[0].scatter(e_targets, e_predictions, label='FT')
axs[0].axline((np.min(e_targets), np.min(e_targets)), slope=1, ls="--", color="red")
axs[0].set_xlabel("Target energy / meV")
axs[0].set_ylabel("Predicted energy / meV")
min_e = np.min(np.array([e_targets, e_predictions])) - 2
max_e = np.max(np.array([e_targets, e_predictions])) + 2
axs[0].set_title("Energy Parity Plot")

# Parity plot for forces
axs[1].scatter(f_targets, f_predictions, alpha=0.5, label='FT')
axs[1].axline((np.min(f_targets), np.min(f_targets)), slope=1, ls="--", color="red")
axs[1].set_xlabel("Target force / meV/Å")
axs[1].set_ylabel("Predicted force / meV/Å")
min_f = np.min(np.array([f_targets, f_predictions])) - 2
max_f = np.max(np.array([f_targets, f_predictions])) + 2
axs[1].set_title("Force Parity Plot")

fig.tight_layout()
plt.show()

r"""
Further fine-tuning examples can be found in the
`AtomisticCookbook <https://atomistic-cookbook.org/examples/pet-finetuning/pet-ft.html>`_
"""





















"""
How to prepare data for training
================================

.. attention::

    This tutorial is showing how to finetune a pretrained model on a new level of theory.
    All you need is a model, such as the universal model PET-MAD and a dataset in xyz 
    format with reference energies (and forces, stress if wanted).


For the small datasets (<10k structures), you can simply provide an XYZ file or an ASE
database to ``metatrain``, and it will handle the data loading for you. Large datasets
(>10k structures) may not fit into the GPU memory. In such cases, it is useful to
pre-process the dataset, save it to disk and load it on the fly during training.

In this tutorial, we will show how to prepare data for training using three different
formats. You can choose the one that best fits your needs.

We start by importing the necessary packages.
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
# Create a XYZ training file (small datasets)
# -------------------------------------------
#
# First, we will show how to create a XYZ file with fields corresponding to the target
# properties. On modern HPC systems, this format is suitable for datasets up to around
# 1M structures. As an example, we will use 100 structures from a file read by ASE_.
# Since files from reference calculations may be located in different directories, we
# first create a list of all path that we want to read from. Here, for simplicity, we
# assume that all files are located in the same directory.
#
# .. _ASE: https://ase-lib.org/

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
