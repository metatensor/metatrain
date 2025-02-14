"""
Training a polarizability model using the TensorBasis in metatrain
==================================================================

This tutorial demonstrates how to train equivariant models in metatrain constructed as
linear combinations of the elements of a tensorial basis with neural networks
(SOAP-based Behler-Parrinello nets) as coefficients of the linear combination.
"""

# %%
#
from glob import glob

import ase.io
import matplotlib.pyplot as plt
import metatensor.torch as mts
import numpy as np
import torch
from featomic.torch.clebsch_gordan import cartesian_to_spherical


# %%
#
# Read some bulk water ASE frames decorated with the polarizability (Cartesian) tensor.
# Extract the polarizability from the ase.Atoms.info dictionary.
frames = ase.io.read("bulk_water_100.xyz", ":")
polarizabilities = np.array([frame.info["alpha"].reshape(3, 3) for frame in frames])

# %%
#
# Create a ``metatensor.torch.TensorMap`` containing the Cartesian polarizability tensor
# values and the respective metadata

cartesian_tensormap = mts.TensorMap(
    keys=mts.Labels.single(),
    blocks=[
        mts.TensorBlock(
            samples=mts.Labels.range("system", len(frames)),
            components=[mts.Labels.range(name, 3) for name in ["xyz_1", "xyz_2"]],
            properties=mts.Labels(["alpha"], torch.tensor([[0]])),
            values=torch.from_numpy(polarizabilities).unsqueeze(-1),
        )
    ],
)

# %%
#
# Extract from the Cartesian polarizability tensor its irreducible spherical components
#
spherical_tensormap = mts.remove_dimension(
    cartesian_to_spherical(cartesian_tensormap, components=["xyz_1", "xyz_2"]),
    "keys",
    "_",
)
#

# %%
#
# We drop the block with ``o3_sigma=-1``, as polarizability should be symmetric and
# therefore any non-zero pseudo-vector component is spurious.
#
spherical_tensormap = mts.drop_blocks(
    spherical_tensormap, mts.Labels(["o3_sigma"], torch.tensor([[-1]]))
)
# %%
#
# Let's save the spherical components of the polarizability tensor to disk
#
mts.save("spherical_polarizability.npz", spherical_tensormap)
# %%
#
# Write the metatrain ``options.yaml`` file for the training of the polarizability
# model
#
# .. literalinclude:: options.yaml
#   :language: yaml
#
#
# The most relevant parts here are the ``architecture`` and ``training_set`` sections.
# The ``architecture`` section specifies the neural network architecture (SOAP-BPNN, in
# this case), the SOAP hyperparameters, and the architecture of the used multi-layer
# perceptron.
# The ``training_set`` section specifies the training data and the target tensorial
# properties. Each spherical component of the target tensorial property is specified by
# a dictionary enumerating the irreps of the SO(3) group that the target tensorial
# property transforms under. These should match the keys of the ``TensorMap`` object
# given in input.
# Here we run for 100 epochs on a 100-structure dataset, but you may want to increase
# use more epochs and more structures for a better model.

# %%
#
# Train the model using the command:
#
# .. code:: bash
#
#    mtt train options.yaml 2> err.log
#
# The ``stderr`` output will be redirected to the ``err.log`` file to avoid seeing too
# many warnings.

# %%
#
# At the end of the training, the model will be saved in the current directory as
# ``model.pt`` and the ``extensions`` directory containing some required files for the
# model to be used. In the ``outputs`` directory, under nested directories named after
# the date and time of the beginning of the training, you will find the training logs
# and model checkpoints.

# %%
#
# In case you need to export a specific checkpoints, you can do so using:
#
# .. code:: bash
#
#    mtt export /path/to/model.ckpt -o model_name.pt
#

# %%
#
# To evaluate the model, we can write the following ``eval.yaml`` file:
#
# .. code:: yaml
#
#    systems:
#        read_from: bulk_water_100.xyz
#        length_unit: angstrom
#

# %%
#
# And then evaluate the model using the command:
#
# .. code:: bash
#
#    mtt eval model.pt eval.yaml -e extensions/ -o outputs.mts 2> err.log
#
# The ``stderr`` output will be redirected to the ``err.log`` file to avoid seeing too
# many warnings.

# %%
#
# After evaluation is completed, three files will be created in the current directory.
# The only one relevant for this example is ``outputs_mtt::polarizability.npz``,
# containing the predicted values of the target tensorial properties as binary
# ``TensorMap`` objects.

# %%
#
# Load the predicted values of the target tensorial properties and the
# training/validation/test indices to slice the predicted values. Since the indices are
# inside the outputs directory, we need to find the latest folder containing the
# indices.
# Since we passed the scalar component of the polarizability as an "energy" target, we
# after loading the predicted values we can update the metadata to reflect the correct
# information about the target.

predicted_polarizabilities = mts.load("outputs_mtt::polarizability.npz")

index_folder = sorted(glob("outputs/*/*/indices"))[-1]
indices = {
    "train": np.loadtxt(f"{index_folder}/training.txt", dtype=int),
    "val": np.loadtxt(f"{index_folder}/validation.txt", dtype=int),
    "test": np.loadtxt(f"{index_folder}/test.txt", dtype=int),
}

# %%
#
# Plot the parity plots of the predicted values of the target tensorial properties
# against the true values for the training, validation, and test sets.

fig, axes = plt.subplots(1, 2, figsize=(10, 5))

for key, ax in zip(spherical_tensormap.keys, axes):
    o3_lambda = key["o3_lambda"]
    o3_sigma = key["o3_sigma"]
    ax.set_aspect("equal")
    ax.set_xlabel("Target")
    ax.set_ylabel("Prediction")
    ax.set_title(f"$\lambda={o3_lambda}$, $\sigma={o3_sigma}$")
    target = spherical_tensormap[key].values
    prediction = predicted_polarizabilities[key].values
    ax.set_xlim(target.min(), target.max())
    ax.set_ylim(target.min(), target.max())
    ax.plot([target.min(), target.max()], [target.min(), target.max()], "--k")

    for tset, idx in indices.items():
        ax.plot(target[idx].flatten(), prediction[idx].flatten(), ".", label=tset)
    ax.legend()
fig.tight_layout()
plt.show()
