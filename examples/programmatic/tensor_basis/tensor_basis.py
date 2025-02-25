"""
Training an equivariant model for the polarizability
====================================================

This tutorial demonstrates how to train an equivariant model for the molecular
polarizability. In this example, the SOAP-BPNN architecture is used to predict
equivariant targets. Internally, this is done using the "tensor basis" construction,
i.e., a linear combinations of the elements of a tensorial basis with the
neural network predicting the invariant coefficients of the linear combination.
"""

# %%
#
import subprocess
import sys
from glob import glob

import ase.io
import matplotlib.pyplot as plt
import metatensor as mts
import numpy as np
from featomic.clebsch_gordan import cartesian_to_spherical


# %%
#
# In addition to the SOAP-BPNN dependencies, training on a tensor target requires the
# ``sphericart-torch`` package. To install it, we will run ``pip install`` from this
# script.
subprocess.check_call([sys.executable, "-m", "pip", "install", "sphericart-torch"])

# %%
#
# Read a subset of 1000 molecules from the QM7x dataset in the XYZ format decorated with
# the polarizability (Cartesian) tensor.
# Extract the polarizability from the ase.Atoms.info dictionary.
#
molecules = ase.io.read("qm7x_reduced_100.xyz", ":")
polarizabilities = np.array(
    [molecule.info["polarizability"].reshape(3, 3) for molecule in molecules]
)

# %%
#
# Create a ``metatensor.torch.TensorMap`` containing the Cartesian polarizability tensor
# values and the respective metadata

cartesian_tensormap = mts.TensorMap(
    keys=mts.Labels.single(),
    blocks=[
        mts.TensorBlock(
            samples=mts.Labels.range("system", len(molecules)),
            components=[mts.Labels.range(name, 3) for name in ["xyz_1", "xyz_2"]],
            properties=mts.Labels(["polarizability"], np.array([[0]])),
            values=polarizabilities[:, :, :, None],
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

# %%
#
# We drop the block with ``o3_sigma=-1``, as polarizability should be symmetric and
# therefore any non-zero pseudo-vector component is spurious.
#
spherical_tensormap = mts.drop_blocks(
    spherical_tensormap, mts.Labels(["o3_sigma"], np.array([[-1]]))
)
# %%
#
# Let's save the spherical components of the polarizability tensor to disk
#
# For now, making each array contiguous is necessary for the save function to work
# (https://github.com/metatensor/metatensor/issues/870)
blocks = []
for block in spherical_tensormap.blocks():
    new_block = mts.TensorBlock(
        samples=block.samples,
        components=block.components,
        properties=block.properties,
        values=np.ascontiguousarray(block.values),
    )
    blocks.append(new_block)
spherical_tensormap = mts.TensorMap(keys=spherical_tensormap.keys, blocks=blocks)

# save
mts.save("spherical_polarizability.mts", spherical_tensormap)
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
# Now that the dataset has been saved, we can train a model on it.
#
# .. code:: bash
#
#    mtt train options.yaml
#
# In this case, we launch the above command from this script
subprocess.run(["mtt", "train", "options.yaml"])

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
# .. literalinclude:: eval.yaml
#   :language: yaml
#

# %%
#
# And then evaluate the model using the command:
#
# .. code:: bash
#
#    mtt eval model.pt eval.yaml -e extensions/ -o outputs.mts
#
# In this case, we launch the above command from this script
subprocess.run(
    ["mtt", "eval", "model.pt", "eval.yaml", "-e", "extensions/", "-o", "outputs.mts"]
)

# %%
#
# After evaluation is completed, three files will be created in the current directory.
# The only one relevant for this example is ``outputs_mtt::polarizability.mts``,
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

predicted_polarizabilities = mts.load("outputs_mtt::polarizability.mts")

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

fig, axes = plt.subplots(1, 2, figsize=(7, 3.5))

for key, ax in zip(spherical_tensormap.keys, axes):
    o3_lambda = key["o3_lambda"]
    o3_sigma = key["o3_sigma"]
    ax.set_aspect("equal")
    ax.set_xlabel("Target")
    ax.set_ylabel("Prediction")
    ax.set_title(rf"$\lambda={o3_lambda}$, $\sigma={o3_sigma}$")
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

# %%
