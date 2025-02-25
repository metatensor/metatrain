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
from glob import glob

import matplotlib.pyplot as plt
import metatensor.torch as mts
import numpy as np


# %%
#
# First, we need to prepare the dataset by saving the polarizability in
# the form of spherical tensors. This is done using the ``metatensor`` library.
#
# .. literalinclude:: save_tensor_map.py
#   :language: python

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

# %%
#
# At the end of the training, the model will be saved in the current directory as
# ``model.pt`` and the ``extensions`` directory containing some required files for the
# model to be used. In the ``outputs`` directory, under nested directories named after
# the date and time of the beginning of the training, you will find the training logs
# and model checkpoints.

# %%
#
# In case you need to export a specific checkpoint (most likely from the ``outputs``
# directory), you can do so using:
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

target_polarizabilities = mts.load("spherical_polarizabilities.mts")
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

for key, ax in zip(target_polarizabilities.keys, axes):
    o3_lambda = key["o3_lambda"]
    o3_sigma = key["o3_sigma"]
    ax.set_aspect("equal")
    ax.set_xlabel("Target")
    ax.set_ylabel("Prediction")
    ax.set_title(rf"$\lambda={o3_lambda}$, $\sigma={o3_sigma}$")
    target = target_polarizabilities[key].values
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
