""".. _classifierexample:

Training a Classifier Model
============================

This tutorial demonstrates how to train a classifier model using metatrain.
The classifier model is a transfer learning architecture that takes a pre-trained
model, freezes its backbone, and trains a small multi-layer perceptron (MLP) on
top of the extracted features for classification tasks.

In this example, we will classify carbon allotropes (diamond, graphite, and graphene).

Creating the Dataset
--------------------

First, we need to create a dataset with different carbon structures. We'll generate
simple structures for diamond, graphite, and graphene, and label them with class
identifiers (0, 1, 2).

"""

# %%

import subprocess

import ase
import ase.io
import matplotlib.pyplot as plt
import numpy as np
from ase.build import bulk, graphene
from metatomic.torch import ModelOutput
from metatomic.torch.ase_calculator import MetatomicCalculator


# %%
#
# We generate structures for three carbon allotropes:
# - Diamond (class 0)
# - Graphite (class 1)
# - Graphene (class 2)

np.random.seed(42)

structures = []

# Generate 100 diamond structures with small random perturbations
for i in range(100):
    diamond = bulk("C", "diamond", a=3.57)
    diamond = diamond * (2, 2, 2)  # Make it bigger
    diamond.rattle(stdev=0.5, seed=i)  # Add random perturbations
    diamond.info["class"] = 0.0  # Label as diamond
    structures.append(diamond)

# Generate 100 graphite structures (using layered graphene-like structures)
for i in range(100):
    # Create a graphite-like structure
    graphite = graphene(formula="C2", size=(3, 3, 1), a=2.46, vacuum=None)
    # Stack two layers
    layer2 = graphite.copy()
    layer2.translate([0, 0, 3.35])
    graphite.extend(layer2)
    graphite.set_cell([graphite.cell[0], graphite.cell[1], [0, 0, 6.7]])
    graphite.rattle(stdev=0.5, seed=i)
    graphite.info["class"] = 1.0  # Label as graphite
    structures.append(graphite)

# Generate 100 graphene structures (single layer)
for i in range(100):
    graphene_struct = graphene(formula="C2", size=(3, 3, 1), a=2.46, vacuum=10.0)
    graphene_struct.rattle(stdev=0.5, seed=i)
    graphene_struct.info["class"] = 2.0  # Label as graphene
    structures.append(graphene_struct)

# Save the structures to a file (these will be used for training)
ase.io.write("carbon_allotropes.xyz", structures)

# %%
#
# Getting a pre-trained universal model
# -------------------------------------
#
# Here, we download a pre-trained model checkpoint that will serve as the backbone
# for our classifier. We will use PET-MAD, a universal interatomic potential for
# materials and molecules.

subprocess.run(
    [
        "wget",
        "https://huggingface.co/lab-cosmo/pet-mad/resolve/v1.0.2/models/pet-mad-v1.0.2.ckpt",
    ],
    check=True,
)

# %%
#
# Training the Classifier
# -----------------------
#
# Now we can train the classifier. The classifier will learn to use the features learned
# by PET-MAD to classify our carbon allotropes.
#
# The key hyperparameters are:
#
# - ``hidden_sizes``: The dimensions of the MLP layers. The last dimension (2 in this
#   case) acts as a bottleneck that can be used to extract collective variables. If
#   collective variables are not needed, this should be set to a larger value.
# - ``model_checkpoint``: Path to the the pre-trained model (here PET-MAD).
#
# .. literalinclude:: options-classifier.yaml
#    :language: yaml

# Here, we run training as a subprocess, in reality you would run this from the command
# line as ``mtt train options-classifier.yaml``.
subprocess.run(
    ["mtt", "train", "options-classifier.yaml", "-o", "classifier.pt"],
    # check=True,
)

# %%
#
# Using the Trained Classifier
# ----------------------------
#
# Once the classifier is trained, we can use it to:
# 1. Predict class labels for new structures
# 2. Extract bottleneck features (collective variables)
#
# Let's test the classifier on some structures:

# Load the model
calc = MetatomicCalculator("classifier.pt")

structures = ase.io.read("carbon_allotropes.xyz", index=":")

# Get predictions
correct_count = 0
for structure in structures:
    prediction = (
        calc.run_model(
            structure,
            {"mtt::class": ModelOutput(per_atom=False)},
        )["mtt::class"]
        .block()
        .values.cpu()
        .squeeze(0)
        .numpy()
    )
    predicted_class = np.argmax(prediction)
    actual_class = int(structure.info["class"])
    if predicted_class == actual_class:
        correct_count += 1

print(f"Classifier accuracy: {(correct_count / len(structures)) * 100:.2f}% correct")

# %%
#
# Now, we extract the features learned by the classifier in our "bottleneck" layer.
# Having only 2 dimensions allows us to easily visualize them. A low dimensionality
# is also necessary if we want to use these features as collective variables in enhanced
# sampling simulations.

# Extract features
bottleneck_features = []
labels = []
for structure in structures:
    features = (
        calc.run_model(
            structure,
            {"features": ModelOutput(per_atom=False)},
        )["features"]
        .block()
        .values.cpu()
        .squeeze(0)
        .numpy()
    )
    bottleneck_features.append(features)
    labels.append(int(structure.info["class"]))
bottleneck_features = np.array(bottleneck_features)
labels = np.array(labels)

# Plot the features for the three classes
plt.figure(figsize=(5, 3))
for class_id in np.unique(labels):
    mask = labels == class_id
    if class_id == 0:
        label = "Diamond"
    elif class_id == 1:
        label = "Graphite"
    else:
        label = "Graphene"
    plt.scatter(
        bottleneck_features[mask, 0],
        bottleneck_features[mask, 1],
        label=label,
        alpha=0.7,
    )
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("Features from Classifier")
plt.legend()
plt.grid()
plt.show()
