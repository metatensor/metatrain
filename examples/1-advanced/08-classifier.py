""".. _classifierexample:

Training a Classifier Model
============================

This tutorial demonstrates how to train a classifier model using metatrain.
The classifier model is a transfer learning architecture that takes a pre-trained
model, freezes its backbone, and trains a small multi-layer perceptron (MLP) on
top of the extracted features for classification tasks.

In this example, we will classify carbon allotropes (diamond, graphite, and graphene)
using a bottleneck size of 2, which can be useful for extracting collective variables
for downstream analysis.

.. note::

   This is an experimental feature. The classifier model is designed for
   system-level classification tasks where each structure belongs to a single class.

Creating the Dataset
--------------------

First, we need to create a dataset with different carbon structures. We'll generate
simple structures for diamond, graphite, and graphene, and label them with class
identifiers (0.0, 1.0, 2.0).

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
# - Diamond (class 0.0): sp3 hybridized, cubic structure
# - Graphite (class 1.0): sp2 hybridized, layered structure
# - Graphene (class 2.0): sp2 hybridized, monolayer

np.random.seed(42)

structures = []

# Generate 20 diamond structures with small random perturbations
for i in range(50):
    diamond = bulk("C", "diamond", a=3.57)
    diamond = diamond * (2, 2, 2)  # Make it bigger
    diamond.rattle(stdev=0.1, seed=i)  # Add random perturbations
    diamond.info["class"] = 0.0  # Label as diamond
    structures.append(diamond)

# Generate 20 graphite structures (using layered graphene-like structures)
for i in range(50):
    # Create a graphite-like structure
    graphite = graphene(formula="C2", size=(3, 3, 1), a=2.46, vacuum=None)
    # Stack two layers
    layer2 = graphite.copy()
    layer2.translate([0, 0, 3.35])
    graphite.extend(layer2)
    graphite.set_cell([graphite.cell[0], graphite.cell[1], [0, 0, 6.7]])
    graphite.rattle(stdev=0.1, seed=i)
    graphite.info["class"] = 1.0  # Label as graphite
    structures.append(graphite)

# Generate 20 graphene structures (single layer)
for i in range(50):
    graphene_struct = graphene(formula="C2", size=(3, 3, 1), a=2.46, vacuum=10.0)
    graphene_struct.rattle(stdev=0.1, seed=i)
    graphene_struct.info["class"] = 2.0  # Label as graphene
    structures.append(graphene_struct)

# Shuffle the structures
np.random.shuffle(structures)

# Save to file
ase.io.write("carbon_allotropes.xyz", structures)

print(f"Generated {len(structures)} structures:")
print("  - Diamond: 50 structures (class 0.0)")
print("  - Graphite: 50 structures (class 1.0)")
print("  - Graphene: 50 structures (class 2.0)")

# %%
#
# Training the Backbone Model
# ----------------------------
#
# First, we need a pre-trained model that can extract features from carbon structures.
# For this example, we'll train a simple SOAP-BPNN model on energy prediction.
# In a real application, you would typically use a pre-trained model on a larger
# dataset.
#
# The training options for the backbone model are as follows:

subprocess.run(
    ["wget","https://huggingface.co/lab-cosmo/pet-mad/resolve/v1.0.2/models/pet-mad-v1.0.2.ckpt"], check=True
)

# %%
#
# Training the Classifier
# -----------------------
#
# Now we can train the classifier on top of the frozen backbone. The classifier
# will learn to map the features extracted by the backbone to the class labels.
#
# The key hyperparameters are:
# - ``hidden_sizes``: The dimensions of the MLP layers. The last dimension (2 in this
#   case) acts as a bottleneck that can be used to extract collective variables.
# - ``model_checkpoint``: Path to the pre-trained backbone model.
#
# The training options for the classifier are:
print("\nTraining the classifier...")
subprocess.run(
    ["mtt", "train", "options-classifier.yaml", "-o", "classifier.pt"], check=True
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

# Create test structures
test_diamond = bulk("C", "diamond", a=3.57) * (2, 2, 2)
test_graphene = graphene(formula="C2", size=(3, 3, 1), a=2.46, vacuum=10.0)

structures = ase.io.read("carbon_allotropes.xyz")

# Get predictions
correct_count = 0
for structure in structures:
    prediction = calc.run_model(
        structure,
        {"mtt::class": ModelOutput(per_atom=False)},
    )["mtt::class"].block().values.cpu().squeeze(0).numpy()
    predicted_class = np.argmax(prediction)
    actual_class = int(structure.info["class"])
    if predicted_class == actual_class:
        correct_count += 1
print(f"Classifier accuracy: {(correct_count/len(structures))*100:.2f}% correct")


# 2D map using the bottleneck features
bottleneck_features = []
labels = []
for structure in structures:
    features = calc.run_model(
        structure,
        {"features": ModelOutput(per_atom=False)},
    )["features"].block().values.cpu().squeeze(0).numpy()
    bottleneck_features.append(features)
    labels.append(int(structure.info["class"]))
bottleneck_features = np.array(bottleneck_features)
labels = np.array(labels)
plt.figure(figsize=(8, 6))
for class_id in np.unique(labels):
    mask = labels == class_id
    plt.scatter(
        bottleneck_features[mask, 0],
        bottleneck_features[mask, 1],
        label=f"Class {class_id}",
        alpha=0.7,
    )
plt.xlabel("Bottleneck Feature 1")
plt.ylabel("Bottleneck Feature 2")
plt.title("2D Bottleneck Features from Classifier")
plt.legend()
plt.grid()
plt.savefig("bottleneck_features.png")
