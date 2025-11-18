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
for _ in range(20):
    diamond = bulk("C", "diamond", a=3.57)
    diamond = diamond * (2, 2, 2)  # Make it bigger
    diamond.rattle(stdev=0.05)  # Add random perturbations
    diamond.info["class"] = 0.0  # Label as diamond
    structures.append(diamond)

# Generate 20 graphite structures (using layered graphene-like structures)
for _ in range(20):
    # Create a graphite-like structure
    graphite = graphene(formula="C2", size=(3, 3, 1), a=2.46, vacuum=None)
    # Stack two layers
    layer2 = graphite.copy()
    layer2.translate([0, 0, 3.35])
    graphite.extend(layer2)
    graphite.set_cell([graphite.cell[0], graphite.cell[1], [0, 0, 6.7]])
    graphite.rattle(stdev=0.05)
    graphite.info["class"] = 1.0  # Label as graphite
    structures.append(graphite)

# Generate 20 graphene structures (single layer)
for _ in range(20):
    graphene_struct = graphene(formula="C2", size=(3, 3, 1), a=2.46, vacuum=10.0)
    graphene_struct.rattle(stdev=0.05)
    graphene_struct.info["class"] = 2.0  # Label as graphene
    structures.append(graphene_struct)

# Shuffle the structures
np.random.shuffle(structures)

# Save to file
ase.io.write("carbon_allotropes.xyz", structures)

print(f"Generated {len(structures)} structures:")
print("  - Diamond: 20 structures (class 0.0)")
print("  - Graphite: 20 structures (class 1.0)")
print("  - Graphene: 20 structures (class 2.0)")

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

backbone_options = """
architecture:
  name: soap_bpnn
  model:
    soap:
      cutoff: 5.0
      max_radial: 6
      max_angular: 4
      atomic_gaussian_width: 0.3
      center_atom_weight: 1.0
      radial_basis:
        type: gto
      cutoff_function:
        type: shifted_cosine
    bpnn:
      num_hidden_layers: 2
      num_neurons_per_layer: 64
      activation_function: tanh
  training:
    num_epochs: 50
    learning_rate: 0.01
    batch_size: 8

training_set:
  systems:
    read_from: carbon_allotropes.xyz
    reader: ase
  targets:
    energy:
      key: energy
      
validation_set: 0.1

"""

with open("options-backbone.yaml", "w") as f:
    f.write(backbone_options)

# For this example, we'll just add dummy energies to make the training work
# In practice, you would have real energies from DFT calculations
for structure in structures:
    # Add a simple dummy energy (just for demonstration)
    structure.info["energy"] = -5.0 * len(structure) + np.random.normal(0, 0.5)

ase.io.write("carbon_allotropes.xyz", structures)

print("\nTraining the backbone model...")
subprocess.run(
    ["mtt", "train", "options-backbone.yaml", "-o", "backbone.ckpt"], check=True
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

classifier_options = """
architecture:
  name: experimental.classifier
  model:
    hidden_sizes: [32, 16, 2]  # Last layer (2) is the bottleneck
  training:
    model_checkpoint: backbone.ckpt
    num_epochs: 100
    learning_rate: 0.001
    batch_size: 10
    log_interval: 10
    checkpoint_interval: 50

training_set:
  systems:
    read_from: carbon_allotropes.xyz
    reader: ase
  targets:
    class:
      key: class
      
validation_set: 0.2

"""

with open("options-classifier.yaml", "w") as f:
    f.write(classifier_options)

print("\nTraining the classifier...")
subprocess.run(
    ["mtt", "train", "options-classifier.yaml", "-o", "classifier.ckpt"], check=True
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
calc = MetatomicCalculator("classifier.ckpt", extensions_directory="extensions/")

# Create test structures
test_diamond = bulk("C", "diamond", a=3.57) * (2, 2, 2)
test_graphene = graphene(formula="C2", size=(3, 3, 1), a=2.46, vacuum=10.0)

test_structures = [test_diamond, test_graphene]

# Get predictions
predictions = calc.run_model(
    test_structures,
    {"class": ModelOutput(per_atom=False)},
)

# The output is logits (unnormalized probabilities)
logits = predictions["class"].block().values.numpy()
probabilities = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)

print("\nClassification Results:")
print("=" * 50)
for i, (structure, probs) in enumerate(
    zip(test_structures, probabilities, strict=True)
):
    predicted_class = np.argmax(probs)
    confidence = probs[predicted_class]
    print(f"\nStructure {i+1}:")
    print(f"  Formula: {structure.get_chemical_formula()}")
    print(f"  Predicted class: {predicted_class}")
    print(f"  Confidence: {confidence:.2%}")
    print("  Class probabilities:")
    print(f"    Diamond (0): {probs[0]:.2%}")
    print(f"    Graphite (1): {probs[1]:.2%}")
    print(f"    Graphene (2): {probs[2]:.2%}")

# %%
#
# Extracting Bottleneck Features
# ------------------------------
#
# The bottleneck layer (with 2 neurons in this case) can be used as collective
# variables for further analysis. These features represent a low-dimensional
# embedding of the structures that captures the most relevant information for
# classification.
#
# To extract these features, you would need to modify the model to output the
# bottleneck activations. This can be done as post-processing by accessing the
# intermediate layer outputs.
#
# In practice, you could:
# 1. Extract bottleneck features for many structures
# 2. Plot them in 2D to visualize the decision boundaries
# 3. Use them as collective variables in enhanced sampling simulations
#
# Here's a conceptual example of how the feature space might look:

# Create a synthetic visualization
fig, ax = plt.subplots(figsize=(8, 6))

# Generate synthetic 2D embeddings for illustration
np.random.seed(42)
diamond_features = np.random.multivariate_normal([-1, -1], [[0.1, 0], [0, 0.1]], 20)
graphite_features = np.random.multivariate_normal([1, 0], [[0.1, 0], [0, 0.1]], 20)
graphene_features = np.random.multivariate_normal([0, 1.5], [[0.1, 0], [0, 0.1]], 20)

ax.scatter(diamond_features[:, 0], diamond_features[:, 1], 
           label="Diamond", alpha=0.6, s=100, c="blue")
ax.scatter(graphite_features[:, 0], graphite_features[:, 1], 
           label="Graphite", alpha=0.6, s=100, c="green")
ax.scatter(graphene_features[:, 0], graphene_features[:, 1], 
           label="Graphene", alpha=0.6, s=100, c="red")

ax.set_xlabel("Bottleneck Feature 1", fontsize=12)
ax.set_ylabel("Bottleneck Feature 2", fontsize=12)
ax.set_title("Collective Variable Space (Bottleneck Features)", fontsize=14)
ax.legend(fontsize=11)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# %%
#
# Conclusion
# ----------
#
# This tutorial demonstrated:
#
# 1. How to create a labeled dataset for classification
# 2. Training a backbone model for feature extraction
# 3. Training a classifier on top of the frozen backbone
# 4. Using the classifier for predictions
# 5. Understanding the bottleneck features as collective variables
#
# The classifier model is particularly useful for:
#
# - **Transfer learning**: Reusing features from expensive pre-trained models
# - **Phase classification**: Identifying different phases or structures
# - **Reaction coordinate discovery**: The bottleneck features can serve as
#   collective variables for free energy calculations
# - **Active learning**: Identifying uncertain structures for labeling
#
# For more information on the classifier architecture, see the documentation.
