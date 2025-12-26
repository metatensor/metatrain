"""
Training with Batch Bounds
===========================

This example demonstrates how to use batch bounds to control the number of atoms
in each training batch. This is useful for:

1. Avoiding out-of-memory errors by limiting the maximum number of atoms per batch
2. Ensuring computational efficiency by setting a minimum number of atoms per batch
3. Creating more consistent batch sizes for stable training

Batch bounds are specified in the training options file and apply to both
training and validation dataloaders.
"""

import metatrain

# %%
# First, let's create a configuration that includes batch bounds.
# We'll use the SOAP-BPNN model as an example.

options = """
device: cpu
base_precision: 64
seed: 42

# Batch bounds configuration
min_atoms_per_batch: 10
max_atoms_per_batch: 100

architecture:
  name: soap_bpnn
  training:
    batch_size: 5
    num_epochs: 10
    learning_rate: 0.01

training_set:
  systems: qm9_reduced_100.xyz
  targets:
    energy:
      key: U0
      unit: hartree

validation_set: 0.1
test_set: 0.0
"""

# %%
# Key points about batch bounds:
#
# - ``min_atoms_per_batch``: Minimum total number of atoms allowed in a batch.
#   Batches with fewer atoms will be rejected (skipped) during training.
#
# - ``max_atoms_per_batch``: Maximum total number of atoms allowed in a batch.
#   Batches with more atoms will be rejected (skipped) during training.
#
# - These bounds apply to the *total* number of atoms across all systems in the batch.
#   For example, with ``batch_size: 5``, if each system has 20 atoms, the batch
#   will have 100 atoms total.
#
# - Both bounds are optional. You can specify just one or both, depending on your needs.

# %%
# Now let's train the model with these batch bounds.
# In a real scenario, you would save this to a YAML file and run:
#
# .. code-block:: bash
#
#     metatrain train options.yaml -o model.pt
#
# The training process will automatically skip batches that fall outside the
# specified atom count bounds, ensuring that your training stays within the
# desired computational constraints.

# %%
# **When to use batch bounds:**
#
# 1. **Memory constraints**: If you're running on limited GPU memory, set
#    ``max_atoms_per_batch`` to prevent out-of-memory errors.
#
# 2. **Variable system sizes**: When your dataset contains systems with very
#    different sizes, batch bounds help ensure consistent memory usage.
#
# 3. **Efficiency**: Setting ``min_atoms_per_batch`` can help avoid inefficient
#    batches with very few atoms.
#
# **Best practices:**
#
# - Monitor your training logs to see if batches are being skipped frequently.
#   If many batches are rejected, you may need to adjust your bounds or batch size.
#
# - Consider the distribution of system sizes in your dataset when setting bounds.
#   Use ``batch_size * average_atoms_per_system`` as a starting point for bounds.
#
# - For datasets with very variable system sizes, you might want to use a smaller
#   ``batch_size`` with appropriate bounds to maintain consistent batch atom counts.
