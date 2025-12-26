"""
Training with Batch Bounds
===========================

This example demonstrates how to use batch bounds to control the number of atoms
in each training batch. This is useful for:

1. Avoiding out-of-memory errors by limiting the maximum number of atoms per batch
2. Ensuring computational efficiency by setting a minimum number of atoms per batch
3. Creating more consistent batch sizes for stable training

Batch bounds are specified in the architecture's training options and apply to both
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

architecture:
  name: soap_bpnn
  training:
    batch_size: 5
    num_epochs: 10
    learning_rate: 0.01
    # Batch bounds configuration
    batch_atom_bounds: [10, 100]

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
# - ``batch_atom_bounds``: A list [min_atoms, max_atoms] specifying bounds.
#   Batches with atom counts outside these bounds will be skipped during training.
#
# - Use ``None`` for either value to disable that bound. For example:
#   - ``batch_atom_bounds: [10, None]`` sets only a minimum
#   - ``batch_atom_bounds: [None, 100]`` sets only a maximum
#   - ``batch_atom_bounds: [None, None]`` disables bounds (default)
#
# - These bounds apply to the *total* number of atoms across all systems in the batch.
#   For example, with ``batch_size: 5``, if each system has 20 atoms, the batch
#   will have 100 atoms total.
#
# - Batches outside the bounds are silently skipped. This is standard practice in
#   graph neural network training, where batch sizes can vary significantly.

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
#    ``batch_atom_bounds: [None, max_atoms]`` to prevent out-of-memory errors.
#
# 2. **Variable system sizes**: When your dataset contains systems with very
#    different sizes, batch bounds help ensure consistent memory usage.
#
# 3. **Efficiency**: Setting ``batch_atom_bounds: [min_atoms, None]`` can help
#    avoid inefficient batches with very few atoms.
#
# **Best practices:**
#
# - Consider the distribution of system sizes in your dataset when setting bounds.
#   Use ``batch_size * average_atoms_per_system`` as a starting point for bounds.
#
# - For datasets with very variable system sizes, you might want to use a smaller
#   ``batch_size`` with appropriate bounds to maintain consistent batch atom counts.
#
# - Remember that skipped batches don't contribute to training, so very restrictive
#   bounds may slow down training. Monitor your training to ensure not too many
#   batches are being skipped.
