"""
Data validation with parity plots for energies and forces
=========================================================

This tutorial shows how to visualise your model output using parity plots. In the
:ref:`train-from-scratch` we learned how to evaluate a trained model on a test set and
save the results to an output file. Here we will show how to create parity plots from
these results.

"""

# %%
# Import necessary libraries
# --------------------------
# ase.io : for reading atomic structures and their properties from xyz files
# matplotlib.pyplot : for creating plots and visualizations
# numpy : for numerical operations and array handling

import ase.io
import matplotlib.pyplot as plt
import numpy as np


# %%
# Load the target and predicted data
# ----------------------------------
# We load the reference (target) data and the model predictions using ASE's read
# function. The ":" means we load all frames from the xyz file. Modify the paths if your
# files are located elsewhere.

targets = ase.io.read("../qm9_reduced_100.xyz", ":")  # reference data (ground truth)
predictions = ase.io.read("output.xyz", ":")  # predicted data from the model


# %%
# Extract the energies from the loaded frames
# -------------------------------------------
# For the targets, the energy is stored in the 'info' dictionary under the key 'U0'.
# For the predictions, ASE stores the total energy accessible via get_total_energy().
# We convert both lists to numpy arrays for easier numerical operations and plotting.

e_targets = np.array([frame.info["U0"] for frame in targets])  # target energies
e_predictions = np.array(
    [frame.get_total_energy() for frame in predictions]
)  # predicted energies


# %%
# Create a parity plot to compare predicted vs. target energies
# -------------------------------------------------------------
# A parity plot shows how close predicted values are to the reference values.
# If the model is perfect, all points should lie on the diagonal y=x line.

# Create a figure and axes for the plot
fig, ax = plt.subplots()

# Scatter plot of target vs. predicted energies
# Each point represents a structure in the dataset
ax.scatter(e_targets, e_predictions)

# Plot a reference diagonal line (y=x) in red dashed style
# This helps to visually see deviations from perfect predictions
ax.axline((np.min(e_targets), np.min(e_targets)), slope=1, ls="--", color="red")

# Label the axes
ax.set_xlabel("Target energy / eV")
ax.set_ylabel("Predicted energy / eV")

# Set axis limits to slightly extend beyond the min and max of the data for better
# visibility
min_val = np.min(np.array([e_targets, e_predictions])) - 2
max_val = np.max(np.array([e_targets, e_predictions])) + 2
ax.set_xlim([min_val, max_val])
ax.set_ylim([min_val, max_val])

# Show the plot
plt.show()
