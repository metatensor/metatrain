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
import ase.io
import matplotlib.pyplot as plt
import numpy as np


# %%
# Load the target and predicted data
targets = ase.io.read(
    "../train_from_scratch/ethanol_reduced_100.xyz", ":"
)  # reference data (ground truth)
predictions = ase.io.read("output.xyz", ":")  # predicted data from the model


# %%
# Extract the energies from the loaded frames
e_targets = np.array([frame.get_total_energy() for frame in targets])  # target energies
e_predictions = np.array(
    [frame.get_total_energy() for frame in predictions]
)  # predicted energies
f_targets = np.array(
    [frame.get_forces().flatten() for frame in targets]
).flatten()  # target forces
f_predictions = np.array(
    [frame.get_forces().flatten() for frame in predictions]
).flatten()  # predicted forces


# %%
# Create parity plots to compare predicted vs. target energies and forces
# -----------------------------------------------------------------------

fig, axs = plt.subplots(1, 2, figsize=(12, 5))

# Parity plot for energies
axs[0].scatter(e_targets, e_predictions)
axs[0].axline((np.min(e_targets), np.min(e_targets)), slope=1, ls="--", color="red")
axs[0].set_xlabel("Target energy / eV")
axs[0].set_ylabel("Predicted energy / eV")
min_e = np.min(np.array([e_targets, e_predictions])) - 2
max_e = np.max(np.array([e_targets, e_predictions])) + 2
axs[0].set_xlim([min_e, max_e])
axs[0].set_ylim([min_e, max_e])
axs[0].set_title("Energy Parity Plot")

# Parity plot for forces
axs[1].scatter(f_targets, f_predictions, alpha=0.5)
axs[1].axline((np.min(f_targets), np.min(f_targets)), slope=1, ls="--", color="red")
axs[1].set_xlabel("Target force / eV/Å")
axs[1].set_ylabel("Predicted force / eV/Å")
min_f = np.min(np.array([f_targets, f_predictions])) - 2
max_f = np.max(np.array([f_targets, f_predictions])) + 2
axs[1].set_xlim([min_f, max_f])
axs[1].set_ylim([min_f, max_f])
axs[1].set_title("Force Parity Plot")

plt.tight_layout()
plt.show()
