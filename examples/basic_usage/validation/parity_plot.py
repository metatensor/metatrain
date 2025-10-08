"""
Data validation with parity plots for energies and forces
===================================

This tutorial shows how to visualise your model output using parity plot. in the
:ref:`train-from-scratch` we learned how to evaluate a trained model on a test set and
save the results to an output file. Here we will show how to create parity plots from
these results.

"""

# %%

import ase.io
import matplotlib.pyplot as plt
import numpy as np


# %%
# load the target and prediction data
targets = ase.io.read("../qm9_reduced_100.xyz", ":")  # TODO: modify path if needed
predictions = ase.io.read("output.xyz", ":")  # TODO: modify path

# %%
# extract the energies
e_targets = np.array([frame.info["U0"] for frame in targets])
e_predictions = np.array([frame.get_total_energy() for frame in predictions])


# %%
#

fig, ax = plt.subplots()

ax.scatter(e_targets, e_predictions)

ax.axline((np.min(e_targets), np.min(e_targets)), slope=1, ls="--", color="red")
ax.set_xlabel("target energy / eV")
ax.set_ylabel("predicted energy / eV")
ax.set_xlim(
    [
        np.min(np.array([e_targets, e_predictions])) - 2,
        np.max(np.array([e_targets, e_predictions])) + 2,
    ]
)
ax.set_ylim(
    [
        np.min(np.array([e_targets, e_predictions])) - 2,
        np.max(np.array([e_targets, e_predictions])) + 2,
    ]
)
