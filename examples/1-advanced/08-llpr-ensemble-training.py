""".. _llprensembleexample:

Training an LLPR-derived shallow ensemble model
================================================

This tutorial demonstrates how to train LLPR-derived shallow ensemble models
using metatrain. Building on the basic LLPR approach, this advanced technique
allows for gradient-based tuning of ensemble weights using a negative log-likelihood
(NLL) loss, which can lead to improved uncertainty estimates.

The LLPR (Last-Layer Prediction Rigidity) approach provides a computationally
efficient method for uncertainty quantification. By training the ensemble weights
with an NLL loss, we can further refine the uncertainty estimates through 
recalibration on the training data.

We first train a baseline model without uncertainties:

.. literalinclude:: options-model.yaml
   :language: yaml

Then we create and train an LLPR ensemble model. There are two main approaches:

1. **Training only the ensemble weights** (recommended for most cases):

.. literalinclude:: options-llpr-ensemble.yaml
   :language: yaml

This approach trains only the last-layer ensemble weights while keeping the
base model's parameters frozen. This is computationally efficient and maintains
the LLPR uncertainty estimates.

2. **Training all parameters** (for more flexibility):

.. literalinclude:: options-llpr-ensemble-full.yaml
   :language: yaml

This approach trains both the ensemble weights and the base model parameters.
Note that this will change the last-layer features, so only the ensemble-based
uncertainties (not the LLPR analytical uncertainties) will be meaningful.

Key hyperparameters:

- ``num_epochs``: If set to a value (e.g., 2), ensemble weight training is enabled.
  If set to ``null``, only LLPR covariance computation is performed.
- ``train_all_parameters``: If ``true``, trains both ensemble weights and base
  model parameters. If ``false`` (default), trains only ensemble weights.
- ``loss``: Use ``"ensemble_nll"`` for negative log-likelihood loss, which is
  recommended for ensemble weight calibration.

You can train these models yourself with the following code:

"""
# %%
#

import subprocess

import ase.io
import matplotlib.pyplot as plt
import numpy as np
from metatomic.torch import ModelOutput
from metatomic.torch.ase_calculator import MetatomicCalculator


# %%
#
# We first train the baseline model without uncertainties, then train the LLPR
# ensemble model with calibrated weights.

# Here, we run training as a subprocess. In practice, you would run this from
# the command line, e.g., ``mtt train options-model.yaml -o model.pt``.

print("Training baseline model...")
subprocess.run(["mtt", "train", "options-model.yaml", "-o", "model.pt"], check=True)

print("Training LLPR ensemble model (weights only)...")
subprocess.run(
    ["mtt", "train", "options-llpr-ensemble.yaml", "-o", "model-llpr-ensemble.pt"],
    check=True,
)

# %%
#
# A detailed step-by-step introduction on how to train a model is provided in
# the :ref:`label_basic_usage` tutorial.
#
# Now let's compare the uncertainties from the trained ensemble with the
# analytical LLPR uncertainties.

# Load some test structures
structures = ase.io.read("ethanol_reduced_100.xyz", ":10")

# Load the ensemble-trained model
calc = MetatomicCalculator(
    "model-llpr-ensemble.pt", extensions_directory="extensions/", device="cpu"
)

# Get predictions with both ensemble and analytical uncertainties
predictions = calc.run_model(
    structures,
    {
        "energy": ModelOutput(per_atom=False),
        "energy_uncertainty": ModelOutput(per_atom=False),  # LLPR analytical
        "energy_ensemble": ModelOutput(per_atom=False),  # ensemble predictions
    },
)

energies = predictions["energy"].block().values.squeeze().numpy()
llpr_uncertainties = (
    predictions["energy_uncertainty"].block().values.squeeze().numpy()
)
ensemble_predictions = predictions["energy_ensemble"].block().values.squeeze().numpy()

# Calculate ensemble mean and standard deviation
ensemble_mean = np.mean(ensemble_predictions, axis=1)
ensemble_std = np.std(ensemble_predictions, axis=1)

print(f"Energies: {energies}")
print(f"LLPR analytical uncertainties: {llpr_uncertainties}")
print(f"Ensemble mean: {ensemble_mean}")
print(f"Ensemble std: {ensemble_std}")

# %%
#
# Let's visualize the comparison between analytical and ensemble uncertainties

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Plot 1: Energy predictions comparison
ax1.scatter(energies, ensemble_mean, alpha=0.6, label="Ensemble mean")
ax1.plot([energies.min(), energies.max()], [energies.min(), energies.max()], "k--")
ax1.set_xlabel("Mean prediction (eV)")
ax1.set_ylabel("Ensemble mean (eV)")
ax1.set_title("Energy predictions: Mean vs Ensemble")
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Uncertainty comparison
ax2.scatter(llpr_uncertainties, ensemble_std, alpha=0.6)
ax2.plot(
    [llpr_uncertainties.min(), llpr_uncertainties.max()],
    [llpr_uncertainties.min(), llpr_uncertainties.max()],
    "k--",
    label="y=x",
)
ax2.set_xlabel("LLPR analytical uncertainty (eV)")
ax2.set_ylabel("Ensemble std (eV)")
ax2.set_title("Uncertainty comparison")
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# %%
#
# After training with the NLL loss, the ensemble standard deviations may differ
# from the analytical LLPR uncertainties. This is expected and can lead to better
# calibrated uncertainties, especially when using ``train_all_parameters: false``.
#
# The ensemble-based uncertainties (standard deviations over ensemble members)
# should be used as the final uncertainty estimates when ``num_epochs`` is set
# to train the ensemble weights.
