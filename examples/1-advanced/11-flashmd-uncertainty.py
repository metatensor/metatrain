r"""
Uncertainties for FlashMD with variance heads and LLPR
======================================================

This tutorial shows how to equip a trained :ref:`FlashMD <arch-flashmd>` model with
uncertainty estimates. The predictions of a FlashMD model (the positions and momenta of
the atoms some time in the future) are uncertain for two reasons:

- **Aleatoric uncertainty:** although MD is deterministic, it is chaotic. Similar
  current positions and momenta can thus lead to different future ones if the time lag
  is large enough, and the model sees this as noise, as if the output were not fully
  determined by its input.
- **Epistemic uncertainty:** the model itself is uncertain, e.g. because the structure
  is far from the training data. This uncertainty decreases with more data.

We estimate both in two cheap steps, on top of a frozen pre-trained model:

1. **Variance heads.** A new head predicts the variance :math:`\sigma^2(x)` of the
   noise on each target (positions, momenta). Only these heads are trained, with the
   Gaussian negative log-likelihood (:ref:`gaussian_nll <loss-functions>`) loss, so the
   mean predictions of the model do not change. This is a mean-variance estimation
   (MVE) network.
2. **LLPR.** The :ref:`LLPR <arch-llpr>` (a `last-layer Laplace approximation
   <https://arxiv.org/abs/2106.14806>`_) is applied to the model to estimate the
   epistemic uncertainty. The difference here compared to the standard LLPR is that
   the heteroscedastic variances predicted by the variance heads are being considered.

Both steps are needed: the variance heads give the aleatoric uncertainty, and the LLPR
the epistemic one.

We will reuse the aluminum dataset of the :ref:`FlashMD tutorial
<sphx_glr_generated_examples_1-advanced_04-flashmd.py>`, which also explains how
FlashMD models are trained and fine-tuned.
"""

# %%
#

import copy
import subprocess

import ase
import ase.build
import ase.io
import ase.units
import matplotlib.pyplot as plt
import numpy as np
from ase.calculators.emt import EMT
from ase.md import VelocityVerlet
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from huggingface_hub import hf_hub_download
from metatomic.torch import ModelOutput
from metatomic_ase import MetatomicCalculator


# %%
#
# Data generation
# ---------------
#
# As in the
# :ref:`FlashMD tutorial <sphx_glr_generated_examples_1-advanced_04-flashmd.py>`, we run
# NVE molecular dynamics of a small box of aluminum with the EMT potential, and
# build pairs of current and future (32 fs later) structures. We also generate test
# pairs at two temperatures: 300 K, like the training data, and 900 K, which is outside
# of the training distribution. The latter is useful for testing the epistemic
# uncertainty of the LLPR, which should be larger for out-of-distribution structures.

time_lag = 32  # in units of the MD time step (1 fs)


def run_nve(temperature_K: float, n_steps: int, seed: int) -> list[ase.Atoms]:
    atoms = ase.build.bulk("Al", "fcc", cubic=True) * (2, 2, 2)
    atoms.calc = EMT()

    # equilibrate with a Langevin thermostat, then run in the NVE ensemble
    MaxwellBoltzmannDistribution(
        atoms, temperature_K=temperature_K, rng=np.random.default_rng(seed)
    )
    dyn = Langevin(
        atoms,
        2 * ase.units.fs,
        temperature_K=temperature_K,
        friction=1 / (100 * ase.units.fs),
        rng=np.random.default_rng(seed),
    )
    dyn.run(500)

    trajectory: list[ase.Atoms] = []
    dyn = VelocityVerlet(atoms, 1 * ase.units.fs)
    dyn.attach(lambda: trajectory.append(copy.deepcopy(atoms)), interval=1)
    dyn.run(n_steps)

    return trajectory


def make_pairs(trajectory: list[ase.Atoms], spacing: int) -> list[ase.Atoms]:
    pairs = []
    for i in range(0, len(trajectory) - time_lag, spacing):
        s = trajectory[i].copy()
        s.calc = None
        s.arrays["future_positions"] = trajectory[i + time_lag].get_positions()
        s.arrays["future_momenta"] = trajectory[i + time_lag].get_momenta()
        pairs.append(s)

    return pairs


train_pairs = make_pairs(run_nve(300, 4000, seed=0), spacing=50)
test_pairs = {
    "300 K": make_pairs(run_nve(300, 1000, seed=1), spacing=50),
    "900 K": make_pairs(run_nve(900, 1000, seed=2), spacing=50),
}

ase.io.write("flashmd.xyz", train_pairs)
print(f"{len(train_pairs)} training structures")

# %%
#
# For the training data, we have used starting structures that are 50 fs apart, which
# is shorter than the velocity autocorrelation time. This is fine for a tutorial, but
# you should space them further apart in practice (see the :ref:`FlashMD tutorial
# <sphx_glr_generated_examples_1-advanced_04-flashmd.py>`).
#
# Training the variance heads
# ---------------------------
#
# We start from the universal pre-trained FlashMD model with the matching time lag. In
# practice, you would usually fine-tune it on your system first, and then add variance
# heads on top of the fine-tuned model.

hf_hub_download(
    repo_id="lab-cosmo/flashmd",
    filename=f"flashmd_pet-omatpes-v2_{time_lag}fs.ckpt",
    local_dir=".",
)

# %%
#
# Variance heads are added with the ``variance_heads`` training hyperparameter, which
# lists the targets that get one. The model to start from is given in the ``finetune``
# section. Everything else in the model (the backbone, the heads predicting the
# targets, the scaler and the atomic baseline) is frozen, and the ``gaussian_nll`` loss
# is used automatically for these targets.
#
# .. note::
#
#    This procedure is equivalent to the faithful heteroscedastic regression of
#    `Stirn et al., 2023 <https://proceedings.mlr.press/v206/stirn23a.html>`_, which
#    ensures that the mean predictions of the model are the same as those of the
#    MSE-trained model. This is in the same spirit as the Laplace approximation (LLPR),
#    which is also applied post hoc and leaves the mean unchanged.
#
# .. note::
#
#    The noise is assumed to be isotropic (one variance per atom, shared by the x, y and
#    z components) and independent across atoms and between positions and momenta.
#    Correlating positions and momenta would need a joint LLPR over both last layers,
#    with four times the cost of the factorization. Correlations across atoms would need
#    a :math:`6N \times 6N` covariance per structure, which can be expensive to store
#    and invert.
#
# .. literalinclude:: options-flashmd-variance.yaml
#    :language: yaml
#
# Here, we run training as a subprocess. In reality you would run this from the command
# line as ``mtt train options-flashmd-variance.yaml -o model-variance.pt``.

subprocess.run(
    ["mtt", "train", "options-flashmd-variance.yaml", "-o", "model-variance.pt"],
    check=True,
)

# %%
#
# The resulting model has two new outputs, ``mtt::aux::position_variance`` and
# ``mtt::aux::momentum_variance``, holding the predicted (aleatoric) variance of each
# target, per atom.
#
# Adding the LLPR
# ---------------
#
# Next, we wrap the model with variance heads with the :ref:`LLPR architecture
# <arch-llpr>`. The LLPR detects the variance outputs and uses them as the noise of the
# last-layer Laplace approximation. This only requires one pass over the training set
# to compute the precision matrix, and one over the validation set to calibrate the
# epistemic part of the uncertainty.
#
# .. literalinclude:: options-flashmd-llpr.yaml
#    :language: yaml

subprocess.run(
    ["mtt", "train", "options-flashmd-llpr.yaml", "-o", "model-llpr.pt"], check=True
)

# %%
#
# Evaluating the uncertainties
# ----------------------------
#
# We load the exported LLPR model as an ASE calculator. Its ``run_model`` method gives
# access to the predicted future ``positions`` and ``momenta``, and to the auxiliary
# outputs. For the positions, ``mtt::aux::position_variance`` is the aleatoric
# variance, and ``mtt::aux::position_uncertainty`` the epistemic uncertainty (as a
# standard deviation). The total variance is their sum, the variance plus the squared
# uncertainty. The momenta have the same outputs, with ``momentum`` in their names. The
# current momenta, which FlashMD needs as an input, are taken from the ``ase.Atoms``.

calc = MetatomicCalculator("model-llpr.pt", extensions_directory="extensions/")

# for each output: the name used in its variance and uncertainty outputs, the array
# with the reference values, and its unit (as text, and for matplotlib)
targets = {
    "positions": {
        "short_name": "position",
        "reference": "future_positions",
        "unit": "Å",
        "plot_unit": "Å",
    },
    "momenta": {
        "short_name": "momentum",
        "reference": "future_momenta",
        "unit": "√(eV·u)",
        "plot_unit": r"$\sqrt{\mathrm{eV \cdot u}}$",
    },
}

outputs = {}
for output_name, target in targets.items():
    for name in [
        output_name,
        f"mtt::aux::{target['short_name']}_variance",
        f"mtt::aux::{target['short_name']}_uncertainty",
    ]:
        outputs[name] = ModelOutput(sample_kind="atom")


def evaluate(pairs: list[ase.Atoms]) -> dict[str, dict[str, np.ndarray]]:
    """Per-atom squared errors and predicted variances of the future positions and
    momenta."""
    predictions = calc.run_model(pairs, outputs)

    results = {}
    for output_name, target in targets.items():
        short_name = target["short_name"]
        mean = predictions[output_name].block().values.squeeze(-1).numpy()
        reference = np.concatenate([s.arrays[target["reference"]] for s in pairs])

        # per atom and Cartesian component
        epistemic_std = predictions[f"mtt::aux::{short_name}_uncertainty"]
        epistemic_std = epistemic_std.block().values
        aleatoric_var = predictions[f"mtt::aux::{short_name}_variance"].block().values

        # average the variances over the three components, like the squared errors
        epistemic_var = (epistemic_std**2).squeeze(-1).mean(-1).double().numpy()
        aleatoric_var = aleatoric_var.squeeze(-1).mean(-1).double().numpy()
        total_var = aleatoric_var + epistemic_var

        results[output_name] = {
            "squared_error": ((mean - reference) ** 2).mean(-1),
            "aleatoric_var": aleatoric_var,
            "epistemic_var": epistemic_var,
            "total_var": total_var,
        }

    return results


results = {name: evaluate(pairs) for name, pairs in test_pairs.items()}

for output_name, target in targets.items():
    unit = target["unit"]

    for name in test_pairs:
        r = results[name][output_name]
        print(
            f"{output_name} at {name}: "
            f"RMSE = {np.sqrt(r['squared_error'].mean()):.4f} {unit}, "
            f"aleatoric std = {np.sqrt(r['aleatoric_var'].mean()):.4f} {unit}, "
            f"epistemic std = {np.sqrt(r['epistemic_var'].mean()):.2e} {unit}, "
            f"total std = {np.sqrt(r['total_var'].mean()):.4f} {unit}"
        )

# %%
#
# If the uncertainties are well calibrated, the predicted variance matches (or is at
# least highly correlated with) the mean squared error. Since there is a single error
# per atom and Cartesian component, we compare them on average, by binning the atoms
# by their predicted variance (left).
# Points on the diagonal are perfectly calibrated. On the right, we compare the
# epistemic part of the uncertainty at the two temperatures. The top row is for the
# positions, the bottom row for the momenta.

fig, axes = plt.subplots(2, 2, figsize=(9, 8))

for (ax_calibration, ax_epistemic), (output_name, target) in zip(
    axes, targets.items(), strict=True
):
    unit = target["plot_unit"]

    # common histogram bins for the two temperatures
    all_epistemic_std = np.concatenate(
        [np.sqrt(results[name][output_name]["epistemic_var"]) for name in test_pairs]
    )
    all_epistemic_std = all_epistemic_std[all_epistemic_std > 0]
    hist_bins = np.logspace(
        np.log10(all_epistemic_std.min()), np.log10(all_epistemic_std.max()), 40
    )

    for name in test_pairs:
        r = results[name][output_name]

        bins = np.array_split(np.argsort(r["total_var"]), 8)
        predicted = [np.sqrt(r["total_var"][b].mean()) for b in bins]
        observed = [np.sqrt(r["squared_error"][b].mean()) for b in bins]
        ax_calibration.plot(predicted, observed, "o-", label=name)

        ax_epistemic.hist(
            np.sqrt(r["epistemic_var"]), bins=hist_bins, alpha=0.6, label=name
        )

    lims = [0.0, max(ax_calibration.get_xlim()[1], ax_calibration.get_ylim()[1])]
    ax_calibration.plot(lims, lims, "k--", lw=1)
    ax_calibration.set_xlim(lims)
    ax_calibration.set_ylim(lims)
    ax_calibration.set_xlabel(f"predicted total std ({unit})")
    ax_calibration.set_ylabel(f"observed RMSE ({unit})")
    ax_calibration.set_title(f"Calibration, {output_name}")
    ax_calibration.legend()

    ax_epistemic.set_xscale("log")
    ax_epistemic.set_xlabel(f"epistemic std ({unit})")
    ax_epistemic.set_ylabel("number of atoms")
    ax_epistemic.set_title(f"Epistemic uncertainty, {output_name}")
    ax_epistemic.legend()

fig.tight_layout()
plt.show()

# %%
#
# At 300 K, the temperature of the training data, the predicted uncertainties match the
# errors for both targets. At 900 K, outside of the training distribution, the
# calibration gets worse, as expected. The epistemic uncertainty of the LLPR is also
# several times larger there, so it can be used to detect such out-of-distribution
# structures.
