r"""
Uncertainties for FlashMD with variance heads and LLPR
======================================================

This tutorial shows how to equip a trained :ref:`FlashMD <arch-flashmd>` model with
uncertainty estimates. The predictions of a FlashMD model (the positions and momenta of
the atoms some time in the future) are uncertain for two reasons:

- **aleatoric** uncertainty: the future is not fully determined by the current positions
  and momenta that the model sees, e.g. because of chaotic dynamics over long time lags.
  This uncertainty depends on the input, and does not go away with more training data;
- **epistemic** uncertainty: the model itself is uncertain, e.g. because the structure
  is far from the training data. This uncertainty decreases with more data.

We estimate both in two cheap steps, on top of a frozen pre-trained model:

1. **Variance heads.** A new head predicts the variance :math:`\sigma^2(x)` of the
   noise on each target. Only these heads are trained, with the Gaussian negative
   log-likelihood (:ref:`gaussian_nll <loss-functions>`) loss, so the mean predictions
   of the model do not change. This is a mean-variance estimation (MVE) network.
2. **Heteroscedastic LLPR.** The :ref:`LLPR <arch-llpr>` (a `last-layer Laplace
   approximation <https://arxiv.org/abs/2106.14806>`_) is applied to the model. Its precision matrix weights each training
   sample by the predicted noise,
   :math:`\Lambda = \sum_i f_i f_i^T / \sigma^2(x_i) + r I`
   (`Immer et al., 2023 <https://arxiv.org/abs/2304.08501>`_), and the predicted
   uncertainty is the total one,
   :math:`\sqrt{\sigma^2(x) + \alpha^2 f^T \Lambda^{-1} f}`.

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
# of the training distribution. The latter is useful to test the epistemic uncertainty of the LLPR, which should be larger for out-of-distribution structures.

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
#    The LLPR is applied post hoc, but an MVE network is normally trained from scratch
#    on the Gaussian NLL, which is expensive for a model like FlashMD. We therefore
#    freeze the pre-trained model and train only the variance heads. Joint NLL training
#    also tends to fit the mean worse than MSE (`Seitzer et al., 2022
#    <https://arxiv.org/abs/2203.09168>`_), and keeping the variance's gradients out of
#    the mean and the shared layers fixes that (`Stirn et al., 2023
#    <https://arxiv.org/abs/2212.09184>`_).
#
# .. note::
#
#    The noise is assumed to be isotropic (one variance per atom, shared by the x, y and
#    z components) and independent across atoms and between positions and momenta.
#    Correlating positions and momenta would need a joint LLPR over both last layers,
#    with four times the cost of the factorization. Correlations across atoms would need
#    a :math:`6N \times 6N` covariance per structure, which can be expensive to store and invert.
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
# variance, and ``mtt::aux::position_uncertainty`` the total uncertainty (as a standard
# deviation). The momenta have the same outputs, with ``momentum`` in their names. The
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

        # average the variances over the three Cartesian components, like the errors
        total = predictions[f"mtt::aux::{short_name}_uncertainty"].block().values ** 2
        aleatoric = predictions[f"mtt::aux::{short_name}_variance"].block().values
        total = total.squeeze(-1).mean(-1).double().numpy()
        aleatoric = aleatoric.squeeze(-1).mean(-1).double().numpy()

        results[output_name] = {
            "squared_error": ((mean - reference) ** 2).mean(-1),
            "aleatoric": aleatoric,
            # clip the round-off errors of the subtraction
            "epistemic": np.clip(total - aleatoric, 0.0, None),
            "total": total,
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
            f"aleatoric std = {np.sqrt(r['aleatoric'].mean()):.4f} {unit}, "
            f"epistemic std = {np.sqrt(r['epistemic'].mean()):.2e} {unit}, "
            f"total std = {np.sqrt(r['total'].mean()):.4f} {unit}"
        )

# %%
#
# If the uncertainties are well calibrated, the predicted variance matches
# (or at least highly correlated to) the mean
# squared error. Since there is a single error per atom and Cartesian component, we
# compare them on average, by binning the atoms by their predicted variance (left).
# Points on the diagonal are perfectly calibrated. On the right, we compare the
# epistemic part of the uncertainty at the two temperatures. The top row is for the
# positions, the bottom row for the momenta.

fig, axes = plt.subplots(2, 2, figsize=(9, 8))

for (ax_calibration, ax_epistemic), (output_name, target) in zip(
    axes, targets.items(), strict=True
):
    unit = target["plot_unit"]

    # common histogram bins for the two temperatures
    all_epistemic = np.concatenate(
        [np.sqrt(results[name][output_name]["epistemic"]) for name in test_pairs]
    )
    all_epistemic = all_epistemic[all_epistemic > 0]
    hist_bins = np.logspace(
        np.log10(all_epistemic.min()), np.log10(all_epistemic.max()), 40
    )

    for name in test_pairs:
        r = results[name][output_name]

        bins = np.array_split(np.argsort(r["total"]), 8)
        predicted = [np.sqrt(r["total"][b].mean()) for b in bins]
        observed = [np.sqrt(r["squared_error"][b].mean()) for b in bins]
        ax_calibration.plot(predicted, observed, "o-", label=name)

        ax_epistemic.hist(
            np.sqrt(r["epistemic"]), bins=hist_bins, alpha=0.6, label=name
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
# errors for both targets. At 900 K, outside of the training distribution (i.e., OOD), the
# calibration gets worse, as expected. As expected, the epistemic uncertainty of the LLPR is several
# times larger there, so it can be used to detect such out-of-distribution structures.
