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
2. **Heteroscedastic LLPR.** The :ref:`LLPR <arch-llpr>` (a last-layer Laplace
   approximation) is applied to the model. Its precision matrix weights each training
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
# of the training distribution.

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
# access to the predicted future ``positions`` and to the auxiliary outputs:
# ``mtt::aux::position_variance`` is the aleatoric variance, and
# ``mtt::aux::position_uncertainty`` the total uncertainty (as a standard deviation).
# The current momenta, which FlashMD needs as an input, are taken from the
# ``ase.Atoms``.

calc = MetatomicCalculator("model-llpr.pt", extensions_directory="extensions/")

outputs = {
    name: ModelOutput(sample_kind="atom")
    for name in [
        "positions",
        "mtt::aux::position_variance",
        "mtt::aux::position_uncertainty",
    ]
}


def evaluate(pairs: list[ase.Atoms]) -> dict[str, np.ndarray]:
    """Per-atom squared errors and predicted variances of the future positions."""
    predictions = calc.run_model(pairs, outputs)

    mean = predictions["positions"].block().values.squeeze(-1).numpy()
    target = np.concatenate([s.arrays["future_positions"] for s in pairs])

    # average the variances over the three Cartesian components, like the errors
    total = predictions["mtt::aux::position_uncertainty"].block().values ** 2
    aleatoric = predictions["mtt::aux::position_variance"].block().values
    total = total.squeeze(-1).mean(-1).double().numpy()
    aleatoric = aleatoric.squeeze(-1).mean(-1).double().numpy()

    return {
        "squared_error": ((mean - target) ** 2).mean(-1),
        "aleatoric": aleatoric,
        # clip the round-off errors of the subtraction
        "epistemic": np.clip(total - aleatoric, 0.0, None),
        "total": total,
    }


results = {name: evaluate(pairs) for name, pairs in test_pairs.items()}

for name, r in results.items():
    print(
        f"{name}: RMSE = {np.sqrt(r['squared_error'].mean()):.4f} A, "
        f"aleatoric std = {np.sqrt(r['aleatoric'].mean()):.4f} A, "
        f"epistemic std = {np.sqrt(r['epistemic'].mean()):.4f} A, "
        f"total std = {np.sqrt(r['total'].mean()):.4f} A"
    )

# %%
#
# If the uncertainties are well calibrated, the predicted variance matches the mean
# squared error. Since there is a single error per atom and Cartesian component, we
# compare them on average, by binning the atoms by their predicted variance (left).
# Points on the diagonal are perfectly calibrated. On the right, we compare the
# epistemic part of the uncertainty at the two temperatures.

fig, (ax_calibration, ax_epistemic) = plt.subplots(1, 2, figsize=(9, 4))

for name, r in results.items():
    bins = np.array_split(np.argsort(r["total"]), 8)
    predicted = [np.sqrt(r["total"][b].mean()) for b in bins]
    observed = [np.sqrt(r["squared_error"][b].mean()) for b in bins]
    ax_calibration.plot(predicted, observed, "o-", label=name)

    ax_epistemic.hist(
        np.sqrt(r["epistemic"]),
        bins=np.logspace(-5, -1.5, 40),
        alpha=0.6,
        label=name,
    )

lims = [0.0, max(ax_calibration.get_xlim()[1], ax_calibration.get_ylim()[1])]
ax_calibration.plot(lims, lims, "k--", lw=1)
ax_calibration.set_xlim(lims)
ax_calibration.set_ylim(lims)
ax_calibration.set_xlabel("predicted total std (A)")
ax_calibration.set_ylabel("observed RMSE (A)")
ax_calibration.set_title("Calibration")
ax_calibration.legend()

ax_epistemic.set_xscale("log")
ax_epistemic.set_xlabel("epistemic std (A)")
ax_epistemic.set_ylabel("number of atoms")
ax_epistemic.set_title("Epistemic uncertainty")
ax_epistemic.legend()

fig.tight_layout()
plt.show()

# %%
#
# At 300 K, the temperature of the training data, the variance heads alone capture the
# error well, and the epistemic uncertainty is negligible. At 900 K, the structures
# and momenta are outside of the training distribution, and the model becomes
# overconfident: the variance heads were only trained at 300 K, and do not extrapolate
# well. The epistemic uncertainty of the LLPR grows several times, with a long tail of
# atoms that are much more uncertain, flagging these structures as unfamiliar. But it
# stays small compared to the aleatoric uncertainty, and does not fix the calibration.
#
# This is expected from a last-layer approximation on top of a frozen backbone. The
# LLPR only accounts for the uncertainty of the last layer of the mean, which is well
# determined by the thousands of per-atom training samples. The epistemic uncertainty
# is therefore more useful as a relative indicator of extrapolation, e.g. to detect
# when a simulation leaves the training distribution, than as a correction of the
# total uncertainty. To get reliable variances in a different regime, the variance
# heads should be trained on data from that regime too.
