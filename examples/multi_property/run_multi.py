r"""
Example of doing multiproperty predictions with "SOAP-BPNN"
===========================================================

This is an example how to use the SOAP-BPNN model to predict isotropic
chemical shieldings :math:`\sigma_{\mathrm{iso}}` and it serves as an
example of how to use metatrain to learn multiple properties with one model.
This tutorial demonstrates how to use the SOAP-BPNN model to predict chemical
shieldings that were computed
using the GIPAW/PBE method. The target DFT GIPAW shieldings of a
dataset of mixed organic crystals containing at most
the elements H and C,are stored in the files
'train.xyz', 'val.xyz' and 'test.xyz'.

The final expression for the shielding is given by:

.. math::

   \begin{aligned}
   \sigma_{\mathrm{iso}}
     &= \sigma_{\mathrm{para}} + \sigma_{\mathrm{dia}} \\
     &\quad + \sigma_{\mathrm{bare}} + \sigma_{\mathrm{shape}} \\
     &\quad + \sigma_{\mathrm{para,oo}} + \sigma_{\mathrm{para,lq}}
   \end{aligned}

where
 :math:`\sigma_{\mathrm{iso}}` is the total isotropic shielding,
 :math:`\sigma_{\mathrm{para}}` is the paramagnetic contribution,
 :math:`\sigma_{\mathrm{dia}}` is the diamagnetic contribution,
 :math:`\sigma_{\mathrm{bare}}` is the bare shielding contribution,
 :math:`\sigma_{\mathrm{shape}}` is the shape contribution,
 :math:`\sigma_{\mathrm{para,oo}}` is the para-oo correction term, and
 :math:`\sigma_{\mathrm{para,lq}}` is the para-lq correction term.

In this example we are going to learn each contribution individually
as well as having one read-out layer for the total isotropic shielding.
This example highlights that different types of contributions can be
learned simultaneously: all contributions in the sum are atom-wise properties,
except for the shape contribution, which is a per-structure property.

The model was trained using the following training options, it is quite a long
input file, as all of the individual contributions have to be defined in the same
input.

.. literalinclude:: options.yaml
   :language: yaml
"""

import subprocess

import ase.io
import matplotlib.pyplot as plt
import numpy as np


# %%
#
# Running the model training
# -------------------------
#
# We run the model training via the command line interface (CLI) of metatensor
# In a suprocess

subprocess.run(["mtt", "train", "options.yaml"], check=True)


# %%
#
# Evalutating the model
# -------------------------
#
# We run the model evaluation via the command line interface (CLI) of metatensor

subprocess.run(
    ["mtt", "eval", "model.pt", "eval.yaml", "-e", "extensions/", "-o", "out.xyz"],
    check=True,
)

# %%
#
# import the prediction results, temporary fix with the property labels

subprocess.run(["sed", "-i", " ", "s/mtt::cs/mtt__cs/g", "out.xyz"], check=True)

frames_pred = ase.io.read("out.xyz", ":")
frames_true = ase.io.read("test.xyz", ":")

# %%
#
# Analyze results

# make a mask of all carbon atoms
mask_C = np.concatenate([frame.get_atomic_numbers() == 6 for frame in frames_true])

Ypred_iso = np.concatenate([frame.arrays["mtt__cs_iso"] for frame in frames_pred])[
    mask_C
]
Ypred_para = np.concatenate(
    [frame.arrays["mtt__cs_iso_para"] for frame in frames_pred]
)[mask_C]

Ytrue_iso = np.concatenate([frame.arrays["cs_iso"] for frame in frames_true])[mask_C]
Ytrue_para = np.concatenate([frame.arrays["cs_iso_para"] for frame in frames_true])[
    mask_C
]

# %%
#
# When continuing to train this model for an extended periods
# the prediction accuracies will improve.

fig, axs = plt.subplots(1, 2, figsize=(10, 5))
axs[0].scatter(Ytrue_iso, Ypred_iso, s=1)
axs[0].set_xlabel("True isotropic shielding")
axs[0].set_ylabel("Predicted isotropic shielding")
axs[0].set_title("Isotropic shielding")
axs[1].scatter(Ytrue_para, Ypred_para, s=1)
axs[1].set_xlabel("True paramagnetic shielding")
axs[1].set_ylabel("Predicted paramagnetic shielding")
axs[1].set_title("Paramagnetic shielding")

fig.tight_layout()
