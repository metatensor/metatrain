.. _architecture-pet:

LLPR
===

The LLPR architecture is a "wrapper" architecture that enables cheap uncertainty
quantification (UQ) via the last-layer prediction rigidity (LLPR) approach proposed
by Bigi et al. :footcite:p:`bigi_mlst_2024` It is compatible with all ``metatrain``
models constructed from NN-based architectures. The implementation of the LLPR as
a separate architecture within ``metatrain`` allows the users to easily compute the
(inverse) covariance matrix, the calibration factors, etc. needed for UQ for an
already-trained model.

Installation
------------

To install the LLPR wrapper architecture, run the following from the root of the repository:

.. code-block:: bash

    pip install metatrain[llpr]

This will install the architecture along with necessary dependencies.

Default Hyperparameters
-----------------------

The default hyperparameters for the PET model are:

.. literalinclude:: ../../../src/metatrain/llpr/default-hypers.yaml
   :language: yaml


Under ``training``, the following hyperparameters are defined:

- ``model_checkpoint``: This should provide the checkpoint to the model for which the
  user wants to perform UQ based on the LLPR approach. Note that the model architecture
  must comply with the requirement that the last-layer features are exposed under the
  convention defined by metatrain.
- ``batch_size``: This defines the batch size used in the computation of last-layer
  features, covariance matrix, etc.
- ``regularizer``: This is the regularizer value (sigma) that is used in applying Eq. 24 of
  Bigi et al. :footcite:p:`bigi_mlst_2024`. If set to ``null``, the internal routine will
  determine the smallest regularizer value that guarantees numerical stability in matrix
  inversion.


To perform uncertainty propagation, one could also generate an ensemble of weights
from the calibrated inverse covariance matrix from the LLPR formalism. To access this
feature within the architecture, one can interact with the following hyperparameters
under ``model`` and under ``ensemble``:

- ``means``: this accepts a dictionary of targets and the names of their corresponding
  last-layer weights. For example, in the case of energy in a PET model, the following
  could be the set of weights to provide:
  - ``node_last_layers.energy.0.energy___0.weight``
  - ``node_last_layers.energy.1.energy___0.weight``
  - ``edge_last_layers.energy.0.energy___0.weight``
  - ``edge_last_layers.energy.1.energy___0.weight``
- ``num_members``: this is a dictionary of targets and the corresponding number of ensemble
  members to sample. Note that a sufficiently large number of members (more than 16) are required
  for robust uncertainty propagation.

References
----------

.. footbibliography::
