r"""
Mean pooling
============

Predicts an intensive scalar property by averaging learned atomic contributions.

A crystal has the same bulk modulus whether it is represented by a primitive
cell or a supercell. Doubling the cell repeats every local atomic environment,
so a sum of local predictions doubles as well. Their mean stays the same.

The extender requests one scalar :math:`a_i` per atom from the architecture.
For a system with :math:`N` contributing atoms, the prediction is

.. math::

    \hat{y} = \frac{1}{N} \sum_{i=1}^{N} a_i .

If atoms are selected, the mean is taken over the selected atoms in each
system. The system-level training label supervises the atomic predictions
through the mean. Autograd can differentiate this reduction. Requests for
explicit gradient blocks are unsupported.

Training
--------

For an architecture that supports extenders and predicts per-atom scalars,
configure mean pooling for a bulk modulus target as shown below. Include
``atomic_baseline`` and ``per_structure_targets`` when the trainer supports them.

.. code-block:: yaml

    architecture:
      training:
        extenders:
          mean_pooling: mtt::bulk_modulus
        atomic_baseline:
          mtt::bulk_modulus: 0.0
        per_structure_targets:
          - mtt::bulk_modulus

The zero atomic baseline keeps the composition model from adding a size-dependent
contribution. Listing the target in ``per_structure_targets`` keeps the trainer
from dividing the prediction and label by the atom count before computing the loss.
"""

from typing_extensions import TypedDict


class Hypers(TypedDict):
    """Hyperparameters for the mean pooling extender."""

    outputs: str
    """Name of the target to predict. It must have sample kind ``"system"``
    and a single scalar property.
    """
