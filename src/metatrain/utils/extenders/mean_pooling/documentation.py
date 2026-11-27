"""
Mean pooling
============

Predict a scalar per system by averaging learned per-atom scalars. For example,
to train on one refractive index per crystal, use the target-name shorthand:

.. code-block:: yaml

    extenders:
      mean_pooling: mtt::refractive_index

The architecture supplies one scalar per selected atom. Each system's output
is the arithmetic mean over its contributing atoms. For an intensive training
target, set its atomic baseline to zero and include it in
``per_structure_targets`` in trainers that support these options. Gradients through
the mean are available via autograd. Requests for explicit gradient blocks are
unsupported.
"""

from typing_extensions import TypedDict


class Hypers(TypedDict):
    """Hyperparameters for the mean pooling extender."""

    outputs: str
    """Name of the scalar system target."""
