r"""
Intensive gap
=============

Does minimum and maximum pooling to get a gap-like intensive property.

Intensive properties like the band gap can't be computed as a sum of
local contributions due to the simple fact that they do not scale
with the size of the system. This hook implements a simple but powerful
idea that was used in this work :footcite:p:`malosso2026transferable`
to predict gaps from local contributions:

.. math::
    E_{gap} = \min_i c_i - \max_i v_i

i.e. it defines the gap as the difference between the maximum and minimum of
two different local contributions :math:`c_i` and :math:`v_i`. The minimum
and maximum can be interpreted as the conduction and valence band edges or
LUMO and HOMO energies, respectively.

The hook takes care of a global scalar output by requesting two separate
local scalar quantities as inputs and performing the min/max pooling.

"""

from typing import Literal, TypedDict

from typing_extensions import NotRequired

from metatrain.utils.hypers import init_with_defaults


class PoolingHypers(TypedDict):
    """Hyperparameters for the per-system pooling step.

    Two pooling types are supported, both controlled by the same ``alpha_max``
    / ``alpha_min`` parameters (sign selects max- vs min-pool, magnitude sets
    the sharpness):

    - ``"smoothmax"`` (default): ``E = (1/alpha) log sum_i exp(alpha h_i)``.
      Recovers a hard max/min as ``|alpha| -> infinity``. Size-intensive up to a
      ``log(N)/|alpha|`` residual.
    - ``"softmax"``: ``E = sum_i softmax(alpha * h_i) * h_i``. A self-weighted
      softmax pool: the softmax weights are computed from the per-atom *values
      themselves*, so the pool attends to the most extreme contributions.
      Strictly intensive (softmax weights sum to 1, removing the
      ``log(N)/|alpha|`` residual of the smoothmax pool) and recovers a hard
      max/min as ``|alpha| -> infinity``.
    """

    type: Literal["smoothmax", "softmax"] = "smoothmax"
    """Pooling type. One of ``"smoothmax"`` or ``"softmax"``."""

    alpha_bottom: float = 20.0
    """Max pooling parameter. ``alpha_bottom > 0`` gives a (smooth/soft) max.
    Larger magnitude -> sharper (closer to a hard max). Used by both pooling
    types."""

    alpha_top: float = -20.0
    """Min pooling parameter. ``alpha_top < 0`` gives a (smooth/soft) min.
    Larger magnitude -> sharper (closer to a hard min). Used by both pooling
    types."""


class HookInputs(TypedDict):
    """Inputs for the minmax hook."""

    bottom: NotRequired[str]
    """Name of the target for the bottom of the gap."""
    top: NotRequired[str]
    """Name of the target for the top of the gap."""


class Hypers(TypedDict):
    """
    Hyperparameters for the global multipole hook.
    """

    pooling: PoolingHypers = init_with_defaults(PoolingHypers)

    inputs: HookInputs | list[HookInputs] = init_with_defaults(HookInputs)

    outputs: str | list[str]
