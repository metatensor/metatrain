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

Pooling
-------

The hard minimum and maximum are replaced by a self-weighted softmax pool,
whose weights are computed from the per-atom values themselves:

.. math::
    E = \sum_i \frac{e^{\alpha h_i}}{\sum_j e^{\alpha h_j}} h_i

The sign of :math:`\alpha` (``sharpness_bottom`` or ``sharpness_top``) selects max-
vs min-pooling and its magnitude sets the sharpness: the pool recovers a hard
max/min as :math:`|\alpha| \to \infty`. The weights sum to one in each system,
so the pooled value is strictly intensive.

"""

from typing_extensions import NotRequired, TypedDict


class HookInputs(TypedDict):
    """Inputs for the minmax hook."""

    bottom: NotRequired[str]
    """Name of the target for the bottom of the gap."""
    top: NotRequired[str]
    """Name of the target for the top of the gap."""


class Hypers(TypedDict):
    """
    Hyperparameters for the intensive gap hook.
    """

    sharpness_bottom: float = 10.0
    """Sharpness of the pooling of the bottom of the gap.

    A positive value gives a soft max, while a negative value gives a soft min.

    A larger magnitude gives a sharper pool (closer to a hard max/min).
    """

    sharpness_top: float = -10.0
    """Sharpness of the pooling of the top of the gap.

    A positive value gives a soft max, while a negative value gives a soft min.

    A larger magnitude gives a sharper pool (closer to a hard max/min).
    """

    inputs: HookInputs | list[HookInputs] | None = None
    """For each output, the names of the inputs to be pooled.

    The amount of input specifications must match the amount of outputs.
    Any missing specification will be determined automatically from the
    output name.
    """

    outputs: str | list[str]
    """Name or list of names for the outputs of the hook.

    If provided, the amount of outputs must match the amount of input
    specifications. If ``None``, the outputs will be set as
    ``mtt::aux::intensive_gap::{output_name.replace('mtt::', '')}``
    for each output name.
    """
