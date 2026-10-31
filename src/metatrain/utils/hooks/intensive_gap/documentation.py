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

Learnable sharpness
-------------------

With ``pooling.learnable_alpha: true`` (only for ``softmax`` pooling), each atom
gets its own sharpness, predicted by the model through two extra per-atom
outputs :math:`u_i` (one for the bottom and one for the top of the gap):

.. math::
    \alpha_i = A \, e^{u_i}, \qquad
    E = \sum_i \frac{e^{\alpha_i h_i}}{\sum_j e^{\alpha_j h_j}} h_i

where :math:`A` is ``alpha_bottom`` or ``alpha_top``, acting as a global scale
whose sign selects max- vs min-pooling. The weights are not invariant to a
constant shift of all :math:`h_i`. :math:`u_i = 0` reduces exactly to the
fixed-sharpness softmax pooling.

"""

from typing import Literal, TypedDict

from typing_extensions import NotRequired

from metatrain.utils.hypers import init_with_defaults


class PoolingHypers(TypedDict):
    """Hyperparameters for the per-system pooling step.

    Two pooling types are supported, both controlled by the same
    ``alpha_bottom`` / ``alpha_top`` parameters (sign selects max- vs min-pool,
    magnitude sets the sharpness):

    - ``"softmax"`` (default): ``E = sum_i softmax(alpha * h_i) * h_i``. A
      self-weighted softmax pool: the softmax weights are computed from the
      per-atom *values themselves*, so the pool attends to the most extreme
      contributions. Strictly intensive (softmax weights sum to 1, removing the
      ``log(N)/|alpha|`` residual of the smoothmax pool) and recovers a hard
      max/min as ``|alpha| -> infinity``.
    - ``"smoothmax"``: ``E = (1/alpha) log sum_i exp(alpha h_i)``. Recovers a
      hard max/min as ``|alpha| -> infinity``, but is not strictly intensive:
      it is size-intensive only up to a ``log(N)/|alpha|`` residual.
    """

    type: Literal["smoothmax", "softmax"] = "softmax"
    """Pooling type. One of ``"softmax"`` or ``"smoothmax"``."""

    alpha_bottom: float = 20.0
    """Max pooling parameter. ``alpha_bottom > 0`` gives a (smooth/soft) max.
    Larger magnitude -> sharper (closer to a hard max). Used by both pooling
    types."""

    alpha_top: float = -20.0
    """Min pooling parameter. ``alpha_top < 0`` gives a (smooth/soft) min.
    Larger magnitude -> sharper (closer to a hard min). Used by both pooling
    types."""

    learnable_alpha: bool = False
    """If ``True``, the sharpness becomes per-atom and learnable:
    ``alpha_i = alpha * exp(u_i)``, where ``u_i`` is predicted by the model and
    ``alpha_bottom`` / ``alpha_top`` act as global scales whose sign selects
    max- vs min-pooling. Only supported with ``"softmax"`` pooling."""


class HookInputs(TypedDict):
    """Inputs for the minmax hook."""

    bottom: NotRequired[str]
    """Name of the target for the bottom of the gap."""
    top: NotRequired[str]
    """Name of the target for the top of the gap."""
    alpha_bottom: NotRequired[str]
    """Name of the per-atom log-sharpness ``u_i`` for the bottom of the gap. Only
    used if ``learnable_alpha`` is ``True``."""
    alpha_top: NotRequired[str]
    """Name of the per-atom log-sharpness ``u_i`` for the top of the gap. Only
    used if ``learnable_alpha`` is ``True``."""


class Hypers(TypedDict):
    """
    Hyperparameters for the intensive gap hook.
    """

    pooling: PoolingHypers = init_with_defaults(PoolingHypers)

    inputs: HookInputs | list[HookInputs] = init_with_defaults(HookInputs)

    outputs: str | list[str]
