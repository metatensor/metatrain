r"""
Intensive gap
=============

Does minimum and maximum pooling to get a gap-like intensive property.

Intensive properties like the band gap can't be computed as a sum of
local contributions due to the simple fact that they do not scale
with the size of the system. This extender implements a simple but powerful
idea that was used in this work :footcite:p:`malosso2026transferable`
to predict gaps from local contributions:

.. math::
    E_{gap} = \min_i c_i - \max_i v_i

i.e. it defines the gap as the difference between the maximum and minimum of
two different local contributions :math:`c_i` and :math:`v_i`. The minimum
and maximum can be interpreted as the conduction and valence band edges or
LUMO and HOMO energies, respectively.

The extender takes care of a global scalar output by requesting two separate
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

Learnable sharpness
-------------------

With ``learnable_sharpness: true``, each atom gets its own sharpness, predicted by
the model through two extra per-atom inputs :math:`u_i` (one for the bottom and one
for the top of the gap):

.. math::
    \alpha_i = A \, e^{u_i}, \qquad
    E = \sum_i \frac{e^{\alpha_i h_i}}{\sum_j e^{\alpha_j h_j}} h_i

where :math:`A` is ``sharpness_bottom`` or ``sharpness_top``, acting as a global
scale whose sign selects max- vs min-pooling (it must not be zero). The pooled
value is still strictly intensive, but the weights are not invariant to a constant
shift of all :math:`h_i`. :math:`u_i = 0` reduces exactly to the fixed-sharpness
pooling.

{{SECTION_INSTALLATION}}

{{SECTION_EXTENDER_HYPERS}}

Additional types to understand hypers
-------------------------------------

.. autoclass:: {{extender_path}}.documentation.ExtenderInputs
    :members:
    :undoc-members:

"""

from typing_extensions import NotRequired, TypedDict


class ExtenderInputs(TypedDict):
    """Inputs to be pooled to compute the intensive gap."""

    bottom: NotRequired[str]
    """Name of the requested input for the bottom of the gap.

    If not provided, it is determined automatically from
    the output name as
    ``mtt::extender_aux::intensive_gap::bottom::{output_name.replace('mtt::', '')}``.

    This input is a per-atom scalar quantity.
    """
    top: NotRequired[str]
    """Name of the requested input for the top of the gap.

    If not provided, it is determined automatically from
    the output name as
    ``mtt::extender_aux::intensive_gap::top::{output_name.replace('mtt::', '')}``.

    This input is a per-atom scalar quantity.
    """
    log_sharpness_bottom: NotRequired[str]
    """Name of the requested input for the per-atom log-sharpness :math:`u_i` of the
    bottom of the gap. Only used if ``learnable_sharpness`` is ``True``.

    If not provided, it is determined automatically from
    the output name as
    ``mtt::extender_aux::intensive_gap::log_sharpness_bottom::{name}``, where ``name``
    is ``output_name.replace('mtt::', '')``.

    This input is a dimensionless per-atom scalar quantity.
    """
    log_sharpness_top: NotRequired[str]
    """Name of the requested input for the per-atom log-sharpness :math:`u_i` of the
    top of the gap. Only used if ``learnable_sharpness`` is ``True``.

    If not provided, it is determined automatically from
    the output name as
    ``mtt::extender_aux::intensive_gap::log_sharpness_top::{name}``, where ``name``
    is ``output_name.replace('mtt::', '')``.

    This input is a dimensionless per-atom scalar quantity.
    """


class Hypers(TypedDict):
    """
    Hyperparameters for the intensive gap extender.
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

    learnable_sharpness: bool = False
    """If ``True``, the sharpness becomes per-atom and learnable:
    ``alpha_i = sharpness * exp(u_i)``, where ``u_i`` is predicted by the model as an
    additional per-atom input (the ``log_sharpness_bottom`` and ``log_sharpness_top``
    inputs), and ``sharpness_bottom`` / ``sharpness_top`` act as global scales whose
    sign selects max- vs min-pooling (they must not be zero).
    """

    inputs: ExtenderInputs | list[ExtenderInputs] | None = None
    """For each output, the names of the inputs to be pooled.

    The amount of input specifications must match the amount of outputs.
    Any missing specification will be determined automatically from the
    output name.
    """

    outputs: str | list[str]
    """Name or list of names for the outputs of the extender.

    The amount of outputs must match the amount of input
    specifications.
    """
