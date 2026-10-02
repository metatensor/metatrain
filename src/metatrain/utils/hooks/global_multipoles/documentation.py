r"""
Global multipoles
=================

Computes a global multipole from local predictions.

Predicting the multipole of a system by simply summing local contributions misses
important contributions from the relative positions of the atoms in the system. As an
example, one can think of predicting the dipole of a system with two atoms of opposite
charge. If the two atoms don't see each other given the cutoff of the model, each atom
will predict a local dipole of zero, and the sum of the two local dipoles will also be
zero. However, it is obvious that this system has a non-zero dipole.

In general, predicting a global multipole of order :math:`\ell` from a simple sum of
local contributions of that :math:`\ell` is not valid whenever the local regions have a
non-zero multipole of order :math:`< \ell`. For example, in the case of the dipole,
whenever the local regions have a net charge (non-zero monopole), the global dipole
can't be computed as the sum of local dipoles. Instead, one needs to add the
contributions from all the lower order multipoles, taking into account their origin
(i.e. the position of the atom). For the case of the dipole, the global dipole can be
computed as:

.. math::
    \mathbf{p} = \sum_i \mathbf{p}_i + \sum_i q_i \mathbf{r}_i

This hook implements this concept by taking care of an output that is global and of
order :math:`\ell`, asking for local contributions of all orders :math:`\leq \ell` and
computing the global multipole accounting for the positions of the atoms.

Origin of the positions
-----------------------

The expression above is only independent of the choice of origin when all multipoles of
order lower than :math:`\ell` vanish. For the dipole that means :math:`\sum_i q_i = 0`,
which is not guaranteed when the charges are learned: translating a system by
:math:`\mathbf{a}` changes the prediction by :math:`\left(\sum_i q_i\right)\mathbf{a}`.

To make the prediction invariant under a rigid translation, the positions of each system
are by default referred to its centre of nuclear charge,

.. math::
    \mathbf{R} = \frac{\sum_i Z_i \mathbf{r}_i}{\sum_i Z_i}

before the multipole is assembled. Because the origin is defined by the system itself it
translates with it, so the dependence cancels exactly, whatever the value of
:math:`\sum_i q_i`. The centre of nuclear charge is a convention used by some electronic
structure codes, but not all.

This is controlled by the ``origin`` hyperparameter: set it to ``"absolute"`` to keep
the positions as they are stored, and recover the origin-dependent behaviour described
above.

.. note::

    This hook only makes sense for non-periodic systems: :math:`\sum_i q_i \mathbf{r}_i`
    depends on how the atoms are wrapped into the cell, and under periodic boundary
    conditions the dipole is only defined modulo a quantum. This is not handled in the
    current implementation.
"""

from typing import Literal, Optional, TypedDict


class Hypers(TypedDict):
    """
    Hyperparameters for the global multipole hook.
    """

    inputs: Optional[str | list[str]] = None
    """
    Name or names for the inputs that this hook will request.

    If ``None``, the hook will request an input named
    ``mtt::aux::local_multipoles::{output_name.replace('mtt::', '')}``
    for each output name.
    """

    outputs: Optional[str | list[str]] = None
    """
    Name or names for the outputs that this hook must produce.

    These targets must be spherical and global, i.e. with sample
    kind ``"system"``.
    """

    origin: Literal["center_of_charge", "absolute"] = "center_of_charge"
    """
    Origin (per-system) that the atomic positions are referred to when assembling the
    multipole.

    - ``"center_of_charge"`` (default): subtract each system's centre of nuclear charge,
      :math:`\\mathbf{R} = \\sum_i Z_i \\mathbf{r}_i / \\sum_i Z_i`, from its positions.
      Because the origin is defined by the system itself it translates with it, so the
      predicted multipole is invariant under a rigid translation whatever the value of
      :math:`\\sum_i q_i`. This is a convention used by some electronic structure codes,
      but not all.
    - ``"absolute"``: use the positions as they are stored, without shifting the origin.
      The predicted multipole is then **not** translationally invariant unless the local
      charges happen to sum to zero: translating a system by :math:`\\mathbf{a}` changes
      it by :math:`\\left(\\sum_i q_i\\right)\\mathbf{a}`. Use this only when the
      absolute frame is meaningful, or to reproduce earlier behaviour.
    """
