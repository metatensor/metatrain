"""
Tensor basis
============

Provides a tensor basis in which to predict spherical tensor targets,
following the approach described in this work :footcite:p:`domina2025representing`.

This extender creates a basis for each target and each angular channel
(``o3_lambda`` block) of the outputs. Then it asks for invariant
coefficients to apply to the basis to produce the target. By using
this extender one can:

- **Use an architecture that produces only scalar outputs**, and still be
  able to predict tensorial targets.
- **Reduce the cost of equivariant models** by asking them to produce only
  scalar outputs, and then use this extender to access the angular momentum
  channels of the target.

"""

from typing import Optional

from typing_extensions import TypedDict

from metatrain.soap_bpnn.documentation import SOAPConfig
from metatrain.utils.hypers import init_with_defaults


class Hypers(TypedDict):
    """
    Hyperparameters for the tensor basis extender.
    """

    soap: SOAPConfig = init_with_defaults(SOAPConfig)
    """Hyperparameters used to compute the spherical expansions from
    which the vector basis will be built. Higher angular momentum
    channels are built by augmenting the order of the vector basis."""

    inputs: Optional[str | list] = None
    """
    Name or names of the targets to use as invariant coefficients
    to apply to the tensor basis.

    If ``None``, they will be set as
    ``mtt::extender_aux::tensor_basis::{output_name.replace('mtt::', '')}``
    for each output name.
    """

    outputs: Optional[str | list] = None
    """
    Name or names of the targets to predict through a tensor basis.

    A separate tensor basis will be built for each target.
    """
