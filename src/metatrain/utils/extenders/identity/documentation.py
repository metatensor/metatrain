"""
Identity
========

This extender just passes the inputs to the outputs without any modification.

For someone who wants to understand how extenders work or wants to implement
their own extender, this is the best point to start, as it is the simplest
extender possible.

It is also very useful for testing and debugging.
"""

from typing import Optional

from typing_extensions import TypedDict


class Hypers(TypedDict):
    """
    Hyperparameters for the identity extender.
    """

    inputs: Optional[str | list[str]] = None
    """The names of the inputs to be passed to the outputs.

    If ``None``, they will be set as
    ``mtt::extender_aux::identity::{output_name.replace('mtt::', '')}``
    for each output name.
    """

    outputs: Optional[str | list[str]] = None
    """The names of the outputs to be produced by the extender."""
