"""
Identity
========

This hook just passes the inputs to the outputs without any modification.

For someone who wants to understand how hooks work or wants to implement
their own hook, this is the best point to start, as it is the simplest
hook possible.

It is also very useful for testing and debugging.
"""

from typing import Optional

from typing_extensions import TypedDict


class Hypers(TypedDict):
    """
    Hyperparameters for the identity hook.
    """

    inputs: Optional[str | list[str]] = None
    """The names of the inputs to be passed to the outputs.

    If ``None``, they will be set as
    ``mtt::aux::identity::{output_name.replace('mtt::', '')}``
    for each output name.
    """

    outputs: Optional[str | list[str]] = None
    """The names of the outputs to be produced by the hook."""
