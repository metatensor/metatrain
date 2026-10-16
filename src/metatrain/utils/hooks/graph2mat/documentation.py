"""
Graph2mat
========

Builds edge features from node features.
"""

from typing import Optional

from typing_extensions import TypedDict




class Hypers(TypedDict):
    """
    Hyperparameters for the graph2mat hook.
    """

    outputs: Optional[str | list[str]] = None
    """The names of the outputs to be produced by the hook."""

    cutoff: float = 8.0
    """The cutoff distance for the neighbor list."""

    edge_lmax: int = 4
    """The maximum degree of the spherical harmonics that embed
    edge vectors."""

    node_hidden_irreps: str = "20x0e + 20x1o + 20x2e + 20x3o + 20x4e"
    
