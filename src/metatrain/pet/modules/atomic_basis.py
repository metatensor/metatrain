import math
from typing import List, Optional

import torch


class ZConditionedReadout(torch.nn.Module):
    """
    Species-conditioned readout module: either a single linear layer or a
    multi-layer perceptron (MLP) with SiLU activations.

    When ``z_conditioned=True``, every layer has its own per-species weight
    matrix and bias selected at runtime via ``species_idx``.  When
    ``z_conditioned=False``, a single shared set of parameters is used
    (equivalent to a standard ``torch.nn.Linear`` / MLP).  The forward
    signature is identical in both cases, so the same module container can
    hold either variant without breaking TorchScript.

    :param in_features: Number of input features per atom.
    :param out_features: Number of output features per atom.
    :param n_species: Number of atomic species (ignored when
        ``z_conditioned=False``).
    :param z_conditioned: Whether to use species-conditioned weights and
        biases for every layer.
    :param hidden_layer_widths: Widths of the hidden layers.  ``None``
        (default) gives a single linear readout
        ``in_features → out_features``.  A non-empty list gives an MLP
        ``in_features → h0 → h1 → … → out_features`` with SiLU activations
        after every hidden layer.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        n_species: int,
        z_conditioned: bool,
        hidden_layer_widths: Optional[List[int]] = None,
    ) -> None:
        super().__init__()

        self.z_conditioned = z_conditioned
        self.in_features = in_features
        self.out_features = out_features
        # When not Z-conditioned we only need a single set of weights
        # (n_species=1), which keeps the parameter layout compatible with
        # downstream code (e.g. LLPR) that accesses weights by named path.
        actual_n_species = n_species if z_conditioned else 1
        self.n_species = actual_n_species

        # Build the sequence of layer dimensions:
        #   in_features → [hidden_layer_widths...] → out_features
        if hidden_layer_widths is None:
            layer_dims: List[int] = [in_features, out_features]
        else:
            layer_dims = [in_features] + hidden_layer_widths + [out_features]

        # Allocate one batched weight matrix and bias per layer.
        # Shapes: (actual_n_species, out_dim, in_dim) and
        #         (actual_n_species, out_dim).
        weights: List[torch.nn.Parameter] = []
        biases: List[torch.nn.Parameter] = []
        for i in range(len(layer_dims) - 1):
            in_dim = layer_dims[i]
            out_dim = layer_dims[i + 1]
            w = torch.nn.Parameter(torch.empty(actual_n_species, out_dim, in_dim))
            b = torch.nn.Parameter(torch.empty(actual_n_species, out_dim))
            torch.nn.init.kaiming_uniform_(w, a=math.sqrt(5))
            bound = 1.0 / math.sqrt(in_dim) if in_dim > 0 else 0.0
            torch.nn.init.uniform_(b, -bound, bound)
            weights.append(w)
            biases.append(b)

        self.weights = torch.nn.ParameterList(weights)
        self.biases = torch.nn.ParameterList(biases)

    def forward(
        self, features: torch.Tensor, species_idx: torch.Tensor
    ) -> torch.Tensor:
        """
        :param features: Tensor of shape ``(n_atoms, in_features)`` for node
            features, or ``(n_atoms, n_neighbours, in_features)`` for edge
            features.
        :param species_idx: Long tensor of shape ``(n_atoms,)`` containing
            the species index for each atom (ignored when
            ``z_conditioned=False``).
        :return: Tensor of shape ``(n_atoms, out_features)`` or
            ``(n_atoms, n_neighbours, out_features)``.
        """
        if self.z_conditioned:
            idx = species_idx
        else:
            # All atoms share the single set of parameters stored at index 0.
            idx = torch.zeros(
                features.shape[0], dtype=torch.long, device=features.device
            )

        # Node features arrive as 2-D; unsqueeze so every path through the
        # layers is uniformly 3-D (..., seq, dim) for the batched matmul.
        is_node = features.dim() == 2
        if is_node:
            features = features.unsqueeze(1)  # (n_atoms, 1, in_features)

        x = features
        n_layers = self.weights.__len__()  # type: ignore[attr-defined]
        for i, (W, b) in enumerate(zip(self.weights, self.biases, strict=True)):
            W_ = W[idx]  # (n_atoms, out_dim, in_dim)
            b_ = b[idx]  # (n_atoms, out_dim)
            x = torch.matmul(x, W_.transpose(-2, -1)) + b_.unsqueeze(1)
            if i < n_layers - 1:
                x = torch.nn.functional.silu(x)

        if is_node:
            x = x.squeeze(1)
        return x
