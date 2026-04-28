import math

import torch


class ZConditionedLinear(torch.nn.Module):
    """
    Species-conditioned linear readout layer implemented as a single batched matmul.

    When ``z_conditioned=True`` (the default), each atomic species has its own
    weight matrix and bias, selected at runtime via ``species_idx``.

    When ``z_conditioned=False``, a single shared weight and bias are used for
    all species (equivalent to a standard ``torch.nn.Linear``).  The forward
    signature is identical in both cases so the same module container can hold
    either variant without breaking TorchScript.

    :param in_features: Number of input features per atom.
    :param out_features: Number of output features per atom.
    :param n_species: Number of atomic species (ignored if ``z_conditioned=False``).
    :param z_conditioned: Whether to use species-conditioned weights and biases.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        n_species: int,
        z_conditioned: bool,
    ) -> None:

        super().__init__()

        self.z_conditioned = z_conditioned
        self.in_features = in_features
        self.out_features = out_features
        # When not Z-conditioned we only need a single set of weights (n_species=1),
        # which keeps the parameter layout identical to ZConditionedLinear so that
        # downstream code (e.g. LLPR) can always refer to ".weight" and ".bias".
        actual_n_species = n_species if z_conditioned else 1
        self.n_species = actual_n_species
        self.weight = torch.nn.Parameter(
            torch.empty(actual_n_species, out_features, in_features)
        )
        self.bias = torch.nn.Parameter(torch.empty(actual_n_species, out_features))
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in = in_features
        bound = 1.0 / math.sqrt(fan_in) if fan_in > 0 else 0.0
        torch.nn.init.uniform_(self.bias, -bound, bound)

    def forward(
        self, features: torch.Tensor, species_idx: torch.Tensor
    ) -> torch.Tensor:
        """
        :param features: Tensor of shape (n_atoms, in_features) or (n_edges,
            in_features).
        :param species_idx: Tensor of shape (n_atoms,) containing the species index
            for each atom (ignored if ``z_conditioned=False``).
        :return: Tensor of shape (n_atoms, out_features) or (n_edges, out_features).
        """
        if self.z_conditioned:
            idx = species_idx
        else:
            # All atoms share the single set of weights stored at index 0.
            idx = torch.zeros(
                features.shape[0], dtype=torch.long, device=features.device
            )
        W = self.weight[idx]  # (n_atoms, out, in)
        b = self.bias[idx]  # (n_atoms, out)
        is_node = features.dim() == 2
        if is_node:
            features = features.unsqueeze(1)  # (n_atoms, 1, in)
        out = torch.matmul(features, W.transpose(-2, -1)) + b.unsqueeze(1)
        if is_node:
            out = out.squeeze(1)
        return out
