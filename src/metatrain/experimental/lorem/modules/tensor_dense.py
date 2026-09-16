"""Clebsch-Gordan tensor product, the TorchScript port of ``e3x.nn.TensorDense``
/ ``e3x.nn.Tensor`` / ``e3x.nn.MessagePass``.

``lorem-jax`` applies ``e3x.nn.TensorDense`` as a self-product on spherical
node features, ``e3x.nn.Tensor`` to mix long-range potentials back into those
features, and ``e3x.nn.MessagePass`` for equivariant message passing. All
three share the same two building blocks, both ported here to match e3x's
actual parameterization (verified against a real ``lorem-jax`` parameter
tree, not just against e3x's docs):

- A per-degree ``Dense`` layer (``e3x.nn.Dense``): unlike a plain
  ``torch.nn.Linear`` applied over the trailing feature axis (which shares
  one weight matrix across every angular-momentum degree), e3x gives each
  degree ``l`` its own weight matrix, with a bias only on the scalar
  (``l=0``) channel. ``_DegreeWiseLinear`` below is that per-degree layer.
- A **learnable, per-``(l1, l2, L)``-triple, per-feature weight** multiplying
  each Clebsch-Gordan coupling term before summing (``e3x.nn.Tensor``'s
  ``kernel``). An earlier version of this module summed every valid triple
  unweighted, which is a strictly less expressive special case (still
  equivariant, just missing a real learnable parameter e3x has).
"""

import math
from typing import List

import torch

from .clebsch_gordan import cg_combine_features, get_cg_coefficients


class _CGBuffer(torch.nn.Module):
    """One ``(2l1+1, 2l2+1, 2L+1)`` Clebsch-Gordan tensor as a buffer."""

    l1: int
    l2: int
    L: int

    def __init__(self, tensor: torch.Tensor, l1: int, l2: int, L: int) -> None:
        super().__init__()
        self.register_buffer("cg", tensor)
        self.l1 = int(l1)
        self.l2 = int(l2)
        self.L = int(L)


class _DegreeWiseLinear(torch.nn.Module):
    """Port of ``e3x.nn.Dense``: one ``Linear`` per angular-momentum degree.

    Applied to a ``(n_atoms, (max_degree+1)**2, in_features)`` spherical
    tensor, degree ``l``'s ``(2l+1)`` rows all go through the *same* weight
    matrix (required for equivariance), but each degree has its *own* weight
    matrix (unlike ``torch.nn.Linear``, which would share one matrix across
    every degree). Only the scalar (``l=0``) channel gets a bias, matching
    e3x (a bias on ``l>0`` would break equivariance).
    """

    def __init__(
        self, in_features: int, out_features: int, max_degree: int, bias: bool = True
    ) -> None:
        super().__init__()
        self.max_degree = int(max_degree)
        self.layers = torch.nn.ModuleList(
            [
                torch.nn.Linear(in_features, out_features, bias=(bias and ell == 0))
                for ell in range(self.max_degree + 1)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """:param x: ``(n_atoms, (max_degree+1)**2, in_features)``."""
        outputs: List[torch.Tensor] = []
        for ell, layer in enumerate(self.layers):
            outputs.append(layer(x[:, ell * ell : (ell + 1) * (ell + 1), :]))
        return torch.cat(outputs, dim=1)


class _CGProduct(torch.nn.Module):
    """Shared TorchScript CG loop. Couplings live on ``self``, not as args."""

    def _init_couplings(
        self,
        couplings: List[torch.Tensor],
        l1: List[int],
        l2: List[int],
        L: List[int],
        out_n_lm: int,
        n_features: int,
    ) -> None:
        self.out_n_lm = int(out_n_lm)
        self.couplings = torch.nn.ModuleList(
            [
                _CGBuffer(tensor, l1[index], l2[index], L[index])
                for index, tensor in enumerate(couplings)
            ]
        )
        # Learnable per-(l1, l2, L)-triple, per-feature weight (e3x.nn.Tensor's
        # "kernel"). Initialized fan-in-per-output-degree normalized (roughly
        # matching e3x's `tensor_lecun_normal`): each output degree L is fed by
        # `n_L` coupling paths, so those paths are scaled by `1/sqrt(n_L)`.
        n_per_L: dict = {}
        for target in L:
            n_per_L[target] = n_per_L.get(target, 0) + 1
        weight = torch.empty(len(couplings), n_features)
        for index, target in enumerate(L):
            std = 1.0 / math.sqrt(max(n_per_L[target], 1))
            torch.nn.init.trunc_normal_(weight[index], std=std, a=-2 * std, b=2 * std)
        self.tensor_weight = torch.nn.Parameter(weight)

    def _couple(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        n_atoms = left.shape[0]
        n_features = left.shape[2]
        output = left.new_zeros((n_atoms, self.out_n_lm, n_features))
        for index, coupling in enumerate(self.couplings):
            l1 = coupling.l1
            l2 = coupling.l2
            L = coupling.L
            coupled = cg_combine_features(
                left[:, l1 * l1 : (l1 + 1) * (l1 + 1), :],
                right[:, l2 * l2 : (l2 + 1) * (l2 + 1), :],
                coupling.cg.to(dtype=left.dtype),
            )
            coupled = coupled * self.tensor_weight[index].to(dtype=left.dtype)
            output[:, L * L : (L + 1) * (L + 1), :] = (
                output[:, L * L : (L + 1) * (L + 1), :] + coupled
            )
        return output


def _build_couplings(
    l1_max: int,
    l2_max: int,
    out_max_degree: int,
    include_pseudotensors: bool,
):
    cg = get_cg_coefficients(max(l1_max, l2_max, out_max_degree))
    l1_list: List[int] = []
    l2_list: List[int] = []
    L_list: List[int] = []
    couplings: List[torch.Tensor] = []
    for l1 in range(l1_max + 1):
        for l2 in range(l2_max + 1):
            for L in range(abs(l1 - l2), min(l1 + l2, out_max_degree) + 1):
                if not include_pseudotensors and (l1 + l2 + L) % 2 != 0:
                    continue
                couplings.append(cg.get((l1, l2, L)).to(torch.float32))
                l1_list.append(l1)
                l2_list.append(l2)
                L_list.append(L)
    return couplings, l1_list, l2_list, L_list


class TensorDense(_CGProduct):
    """Linear feature projections followed by a CG self-product.

    Port of ``e3x.nn.TensorDense(use_fused_tensor=False)``. Input and output
    layout is ``(n_atoms, (L+1)**2, n_features)``, with real spherical
    harmonics ordered ``m = -ℓ … +ℓ`` inside each ``ℓ`` (the same order as
    the LOREM backbone).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        in_max_degree: int,
        out_max_degree: int,
        include_pseudotensors: bool = False,
        use_bias: bool = False,
    ) -> None:
        super().__init__()
        if in_max_degree < 0 or out_max_degree < 0:
            raise ValueError("max_degree must be >= 0")
        if out_max_degree > in_max_degree * 2:
            raise ValueError(
                f"out_max_degree ({out_max_degree}) cannot exceed "
                f"2 * in_max_degree ({in_max_degree})."
            )

        self.in_max_degree = int(in_max_degree)
        self.out_max_degree = int(out_max_degree)
        self.in_n_lm = (self.in_max_degree + 1) * (self.in_max_degree + 1)
        self.out_features = int(out_features)

        # e3x.nn.TensorDense projects with *one* degree-wise Dense to
        # `2 * out_features`, then splits the result into the two self-product
        # factors -- not two independently-parameterized projections. Kept as
        # one module (not two) so a lorem-jax checkpoint's single `dense`
        # kernel maps onto it directly.
        self.dense = _DegreeWiseLinear(
            in_features, 2 * out_features, self.in_max_degree, bias=use_bias
        )

        couplings, l1_list, l2_list, L_list = _build_couplings(
            self.in_max_degree,
            self.in_max_degree,
            out_max_degree,
            include_pseudotensors,
        )
        self._init_couplings(
            couplings,
            l1_list,
            l2_list,
            L_list,
            (self.out_max_degree + 1) * (self.out_max_degree + 1),
            self.out_features,
        )

    def forward(self, spherical: torch.Tensor) -> torch.Tensor:
        """:param spherical: ``(n_atoms, in_n_lm, in_features)``."""
        projected = self.dense(spherical)
        a, b = projected[..., : self.out_features], projected[..., self.out_features :]
        return self._couple(a, b)


class TensorProduct(_CGProduct):
    """CG product of two spherical tensors with a shared feature width.

    Port of ``e3x.nn.Tensor(..., include_pseudotensors=False)`` -- no
    internal projection, just the weighted CG coupling of two given inputs.
    """

    def __init__(
        self,
        left_max_degree: int,
        right_max_degree: int,
        out_max_degree: int,
        include_pseudotensors: bool = False,
        n_features: int = 1,
    ) -> None:
        super().__init__()
        self.left_max_degree = int(left_max_degree)
        self.right_max_degree = int(right_max_degree)
        self.out_max_degree = int(out_max_degree)

        couplings, l1_list, l2_list, L_list = _build_couplings(
            self.left_max_degree,
            self.right_max_degree,
            out_max_degree,
            include_pseudotensors,
        )
        self._init_couplings(
            couplings,
            l1_list,
            l2_list,
            L_list,
            (self.out_max_degree + 1) * (self.out_max_degree + 1),
            n_features,
        )

    def forward(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        """``left`` / ``right`` are ``(n_atoms, n_lm_*, n_features)``."""
        return self._couple(left, right)


class EquivariantMessagePass(torch.nn.Module):
    """Port of one lorem-jax equivariant message-passing step.

    lorem-jax's ``num_message_passing`` loop updates both scalar *and*
    spherical node features each iteration. The spherical half is::

        messages[i] = sum_(j in N(i)) tensor(
            nodes_spherical[j], filter(edges_basis[ij])
        )
        nodes_spherical = tensor_combine(dense_x(nodes_spherical), dense_m(messages))

    (``e3x.nn.MessagePass`` for the first line, ``e3x.nn.Dense`` +
    ``e3x.nn.Tensor`` for the second). ``filter`` and both ``dense_*`` are
    :class:`_DegreeWiseLinear`; the two couplings are separate
    :class:`TensorProduct` instances (each with its own learnable
    ``tensor_weight`` -- e3x gives ``MessagePass`` and the combine step
    independent kernels).
    """

    def __init__(
        self, num_features: int, max_degree: int, include_pseudotensors: bool = False
    ) -> None:
        super().__init__()
        self.max_degree = int(max_degree)
        self.filter = _DegreeWiseLinear(
            num_features, num_features, self.max_degree, bias=False
        )
        self.message_tensor = TensorProduct(
            self.max_degree,
            self.max_degree,
            self.max_degree,
            include_pseudotensors=include_pseudotensors,
            n_features=num_features,
        )
        self.combine_dense_x = _DegreeWiseLinear(
            num_features, num_features, self.max_degree, bias=False
        )
        self.combine_dense_m = _DegreeWiseLinear(
            num_features, num_features, self.max_degree, bias=False
        )
        self.combine_tensor = TensorProduct(
            self.max_degree,
            self.max_degree,
            self.max_degree,
            include_pseudotensors=include_pseudotensors,
            n_features=num_features,
        )

    def forward(
        self,
        nodes_spherical: torch.Tensor,
        edges_basis: torch.Tensor,
        centers: torch.Tensor,
        neighbors: torch.Tensor,
    ) -> torch.Tensor:
        """
        :param nodes_spherical: ``(n_atoms, n_lm, F)`` current spherical features.
        :param edges_basis: ``(n_edges, n_lm, F)`` per-edge spherical "basis"
            to filter (e.g. edge-coefficients times spherical harmonics).
        :param centers: ``(n_edges,)`` destination atom index (``i``).
        :param neighbors: ``(n_edges,)`` source atom index (``j``).
        :return: ``(n_atoms, n_lm, F)`` updated spherical features.
        """
        n_atoms = nodes_spherical.shape[0]
        filtered = self.filter(edges_basis)
        gathered = nodes_spherical[neighbors]
        products = self.message_tensor(gathered, filtered)
        messages = nodes_spherical.new_zeros(
            (n_atoms, products.shape[1], products.shape[2])
        )
        if products.shape[0] > 0:
            messages.index_add_(0, centers, products)
        return self.combine_tensor(
            self.combine_dense_x(nodes_spherical),
            self.combine_dense_m(messages),
        )
