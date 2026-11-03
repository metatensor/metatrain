"""Clebsch-Gordan tensor product used by the short-range and long-range blocks.

Both layers evaluate as e3x does: weights live on a dense degree grid,
are expanded to ``(ℓ, m)`` resolution with precomputed index buffers, and
are contracted with a dense Clebsch-Gordan tensor in one einsum.

- Each angular-momentum degree has its own linear weight. Only the scalar
  (degree 0) channel has a bias.
- Each allowed ``(l1, l2, L)`` coupling has its own learnable weight. The
  weight grid also holds the disallowed triples, as e3x's does; their
  Clebsch-Gordan block is zero, so they stay at zero and get no gradient.
"""

import math

import torch

from .clebsch_gordan import degree_index, dense_clebsch_gordan


# standard deviation of a unit normal truncated at +-2, which e3x (and flax's
# lecun_normal) divides out so initial weights keep the intended variance
_TRUNCATED_NORMAL_STD = 0.87962566103423978


class _DegreeWiseLinear(torch.nn.Module):
    """Port of ``e3x.nn.Dense``: one weight matrix per angular-momentum degree.

    Applied to a ``(n_atoms, (max_degree+1)**2, in_features)`` spherical
    tensor, degree ``l``'s ``(2l+1)`` rows all go through the *same* matrix
    (required for equivariance), but each degree has its *own* matrix. Only
    the scalar (``l=0``) channel gets a bias, as in e3x (a bias on ``l>0``
    would break equivariance). ``weight[l]`` is ``(in, out)``, the layout of
    an e3x kernel. Initialized like e3x: ``lecun_normal`` kernels (a normal
    truncated at +-2 standard deviations, rescaled to variance
    ``1 / in_features``) and a zero bias.
    """

    def __init__(
        self, in_features: int, out_features: int, max_degree: int, bias: bool = True
    ) -> None:
        super().__init__()
        weight = torch.empty(max_degree + 1, in_features, out_features)
        torch.nn.init.trunc_normal_(weight, a=-2.0, b=2.0)
        self.weight = torch.nn.Parameter(
            weight / (_TRUNCATED_NORMAL_STD * math.sqrt(in_features))
        )
        if bias:
            self.bias = torch.nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)
        self.register_buffer("degree", degree_index(max_degree), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """:param x: ``(n_atoms, (max_degree+1)**2, in_features)``."""
        out = torch.einsum("nmi,mio->nmo", x, self.weight[self.degree])
        bias = self.bias
        if bias is not None:
            out[:, 0] += bias
        return out


class _CGProduct(torch.nn.Module):
    """Weighted Clebsch-Gordan product of two spherical tensors.

    ``tensor_weight`` is e3x's ``(L1+1, L2+1, L3+1, F)`` kernel grid,
    disallowed triples included (their Clebsch-Gordan block is zero), so a
    lorem-jax kernel copies over as is.
    """

    def _init_product(
        self,
        l1_max: int,
        l2_max: int,
        out_max_degree: int,
        n_features: int,
        even_parity: bool = False,
    ) -> None:
        # e3x's parity mask without pseudotensors: the output parity is
        # (-1)^(l1 + l2) and must be (-1)^L, unless ``even_parity`` keeps every
        # even l1 + l2 path (e3x's parity +1 slot with pseudotensors on)
        def allowed(l1: int, l2: int, L: int) -> bool:
            if even_parity:
                return (l1 + l2) % 2 == 0
            return (l1 + l2 + L) % 2 == 0

        cg = dense_clebsch_gordan(l1_max, l2_max, out_max_degree, allowed)
        # stored in float64 and cast where used (like the BEC head's buffer), so
        # a float64 model gets exact coefficients
        self.register_buffer("cg", cg.to(torch.float64), persistent=False)
        # row of the flattened weight grid for each (lm1, lm2, lm3) entry
        degree1 = degree_index(l1_max)[:, None, None]
        degree2 = degree_index(l2_max)[None, :, None]
        degree3 = degree_index(out_max_degree)[None, None, :]
        n2, n3 = l2_max + 1, out_max_degree + 1
        self.register_buffer(
            "weight_index",
            (degree1 * n2 + degree2) * n3 + degree3,
            persistent=False,
        )

        # e3x's ``tensor_lecun_normal``: output degree L is fed by n_L coupling
        # paths, each scaled 1/sqrt(n_L).
        paths = torch.tensor(
            [
                [
                    [
                        abs(l1 - l2) <= L <= l1 + l2 and allowed(l1, l2, L)
                        for L in range(out_max_degree + 1)
                    ]
                    for l2 in range(l2_max + 1)
                ]
                for l1 in range(l1_max + 1)
            ]
        )
        std = 1.0 / paths.sum(dim=(0, 1)).clamp(min=1).sqrt() / _TRUNCATED_NORMAL_STD
        weight = torch.empty(*paths.shape, n_features)
        torch.nn.init.trunc_normal_(weight, a=-2.0, b=2.0)
        self.tensor_weight = torch.nn.Parameter(
            weight * std[:, None] * paths[..., None]
        )

    def contract(self, outer: torch.Tensor) -> torch.Tensor:
        """``(n, n_lm1, n_lm2, F)`` outer products ``->`` ``(n, n_lm3, F)``.

        The product is linear in the outer product, so a sum of outer products
        (messages scattered onto atoms) can be contracted once.
        """
        n_features = self.tensor_weight.shape[-1]
        weight = self.tensor_weight.reshape(-1, n_features)[self.weight_index]
        kernel = weight * self.cg.to(dtype=weight.dtype).unsqueeze(-1)
        # a matmul batched over features
        return torch.einsum("nijf,ijkf->nkf", outer, kernel)

    def _couple(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        """``(n, n_lm1, F) x (n, n_lm2, F) -> (n, n_lm3, F)``."""
        return self.contract(left.unsqueeze(2) * right.unsqueeze(1))


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
        use_bias: bool = False,
        even_parity: bool = False,
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
        self.out_features = int(out_features)

        # e3x.nn.TensorDense projects with *one* degree-wise Dense to
        # `2 * out_features`, then splits the result into the two self-product
        # factors, so a lorem-jax checkpoint's single `dense` kernel maps onto it.
        self.dense = _DegreeWiseLinear(
            in_features, 2 * out_features, self.in_max_degree, bias=use_bias
        )
        self._init_product(
            self.in_max_degree,
            self.in_max_degree,
            self.out_max_degree,
            self.out_features,
            even_parity,
        )

    def forward(self, spherical: torch.Tensor) -> torch.Tensor:
        """:param spherical: ``(n_atoms, (in_max_degree + 1) ** 2, in_features)``."""
        projected = self.dense(spherical)
        a, b = projected[..., : self.out_features], projected[..., self.out_features :]
        return self._couple(a, b)


class TensorProduct(_CGProduct):
    """CG product of two spherical tensors with a shared feature width.

    Port of ``e3x.nn.Tensor(include_pseudotensors=False)``: no
    internal projection, just the weighted CG coupling of two given inputs.
    """

    def __init__(
        self,
        left_max_degree: int,
        right_max_degree: int,
        out_max_degree: int,
        n_features: int = 1,
    ) -> None:
        super().__init__()
        self.left_max_degree = int(left_max_degree)
        self.right_max_degree = int(right_max_degree)
        self.out_max_degree = int(out_max_degree)
        self._init_product(
            self.left_max_degree,
            self.right_max_degree,
            self.out_max_degree,
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
            filter(edges_basis[ij]), nodes_spherical[j]
        )
        nodes_spherical = tensor_combine(dense_x(nodes_spherical), dense_m(messages))

    (``e3x.nn.MessagePass`` for the first line, ``e3x.nn.Dense`` +
    ``e3x.nn.Tensor`` for the second). The filter is the *first* factor, as
    in e3x's ``_Conv.__call__``; the kernel's first degree axis belongs to it.
    ``filter`` and both ``dense_*`` are :class:`_DegreeWiseLinear`; the two
    couplings are separate :class:`TensorProduct` instances (e3x gives
    ``MessagePass`` and the combine step independent kernels).
    """

    def __init__(self, num_features: int, max_degree: int) -> None:
        super().__init__()
        self.max_degree = int(max_degree)
        self.filter = _DegreeWiseLinear(
            num_features, num_features, self.max_degree, bias=False
        )
        self.message_tensor = TensorProduct(
            self.max_degree,
            self.max_degree,
            self.max_degree,
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
        # sum the per-edge outer products onto atoms before contracting with
        # the kernel: same result, n_neighbors times fewer contraction FLOPs
        outer = filtered.unsqueeze(2) * gathered.unsqueeze(1)
        summed = outer.new_zeros((n_atoms,) + outer.shape[1:])
        if outer.shape[0] > 0:
            summed.index_add_(0, centers, outer)
        messages = self.message_tensor.contract(summed)
        return self.combine_tensor(
            self.combine_dense_x(nodes_spherical),
            self.combine_dense_m(messages),
        )
