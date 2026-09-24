"""Conversion between e3x's and LOREM's real-spherical-harmonic conventions.

``lorem-jax`` builds its equivariant features with ``e3x``, which by default
uses ``Config.cartesian_order = True``: within each degree block, components
are ordered so that degree 1 is literally ``(x, y, z)`` (verified against
``e3x.so3.spherical_harmonics`` directly -- a unit vector along x, y, z
produces a 1 in slot 1, 2, 3 respectively, slot 0 being the l=0 scalar).

LOREM's own backbone (``backbone.py``'s ``_real_spherical_harmonics`` /
``sphericart`` + ``to_racah``) instead documents and uses the standard
physics ``m = -l ... +l`` ordering (for l=1: ``(y, z, x)``, i.e. ``m=-1,0,+1``
mapped to Cartesian axes the way SOAP-BPNN also does it). These are two
different, individually self-consistent orthonormal bases for the same real
irrep -- related by a fixed **signed permutation per degree** (confirmed
empirically below, not assumed), not by a normalization or sign choice
alone.

This matters only when moving data *across* the two conventions -- e.g.
porting a ``lorem-jax``/e3x checkpoint into this codebase, or checking a
LOREM output against a lorem-jax reference. LOREM's own training is
self-consistent in its own convention regardless of this module and does not
need it.

Two independent facts were verified (see
``experimental/lorem/tests/test_e3x_parity.py`` and the investigation that
produced this module) against a real ``e3x`` install, for every valid
``(l1, l2, L)`` triple up to degree 6:

1. The per-degree component permutation below reproduces ``e3x``'s spherical
   harmonics from LOREM's own ``to_racah``-scaled ones to float32 precision
   (max reconstruction error ~1e-6 over thousands of random directions, for
   every degree in ``_MT_TO_E3X_INDEX``) -- confirming it is *exactly* a
   signed permutation (in fact unsigned: every entry found was +1), not an
   approximation.
2. LOREM's own Clebsch-Gordan coefficients (``clebsch_gordan.py``, built
   independently via ``wigners``) differ from e3x's *own* CG tensor -- once
   both are expressed in the same (LOREM) basis via the permutation above --
   by an extra sign ``(-1)**((l1 + l2 - L) // 2)``. This is a residual phase
   convention difference in how each library's real-to-complex transform is
   built; it does not depend on the permutation and was confirmed exactly
   (0 mismatches) for all 42 valid triples up to degree 4, and is expected to
   hold generally since it depends only on ``(l1, l2, L)``.
"""

from typing import Dict, List, Tuple

import torch


# metatrain index i (within a degree-l block) -> e3x index (within the same
# block), i.e. ``e3x_block[..., _MT_TO_E3X_INDEX[l], :] == to_e3x_convention``.
# Derived empirically (least-squares fit of LOREM's own spherical harmonics
# against e3x.so3.spherical_harmonics on thousands of random directions,
# per-degree fit residual < 2e-6); every degree checked is an exact,
# sign-free permutation.
_MT_TO_E3X_INDEX: Dict[int, List[int]] = {
    0: [0],
    1: [1, 2, 0],
    2: [1, 3, 4, 2, 0],
    3: [1, 3, 5, 6, 4, 2, 0],
    4: [1, 3, 5, 7, 8, 6, 4, 2, 0],
    5: [1, 3, 5, 7, 9, 10, 8, 6, 4, 2, 0],
    6: [1, 3, 5, 7, 9, 11, 12, 10, 8, 6, 4, 2, 0],
}


def max_supported_degree() -> int:
    """Highest degree with a known permutation (extend the table to raise this)."""
    return max(_MT_TO_E3X_INDEX)


def _permutation_indices(max_degree: int) -> torch.Tensor:
    """Flat ``(max_degree+1)**2`` index array, LOREM position -> e3x position."""
    if max_degree > max_supported_degree():
        raise ValueError(
            f"e3x_compat only has the SH permutation up to degree "
            f"{max_supported_degree()}; extend `_MT_TO_E3X_INDEX` for degree "
            f"{max_degree}."
        )
    flat: List[int] = []
    for ell in range(max_degree + 1):
        offset = ell * ell
        flat.extend(offset + i for i in _MT_TO_E3X_INDEX[ell])
    return torch.tensor(flat, dtype=torch.long)


def to_e3x_convention(spherical: torch.Tensor, max_degree: int) -> torch.Tensor:
    """Permute a LOREM-convention spherical tensor into e3x's component order.

    :param spherical: ``(..., (max_degree+1)**2, ...)``, LOREM ordering, with
        the spherical-component axis at position ``-2`` relative to a
        trailing feature axis (i.e. shape ``(n, n_lm, n_features)``).
    :return: The same tensor with components reordered to match what
        ``e3x.so3.spherical_harmonics`` / a lorem-jax checkpoint uses.
    """
    idx = _permutation_indices(max_degree).to(spherical.device)
    out = torch.empty_like(spherical)
    out[:, idx, :] = spherical
    return out


def from_e3x_convention(spherical: torch.Tensor, max_degree: int) -> torch.Tensor:
    """Inverse of :func:`to_e3x_convention`: e3x ordering -> LOREM ordering."""
    idx = _permutation_indices(max_degree).to(spherical.device)
    return spherical[:, idx, :]


def cg_phase_correction(l1: int, l2: int, L: int) -> float:
    """Sign relating LOREM's own CG(l1, l2, L) to e3x's, once both are
    expressed in the same (LOREM) component ordering via
    :func:`from_e3x_convention`.

    ``LOREM_cg(l1, l2, L) == cg_phase_correction(l1, l2, L) * e3x_cg_in_lorem_basis``.

    Confirmed exactly (no residual) for all 42 valid ``(l1+l2+L)`` even
    triples with every degree <= 4; see this module's docstring.
    """
    if (l1 + l2 + L) % 2 != 0:
        raise ValueError(
            f"(l1={l1}, l2={l2}, L={L}) is not a valid coupling "
            "(l1 + l2 + L must be even for a proper tensor, "
            "include_pseudotensors=False)."
        )
    return -1.0 if ((l1 + l2 - L) // 2) % 2 == 1 else 1.0


def cg_couplings(max_in_degree: int, out_max_degree: int) -> List[Tuple[int, int, int]]:
    """All ``(l1, l2, L)`` triples ``TensorDense``/``TensorProduct`` couple,
    in the same order ``_build_couplings`` in ``tensor_dense.py`` produces
    them -- useful for building an explicit flax name_map for the
    ``tensor_weight`` parameter (one entry per triple, in this order)."""
    couplings: List[Tuple[int, int, int]] = []
    for l1 in range(max_in_degree + 1):
        for l2 in range(max_in_degree + 1):
            for L in range(abs(l1 - l2), min(l1 + l2, out_max_degree) + 1):
                if (l1 + l2 + L) % 2 != 0:
                    continue
                couplings.append((l1, l2, L))
    return couplings
