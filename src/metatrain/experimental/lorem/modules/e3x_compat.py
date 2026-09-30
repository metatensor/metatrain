"""Reorder spherical-harmonic components between this code and e3x.

Only needed when loading an external checkpoint that used e3x's component
order. Training in this package does not use it.

Degree 1 in e3x is ``(x, y, z)``. Here it is ``(y, z, x)`` (``m = -1, 0, +1``).
The map is a fixed permutation per degree (``_MT_TO_E3X_INDEX``), plus a sign
``(-1)**((l1 + l2 - L) // 2)`` on Clebsch-Gordan coefficients.
"""

from typing import Dict, List, Tuple

import torch


# metatrain index within a degree -> e3x index within the same degree.
# Degree 1 is the ``(y, z, x)`` -> ``(x, y, z)`` swap. Higher degrees follow
# the same reversal. Extend this table to support a higher ``max_degree``.
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

    Checked against e3x for every valid triple with all degrees <= 4
    (``tests/test_e3x_parity.py``).
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
    in the order of ``tensor_dense._build_couplings`` (one ``tensor_weight``
    row per triple)."""
    couplings: List[Tuple[int, int, int]] = []
    for l1 in range(max_in_degree + 1):
        for l2 in range(max_in_degree + 1):
            for L in range(abs(l1 - l2), min(l1 + l2, out_max_degree) + 1):
                if (l1 + l2 + L) % 2 != 0:
                    continue
                couplings.append((l1, l2, L))
    return couplings
