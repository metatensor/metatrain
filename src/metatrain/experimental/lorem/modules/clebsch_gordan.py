"""Real Clebsch-Gordan coefficients, matching SOAP-BPNN's convention.

Used by ``TensorDense`` / ``TensorProduct`` (the ``e3x.nn.TensorDense`` port)
and by the Born-effective-charge head. Coefficients are built at init time
with ``wigners``; :func:`dense_clebsch_gordan` lays them out with e3x's signs
so one einsum evaluates every coupling.
"""

from typing import Callable, Dict, Tuple

import numpy as np
import torch
import wigners


class ClebschGordanReal:
    """Real CG tensors keyed by ``(l1, l2, L)``."""

    def __init__(self) -> None:
        self._cgs: Dict[Tuple[int, int, int], torch.Tensor] = {}

    def get(self, key: Tuple[int, int, int]) -> torch.Tensor:
        if key not in self._cgs:
            self._add(key[0], key[1], key[2])
        return self._cgs[key]

    def _add(self, l1: int, l2: int, L: int) -> None:
        maxx = max(l1, l2, L)
        r2c = {ell: _real2complex(ell) for ell in range(maxx + 1)}
        c2r = {ell: np.conjugate(r2c[ell]).T for ell in range(maxx + 1)}

        complex_cg = _complex_clebsch_gordan_matrix(l1, l2, L)
        real_cg = (r2c[l1].T @ complex_cg.reshape(2 * l1 + 1, -1)).reshape(
            complex_cg.shape
        )
        real_cg = real_cg.swapaxes(0, 1)
        real_cg = (r2c[l2].T @ real_cg.reshape(2 * l2 + 1, -1)).reshape(real_cg.shape)
        real_cg = real_cg.swapaxes(0, 1)
        real_cg = real_cg @ c2r[L].T

        rcg = np.real(real_cg) if (l1 + l2 + L) % 2 == 0 else np.imag(real_cg)
        almost_zero = np.where(np.logical_and(np.abs(rcg) > 0, np.abs(rcg) < 1e-14))
        rcg[almost_zero] = 0.0
        self._cgs[(l1, l2, L)] = torch.tensor(rcg, dtype=torch.float64)


def cg_phase_correction(l1: int, l2: int, L: int) -> float:
    """Sign relating these Clebsch-Gordan coefficients to e3x's.

    ``e3x_cg(l1, l2, L) == cg_phase_correction(l1, l2, L) * cg(l1, l2, L)``,
    both in the ``m = -ℓ … +ℓ`` order. Tested against e3x for every triple
    with all degrees <= 4 (``tests/test_e3x_parity.py``), and checked once
    up to degree 6, the paper default. For odd
    ``l1 + l2 + L`` (two proper tensors coupled into a pseudotensor, as in the
    Born-charge head) the same formula holds, with ``//`` rounding down.
    """
    if not abs(l1 - l2) <= L <= l1 + l2:
        raise ValueError(f"(l1={l1}, l2={l2}, L={L}) is not a valid coupling.")
    return -1.0 if ((l1 + l2 - L) // 2) % 2 == 1 else 1.0


def dense_clebsch_gordan(
    l1_max: int,
    l2_max: int,
    l3_max: int,
    allowed: Callable[[int, int, int], bool],
) -> torch.Tensor:
    """Every allowed coupling in one ``(n_lm1, n_lm2, n_lm3)`` tensor.

    Block ``(l1, l2, L)`` holds e3x's coefficients (so lorem-jax kernels copy
    over unchanged) when ``allowed(l1, l2, L)``, and zeros otherwise: the zeros
    are the parity mask, applied once here instead of on every forward.
    """
    cg = ClebschGordanReal()
    out = torch.zeros(
        ((l1_max + 1) ** 2, (l2_max + 1) ** 2, (l3_max + 1) ** 2), dtype=torch.float64
    )
    for l1 in range(l1_max + 1):
        for l2 in range(l2_max + 1):
            for L in range(abs(l1 - l2), min(l1 + l2, l3_max) + 1):
                if allowed(l1, l2, L):
                    out[
                        l1**2 : (l1 + 1) ** 2,
                        l2**2 : (l2 + 1) ** 2,
                        L**2 : (L + 1) ** 2,
                    ] = cg.get((l1, l2, L)) * cg_phase_correction(l1, l2, L)
    return out


def degree_index(max_degree: int) -> torch.Tensor:
    """The degree of each ``(ℓ, m)`` row: ``[0, 1, 1, 1, 2, …]``."""
    return torch.tensor(
        [ell for ell in range(max_degree + 1) for _ in range(2 * ell + 1)]
    )


def _real2complex(L: int) -> np.ndarray:
    result = np.zeros((2 * L + 1, 2 * L + 1), dtype=np.complex128)
    inv_sqrt_2 = 1.0 / np.sqrt(2)
    for m in range(-L, L + 1):
        if m < 0:
            result[L - m, L + m] = inv_sqrt_2 * 1j * (-1) ** m
            result[L + m, L + m] = -inv_sqrt_2 * 1j
        if m == 0:
            result[L, L] = 1.0
        if m > 0:
            result[L + m, L + m] = inv_sqrt_2 * (-1) ** m
            result[L - m, L + m] = inv_sqrt_2
    return result


def _complex_clebsch_gordan_matrix(l1: int, l2: int, L: int) -> np.ndarray:
    if np.abs(l1 - l2) > L or np.abs(l1 + l2) < L:
        return np.zeros((2 * l1 + 1, 2 * l2 + 1, 2 * L + 1), dtype=np.double)
    return wigners.clebsch_gordan_array(l1, l2, L)
