"""Real Clebsch-Gordan coefficients, matching SOAP-BPNN's convention.

Used by ``TensorDense`` / ``TensorProduct`` (the ``e3x.nn.TensorDense`` port)
and by the Born-effective-charge head. Coefficients are built at init time
with ``wigners``; the forward path is TorchScript-friendly einsum.
"""

from typing import Dict, Tuple

import numpy as np
import torch
import wigners


def cg_combine_features(
    left: torch.Tensor, right: torch.Tensor, coefficients: torch.Tensor
) -> torch.Tensor:
    """Couple two feature-wise spherical tensors.

    :param left: ``(n_atoms, 2*l1+1, n_features)``
    :param right: ``(n_atoms, 2*l2+1, n_features)``
    :param coefficients: ``(2*l1+1, 2*l2+1, 2*L+1)``
    :return: ``(n_atoms, 2*L+1, n_features)``
    """
    return torch.einsum("n i f, n j f, i j k -> n k f", left, right, coefficients)


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


def get_cg_coefficients(l_max: int) -> ClebschGordanReal:
    """All ``(l1, l2, L)`` couplings with each index at most ``l_max``."""
    cg_object = ClebschGordanReal()
    for l1 in range(l_max + 1):
        for l2 in range(l_max + 1):
            for L in range(abs(l1 - l2), min(l1 + l2, l_max) + 1):
                cg_object.get((l1, l2, L))
    return cg_object


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
