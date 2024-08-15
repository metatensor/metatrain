import numpy as np
import torch
import wigners


def cg_combine_l1l2L(tensor12, cg_tensor):
    # print(tensor12.shape)
    # print(cg_tensor.shape)
    out_tensor = tensor12 @ cg_tensor.reshape(
        cg_tensor.shape[0] * cg_tensor.shape[1], cg_tensor.shape[2]
    )
    return out_tensor.swapaxes(1, 2) #/ (cg_tensor.shape[0]*cg_tensor.shape[1]*cg_tensor.shape[2])


def get_cg_coefficients(l_max):
    cg_object = ClebschGordanReal()
    for l1 in range(l_max + 1):
        for l2 in range(l_max + 1):
            for L in range(abs(l1 - l2), min(l1 + l2, l_max) + 1):
                cg_object._add(l1, l2, L)
    return cg_object


class ClebschGordanReal:

    def __init__(self):
        self._cgs = {}

    def _add(self, l1, l2, L):
        # print(f"Adding new CGs with l1={l1}, l2={l2}, L={L}")

        if self._cgs is None:
            raise ValueError("Trying to add CGs when not initialized... exiting")

        if (l1, l2, L) in self._cgs:
            raise ValueError("Trying to add CGs that are already present... exiting")

        maxx = max(l1, max(l2, L))

        # real-to-complex and complex-to-real transformations as matrices
        r2c = {}
        c2r = {}
        for l in range(0, maxx + 1):
            r2c[l] = _real2complex(l)
            c2r[l] = np.conjugate(r2c[l]).T

        complex_cg = _complex_clebsch_gordan_matrix(l1, l2, L)

        real_cg = (r2c[l1].T @ complex_cg.reshape(2 * l1 + 1, -1)).reshape(
            complex_cg.shape
        )

        real_cg = real_cg.swapaxes(0, 1)
        real_cg = (r2c[l2].T @ real_cg.reshape(2 * l2 + 1, -1)).reshape(real_cg.shape)
        real_cg = real_cg.swapaxes(0, 1)

        real_cg = real_cg @ c2r[L].T

        if (l1 + l2 + L) % 2 == 0:
            rcg = np.real(real_cg)
        else:
            rcg = np.imag(real_cg)

        # Zero any possible (and very rare) near-zero elements
        where_almost_zero = np.where(
            np.logical_and(np.abs(rcg) > 0, np.abs(rcg) < 1e-14)
        )
        if len(where_almost_zero[0] != 0):
            print("INFO: Found almost-zero CG!")
        for i0, i1, i2 in zip(
            where_almost_zero[0], where_almost_zero[1], where_almost_zero[2]
        ):
            rcg[i0, i1, i2] = 0.0

        self._cgs[(l1, l2, L)] = torch.tensor(rcg)

    def get(self, key):
        if key in self._cgs:
            return self._cgs[key]
        else:
            self._add(key[0], key[1], key[2])
            return self._cgs[key]


def _real2complex(L):
    """
    Computes a matrix that can be used to convert from real to complex-valued
    spherical harmonics(coefficients) of order L.

    It's meant to be applied to the left, ``real2complex @ [-L..L]``.
    """
    result = np.zeros((2 * L + 1, 2 * L + 1), dtype=np.complex128)

    I_SQRT_2 = 1.0 / np.sqrt(2)

    for m in range(-L, L + 1):
        if m < 0:
            result[L - m, L + m] = I_SQRT_2 * 1j * (-1) ** m
            result[L + m, L + m] = -I_SQRT_2 * 1j

        if m == 0:
            result[L, L] = 1.0

        if m > 0:
            result[L + m, L + m] = I_SQRT_2 * (-1) ** m
            result[L - m, L + m] = I_SQRT_2

    return result


def _complex_clebsch_gordan_matrix(l1, l2, L):
    if np.abs(l1 - l2) > L or np.abs(l1 + l2) < L:
        return np.zeros((2 * l1 + 1, 2 * l2 + 1, 2 * L + 1), dtype=np.double)
    else:
        return wigners.clebsch_gordan_array(l1, l2, L)
