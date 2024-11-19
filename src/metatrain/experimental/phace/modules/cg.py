import numpy as np
import torch
import wigners


try:
    import mops.torch
except ImportError:
    pass
from typing import Dict


def cg_combine_l1l2(
    tensor_A, tensor_B, tensor_C, indices_A, indices_B, indices_output, split_sizes
):

    assert tensor_A.shape[0] == tensor_B.shape[0]
    shape_0 = tensor_A.shape[0]

    assert tensor_A.shape[2] == tensor_B.shape[2]
    shape_2 = tensor_A.shape[2]

    tensor_A = tensor_A.swapaxes(1, 2).reshape(shape_0 * shape_2, tensor_A.shape[1])
    tensor_B = tensor_B.swapaxes(1, 2).reshape(shape_0 * shape_2, tensor_B.shape[1])

    output_size = int(split_sizes.sum().item())
    mops_result = mops.torch.sparse_accumulation_of_products(
        tensor_A, tensor_B, tensor_C, indices_A, indices_B, indices_output, output_size
    )

    split_sizes_list: List[int] = split_sizes.tolist()
    L_splits = torch.split(mops_result, split_sizes_list, dim=1)
    result = []
    for L_split, split_size in zip(L_splits, split_sizes_list):
        result.append(L_split.reshape(shape_0, shape_2, split_size).swapaxes(1, 2))
    return result


def cgs_to_sparse(cgs, l_max):
    sparse_cgs = {}
    for l1 in range(l_max + 1):
        for l2 in range(l_max + 1):
            base_M_index = 0
            split_sizes = []
            i1 = []
            i2 = []
            I = []
            C = []
            for L in range(abs(l1 - l2), min(l1 + l2, l_max) + 1):
                dense_cg_matrix = cgs[f"{l1}_{l2}_{L}"]
                where_nonzero = torch.nonzero(dense_cg_matrix, as_tuple=True)
                m1, m2, M = where_nonzero
                nonzero_coeffs = dense_cg_matrix[where_nonzero]
                i1.append(m1)
                i2.append(m2)
                I.append(M + base_M_index)
                C.append(nonzero_coeffs)
                split_sizes.append(2 * L + 1)
                base_M_index += 2 * L + 1
            sparse_cgs[f"{l1}_{l2}"] = {}
            sparse_cgs[f"{l1}_{l2}"]["i1"] = torch.concatenate(i1).to(torch.int32)
            sparse_cgs[f"{l1}_{l2}"]["i2"] = torch.concatenate(i2).to(torch.int32)
            sparse_cgs[f"{l1}_{l2}"]["I"] = torch.concatenate(I).to(torch.int32)
            sparse_cgs[f"{l1}_{l2}"]["C"] = torch.concatenate(C)
            sparse_cgs[f"{l1}_{l2}"]["split_sizes"] = torch.tensor(split_sizes)
    return sparse_cgs


def cgs_to_device_dtype(
    cgs: Dict[str, Dict[str, torch.Tensor]], device: torch.device, dtype: torch.dtype
):
    cgs_device: Dict[str, Dict[str, torch.Tensor]] = {}
    for key, value in cgs.items():
        cgs_device[key] = {}
        for k, v in value.items():
            if k == "split_sizes":
                cgs_device[key][k] = v
            elif k == "C":
                cgs_device[key][k] = v.to(device, dtype)
            else:
                cgs_device[key][k] = v.to(device)
    return cgs_device


def cg_combine_l1l2L(tensor12, cg_tensor):
    out_tensor = tensor12 @ cg_tensor.reshape(
        cg_tensor.shape[0] * cg_tensor.shape[1], cg_tensor.shape[2]
    )
    return out_tensor.swapaxes(
        1, 2
    )  # / (cg_tensor.shape[0]*cg_tensor.shape[1]*cg_tensor.shape[2])


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
