import copy

import numpy as np
import scipy as sp
import scipy.optimize
from scipy import integrate
from scipy.special import jv
from scipy.special import spherical_jn as j_l

from .splines import generate_splines


def Jn(r, n):
    return np.sqrt(np.pi / (2 * r)) * jv(n + 0.5, r)


def Jn_zeros(n, nt):
    zeros_j = np.zeros((n + 1, nt), dtype=np.float64)
    zeros_j[0] = np.arange(1, nt + 1) * np.pi
    points = np.arange(1, nt + n + 1) * np.pi
    roots = np.zeros(nt + n, dtype=np.float64)
    for i in range(1, n + 1):
        for j in range(nt + n - i):
            roots[j] = scipy.optimize.brentq(Jn, points[j], points[j + 1], (i,))
        points = roots
        zeros_j[i][:nt] = roots[:nt]
    return zeros_j


def get_le_spliner(E_max, r_cut, normalize):

    l_big = 50
    n_big = 50
    z_ln = Jn_zeros(l_big, n_big)
    z_nl = z_ln.T
    E_nl = z_nl**2

    l_max = np.where(E_nl[0, :] <= E_max)[0][-1]

    n_max_l = []
    for l in range(l_max + 1):
        n_max_l.append(np.where(E_nl[:, l] <= E_max)[0][-1] + 1)
    n_max_l = np.array(n_max_l)

    def R_nl(n, el, r):
        # Un-normalized LE radial basis functions
        return j_l(el, z_nl[n, el] * r / r_cut)

    def N_nl(n, el):
        # Normalization factor for LE basis functions, excluding the a**(-1.5) factor
        def function_to_integrate_to_get_normalization_factor(x):
            return j_l(el, x) ** 2 * x**2

        integral, _ = sp.integrate.quadrature(
            function_to_integrate_to_get_normalization_factor,
            0.0,
            z_nl[n, el],
            maxiter=200,
        )
        return (1.0 / z_nl[n, el] ** 3 * integral) ** (-0.5)

    def index_to_nl(index, n_max_l):

        n = copy.deepcopy(index)
        for l in range(l_max + 1):
            n -= n_max_l[l]
            if n < 0:
                break

        return n + n_max_l[l], l

    def laplacian_eigenstate_basis(index, r):
        n, el = index_to_nl(index, n_max_l)
        R = np.zeros_like(r)
        for i in range(r.shape[0]):
            R[i] = R_nl(n, el, r[i])
        return_array = N_nl(n, el) * R * r_cut ** (-1.5)
        if normalize:
            # normalize by square root of sphere volume, excluding sqrt(4pi) which is included in the SH
            return_array *= np.sqrt((1 / 3) * r_cut**3)
        return return_array

    normalization_check_integral, _ = sp.integrate.quadrature(
        lambda x: laplacian_eigenstate_basis(2, x) ** 2 * x**2, 0.0, r_cut, maxiter=200
    )
    if normalize:
        normalization_check_integral /= (1 / 3) * r_cut**3
    if abs(normalization_check_integral - 1) > 1e-6:
        raise ValueError("normalization of radial basis FAILED")

    def laplacian_eigenstate_basis_derivative(index, r):
        delta = 1e-6
        all_derivatives_except_at_zero = (
            laplacian_eigenstate_basis(index, r[1:] + delta)
            - laplacian_eigenstate_basis(index, r[1:] - delta)
        ) / (2.0 * delta)
        derivative_at_zero = (
            laplacian_eigenstate_basis(index, np.array([delta / 10.0]))
            - laplacian_eigenstate_basis(index, np.array([0.0]))
        ) / (delta / 10.0)
        return np.concatenate([derivative_at_zero, all_derivatives_except_at_zero])

    n_max_l = [int(n_max) for n_max in n_max_l]

    return n_max_l, generate_splines(
        laplacian_eigenstate_basis,
        laplacian_eigenstate_basis_derivative,
        np.sum(n_max_l),
        r_cut,
    )
