import copy
import os

import numpy as np

from .splines import generate_splines
from physical_basis import PhysicalBasis


def get_physical_basis_spliner(E_max, r_cut, normalize):

    l_max = 50
    n_max = 50
    a = 10.0  # by construction of the files

    physical_basis = PhysicalBasis()
    E_ln = physical_basis.E_ln
    E_nl = E_ln.T
    l_max_new = np.where(E_nl[0, :] <= E_max)[0][-1]
    if l_max_new > l_max:
        raise ValueError("l_max too large, try decreasing E_max")
    else:
        l_max = l_max_new

    n_max_l = []
    for l in range(l_max + 1):
        n_max_l.append(np.where(E_nl[:, l] <= E_max)[0][-1] + 1)
    if n_max_l[0] > n_max:
        raise ValueError("n_max too large, try decreasing E_max")

    def function_for_splining(n, l, x):
        ret = physical_basis.compute(n, l, x)
        if normalize:
            # normalize by square root of sphere volume, excluding sqrt(4pi) which is included in the SH
            ret *= np.sqrt((1 / 3) * r_cut**3)
        return ret

    def function_for_splining_derivative(n, l, x):
        ret = physical_basis.compute_derivative(n, l, x)
        if normalize:
            # normalize by square root of sphere volume, excluding sqrt(4pi) which is included in the SH
            ret *= np.sqrt((1 / 3) * r_cut**3)
        return ret

    def index_to_nl(index, n_max_l):
        # FIXME: should probably use cumsum
        n = copy.deepcopy(index)
        for l in range(l_max + 1):
            n -= n_max_l[l]
            if n < 0:
                break
        return n + n_max_l[l], l

    def function_for_splining_index(index, r):
        n, l = index_to_nl(index, n_max_l)
        return function_for_splining(n, l, r)

    def function_for_splining_index_derivative(index, r):
        n, l = index_to_nl(index, n_max_l)
        return function_for_splining_derivative(n, l, r)

    spliner = generate_splines(
        function_for_splining_index,
        function_for_splining_index_derivative,
        np.sum(n_max_l),
        a,
    )
    print("Number of spline points:", len(spliner.spline_positions))

    n_max_l = [int(n_max) for n_max in n_max_l]
    return n_max_l, spliner
