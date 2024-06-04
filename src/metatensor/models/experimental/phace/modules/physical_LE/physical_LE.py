import copy
import os

import numpy as np

from ..splines import generate_splines


# All these periodic functions are zeroed for the (unlikely) case where r > 10*r_0
# which is outside the domain where the eigenvalue equation was solved


def s(n, x):
    return np.sin(np.pi * (n + 1.0) * x / 10.0)


def ds(n, x):
    return np.pi * (n + 1.0) * np.cos(np.pi * (n + 1.0) * x / 10.0) / 10.0


def c(n, x):
    return np.cos(np.pi * (n + 0.5) * x / 10.0)


def dc(n, x):
    return -np.pi * (n + 0.5) * np.sin(np.pi * (n + 0.5) * x / 10.0) / 10.0


def get_physical_le_spliner(E_max, r_cut, normalize):

    l_max = 50
    n_max = 50
    n_max_big = 200

    a = 10.0  # by construction of the files

    dir_path = os.path.dirname(os.path.realpath(__file__))

    E_ln = np.load(os.path.join(dir_path, "eigenvalues.npy"))
    eigenvectors = np.load(os.path.join(dir_path, "eigenvectors.npy"))

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
        ret = np.zeros_like(x)
        for m in range(n_max_big):
            ret += (
                eigenvectors[l][m, n] * c(m, x)
                if l % 2 == 0
                else eigenvectors[l][m, n] * s(m, x)
            )
        if normalize:
            # normalize by square root of sphere volume, excluding sqrt(4pi) which is included in the SH
            ret *= np.sqrt((1 / 3) * r_cut**3)
        return ret

    def function_for_splining_derivative(n, l, x):
        ret = np.zeros_like(x)
        for m in range(n_max_big):
            ret += (
                eigenvectors[l][m, n] * dc(m, x)
                if l % 2 == 0
                else eigenvectors[l][m, n] * ds(m, x)
            )
        if normalize:
            # normalize by square root of sphere volume, excluding sqrt(4pi) which is included in the SH
            ret *= np.sqrt((1 / 3) * r_cut**3)
        return ret

    """
    import matplotlib.pyplot as plt
    r = np.linspace(0.01, a-0.001, 1000)
    l = 0
    for n in range(n_max_l[l]):
        plt.plot(r, function_for_splining(n, l, r), label=str(n))
    plt.plot([0.0, a], [0.0, 0.0], "black")
    plt.xlim(0.0, a)
    plt.legend()
    plt.savefig("radial-real.pdf")
    """

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
    # print("Number of spline points:", len(spliner.spline_positions))

    n_max_l = [int(n_max) for n_max in n_max_l]
    return n_max_l, spliner
