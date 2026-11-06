"""
Functions for performing CG coupling.

These functions are just copied from featomic, to avoid having featomic
as a dependency because it adds extensions to models.

Once these functions are moved to metatomic, we can remove them
from here and import them from metatomic.
"""

import torch
from metatensor.torch import Labels, TensorMap


def cg_couple(
    array: torch.Tensor,
    o3_lambdas: list[int],
    cg_coefficients: TensorMap,
    cg_backend: str,
) -> list[torch.Tensor]:
    """
    Go from an uncoupled product basis that behave like a product of spherical harmonics
    to a coupled basis that behaves like a single spherical harmonic.

    The ``array`` shape should be ``(n_s, 2 * l1 + 1, 2 * l2 + 1, n_q)``.
    ``n_s`` is the number of samples, and ``n_q`` is the number of properties.

    This function will output a list of arrays, whose shape will be ``[n_s, (2 *
    o3_lambda+1), n_q]``, with the requested ``o3_lambdas``.

    These arrays will contain the result of projecting from a product of spherical
    harmonic with degree ``l1`` and ``l2`` to a single spherical harmonic of degree
    ``o3_lambdas``; using Clebsch-Gordan coefficients. The values for ``l1`` and ``l2``
    are inferred from the size of the middle axis of the input array.

    The Clebsch-Gordan coefficients are cached in ``cg_coefficients``. These must be
    computed by :py:func:`calculate_cg_coefficients`.

    The operation is dispatched such that numpy arrays or torch tensors are
    automatically handled.

    :param array: input array with shape ``[n_s, 2 * l1 + 1, 2 * l2 + 1, n_q]``
    :param o3_lambdas: list of degrees of spherical harmonics to compute
    :param cg_coefficients: CG coefficients as returned by
        :py:func:`calculate_cg_coefficients` with the same ``cg_backed`` given to this
        function
    :param cg_backend: specifies the backend to use for the calculation. It can be
        ``"python-dense"``, or ``"python-sparse"``.

    :return: A list of array, one for each ``o3_lambda``
    """

    assert len(array.shape) == 4

    l1 = (array.shape[1] - 1) // 2
    l2 = (array.shape[2] - 1) // 2

    if cg_backend == "python-sparse":
        arrays = {}
        for m1 in range(2 * l1 + 1):
            for m2 in range(2 * l2 + 1):
                arrays[str((m1, m2))] = array[:, m1, m2, :]

        return [
            _cg_couple_sparse(arrays, l1, l2, o3_lambda, cg_coefficients)
            for o3_lambda in o3_lambdas
        ]
    elif cg_backend == "python-dense":
        n_samples = array.shape[0]
        n_properties = array.shape[3]

        # Move properties next to samples
        array = torch.permute(array, [0, 3, 1, 2])

        array = array.reshape(n_samples * n_properties, 2 * l1 + 1, 2 * l2 + 1)

        results = []
        for o3_lambda in o3_lambdas:
            result = _cg_couple_dense(array, o3_lambda, cg_coefficients)

            # Separate samples and properties
            result = result.reshape(n_samples, n_properties, -1)

            # Back to TensorBlock-like axes
            result = result.swapaxes(1, 2)

            results.append(result)

        return results

    else:
        raise ValueError(
            f"invalid `cg_backend`, got '{cg_backend}', "
            "only 'python-dense', or 'python-sparse' are supported"
        )


def _cg_couple_sparse(
    arrays: dict[str, torch.Tensor],
    l1: int,
    l2: int,
    o3_lambda: int,
    cg_coefficients: TensorMap,
) -> torch.Tensor:
    """
    Couple two spherical harmonics (of degree ``l1`` and ``l2``) to a single one (of
    degree ``o3_lambda``) using CG coefficients. This is a "dense" implementation, using
    all CG coefficients at the same time.

    :param arrays: dictionary of arrays for the different ``(m1, m2)`` corresponding to
        ``l1`` and ``l2``. Each array in the dictionary is expected to have a shape
        ``[samples, properties]`` and correspond to a single ``(m1, m2)`` pair.
    :param l1: degree of the first spherical harmonic
    :param l2: degree of the first spherical harmonic
    :param o3_lambda: value of lambda for the output spherical harmonic
    :param cg_coefficients: CG coefficients as returned by
        :py:func:`calculate_cg_coefficients` with ``cg_backed="python-sparse"``

    :return: array of shape ``[samples, 2 * o3_lambda + 1, properties]``
    """

    array = list(arrays.values())[0]
    assert len(array.shape) == 2

    output = torch.zeros(
        (2 * o3_lambda + 1, array.shape[0], array.shape[1]),
        dtype=array.dtype,
        device=array.device,
    )

    # Fill in each mu component of the output array in turn
    cg_l1l2lam = cg_coefficients.block({"l1": l1, "l2": l2, "lambda": o3_lambda})
    cg_l1l2lam_samples = cg_l1l2lam.samples
    for i in range(len(cg_l1l2lam_samples)):
        m1m2mu = cg_l1l2lam_samples.entry(i)
        m1 = int(m1m2mu[0])
        m2 = int(m1m2mu[1])
        mu = int(m1m2mu[2])
        # Broadcast arrays, multiply together and with CG coeff
        output[mu, :, :] += (
            arrays[str((m1, m2))]
            * (-1) ** (l1 + l2 + o3_lambda)
            * cg_l1l2lam.values[i, 0]
        )

    return output.swapaxes(0, 1)


def _cg_couple_dense(
    array: torch.Tensor,
    o3_lambda: int,
    cg_coefficients: TensorMap,
) -> torch.Tensor:
    """
    Couple two spherical harmonics (of degree ``l1`` and ``l2``) to a single one (of
    degree ``o3_lambda``) using CG coefficients. This is a "dense" implementation, using
    all CG coefficients at the same time.

    :param array: input array, we expect a shape of ``[N, 2*l1 + 1, 2*l2 + 1]``
    :param o3_lambda: value of lambda for the output spherical harmonic
    :param cg_coefficients: CG coefficients as returned by
        :py:func:`calculate_cg_coefficients` with ``cg_backed="python-dense"``

    :return: array of shape ``[N, 2 * o3_lambda + 1]``
    """
    assert len(array.shape) == 3

    l1 = (array.shape[1] - 1) // 2
    l2 = (array.shape[2] - 1) // 2

    key = Labels(
        ["l1", "l2", "lambda"],
        torch.tensor([[l1, l2, o3_lambda]], device=cg_coefficients.keys.values.device),
    )

    cg_l1l2lam = (-1) ** (l1 + l2 + o3_lambda) * cg_coefficients.block(key).values

    return torch.tensordot(array, cg_l1l2lam[0, ..., 0], ([2, 1], [1, 0]))


def cg_uncouple(
    arrays: list[torch.Tensor],
    l1: int,
    l2: int,
    o3_lambdas: list[int],
    cg_coefficients: TensorMap,
    cg_backend: str,
) -> torch.Tensor:
    """
    Exact inverse of :py:func:`cg_couple`.

    Go from a coupled basis that behaves like a single spherical harmonic back to the
    uncoupled product basis that behaves like a product of two spherical harmonics of
    degrees ``l1`` and ``l2``.

    The coupling performed by :py:func:`cg_couple` is the orthogonal transformation

    ``T^{L}_{M} = \\sum_{m1 m2} (-1)^{l1 + l2 + L} <l1 m1 l2 m2|L M> T^{l1}_{m1}
    T^{l2}_{m2}``

    Because the (real) Clebsch-Gordan coefficients are orthogonal and complete over the
    full set of ``L`` in ``range(|l1 - l2|, l1 + l2 + 1)``, this can be inverted exactly
    by applying the transpose:

    ``T^{l1}_{m1} T^{l2}_{m2} = \\sum_{L M} (-1)^{l1 + l2 + L} <l1 m1 l2 m2|L M>
    T^{L}_{M}``

    The same sign convention and the same cached coefficients as :py:func:`cg_couple`
    are used, so ``cg_uncouple(cg_couple(...))`` recovers the original array (summing
    over whichever ``o3_lambdas`` are provided).

    :param arrays: list of arrays, one for each entry in ``o3_lambdas``, each with shape
        ``[n_s, 2 * o3_lambda + 1, n_q]``. ``n_s`` and ``n_q`` must be the same for all
        the arrays.
    :param l1: degree of the first spherical harmonic to recover
    :param l2: degree of the second spherical harmonic to recover
    :param o3_lambdas: degrees of the coupled spherical harmonics given in ``arrays``,
        in the same order as ``arrays``. For an exact inversion this should contain all
        of ``range(|l1 - l2|, l1 + l2 + 1)``; any missing degree is treated as a block
        of zeros.
    :param cg_coefficients: CG coefficients as returned by
        :py:func:`calculate_cg_coefficients` with the same ``cg_backend`` given to this
        function
    :param cg_backend: ``"python-dense"`` or ``"python-sparse"``

    :return: array with shape ``[n_s, 2 * l1 + 1, 2 * l2 + 1, n_q]``
    """
    assert len(arrays) == len(o3_lambdas)
    assert len(arrays) > 0

    n_s = arrays[0].shape[0]
    n_q = arrays[0].shape[2]

    output = torch.zeros(
        (n_s, 2 * l1 + 1, 2 * l2 + 1, n_q),
        dtype=arrays[0].dtype,
        device=arrays[0].device,
    )

    if cg_backend == "python-sparse":
        for o3_lambda, array in zip(o3_lambdas, arrays, strict=True):
            cg_l1l2lam = cg_coefficients.block(
                {"l1": l1, "l2": l2, "lambda": o3_lambda}
            )
            sign = (-1) ** (l1 + l2 + o3_lambda)
            samples = cg_l1l2lam.samples
            for i in range(len(samples)):
                m1m2mu = samples.entry(i)
                m1 = int(m1m2mu[0])
                m2 = int(m1m2mu[1])
                mu = int(m1m2mu[2])
                output[:, m1, m2, :] += array[:, mu, :] * sign * cg_l1l2lam.values[i, 0]

        return output

    elif cg_backend == "python-dense":
        for o3_lambda, array in zip(o3_lambdas, arrays, strict=True):
            # cg block has shape (1, 2*l1 + 1, 2*l2 + 1, 2*o3_lambda + 1, 1)
            cg_l1l2lam = (-1) ** (l1 + l2 + o3_lambda) * cg_coefficients.block(
                {"l1": l1, "l2": l2, "lambda": o3_lambda}
            ).values
            # contract the `mu` axis of `array` with the `mu` axis of the coefficients
            # array (n_s, 2*o3_lambda + 1, n_q) . (2*l1+1, 2*l2+1, 2*o3_lambda+1)
            #   -> (n_s, n_q, 2*l1 + 1, 2*l2 + 1)
            contribution = torch.tensordot(array, cg_l1l2lam[0, ..., 0], ([1], [2]))
            # -> (n_s, 2*l1 + 1, 2*l2 + 1, n_q)
            output = output + torch.permute(contribution, [0, 2, 3, 1])

        return output

    else:
        raise ValueError(
            f"invalid `cg_backend`, got '{cg_backend}', "
            "only 'python-dense', or 'python-sparse' are supported"
        )
