import ase.io
import numpy as np
import torch
from metatensor.torch import Labels, TensorBlock, TensorMap


def l2_components_from_matrix(A):
    A = A.reshape(3, 3)

    l2_A = np.empty((5,))
    l2_A[0] = (A[0, 1] + A[1, 0]) / 2.0
    l2_A[1] = (A[1, 2] + A[2, 1]) / 2.0
    l2_A[2] = (2.0 * A[2, 2] - A[0, 0] - A[1, 1]) / ((2.0) * np.sqrt(3.0))
    l2_A[3] = (A[0, 2] + A[2, 0]) / 2.0
    l2_A[4] = (A[0, 0] - A[1, 1]) / 2.0

    return l2_A


def dump_spherical_targets(path_in, path_out):
    # Takes polarizabilities from a dataset in Cartesian format, converts them to
    # spherical coordinates, and saves them in metatensor format (suitable for
    # training a model with spherical targets).

    structures = ase.io.read(path_in, ":")

    polarizabilities_l2 = np.array(
        [
            l2_components_from_matrix(structure.info["polarizability"])
            for structure in structures
        ]
    )

    samples = Labels(
        names=["system"],
        values=torch.arange(len(structures)).reshape(-1, 1),
    )

    properties = Labels.single()

    components_l2 = Labels(
        names=["o3_mu"],
        values=torch.tensor([[-2], [-1], [0], [1], [2]]),
    )

    keys = Labels(
        names=["o3_lambda", "o3_sigma"],
        values=torch.tensor([[2, 1]]),
    )

    tensor_map = TensorMap(
        keys=keys,
        blocks=[
            TensorBlock(
                values=torch.tensor(polarizabilities_l2, dtype=torch.float64).reshape(
                    100, 5, 1
                ),
                samples=samples,
                components=[components_l2],
                properties=properties,
            ),
        ],
    )

    tensor_map.save(path_out)
