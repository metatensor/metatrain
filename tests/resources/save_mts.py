import ase.io
import numpy as np
import torch
from metatensor.torch import Labels, TensorBlock, TensorMap


def _complex_to_real_spherical_harmonics_transform(ell: int):
    """
    Generate the transformation matrix from complex spherical harmonics
    to real spherical harmonics for a given l.
    Returns a transformation matrix of shape ((2l+1), (2l+1)).
    """
    if ell < 0 or not isinstance(ell, int):
        raise ValueError("l must be a non-negative integer.")

    # The size of the transformation matrix is (2l+1) x (2l+1)
    size = 2 * ell + 1
    T = np.zeros((size, size), dtype=complex)

    for m in range(-ell, ell + 1):
        m_index = m + ell  # Index in the matrix
        if m > 0:
            # Real part of Y_{l}^{m}
            T[m_index, ell + m] = 1 / np.sqrt(2)
            T[m_index, ell - m] = 1 / np.sqrt(2) * (-1) ** m
        elif m < 0:
            # Imaginary part of Y_{l}^{|m|}
            T[m_index, ell + abs(m)] = 1j / np.sqrt(2)
            T[m_index, ell - abs(m)] = -1j / np.sqrt(2) * (-1)**abs(m)
        else:  # m == 0
            # Y_{l}^{0} remains unchanged
            T[m_index, ell] = 1

    # Return the transformation matrix to convert complex to real spherical harmonics
    return T


structures = ase.io.read("qm7x_reduced_100.xyz", ":")

polarizabilities = np.array(
    [structure.info["polarizability"] for structure in structures]
)
# polarizabilities = torch.tensor(polarizabilities, dtype=torch.float64)

# conversion to spherical from here
# https://journals.aps.org/prl/supplemental/10.1103/PhysRevLett.120.036002/SI.pdf

polarizabilities_xx_xy_xz_yy_yz_zz = polarizabilities[:, [0, 1, 2, 4, 5, 8]]

l0_transformation_matrix = np.array(
    [-1.0 / np.sqrt(3.0), 0.0, 0.0, -1.0 / np.sqrt(3.0), 0.0, -1.0 / np.sqrt(3.0)]
)

l2_transformation_matrix = np.array(
    [
        [1.0 / 2.0, 0.0, -1.0 / np.sqrt(6.0), 0.0, 1.0 / 2.0],
        [-1.0j / 2.0, 0.0, 0.0, 0.0, 1.0j / 2.0],
        [0.0, 1.0 / 2.0, 0.0, -1.0 / 2.0, 0.0],
        [-1.0 / 2.0, 0.0, -1.0 / np.sqrt(6.0), 0.0, -1.0 / 2.0],
        [0.0, -1.0j / 2.0, 0.0, -1.0j / 2.0, 0.0],
        [0.0, 0.0, 2.0 / np.sqrt(6.0), 0.0, 0.0],
    ]
)

polarizabilities_l0 = polarizabilities_xx_xy_xz_yy_yz_zz @ l0_transformation_matrix
polarizabilities_l2 = polarizabilities_xx_xy_xz_yy_yz_zz @ l2_transformation_matrix

complex_to_real = _complex_to_real_spherical_harmonics_transform(2)
polarizabilities_l2 = polarizabilities_l2 @ complex_to_real.T

assert polarizabilities_l2.imag.max() < 1e-10
polarizabilities_l2 = polarizabilities_l2.real

samples = Labels(
    names=["system"],
    values=torch.arange(len(structures)).reshape(-1, 1),
)

properties = Labels.single()

components_l0 = Labels(names=["o3_mu"], values=torch.tensor([[0]]))

components_l2 = Labels(
    names=["o3_mu"],
    values=torch.tensor([[-2], [-1], [0], [1], [2]]),
)

# keys = Labels(
#     names=["o3_lambda", "o3_sigma"],
#     values=torch.tensor([[0, 1], [2, 1]]),
# )

keys = Labels(
    names=["o3_lambda", "o3_sigma"],
    values=torch.tensor([[2, 1]]),
)

tensor_map = TensorMap(
    keys=keys,
    blocks=[
        # TensorBlock(
        #     values=torch.tensor(polarizabilities_l0, dtype=torch.float64).reshape(
        #         100, 1, 1
        #     ),
        #     samples=samples,
        #     components=[components_l0],
        #     properties=properties,
        # ),
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

tensor_map.save("qm7x_reduced_100.npz")
