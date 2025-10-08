import numpy as np
import spherical
import torch
from metatomic.torch import System
from scipy.spatial.transform import Rotation


def get_random_rotation() -> Rotation:
    return Rotation.random()


def rotate_system(system: System, rotation: Rotation) -> System:
    """
    Rotate a System object using a given rotation.

    :param system: The System object to be rotated.
    :param rotation: A scipy.spatial.transform.Rotation object representing the rotation
    :return: A new System object with rotated positions and cell.
    """
    rotated_positions = (
        system.positions.detach().cpu().numpy() @ np.array(rotation.as_matrix()).T
    )
    rotated_cell = system.cell.detach().cpu().numpy() @ np.array(rotation.as_matrix()).T
    return System(
        positions=torch.tensor(
            rotated_positions, device=system.device, dtype=system.positions.dtype
        ),
        cell=torch.tensor(rotated_cell, device=system.device, dtype=system.cell.dtype),
        types=system.types,
        pbc=system.pbc,
    )


def rotate_spherical_tensor(
    spherical_tensor: np.ndarray, rotation: Rotation
) -> np.ndarray:
    """
    Rotate a spherical tensor using a given rotation.

    :param spherical_tensor: A numpy array of shape (n_samples, 2*l+1, n_properties)
        representing the spherical tensor to be rotated.
    :param rotation: A scipy.spatial.transform.Rotation object representing the rotation
    :return: A numpy array of the same shape as spherical_tensor, representing the
        rotated spherical tensor.
    """
    # the spherical tensor is a tensor of shape (n_samples, 2*l+1, n_properties)
    L = (spherical_tensor.shape[1] - 1) // 2
    rotated_spherical_tensor = (
        spherical_tensor.swapaxes(-1, -2) @ calculate_wigner_D(rotation, L).T
    ).swapaxes(-1, -2)
    return rotated_spherical_tensor


def calculate_wigner_D(rotation: Rotation, L: int) -> np.ndarray:
    """
    Calculate the Wigner D matrix for a given rotation and angular momentum L.

    :param rotation: A scipy.spatial.transform.Rotation object representing the rotation
    :param L: The angular momentum quantum number (non-negative integer)
    :return: A numpy array of shape (2*L+1, 2*L+1) representing the Wigner D matrix in
        the real spherical harmonics basis.
    """
    # We initialize the Wigner calculator from the quaternionic library...
    wigner = spherical.Wigner(L)
    # ...and we also initialize the transformation matrix from complex to real
    complex_to_real_transform = complex_to_real_spherical_harmonics_transform(L)

    # Obtaining the quaternion associated with the rotation by means of the scipy
    # libtary
    quaternion_scipy = rotation.as_quat()
    # Change convention to be consistent with the one of the quaternionic library
    quaternion_quaternionic = scipy_quaternion_to_quaternionic(quaternion_scipy)
    # applying the quaternion to the Wigner D to obtain the actual values
    wigners_R = wigner.D(quaternion_quaternionic)

    # We now extract the values of the Wigner D matrices and transform them to real
    wigner_D_matrix_complex = np.zeros((2 * L + 1, 2 * L + 1), dtype=np.complex128)
    for m in range(-L, L + 1):
        for mp in range(-L, L + 1):
            # This is the procedure that gives the correct indexing of the Wigner
            # D matrices, note that the quaternionic library uses a convention such
            # that the resulting matrix is the complex conjugate of the one that we
            # expect from the rotation, and so we take the complex conjugation
            wigner_D_matrix_complex[m + L, mp + L] = (
                wigners_R[wigner.Dindex(L, m, mp)]
            ).conj()

    # We finally transform everything in the real representation...
    wigner_D_matrix = (
        complex_to_real_transform.conj()
        @ wigner_D_matrix_complex
        @ complex_to_real_transform.T
    )
    # ...and check that we do not have imaginary contributions
    assert np.allclose(wigner_D_matrix.imag, 0.0)  # check that the matrix is real

    return wigner_D_matrix.real


def complex_to_real_spherical_harmonics_transform(ell: int) -> np.ndarray:
    """
    Generates the transformation matrix from complex spherical harmonics to real
    spherical harmonics for a given l.

    :param ell: The angular momentum quantum number (non-negative integer).
    :return: Returns a transformation matrix of shape ((2l+1), (2l+1)).
    """

    if ell < 0 or not isinstance(ell, int):
        raise ValueError("l must be a non-negative integer.")

    # The size of the transformation matrix is (2l+1) x (2l+1)
    size = 2 * ell + 1
    U = np.zeros((size, size), dtype=complex)

    for m in range(-ell, ell + 1):
        m_index = m + ell  # Index in the matrix
        if m > 0:
            # Real part of Y_{l}^{m}
            U[m_index, ell + m] = 1 / np.sqrt(2) * (-1) ** m
            U[m_index, ell - m] = 1 / np.sqrt(2)
        elif m < 0:
            # Imaginary part of Y_{l}^{|m|}
            U[m_index, ell + abs(m)] = -1j / np.sqrt(2) * (-1) ** m
            U[m_index, ell - abs(m)] = 1j / np.sqrt(2)
        else:  # m == 0
            # Y_{l}^{0} remains unchanged
            U[m_index, ell] = 1

    return U


def scipy_quaternion_to_quaternionic(q_scipy: np.ndarray) -> np.ndarray:
    """
    This function convert a quaternion obtained from the scipy library to the format
    used by the quaternionic library.

    :param q_scipy: A numpy array of shape (4,) representing the quaternion in
        'xyzw' format (scipy).
    :return: A numpy array of shape (4,) representing the quaternion in 'wxyz'
        format (quaternionic).
    """
    qx, qy, qz, qw = q_scipy
    q_quaternion = np.array([qw, qx, qy, qz])
    return q_quaternion


def rotation_matrix_conversion_order(rotation_matrix: np.ndarray) -> np.ndarray:
    """
    This function is used to convert a rotation matrix from the format (y, z, x)
    to (x, y, z).

    :param rotation_matrix: A numpy array of shape (3, 3) representing the rotation
        matrix in 'yzx' format (spherical harmonics convention).
    :return: A numpy array of shape (3, 3) representing the rotation matrix in
        'xyz' format (Cartesian convention, used by scipy).
    """
    converted_matrix = rotation_matrix[[2, 0, 1], :]
    converted_matrix = converted_matrix[:, [2, 0, 1]]
    return converted_matrix
