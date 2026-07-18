import numpy as np
import torch
from metatomic.torch import System
from metatomic.torch.o3 import O3Transformation
from scipy.spatial.transform import Rotation


def get_random_rotation() -> Rotation:
    """Generate a random 3D rotation.

    :return: A scipy.spatial.transform.Rotation object representing the random rotation.
    """
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
    transformation = O3Transformation(
        torch.tensor(rotation.as_matrix(), dtype=torch.float64), L
    )
    return transformation.wigner_D_matrix(L).numpy()


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
