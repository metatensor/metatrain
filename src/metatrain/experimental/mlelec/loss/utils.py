from typing import List, Union

import ase
import numpy as np
import torch


def pyscf_orbital_order(
    matrix: Union[torch.Tensor, np.ndarray, List[Union[torch.Tensor, np.ndarray]]],
    frames: Union[List[ase.Atoms], ase.Atoms],
    orbital: dict,
    to_spherical: bool = True,
):
    """
    Reorder the orbital indices for PySCF fock/overlap matrices.

    This function supports a single matrix, a stacked matrix (3D tensor/array),
    or a list of matrices (which may be ragged). It reorders the l=1 orbitals.

    When `to_spherical` is True, the l=1 orbitals are re-ordered from Cartesian
    ordering ([x, y, z]) to spherical harmonics ordering ([-1, 0, 1]). When False,
    the process is reversed.

    Args:
        matrix (Union[torch.Tensor, np.ndarray, List[Union[torch.Tensor, np.ndarray]]]):
            The matrix or list of matrices to reorder. For multiple frames, matrices
            can be provided as a stacked tensor/array with shape (n_frames, N, N) or as
            a list.
        frames (Union[List[ase.Atoms], ase.Atoms]): A single frame or a list of ASE
            Atoms.
        orbital (dict): Dictionary of orbitals for each atom type in the frames.
            Each entry should be an iterable of orbital descriptors (n, l, _).
        to_spherical (bool): Flag indicating whether to convert to spherical order
            (True) or back to Cartesian (pyscf) order (False).

    Returns:
        Reordered matrix or list of matrices, preserving the type and structure of the
        input.
        If a single frame is provided, a single matrix is returned.
    """

    def compute_indices(
        frame: ase.Atoms, orbital: dict, to_spherical: bool
    ) -> List[int]:
        """
        Compute the reordering indices for a given frame.

        For each atom in the frame, the orbital ordering is updated.
        For l=1 orbitals, if converting to spherical order, the new index order is:
            [iorb + 1, iorb + 2, iorb]
        and if converting back:
            [iorb + 2, iorb, iorb + 1]
        For other orbitals, indices remain sequential.
        """
        indices = []
        iorb = 0
        for atom_type in list(frame.numbers):
            cur = None  # reset per atom as in the original implementation
            for a in orbital[atom_type]:
                n, l, _ = a
                if cur != (n, l):
                    if l == 1:
                        if to_spherical:
                            indices += [iorb + 1, iorb + 2, iorb]
                        else:
                            indices += [iorb + 2, iorb, iorb + 1]
                    else:
                        indices += list(range(iorb, iorb + 2 * l + 1))
                    iorb += 2 * l + 1
                    cur = (n, l)
        return indices

    # Normalize frames to a list for uniform handling.
    single_frame = False

    if not isinstance(frames, list) and not isinstance(frames, tuple):
        frames = [frames]
        single_frame = True

    # Normalize matrices to a list of matrices.
    matrix_list = []
    if isinstance(matrix, list):
        matrix_list = matrix
    elif isinstance(matrix, (np.ndarray, torch.Tensor)):
        if matrix.ndim == 2:
            matrix_list = [matrix]
        elif matrix.ndim == 3:
            # Convert to list along the first axis.
            matrix_list = [matrix[i] for i in range(matrix.shape[0])]
        else:
            raise ValueError(
                "Unsupported matrix dimension. Expected 2 or 3 dimensions."
            )
    else:
        raise TypeError("matrix must be a torch.Tensor, np.ndarray, or list thereof.")

    if len(frames) != len(matrix_list):
        raise ValueError(
            (
                f"Number of frames ({len(frames)}) and number of "
                f"matrices ({len(matrix_list)}) do not match."
            )
        )

    # Determine the type (torch or numpy) from the first matrix in the list.
    is_torch = isinstance(matrix_list[0], torch.Tensor)
    fixed_matrices = []

    for mat, frame in zip(matrix_list, frames):
        idx = compute_indices(frame, orbital, to_spherical)
        if is_torch:
            idx_tensor = torch.tensor(idx, dtype=torch.long, device=mat.device)
            reordered = mat[idx_tensor][:, idx_tensor]
        else:
            idx_arr = np.array(idx)
            reordered = mat[np.ix_(idx_arr, idx_arr)]
        fixed_matrices.append(reordered)

    if single_frame:
        return fixed_matrices[0]
    else:
        return fixed_matrices
