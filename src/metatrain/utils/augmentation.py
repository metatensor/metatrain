import random
from typing import Dict, List, Tuple

import numpy as np
import torch
from metatensor.torch import TensorBlock, TensorMap
from metatensor.torch.atomistic import System
from scipy.spatial.transform import Rotation

from .data import TargetInfo


def get_random_rotation():
    return Rotation.random()


def get_random_inversion():
    return random.choice([1, -1])


class RotationalAugmenter:
    def __init__(self, target_info_dict: Dict[str, TargetInfo]):
        # checks on targets
        for target_info in target_info_dict.values():
            if target_info.is_cartesian:
                if len(target_info.layout.block(0).components) != 1:
                    raise ValueError(
                        "RotationalAugmenter only supports Cartesian targets "
                        "with `rank=1`."
                    )

        self.target_info_dict = target_info_dict

        self.wigner = None
        self.complex_to_real_spherical_harmonics_transforms = {}
        is_any_target_spherical = any(
            target_info.is_spherical for target_info in target_info_dict.values()
        )
        if is_any_target_spherical:
            try:
                import spherical
            except ImportError:
                # quaternionic (used below) is a dependency of spherical
                raise ImportError(
                    "To use spherical targets with nanoPET, please install the "
                    "`spherical` package with `pip install spherical`."
                )
            largest_l = max(
                (len(block.components[0]) - 1) // 2
                for target_info in target_info_dict.values()
                if target_info.is_spherical
                for block in target_info.layout.blocks()
            )
            self.wigner = spherical.Wigner(largest_l)
            for ell in range(largest_l + 1):
                self.complex_to_real_spherical_harmonics_transforms[ell] = (
                    _complex_to_real_spherical_harmonics_transform(ell)
                )

    def apply_random_augmentations(
        self, systems: List[System], targets: Dict[str, TensorMap]
    ) -> Tuple[List[System], Dict[str, TensorMap]]:
        """
        Apply a random augmentation to a number of ``System`` objects and its targets.
        """

        rotations = [get_random_rotation() for _ in range(len(systems))]
        inversions = [get_random_inversion() for _ in range(len(systems))]
        transformations = [
            torch.from_numpy(r.as_matrix() * i) for r, i in zip(rotations, inversions)
        ]

        wigner_D_matrices = {}
        if self.wigner is not None:
            scipy_quaternions = [r.as_quat() for r in rotations]
            quaternionic_quaternions = [
                _scipy_quaternion_to_quaternionic(q) for q in scipy_quaternions
            ]
            wigner_D_matrices_complex = [
                self.wigner.D(q) for q in quaternionic_quaternions
            ]
            for target_name in targets.keys():
                target_info = self.target_info_dict[target_name]
                if target_info.is_spherical:
                    for block in target_info.layout.blocks():
                        ell = (len(block.components[0]) - 1) // 2
                        if ell not in wigner_D_matrices:  # skip if already computed
                            wigner_D_matrices_l = []
                            for wigner_D_matrix_complex in wigner_D_matrices_complex:
                                wigner_D_matrix = np.zeros(
                                    (2 * ell + 1, 2 * ell + 1), dtype=np.complex128
                                )
                                for mp in range(-ell, ell + 1):
                                    for m in range(-ell, ell + 1):
                                        wigner_D_matrix[m + ell, mp + ell] = (
                                            wigner_D_matrix_complex[
                                                self.wigner.Dindex(ell, m, mp)
                                            ]
                                        ).conj()
                                U = self.complex_to_real_spherical_harmonics_transforms[
                                    ell
                                ]
                                wigner_D_matrix = U.conj() @ wigner_D_matrix @ U.T
                                assert np.allclose(wigner_D_matrix.imag, 0.0)
                                wigner_D_matrix = wigner_D_matrix.real
                                wigner_D_matrices_l.append(
                                    torch.from_numpy(wigner_D_matrix)
                                )
                            wigner_D_matrices[ell] = wigner_D_matrices_l

        return _apply_random_augmentations(
            systems, targets, transformations, wigner_D_matrices
        )


def _apply_wigner_D_matrices(
    systems: List[System],
    target_tmap: TensorMap,
    transformations: List[torch.Tensor],
    wigner_D_matrices: Dict[int, List[torch.Tensor]],
) -> TensorMap:
    new_blocks: List[TensorBlock] = []
    for key, block in target_tmap.items():
        ell, sigma = int(key[0]), int(key[1])
        values = block.values
        if "atom" in block.samples.names:
            split_values = torch.split(
                values, [len(system.positions) for system in systems]
            )
        else:
            split_values = torch.split(values, [1 for _ in systems])
        new_values = []
        ell = (len(block.components[0]) - 1) // 2
        for v, transformation, wigner_D_matrix in zip(
            split_values, transformations, wigner_D_matrices[ell]
        ):
            is_inverted = torch.det(transformation) < 0
            new_v = v.clone()
            if is_inverted:  # inversion
                new_v = new_v * (-1) ** ell * sigma
            # fold property dimension in, apply transformation, unfold property dim
            new_v = new_v.transpose(1, 2)
            new_v = new_v @ wigner_D_matrix.T
            new_v = new_v.transpose(1, 2)
            new_values.append(new_v)
        new_values = torch.concatenate(new_values)
        new_block = TensorBlock(
            values=new_values,
            samples=block.samples,
            components=block.components,
            properties=block.properties,
        )
        new_blocks.append(new_block)

    return TensorMap(
        keys=target_tmap.keys,
        blocks=new_blocks,
    )


@torch.jit.script  # script for speed
def _apply_random_augmentations(  # pragma: no cover
    systems: List[System],
    targets: Dict[str, TensorMap],
    transformations: List[torch.Tensor],
    wigner_D_matrices: Dict[int, List[torch.Tensor]],
) -> Tuple[List[System], Dict[str, TensorMap]]:
    # Apply the transformations to the systems
    new_systems: List[System] = []
    for system, transformation in zip(systems, transformations):
        new_system = System(
            positions=system.positions @ transformation.T,
            types=system.types,
            cell=system.cell @ transformation.T,
            pbc=system.pbc,
        )
        for nl_options in system.known_neighbor_lists():
            old_nl = system.get_neighbor_list(nl_options)
            new_system.add_neighbor_list(
                nl_options,
                TensorBlock(
                    values=(old_nl.values.squeeze(-1) @ transformation.T).unsqueeze(-1),
                    samples=old_nl.samples,
                    components=old_nl.components,
                    properties=old_nl.properties,
                ),
            )
        new_systems.append(new_system)

    # Apply the transformation to the targets
    new_targets: Dict[str, TensorMap] = {}
    for name, target_tmap in targets.items():
        is_scalar = False
        if len(target_tmap.blocks()) == 1:
            if len(target_tmap.block().components) == 0:
                is_scalar = True

        is_spherical = all(
            len(block.components) == 1 and block.components[0].names == ["o3_mu"]
            for block in target_tmap.blocks()
        )

        if is_scalar:
            # no change for energies
            energy_block = TensorBlock(
                values=target_tmap.block().values,
                samples=target_tmap.block().samples,
                components=target_tmap.block().components,
                properties=target_tmap.block().properties,
            )
            if target_tmap.block().has_gradient("positions"):
                # transform position gradients:
                block = target_tmap.block().gradient("positions")
                position_gradients = block.values.squeeze(-1)
                split_sizes_forces = [system.positions.shape[0] for system in systems]
                split_position_gradients = torch.split(
                    position_gradients, split_sizes_forces
                )
                position_gradients = torch.cat(
                    [
                        split_position_gradients[i] @ transformations[i].T
                        for i in range(len(systems))
                    ]
                )
                energy_block.add_gradient(
                    "positions",
                    TensorBlock(
                        values=position_gradients.unsqueeze(-1),
                        samples=block.samples,
                        components=block.components,
                        properties=block.properties,
                    ),
                )
            if target_tmap.block().has_gradient("strain"):
                # transform strain gradients (rank 2 tensor):
                block = target_tmap.block().gradient("strain")
                strain_gradients = block.values.squeeze(-1)
                split_strain_gradients = torch.split(strain_gradients, 1)
                new_strain_gradients = torch.stack(
                    [
                        transformations[i]
                        @ split_strain_gradients[i].squeeze(0)
                        @ transformations[i].T
                        for i in range(len(systems))
                    ],
                    dim=0,
                )
                energy_block.add_gradient(
                    "strain",
                    TensorBlock(
                        values=new_strain_gradients.unsqueeze(-1),
                        samples=block.samples,
                        components=block.components,
                        properties=block.properties,
                    ),
                )
            new_targets[name] = TensorMap(
                keys=target_tmap.keys,
                blocks=[energy_block],
            )

        elif is_spherical:
            new_targets[name] = _apply_wigner_D_matrices(
                systems, target_tmap, transformations, wigner_D_matrices
            )

        else:
            # transform Cartesian vector:
            block = target_tmap.block()
            vectors = block.values
            if "atom" in target_tmap.block().samples.names:
                split_vectors = torch.split(
                    vectors, [len(system.positions) for system in systems]
                )
            else:
                split_vectors = torch.split(vectors, [1 for _ in systems])
            new_vectors = []
            for v, transformation in zip(split_vectors, transformations):
                # fold property dimension in, apply transformation, unfold property dim
                new_v = v.transpose(1, 2)
                new_v = new_v @ transformation.T
                new_v = new_v.transpose(1, 2)
                new_vectors.append(new_v)
            new_vectors = torch.cat(new_vectors)
            new_targets[name] = TensorMap(
                keys=target_tmap.keys,
                blocks=[
                    TensorBlock(
                        values=new_vectors,
                        samples=block.samples,
                        components=block.components,
                        properties=block.properties,
                    )
                ],
            )

    return new_systems, new_targets


def _complex_to_real_spherical_harmonics_transform(ell: int):
    # Generates the transformation matrix from complex spherical harmonics
    # to real spherical harmonics for a given l.
    # Returns a transformation matrix of shape ((2l+1), (2l+1)).

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


def _scipy_quaternion_to_quaternionic(q_scipy):
    # This function convert a quaternion obtained from the scipy library to the format
    # used by the quaternionic library.
    # Note: 'xyzw' is the format used by scipy.spatial.transform.Rotation
    # while 'wxyz' is the format used by quaternionic.
    qx, qy, qz, qw = q_scipy
    q_quaternion = np.array([qw, qx, qy, qz])
    return q_quaternion
