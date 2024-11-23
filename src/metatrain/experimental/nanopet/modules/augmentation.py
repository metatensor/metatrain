import random
from typing import Dict, List, Tuple

import numpy as np
import torch
from metatensor.torch import TensorBlock, TensorMap
from metatensor.torch.atomistic import System
from scipy.spatial.transform import Rotation

from ....utils.data import TargetInfo


def get_random_augmentation():

    transformation = Rotation.random().as_matrix()
    invert = random.choice([True, False])
    if invert:
        transformation *= -1
    return transformation


class RotationalAugmenter:
    def __init__(self, target_info_dict: Dict[str, TargetInfo]):
        # checks on targets
        for target_info in target_info_dict.values():
            if target_info.is_cartesian:
                if len(target_info.layout.components) != 1:
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
                len(block.components[0]) // 2 - 1
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

        transformations = [
            torch.from_numpy(get_random_augmentation()) for _ in range(len(systems))
        ]

        wigner_D_matrices = {}
        if self.wigner is not None:
            import quaternionic

            quaternionic_rotations = [
                quaternionic.array.from_rotation_matrix(t.numpy())
                for t in transformations
            ]
            wigner_D_matrices_complex = [
                self.wigner.D(R) for R in quaternionic_rotations
            ]
            for target_name in targets.keys():
                target_info = self.target_info_dict[target_name]
                if target_info.is_spherical:
                    for block in target_info.layout.block():
                        ell = len(block.components[0]) // 2 - 1
                        if ell not in wigner_D_matrices:
                            wigner_D_matrices_l = []
                            for wigner_D_matrix_complex in wigner_D_matrices_complex:
                                wigner_D_matrix = np.zeros(
                                    (2 * ell + 1, 2 * ell + 1), dtype=np.complex128
                                )
                                for mp in range(-ell, ell + 1):
                                    for m in range(-ell, ell + 1):
                                        wigner_D_matrix[mp + ell, m + ell] = (
                                            wigner_D_matrix_complex[
                                                self.wigner.Dindex(ell, mp, m)
                                            ]
                                        )
                                U = self.complex_to_real_spherical_harmonics_transforms[
                                    ell
                                ]
                                wigner_D_matrix = U @ wigner_D_matrix @ U.T.conj()
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
        ell = len(block.components[0]) // 2 - 1
        for v, transformation, wigner_D_matrix in zip(
            split_values, transformations, wigner_D_matrices[ell]
        ):
            is_inverted = torch.det(transformation) < 0
            new_v = v.clone()
            if is_inverted and sigma == -1:  # inversion
                new_v = -new_v
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


# script for speed
@torch.jit.script
def _apply_random_augmentations(
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
        assert len(target_tmap.blocks()) == 1
        is_scalar = len(target_tmap.block().components) == 0
        is_spherical = all(
            len(block.components) == 1 and block.components[0].names == ["o3_mu"]
            for block in target_tmap.blocks()
        )

        # for now, only accept vectors if they only have one subtarget/property
        if not is_scalar:
            assert target_tmap.block().values.shape[-1] == 1

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
            T[m_index, ell + abs(m)] = -1j / np.sqrt(2)
            T[m_index, ell - abs(m)] = 1j / np.sqrt(2) * (-1) ** abs(m)
        else:  # m == 0
            # Y_{l}^{0} remains unchanged
            T[m_index, ell] = 1

    # Return the transformation matrix to convert complex to real spherical harmonics
    return T
