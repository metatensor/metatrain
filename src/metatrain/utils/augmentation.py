import random
from typing import Dict, List, Optional, Tuple

import metatensor.torch as mts
import numpy as np
import torch
from metatensor.torch import TensorBlock, TensorMap
from metatomic.torch import System, register_autograd_neighbors
from scipy.spatial.transform import Rotation

from . import torch_jit_script_unless_coverage
from .data import TargetInfo


def get_random_rotation():
    return Rotation.random()


def get_random_inversion():
    return random.choice([1, -1])


class RotationalAugmenter:
    """
    A class to apply random rotations and inversions to a set of systems and their
    targets.

    :param target_info_dict: A dictionary mapping target names to their corresponding
        :class:`TargetInfo` objects. This is used to determine the type of targets and
        how to apply the augmentations.
    """

    def __init__(
        self,
        target_info_dict: Dict[str, TargetInfo],
        extra_data_info_dict: Optional[Dict[str, TargetInfo]] = None,
    ):
        # checks on targets
        for target_info in target_info_dict.values():
            if target_info.is_cartesian:
                if len(target_info.layout.block(0).components) > 2:
                    raise ValueError(
                        "RotationalAugmenter only supports Cartesian targets "
                        "with `rank<=2`."
                    )

        self.target_info_dict = target_info_dict
        if extra_data_info_dict is None:
            extra_data_info_dict = {}
        self.extra_data_info_dict = extra_data_info_dict

        self.wigner = None
        self.complex_to_real_spherical_harmonics_transforms = {}
        is_any_target_spherical = any(
            target_info.is_spherical for target_info in target_info_dict.values()
        )
        is_any_extra_data_spherical = any(
            extra_data_info.is_spherical
            for extra_data_info in extra_data_info_dict.values()
        )
        if is_any_target_spherical or is_any_extra_data_spherical:
            try:
                import spherical
            except ImportError as e:
                # quaternionic (used below) is a dependency of spherical
                raise ImportError(
                    "To perform data augmentation on spherical targets, please "
                    "install the `spherical` package with `pip install spherical`."
                ) from e

            largest_l_targets = -1
            largest_l_extra_data = -1
            if is_any_target_spherical:
                largest_l_targets = max(
                    (len(block.components[0]) - 1) // 2
                    for target_info in target_info_dict.values()
                    if target_info.is_spherical
                    for block in target_info.layout.blocks()
                )
            if is_any_extra_data_spherical:
                largest_l_extra_data = max(
                    (len(block.components[0]) - 1) // 2
                    for extra_data_info in extra_data_info_dict.values()
                    if extra_data_info.is_spherical
                    for block in extra_data_info.layout.blocks()
                )
            largest_l = max(largest_l_targets, largest_l_extra_data)

            self.wigner = spherical.Wigner(largest_l)
            for ell in range(largest_l + 1):
                self.complex_to_real_spherical_harmonics_transforms[ell] = (
                    _complex_to_real_spherical_harmonics_transform(ell)
                )

    def apply_random_augmentations(
        self,
        systems: List[System],
        targets: Dict[str, TensorMap],
        extra_data: Optional[Dict[str, TensorMap]] = None,
    ) -> Tuple[List[System], Dict[str, TensorMap], Dict[str, TensorMap]]:
        """
        Apply a random augmentation to a number of ``System`` objects and its targets.

        :param systems: A list of :class:`System` objects to be augmented.
        :param targets: A dictionary mapping target names to their corresponding
            :class:`TensorMap` objects. These are the targets to be augmented.

        :return: A tuple containing the augmented systems and targets.
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
            tensormap_dicts = (
                [targets, extra_data] if extra_data is not None else [targets]
            )
            info_dicts = (
                [self.target_info_dict, self.extra_data_info_dict]
                if extra_data is not None
                else [self.target_info_dict]
            )
            for tensormap_dict, info_dict in zip(tensormap_dicts, info_dicts):
                for name in tensormap_dict.keys():
                    if name.endswith("_mask"):
                        # skip loss masks
                        continue
                    tensormap_info = info_dict[name]
                    if tensormap_info.is_spherical:
                        for block in tensormap_info.layout.blocks():
                            ell = (len(block.components[0]) - 1) // 2
                            U = self.complex_to_real_spherical_harmonics_transforms[ell]
                            if ell not in wigner_D_matrices:  # skip if already computed
                                wigner_D_matrices_l = []
                                for (
                                    wigner_D_matrix_complex
                                ) in wigner_D_matrices_complex:
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

                                    wigner_D_matrix = U.conj() @ wigner_D_matrix @ U.T
                                    assert np.allclose(wigner_D_matrix.imag, 0.0)
                                    wigner_D_matrix = wigner_D_matrix.real
                                    wigner_D_matrices_l.append(
                                        torch.from_numpy(wigner_D_matrix)
                                    )
                                wigner_D_matrices[ell] = wigner_D_matrices_l

        return _apply_random_augmentations(
            systems, targets, transformations, wigner_D_matrices, extra_data=extra_data
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
            # fold property dimension in, apply transformation,
            # unfold property dimension
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


@torch_jit_script_unless_coverage  # script for speed
def _apply_random_augmentations(
    systems: List[System],
    targets: Dict[str, TensorMap],
    transformations: List[torch.Tensor],
    wigner_D_matrices: Dict[int, List[torch.Tensor]],
    extra_data: Optional[Dict[str, TensorMap]] = None,
) -> Tuple[List[System], Dict[str, TensorMap], Dict[str, TensorMap]]:
    # Apply the transformations to the systems

    new_systems: List[System] = []
    for system, transformation in zip(systems, transformations):
        new_system = System(
            positions=system.positions @ transformation.T,
            types=system.types,
            cell=system.cell @ transformation.T,
            pbc=system.pbc,
        )
        for data_name in system.known_data():
            data = system.get_data(data_name)
            # check if this data is easy to handle (scalar/vector), otherwise error out
            if len(data) != 1:
                raise ValueError(
                    f"System data '{data_name}' has {len(data)} blocks, which is not "
                    "supported by RotationalAugmenter. Only scalar and vector data are "
                    "supported."
                )
            if len(data.block().components) == 0:
                # scalar data, no change
                new_system.add_data(data_name, data)
            elif len(data.block().components) == 1 and data.block().components[
                0
            ].names == ["xyz"]:
                new_system.add_data(
                    data_name,
                    TensorMap(
                        keys=data.keys,
                        blocks=[
                            TensorBlock(
                                values=(
                                    data.block().values.swapaxes(-1, -2)
                                    @ transformation.T
                                ).swapaxes(-1, -2),
                                samples=data.block().samples,
                                components=data.block().components,
                                properties=data.block().properties,
                            )
                        ],
                    ),
                )
            else:
                raise ValueError(
                    f"System data '{data_name}' has components "
                    f"{data.block().components}, which are not supported by "
                    "RotationalAugmenter. Only scalar and vector data are supported."
                )
        for options in system.known_neighbor_lists():
            neighbors = mts.detach_block(system.get_neighbor_list(options))

            neighbors.values[:] = (
                neighbors.values.squeeze(-1) @ transformation.T
            ).unsqueeze(-1)

            register_autograd_neighbors(system, neighbors)
            new_system.add_neighbor_list(options, neighbors)
        new_systems.append(new_system)

    # Apply the transformation to the targets and extra data
    new_targets: Dict[str, TensorMap] = {}
    new_extra_data: Dict[str, TensorMap] = {}

    # Do not transform any masks present in extra_data
    if extra_data is not None:
        mask_keys: List[str] = []
        for key in extra_data.keys():
            if key.endswith("_mask"):
                mask_keys.append(key)
        for key in mask_keys:
            new_extra_data[key] = extra_data.pop(key)

    for tensormap_dict, new_dict in zip(
        [targets, extra_data], [new_targets, new_extra_data]
    ):
        if tensormap_dict is None:
            continue
        assert tensormap_dict is not None
        for name, original_tmap in tensormap_dict.items():
            is_scalar = False
            if len(original_tmap.blocks()) == 1:
                if len(original_tmap.block().components) == 0:
                    is_scalar = True

            is_cartesian = False
            if len(original_tmap.blocks()) == 1:
                if len(original_tmap.block().components) > 0:
                    if "xyz" in original_tmap.block().components[0].names[0]:
                        is_cartesian = True

            is_spherical = all(
                len(block.components) == 1 and block.components[0].names == ["o3_mu"]
                for block in original_tmap.blocks()
            )

            if is_scalar:
                # no change for energies
                energy_block = TensorBlock(
                    values=original_tmap.block().values,
                    samples=original_tmap.block().samples,
                    components=original_tmap.block().components,
                    properties=original_tmap.block().properties,
                )
                if original_tmap.block().has_gradient("positions"):
                    # transform position gradients:
                    block = original_tmap.block().gradient("positions")
                    position_gradients = block.values.squeeze(-1)
                    split_sizes_forces = [
                        system.positions.shape[0] for system in systems
                    ]
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
                if original_tmap.block().has_gradient("strain"):
                    # transform strain gradients (rank-2 tensor):
                    block = original_tmap.block().gradient("strain")
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
                new_dict[name] = TensorMap(
                    keys=original_tmap.keys,
                    blocks=[energy_block],
                )

            elif is_spherical:
                new_dict[name] = _apply_wigner_D_matrices(
                    systems, original_tmap, transformations, wigner_D_matrices
                )

            elif is_cartesian:
                rank = len(original_tmap.block().components)
                if rank == 1:
                    # transform Cartesian vector:
                    block = original_tmap.block()
                    vectors = block.values
                    if "atom" in original_tmap.block().samples.names:
                        split_vectors = torch.split(
                            vectors, [len(system.positions) for system in systems]
                        )
                    else:
                        split_vectors = torch.split(vectors, [1 for _ in systems])
                    new_vectors = []
                    for v, transformation in zip(split_vectors, transformations):
                        # fold property dimension in, apply transformation,
                        # unfold property dimension
                        new_v = v.transpose(1, 2)
                        new_v = new_v @ transformation.T
                        new_v = new_v.transpose(1, 2)
                        new_vectors.append(new_v)
                    new_vectors = torch.cat(new_vectors)
                    new_dict[name] = TensorMap(
                        keys=original_tmap.keys,
                        blocks=[
                            TensorBlock(
                                values=new_vectors,
                                samples=block.samples,
                                components=block.components,
                                properties=block.properties,
                            )
                        ],
                    )
                elif rank == 2:
                    # transform Cartesian rank-2 tensor:
                    block = original_tmap.block()
                    tensor = block.values
                    if "atom" in original_tmap.block().samples.names:
                        split_tensors = torch.split(
                            tensor, [len(system.positions) for system in systems]
                        )
                    else:
                        split_tensors = torch.split(tensor, [1 for _ in systems])
                    new_tensors = []
                    for tensor, transformation in zip(split_tensors, transformations):
                        new_tensor = torch.einsum(
                            "Aa,iabp,bB->iABp", transformation, tensor, transformation.T
                        )
                        new_tensors.append(new_tensor)
                    new_tensors = torch.cat(new_tensors)
                    new_dict[name] = TensorMap(
                        keys=original_tmap.keys,
                        blocks=[
                            TensorBlock(
                                values=new_tensors,
                                samples=block.samples,
                                components=block.components,
                                properties=block.properties,
                            )
                        ],
                    )

    return new_systems, new_targets, new_extra_data


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
