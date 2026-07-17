from typing import Dict, List, Optional, Tuple

import torch
from metatensor.torch import TensorMap
from metatomic.torch import System
from metatomic.torch.o3 import (
    O3Transformation,
    random_transformations,
    transform_system,
    transform_tensor,
)

from .data import TargetInfo


class RotationalAugmenter:
    """
    Applies random O(3) rotations to a set of systems and their targets.

    :param target_info_dict: A dictionary mapping target names to their corresponding
        :class:`TargetInfo` objects.
    :param extra_data_info_dict: An optional dictionary mapping extra data names to
        their corresponding :class:`TargetInfo` objects.
    """

    def __init__(
        self,
        target_info_dict: Dict[str, TargetInfo],
        extra_data_info_dict: Optional[Dict[str, TargetInfo]] = None,
    ):
        for target_info in target_info_dict.values():
            if target_info.is_cartesian:
                if len(target_info.layout.block(0).components) > 2:
                    raise ValueError(
                        "RotationalAugmenter only supports Cartesian targets "
                        "with `rank<=2`."
                    )

        if extra_data_info_dict is None:
            extra_data_info_dict = {}
        self._max_angular_momentum = _max_angular_momentum(
            target_info_dict, extra_data_info_dict
        )

    def apply_random_augmentations(
        self,
        systems: List[System],
        targets: Dict[str, TensorMap],
        extra_data: Optional[Dict[str, TensorMap]] = None,
    ) -> Tuple[List[System], Dict[str, TensorMap], Dict[str, TensorMap]]:
        """
        Applies random O(3) augmentations to systems, targets, and optional extra data.

        :param systems: A list of :class:`System` objects.
        :param targets: A dictionary mapping target names to :class:`TensorMap` objects.
        :param extra_data: An optional dictionary of additional :class:`TensorMap`
            objects to augment alongside targets.
        :return: A tuple of augmented systems, targets, and extra data.
        """
        transformations = random_transformations(
            len(systems),
            self._max_angular_momentum,
            device=torch.device("cpu"),
            dtype=systems[0].positions.dtype,
            include_inversions=True,
        )
        return self._apply(systems, targets, transformations, extra_data=extra_data)

    def apply_augmentations(
        self,
        systems: List[System],
        targets: Dict[str, TensorMap],
        transformations: List[torch.Tensor],
        extra_data: Optional[Dict[str, TensorMap]] = None,
    ) -> Tuple[List[System], Dict[str, TensorMap], Dict[str, TensorMap]]:
        """
        Applies the given O(3) transformations to systems, targets, and optional extra
        data.

        :param systems: A list of :class:`System` objects.
        :param targets: A dictionary mapping target names to :class:`TensorMap` objects.
        :param transformations: A list of 3x3 orthogonal :class:`torch.Tensor` matrices,
            one per system. Matrices with determinant -1 are improper rotations.
        :param extra_data: An optional dictionary of additional :class:`TensorMap`
            objects to augment alongside targets.
        :return: A tuple of augmented systems, targets, and extra data.
        """
        o3_transformations = [
            O3Transformation(matrix, self._max_angular_momentum)
            for matrix in transformations
        ]
        return self._apply(systems, targets, o3_transformations, extra_data=extra_data)

    def _apply(
        self,
        systems: List[System],
        targets: Dict[str, TensorMap],
        transformations: List[O3Transformation],
        extra_data: Optional[Dict[str, TensorMap]] = None,
    ) -> Tuple[List[System], Dict[str, TensorMap], Dict[str, TensorMap]]:
        new_systems = [
            transform_system(system, transformation)
            for system, transformation in zip(systems, transformations, strict=True)
        ]

        # The "system" sample label is the absolute dataset index of each system
        # (see ``dataset.py``), not its position in ``systems``, so it must be
        # recovered from the batch itself rather than assumed to be range(N).
        system_ids = _batch_system_ids(targets, extra_data, len(systems))

        new_targets = {
            name: transform_tensor(tmap, systems, transformations, system_ids)
            for name, tmap in targets.items()
        }

        new_extra_data: Dict[str, TensorMap] = {}
        if extra_data is not None:
            for name, tmap in extra_data.items():
                if name.endswith("_mask"):
                    # loss masks are not physical quantities and must not be rotated
                    new_extra_data[name] = tmap
                else:
                    new_extra_data[name] = transform_tensor(
                        tmap, systems, transformations, system_ids
                    )

        return new_systems, new_targets, new_extra_data


def _batch_system_ids(
    targets: Dict[str, TensorMap],
    extra_data: Optional[Dict[str, TensorMap]],
    n_systems: int,
) -> Optional[torch.Tensor]:
    """Recover the "system" label value assigned to each of the ``n_systems``
    systems in this batch, in the same order as the ``systems`` list.

    Every non-mask target/extra-data tensor covers all systems in the batch, so the
    first one found with a "system" samples column is used; its distinct values, in
    order of first appearance, give the per-system labels.
    """
    for tensormap_dict in (targets, extra_data or {}):
        for tmap in tensormap_dict.values():
            for block in tmap.blocks():
                if "system" not in block.samples.names:
                    continue
                column = block.samples.column("system")
                seen: Dict[int, None] = {}
                for value in column.tolist():
                    seen.setdefault(value, None)
                if len(seen) == n_systems:
                    return torch.tensor(list(seen.keys()), dtype=torch.int32)
    return None


def _max_angular_momentum(
    target_info_dict: Dict[str, TargetInfo],
    extra_data_info_dict: Dict[str, TargetInfo],
) -> int:
    """Largest angular momentum among all spherical targets/extra data, so the
    Wigner-D cache built for each transformation covers every ``ell`` it will be
    asked to rotate."""
    max_ell = 0
    for info_dict in (target_info_dict, extra_data_info_dict):
        for name, info in info_dict.items():
            if name.endswith("_mask") or not info.is_spherical:
                continue
            for block in info.layout.blocks():
                for component in block.components:
                    max_ell = max(max_ell, (len(component) - 1) // 2)
    return max_ell
