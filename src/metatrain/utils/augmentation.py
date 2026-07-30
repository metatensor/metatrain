import functools
from typing import (
    AbstractSet,
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
    Union,
)

import torch
from metatensor.torch import Labels, TensorBlock, TensorMap
from metatomic.torch import System
from metatomic.torch.o3 import O3Transformations, random_transformations

from .data import TargetInfo


#: Key under which :meth:`O3Augmenter.apply_random_system_augmentations` records the
#: transformation it applied, so that :meth:`O3Augmenter.undo_augmentation` can invert
#: it on the model's predictions.
AUGMENTATION_NAME = "mtt::aux::augmentation"


class O3Augmenter:
    """
    Applies random O(3) transformations to a set of systems and their targets.

    :param target_info_dict: A dictionary mapping target names to their corresponding
        :class:`TargetInfo` objects.
    :param extra_data_info_dict: An optional dictionary mapping extra data names to
        their corresponding :class:`TargetInfo` objects.
    :param group: which transformations :meth:`apply_random_augmentations` samples
        from: ``"O3"`` (the default) draws uniform rotations and improper rotations,
        while ``"inversions"`` draws only the identity and the inversion, for
        architectures that are already rotation-equivariant by construction.
    """

    def __init__(
        self,
        target_info_dict: Dict[str, TargetInfo],
        extra_data_info_dict: Optional[Dict[str, TargetInfo]] = None,
        group: str = "O3",
    ):
        if group not in ("O3", "inversions"):
            raise ValueError(
                f"unknown transformation group '{group}', expected 'O3' or 'inversions'"
            )
        self._group = group
        if extra_data_info_dict is None:
            extra_data_info_dict = {}
        self._max_angular_momentum = _max_angular_momentum(
            target_info_dict, extra_data_info_dict
        )
        self._target_names = set(target_info_dict)
        # Extra data declared as physical quantities, i.e. the ones whose tensor
        # character the user has described. Only these are transformed when the
        # per-target split is active; see ``apply_random_system_augmentations``.
        self._declared_extra_data = set(extra_data_info_dict)

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
        dtype = systems[0].positions.dtype
        transformations = self.sample_transformations(len(systems), dtype=dtype)
        return self._apply(systems, targets, transformations, extra_data=extra_data)

    def sample_transformations(
        self,
        n_systems: int,
        dtype: torch.dtype,
    ) -> List[O3Transformation]:
        """
        Draws one random transformation per system from the configured group.

        Exposed separately from :meth:`apply_random_augmentations` so that a caller
        which needs to know *which* transformation was applied can sample first and
        apply second, rather than trying to recover it afterwards. Density losses do
        this: their metric matrices are built in the unrotated frame, so they must
        undo the augmentation on their residual.

        :param n_systems: Number of transformations to draw.
        :param dtype: Floating point dtype of the transformation matrices.
        :return: One transformation per system.
        """
        if self._group == "inversions":
            signs = torch.randint(0, 2, (n_systems,)) * 2 - 1
            return [
                O3Transformation(
                    sign * torch.eye(3, dtype=dtype), self._max_angular_momentum
                )
                for sign in signs
            ]
        return random_transformations(
            n_systems,
            self._max_angular_momentum,
            device=torch.device("cpu"),
            dtype=dtype,
            add_inversions=True,
        )

    def apply_random_system_augmentations(
        self,
        systems: List[System],
        targets: Dict[str, TensorMap],
        extra_data: Optional[Dict[str, TensorMap]] = None,
        original_frame: AbstractSet[str] = frozenset(),
    ) -> Tuple[List[System], Dict[str, TensorMap], Dict[str, TensorMap]]:
        """
        Augments the systems, keeping ``original_frame`` targets in the dataset frame.

        The alternative to :meth:`apply_random_augmentations`: rather than rotating
        everything and comparing in the augmented frame, the targets named in
        ``original_frame`` are left as the dataset stores them, and
        :meth:`undo_augmentation` maps their predictions back before the loss is
        taken. The two are equivalent for a rotationally invariant loss -- the
        residual of one is the rotation of the other -- but only this ordering leaves
        a target's reference data in the frame it was computed in.

        Every other target is augmented exactly as :meth:`apply_random_augmentations`
        would, so mixing the two kinds in one run leaves each of them with the loss it
        would have had on its own.

        Extra data follows its target: an entry named ``"<target>_..."`` is treated as
        belonging to that target. Only extra data **declared** in the augmenter's
        ``extra_data_info_dict`` is ever transformed, since only those have a
        described tensor character; bookkeeping payloads are passed through.

        The applied transformation is recorded in ``extra_data`` under
        :data:`AUGMENTATION_NAME`.

        :param systems: A list of :class:`System` objects.
        :param targets: A dictionary mapping target names to :class:`TensorMap`
            objects.
        :param extra_data: An optional dictionary of additional :class:`TensorMap`
            objects.
        :param original_frame: Names of the targets to leave in the dataset's frame.
        :return: A tuple of augmented systems, the targets, and the extra data with
            the applied transformation recorded in it.
        """
        dtype = systems[0].positions.dtype
        transformations = self.sample_transformations(len(systems), dtype=dtype)
        new_systems = [
            transform_system(system, transformation)
            for system, transformation in zip(systems, transformations, strict=True)
        ]

        n_systems = len(systems)

        def _transform(tmap: TensorMap) -> TensorMap:
            return transform_tensor(
                tmap, systems, transformations, _tensor_system_ids(tmap, n_systems)
            )

        new_targets = {
            name: tmap if name in original_frame else _transform(tmap)
            for name, tmap in targets.items()
        }

        new_extra_data: Dict[str, TensorMap] = {}
        for name, tmap in (extra_data or {}).items():
            if (
                name in self._declared_extra_data
                and self._owning_target(name) not in original_frame
            ):
                new_extra_data[name] = _transform(tmap)
            else:
                new_extra_data[name] = tmap

        new_extra_data[AUGMENTATION_NAME] = _pack_transformations(
            [transformation.matrix for transformation in transformations]
        )
        return new_systems, new_targets, new_extra_data

    def undo_augmentation(
        self,
        predictions: Dict[str, TensorMap],
        systems: List[System],
        extra_data: Optional[Dict[str, TensorMap]] = None,
        original_frame: AbstractSet[str] = frozenset(),
    ) -> Dict[str, TensorMap]:
        """
        Maps predictions back into the frame their targets are stored in.

        The inverse of :meth:`apply_random_system_augmentations`, to be called on a
        model's output before the loss is taken. Only the targets named in
        ``original_frame`` are mapped back; the rest were compared in the augmented
        frame and must stay there. If no augmentation was recorded -- during
        validation, or when :meth:`apply_random_augmentations` was used instead --
        the predictions are returned untouched, so callers need no conditional.

        :param predictions: The model's outputs.
        :param systems: The systems the predictions were made on.
        :param extra_data: The batch's extra data, holding the recorded
            transformation.
        :param original_frame: Names of the targets to map back. Must match what was
            passed to :meth:`apply_random_system_augmentations`.
        :return: The predictions, each in the frame its target is stored in.
        """
        if extra_data is None or AUGMENTATION_NAME not in extra_data:
            return predictions
        to_undo = [name for name in predictions if name in original_frame]
        if not to_undo:
            return predictions

        reference = predictions[to_undo[0]].block(0).values
        matrices = _unpack_transformations(extra_data[AUGMENTATION_NAME]).to(
            dtype=reference.dtype, device=reference.device
        )
        # O(3) matrices are orthogonal, so the inverse is the transpose. This holds
        # for improper rotations too, so reflections need no special handling.
        inverse = [
            O3Transformation(matrix.T.contiguous(), self._max_angular_momentum)
            for matrix in matrices
        ]
        n_systems = len(systems)
        return {
            name: (
                transform_tensor(
                    tmap, systems, inverse, _tensor_system_ids(tmap, n_systems)
                )
                if name in original_frame
                else tmap
            )
            for name, tmap in predictions.items()
        }

    def _owning_target(self, extra_data_name: str) -> Optional[str]:
        """The target an extra-data entry belongs to.

        Uses the ``"<target>_..."`` naming convention that metatrain's own extra data
        follows (masks, metric matrices, projections).

        :param extra_data_name: Key of the entry in ``extra_data``.
        :return: The longest matching target name, or ``None`` if it belongs to none.
        """
        owners = [
            name
            for name in self._target_names
            if extra_data_name.startswith(f"{name}_")
        ]
        return max(owners, key=len) if owners else None

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
        o3_transformations = O3Transformations(
            torch.stack(transformations), self._max_angular_momentum
        )
        return self._apply(systems, targets, o3_transformations, extra_data=extra_data)

    def _apply(
        self,
        systems: List[System],
        targets: Dict[str, TensorMap],
        transformations: O3Transformations,
        extra_data: Optional[Dict[str, TensorMap]] = None,
    ) -> Tuple[List[System], Dict[str, TensorMap], Dict[str, TensorMap]]:
        new_systems = transformations.transform_systems(systems)

        n_systems = len(systems)

        def _transform(tmap: TensorMap) -> TensorMap:
            return transformations.transform_tensormap(
                tmap, _tensor_system_ids(tmap, n_systems)
            )

        new_targets = {name: _transform(tmap) for name, tmap in targets.items()}

        new_extra_data: Dict[str, TensorMap] = {}
        if extra_data is not None:
            for name, tmap in extra_data.items():
                if name.endswith("_mask"):
                    # loss masks are not physical quantities and must not be rotated
                    new_extra_data[name] = tmap
                else:
                    new_extra_data[name] = _transform(tmap)

        return new_systems, new_targets, new_extra_data


def original_frame_targets(loss_hypers: Union[str, Dict[str, Any], None]) -> Set[str]:
    """
    The targets whose losses must be evaluated in the frame the dataset stores them in.

    Read from :attr:`~metatrain.utils.loss.LossInterface.evaluate_in_original_frame`,
    so a loss opts in without any trainer having to know about it.

    :param loss_hypers: The trainer's ``loss`` hyperparameter, keyed by target name.
    :return: The names of those targets; empty if none.
    """
    from .loss import LossType

    if not isinstance(loss_hypers, dict):
        # the shorthand `loss: mse`, i.e. one loss type for every target
        return set()

    names = set()
    for target_name, spec in loss_hypers.items():
        loss_type = spec.get("type") if isinstance(spec, dict) else None
        if loss_type is None:
            continue
        if LossType.from_key(loss_type).cls.evaluate_in_original_frame:
            names.add(target_name)
    return names


def get_augmentation_transform(
    augmenter: O3Augmenter, original_frame: AbstractSet[str]
) -> Callable:
    """
    Selects the augmentation workflow a run's targets require.

    :param augmenter: The augmenter to draw transformations from.
    :param original_frame: Targets to keep in the dataset's frame, from
        :func:`original_frame_targets`. When empty, the ordinary
        augment-everything workflow is used.
    :return: The collate transform to use for training augmentation.
    """
    if not original_frame:
        return augmenter.apply_random_augmentations
    return functools.partial(
        augmenter.apply_random_system_augmentations, original_frame=original_frame
    )


def _pack_transformations(matrices: List[torch.Tensor]) -> TensorMap:
    """Store one 3x3 transformation per system as a flat, invariant payload.

    Kept without a component axis so that it is inert to any further augmentation:
    it describes a transformation, and is not itself a physical quantity to rotate.

    :param matrices: One 3x3 orthogonal matrix per system.
    :return: A :class:`TensorMap` with one ``(n_systems, 9)`` block.
    """
    values = torch.stack([matrix.reshape(-1) for matrix in matrices])
    n_systems = values.shape[0]
    return TensorMap(
        Labels.single().to(device=values.device),
        [
            TensorBlock(
                values=values,
                samples=Labels(
                    names=["system"],
                    values=torch.arange(
                        n_systems, dtype=torch.int32, device=values.device
                    ).reshape(-1, 1),
                ),
                components=[],
                properties=Labels(
                    names=["matrix_element"],
                    values=torch.arange(
                        9, dtype=torch.int32, device=values.device
                    ).reshape(-1, 1),
                ),
            )
        ],
    )


def _unpack_transformations(packed: TensorMap) -> torch.Tensor:
    """Recover the transformations stored by :func:`_pack_transformations`.

    :param packed: The :data:`AUGMENTATION_NAME` entry of ``extra_data``.
    :return: A ``(n_systems, 3, 3)`` tensor.
    """
    return packed.block().values.reshape(-1, 3, 3)


def _tensor_system_ids(tensor: TensorMap, n_systems: int) -> Optional[torch.Tensor]:
    """Recover the "system" label value assigned to each of the ``n_systems``
    systems in this batch, in the same order as the ``systems`` list, as used by
    this specific tensor.

    The "system" sample label is normally the absolute dataset index of each system
    (see ``dataset.py``), but some collate transforms (e.g. atomic-basis target
    preparation) reindex it to a batch-local ``0..n_systems-1`` before augmentation
    runs. Different tensors in the same batch can therefore use different "system"
    numbering, so the mapping must be recovered independently for each tensor rather
    than shared across the whole batch.

    :param tensor: the tensor to recover the per-system "system" label values from.
    :param n_systems: the number of systems in the batch.
    :return: a tensor of ``n_systems`` "system" label values, in the same order as the
        ``systems`` list, or ``None`` if no block of ``tensor`` has a "system" samples
        column with exactly ``n_systems`` distinct values.
    """
    for block in tensor.blocks():
        if "system" not in block.samples.names:
            continue
        column = block.samples.column("system")
        # order-preserving dedup: the first-appearance order must match `systems`
        seen = dict.fromkeys(column.tolist())
        if len(seen) == n_systems:
            return torch.tensor(
                list(seen.keys()), dtype=torch.int32, device=column.device
            )
    return None


def _max_angular_momentum(
    target_info_dict: Dict[str, TargetInfo],
    extra_data_info_dict: Dict[str, TargetInfo],
) -> int:
    """Largest angular momentum among all spherical targets/extra data, so the
    Wigner-D cache built for each transformation covers every ``ell`` it will be
    asked to rotate.

    :param target_info_dict: A dictionary mapping target names to their corresponding
        :class:`TargetInfo` objects.
    :param extra_data_info_dict: A dictionary mapping extra data names to their
        corresponding :class:`TargetInfo` objects.
    :return: The largest angular momentum ``ell`` found among all spherical
        targets/extra data.
    """
    max_ell = 0
    for info_dict in (target_info_dict, extra_data_info_dict):
        for name, info in info_dict.items():
            if name.endswith("_mask") or not info.is_spherical:
                continue
            for block in info.layout.blocks():
                for component in block.components:
                    max_ell = max(max_ell, (len(component) - 1) // 2)
    return max_ell
