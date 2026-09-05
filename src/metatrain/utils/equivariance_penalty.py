"""On-line equivariance-error penalty, shared across architectures.

The idea: instead of a single random O(3) augmentation per system (the usual
rotational-augmentation training scheme), augment each system
``num_augmentations`` times, evaluate the model on all of them, map every
prediction back to the original (untransformed) frame, and reduce them to a
mean and a variance over augmentations, per system. Training on
``MSE(mean, target) + weight * mean(variance)`` directly penalizes the model
for disagreeing with itself under rotation, which is what
``metatomic.torch.o3.SymmetrizedModel``'s equivariance error already
*measures* (see ``mtt eval``'s ``equivariance`` option) -- this is a cheap,
random-sample approximation to the same quantity, used as a training signal
instead.

This module provides the pieces that are architecture-agnostic:

* :func:`equivariance_variance_output_name` -- the
  ``{target}_equivariance_variance`` auxiliary-key naming convention (distinct
  from :func:`~metatrain.utils.ensemble.uncertainty_output_name`: this is the
  spread of one model's predictions over rotated copies of the same system,
  not the spread across ensemble members).
* :func:`reduce_over_augmentations` -- reduces a batch produced by
  :meth:`~metatrain.utils.augmentation.O3Augmenter.replicate_and_augment` (and
  mapped back to the original frame by
  :meth:`~metatrain.utils.augmentation.O3Augmenter.undo_augmentation`) to a
  per-system mean and variance.

Everything else -- drawing and applying the augmentations, calling the model,
and combining the mean/variance into a loss term -- is trainer/loss code, see
``metatrain.pet.trainer`` and
:class:`~metatrain.utils.loss.EquivariancePenaltyLoss`.
"""

from typing import Dict, List, Tuple

import torch
from metatensor.torch import Labels, TensorBlock, TensorMap


def equivariance_variance_output_name(target_name: str) -> str:
    """Name of the ``_equivariance_variance`` auxiliary output for a target.

    Mirrors :func:`~metatrain.utils.ensemble.uncertainty_output_name`'s naming
    convention, under a different suffix so the two auxiliary quantities
    (variance across ensemble members vs. variance across augmentations of the
    same model) can never collide if both are ever used on the same target.

    :param target_name: name of the base target (e.g. ``"energy"``).
    :return: the name of the corresponding auxiliary output.
    """
    if target_name == "energy":
        return "energy_equivariance_variance"
    return "mtt::aux::" + target_name.replace("mtt::", "") + "_equivariance_variance"


def reduce_over_augmentations(
    tensor_map: TensorMap, num_augmentations: int
) -> Tuple[TensorMap, TensorMap]:
    """Reduces every ``num_augmentations`` consecutive "system" samples to one.

    Expects ``tensor_map`` to come from a batch built by
    :meth:`~metatrain.utils.augmentation.O3Augmenter.replicate_and_augment`
    (each original system replicated ``num_augmentations`` times,
    consecutively, before augmenting) and mapped back to the original frame by
    :meth:`~metatrain.utils.augmentation.O3Augmenter.undo_augmentation`: its
    "system" sample values are then ``0, 1, ..., n_replicas - 1``, in that same
    replicated order, so augmentation ``a`` of original system ``i`` has
    "system" value ``i * num_augmentations + a``.

    Reduction only ever needs to select rows by their "system" value modulo
    ``num_augmentations``, with no explicit grouping or sorting: taking every
    row whose "system" value is congruent to ``a`` (mod ``num_augmentations``)
    selects exactly one (the ``a``-th) replica of every original system, in
    original-system order, since replicas of the same original system share
    everything about their row *except* the values themselves (same atoms, in
    the same order -- rotation does not reorder atoms). Stacking the
    ``num_augmentations`` such selections along a new leading axis and
    reducing over it is then exactly the ensemble mean/variance pattern
    already used for a :mod:`~metatrain.utils.ensemble` shallow ensemble.

    :param tensor_map: predictions for a batch of ``n_systems *
        num_augmentations`` (replicated, augmented, back-transformed) systems.
    :param num_augmentations: number of consecutive replicas per original
        system.
    :return: ``(mean, variance)``, each a :class:`TensorMap` with as many
        "system" values as there were original systems (``0`` ..
        ``n_systems - 1``), the mean and (unbiased) variance of the
        ``num_augmentations`` replicas' predictions.
    """
    mean_blocks: List[TensorBlock] = []
    variance_blocks: List[TensorBlock] = []
    for key, block in tensor_map.items():
        if len(block.gradients_list()) != 0:
            raise NotImplementedError(
                "reduce_over_augmentations does not support blocks with "
                f"explicit gradients (block {key.print()} has "
                f"{block.gradients_list()})"
            )
        sample_values = block.samples.values
        system_column = block.samples.names.index("system")
        system = sample_values[:, system_column]
        slot = system % num_augmentations

        slices = [block.values[slot == a] for a in range(num_augmentations)]
        stacked = torch.stack(slices, dim=0)
        mean_values = stacked.mean(dim=0)
        variance_values = stacked.var(dim=0, unbiased=True)

        representative = slot == 0
        new_sample_values = sample_values[representative].clone()
        new_sample_values[:, system_column] = (
            new_sample_values[:, system_column] // num_augmentations
        )
        new_samples = Labels(block.samples.names, new_sample_values)

        mean_blocks.append(
            TensorBlock(
                values=mean_values,
                samples=new_samples,
                components=block.components,
                properties=block.properties,
            )
        )
        variance_blocks.append(
            TensorBlock(
                values=variance_values,
                samples=new_samples,
                components=block.components,
                properties=block.properties,
            )
        )
    return (
        TensorMap(tensor_map.keys, mean_blocks),
        TensorMap(tensor_map.keys, variance_blocks),
    )


def reduce_predictions_over_augmentations(
    predictions: Dict[str, TensorMap],
    num_augmentations: int,
    variance_targets: List[str],
) -> Dict[str, TensorMap]:
    """Reduces every prediction in a batch over augmentations, in one call.

    Every entry of ``predictions`` is reduced to its mean over augmentations
    (see :func:`reduce_over_augmentations`), so the returned dict has exactly
    the same keys and shapes a single, non-augmented forward pass would have
    produced -- every other consumer (the scaler, ``average_by_num_atoms``,
    plain losses, RMSE/MAE accumulation) needs no awareness that augmentation
    happened at all. For the targets listed in ``variance_targets``, the
    variance is *also* included, under
    :func:`equivariance_variance_output_name`, for
    :class:`~metatrain.utils.loss.EquivariancePenaltyLoss` to consume.

    :param predictions: predictions for a batch of ``n_systems *
        num_augmentations`` systems, already mapped back to the original
        frame (see :meth:`~metatrain.utils.augmentation.O3Augmenter
        .undo_augmentation`).
    :param num_augmentations: number of consecutive replicas per original
        system.
    :param variance_targets: names of the targets to also expose the
        augmentation variance for.
    :return: one mean prediction per entry of ``predictions``, plus the
        variance for each of ``variance_targets``.
    """
    reduced: Dict[str, TensorMap] = {}
    for name, tensor_map in predictions.items():
        mean, variance = reduce_over_augmentations(tensor_map, num_augmentations)
        reduced[name] = mean
        if name in variance_targets:
            reduced[equivariance_variance_output_name(name)] = variance
    return reduced
