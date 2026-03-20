"""Utilities for attaching per-system data to :class:`System` objects."""

from typing import Callable, Dict, List, Tuple

import torch
from metatensor.torch import Labels, TensorBlock, TensorMap
from metatomic.torch import System


def get_system_data_transform(
    data_keys: List[str],
) -> Callable[
    [List[System], Dict[str, TensorMap], Dict[str, TensorMap]],
    Tuple[List[System], Dict[str, TensorMap], Dict[str, TensorMap]],
]:
    """Return a :class:`CollateFn` callable that moves per-system scalar data
    from the ``extra`` dict into each :class:`System` via ``add_data``.

    After ``group_and_join`` the extra dict contains batched
    :class:`TensorMap` objects keyed by the requested names (e.g.
    ``"mtt::charge"``, ``"mtt::spin"``), with the ``"system"`` sample
    dimension indexed 0, 1, … matching the position of each system in the
    batch list.  This callable re-attaches those per-system scalars to the
    :class:`System` objects so that models can read them with
    ``system.get_data(key)``.

    NaN values are treated as missing: the corresponding system will not have
    the data key attached and the model will fall back to its own default.

    :param data_keys: List of extra_data keys to route from the ``extra``
        dict into each system, e.g. ``["mtt::charge", "mtt::spin"]``.
    :return: A three-argument callable
        ``(systems, targets, extra) -> (systems, targets, extra)``.
    """

    def transform(
        systems: List[System],
        targets: Dict[str, TensorMap],
        extra: Dict[str, TensorMap],
    ) -> Tuple[List[System], Dict[str, TensorMap], Dict[str, TensorMap]]:
        for key in data_keys:
            if key not in extra:
                continue
            prop_name = key.split("::")[-1]
            block = extra[key].block()
            if block.samples.names != ["system"]:
                raise ValueError(
                    f"get_system_data_transform expects per-system TensorMaps "
                    f"(samples=['system']), got samples={block.samples.names} "
                    f"for key '{key}'. Per-atom extra data cannot be attached "
                    "to System objects this way."
                )
            for row_idx in range(len(block.samples)):
                val = block.values[row_idx : row_idx + 1]  # shape [1, n_props]
                if torch.isnan(val).any():
                    continue
                systems[row_idx].add_data(
                    key,
                    TensorMap(
                        keys=Labels.single(),
                        blocks=[
                            TensorBlock(
                                values=val,
                                samples=Labels(
                                    "system",
                                    torch.tensor(
                                        [[row_idx]],
                                        device=val.device,
                                        dtype=torch.int32,
                                    ),
                                ),
                                components=[],
                                properties=Labels.range(prop_name, val.shape[-1]),
                            )
                        ],
                    ),
                )
        return systems, targets, extra

    return transform
