"""Last-layer features of block-aligned targets.

Architectures expose the last-layer features of a target either shared by all
the target's blocks, as the first block of ``last_layer_feature_size`` features,
or block-aligned: one block per target block, with the target's keys and
components. The feature sizes of block-aligned targets are listed in
``last_layer_feature_sizes`` (target name -> one size per block).
"""

from typing import List

import torch
from metatensor.torch import Labels, TensorBlock, TensorMap


def block_aligned_last_layer_features(
    block_values: List[torch.Tensor],
    samples: Labels,
    keys: Labels,
    components_per_block: List[List[Labels]],
) -> TensorMap:
    """Wrap per-block last-layer features into a block-aligned ``TensorMap``.

    :param block_values: one ``(n_samples, n_components, n_features)`` tensor
        per target block, in the target's block order.
    :param samples: samples labels shared by all feature blocks.
    :param keys: the target's keys.
    :param components_per_block: the target's component labels, per block.
    :return: the block-aligned last-layer features.
    """
    blocks: List[TensorBlock] = []
    for block_index, components in enumerate(components_per_block):
        values = block_values[block_index]
        if len(components) == 0:
            values = values.squeeze(1)
        blocks.append(
            TensorBlock(
                values=values,
                samples=samples,
                components=components,
                properties=Labels(
                    names=["feature"],
                    values=torch.arange(
                        values.shape[-1], device=values.device
                    ).unsqueeze(-1),
                ),
            )
        )
    return TensorMap(keys=keys, blocks=blocks)
