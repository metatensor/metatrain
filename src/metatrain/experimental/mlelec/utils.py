from typing import Tuple, Optional
import torch

import metatensor.torch as mts
from metatensor.torch import Labels, TensorBlock


def split_node_edge_targets(tensor):
    node_blocks = []
    edge_blocks = []
    node_keys = []
    edge_keys = []
    for k, b in tensor.items():
        if k["s2_pi"] == 0 and k["first_atom_type"] == k["second_atom_type"]:
            node_blocks.append(
                mts.TensorBlock(
                    samples=mts.Labels(
                        ["system", "atom"],
                        torch.tensor(b.samples.view(["system", "first_atom"])),
                    ),
                    components=b.components,
                    properties=b.properties,
                    values=b.values,
                )
            )
            node_keys.append(k.values[:3])
        else:
            edge_blocks.append(b)
            edge_keys.append(k.values)
    node = mts.TensorMap(
        mts.Labels(["o3_lambda", "o3_sigma", "center_type"], torch.stack(node_keys)),
        node_blocks,
    )
    edge = mts.TensorMap(
        mts.Labels(
            [
                "o3_lambda",
                "o3_sigma",
                "s2_pi",
                "first_atom_type",
                "second_atom_type",
            ],
            torch.stack(edge_keys),
        ),
        edge_blocks,
    )
    return node, edge


def symmetrize_samples(
    block: TensorBlock, second_atom_type: Optional[int] = None
) -> Tuple[TensorBlock]:
    """
    Symmetrizes the samples dimension of a tensor block.
    """

    # Define the samples
    all_samples = block.samples
    sample_names = all_samples.names

    # Permute the samples to get the negative samples
    permuted_samples = all_samples.permute([0, 2, 1, 3, 4, 5]).values.clone()
    permuted_samples[:, -3:] *= -1
    values = block.values  # .clone() # TODO: is this to be cloned?

    # Find the indices of the samples to symmetrize
    idx_to_symmetrize = all_samples.select(
        Labels(
            names=sample_names,
            values=permuted_samples,
        )
    )

    # Symmetrize the sample values
    values_plus = values + values[idx_to_symmetrize]
    values_minus = values - values[idx_to_symmetrize]

    reduced_samples_mask = (
        (all_samples.values[:, 1] < all_samples.values[:, 2])
        & torch.isclose(
            torch.linalg.norm(1.0 * all_samples.values[:, -3:]), torch.tensor(0.0)
        )
    ) | (
        (all_samples.values[:, 1] <= all_samples.values[:, 2])
        & (
            ~torch.isclose(
                torch.linalg.norm(1.0 * all_samples.values[:, -3:]), torch.tensor(0.0)
            )
        )
    )
    reduced_samples = Labels(
        sample_names,
        all_samples.values[reduced_samples_mask],
    )
    values_plus = values_plus[reduced_samples_mask]
    values_minus = values_minus[reduced_samples_mask]

    if second_atom_type is not None:
        properties = block.properties.insert(
            1,
            "neighbor_2_type",
            torch.tensor(block.properties.values.shape[0] * [second_atom_type]),
        )
    else:
        properties = block.properties

    block_plus = TensorBlock(
        samples=reduced_samples,
        values=values_plus,
        components=block.components,
        properties=properties,
    )
    block_minus = TensorBlock(
        samples=reduced_samples,
        values=values_minus,
        components=block.components,
        properties=properties,
    )

    return block_plus, block_minus
