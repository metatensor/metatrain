from typing import List, Tuple, Union

import torch
from metatensor.torch import Labels, TensorBlock, TensorMap

from metatensor.models.utils.data import Dataset, get_atomic_types


def calculate_composition_weights(
    datasets: Union[Dataset, List[Dataset]], property: str
) -> Tuple[torch.Tensor, List[int]]:
    """Calculate the composition weights for a dataset.

    For now, it assumes per-system properties.

    :param dataset: Dataset to calculate the composition weights for.
    :returns: Composition weights for the dataset, as well as the
        list of species that the weights correspond to.
    """
    if not isinstance(datasets, list):
        datasets = [datasets]

    # Note: `atomic_types` are sorted, and the composition weights are sorted as
    # well, because the species are sorted in the composition features.
    atomic_types = sorted(get_atomic_types(datasets))

    targets = torch.stack(
        [sample[property].block().values for dataset in datasets for sample in dataset]
    )
    targets = targets.squeeze(dim=(1, 2))  # remove component and property dimensions

    structure_list = [sample["system"] for dataset in datasets for sample in dataset]

    dtype = structure_list[0].positions.dtype
    composition_features = torch.empty(
        (len(structure_list), len(atomic_types)), dtype=dtype
    )
    for i, structure in enumerate(structure_list):
        for j, s in enumerate(atomic_types):
            composition_features[i, j] = torch.sum(structure.types == s)

    regularizer = 1e-20
    while regularizer:
        if regularizer > 1e5:
            raise RuntimeError(
                "Failed to solve the linear system to calculate the "
                "composition weights. The dataset is probably too small "
                "or ill-conditioned."
            )
        try:
            solution = torch.linalg.solve(
                composition_features.T @ composition_features
                + regularizer
                * torch.eye(
                    composition_features.shape[1],
                    dtype=composition_features.dtype,
                    device=composition_features.device,
                ),
                composition_features.T @ targets,
            )
            break
        except torch._C._LinAlgError:
            regularizer *= 10.0

    return solution, atomic_types


def apply_composition_contribution(
    atomic_property: TensorMap, composition_weights: torch.Tensor
) -> TensorMap:
    """Apply the composition contribution to an atomic property.

    :param atomic_property: Atomic property to apply the composition contribution to.
    :param composition_weights: Composition weights to apply.
    :returns: Atomic property with the composition contribution applied.
    """

    new_keys: List[int] = []
    new_blocks: List[TensorBlock] = []
    for key, block in atomic_property.items():
        atomic_type = int(key.values.item())
        new_keys.append(atomic_type)
        new_values = block.values + composition_weights[atomic_type]
        new_blocks.append(
            TensorBlock(
                values=new_values,
                samples=block.samples,
                components=block.components,
                properties=block.properties,
            )
        )

    new_keys_labels = Labels(
        names=["center_type"],
        values=torch.tensor(new_keys, device=new_blocks[0].values.device).reshape(
            -1, 1
        ),
    )

    return TensorMap(keys=new_keys_labels, blocks=new_blocks)
