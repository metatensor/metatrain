from typing import List, Tuple

import rascaline.torch
import torch
from metatensor.learn.data.dataset import _BaseDataset
from metatensor.torch import Labels, TensorBlock, TensorMap

from metatensor.models.utils.data import get_all_species


def calculate_composition_weights(
    datasets: _BaseDataset, property: str
) -> Tuple[torch.Tensor, List[int]]:
    """Calculate the composition weights for a dataset.

    For now, it assumes per-system properties.

    :param dataset: Dataset to calculate the composition weights for.
    :returns: Composition weights for the dataset, as well as the
        list of species that the weights correspond to.
    """

    # Get the target for each system in the dataset
    # TODO: the dataset will be iterable once metatensor PR #500 merged.
    targets = torch.stack(
        [
            dataset[sample_id]._asdict()[property].block().values
            for dataset in datasets
            for sample_id in range(len(dataset))
        ]
    )

    # Get the composition for each system in the dataset
    composition_calculator = rascaline.torch.AtomicComposition(per_structure=True)
    # TODO: the dataset will be iterable once metatensor PR #500 merged.
    composition_features = composition_calculator.compute(
        [
            dataset[sample_id]._asdict()["system"]
            for dataset in datasets
            for sample_id in range(len(dataset))
        ]
    )
    composition_features = composition_features.keys_to_properties("species_center")
    composition_features = composition_features.block().values

    targets = targets.squeeze(dim=(1, 2))  # remove component and property dimensions

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

    species = get_all_species(datasets)
    # note that this is sorted, and the composition weights are sorted
    # as well, because the species are sorted in the composition features

    return solution, species


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
        atomic_species = int(key.values.item())
        new_keys.append(atomic_species)
        new_values = block.values + composition_weights[atomic_species]
        new_blocks.append(
            TensorBlock(
                values=new_values,
                samples=block.samples,
                components=block.components,
                properties=block.properties,
            )
        )

    new_keys_labels = Labels(
        names=["species_center"],
        values=torch.tensor(new_keys, device=new_blocks[0].values.device).reshape(
            -1, 1
        ),
    )

    return TensorMap(keys=new_keys_labels, blocks=new_blocks)
