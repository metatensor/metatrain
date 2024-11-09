from typing import List, Tuple, Union

import torch

from ....utils.data.dataset import Dataset, get_atomic_types


def calculate_composition_weights(
    datasets: Union[Dataset, List[Dataset]], property: str
) -> Tuple[torch.Tensor, List[int]]:
    """Calculate the composition weights for a dataset.

    It assumes per-system properties.

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

    total_num_structures = sum([len(dataset) for dataset in datasets])
    dtype = datasets[0][0]["system"].positions.dtype
    composition_features = torch.empty(
        (total_num_structures, len(atomic_types)), dtype=dtype
    )
    structure_index = 0
    for dataset in datasets:
        for sample in dataset:
            structure = sample["system"]
            for j, s in enumerate(atomic_types):
                composition_features[structure_index, j] = torch.sum(
                    structure.types == s
                )
            structure_index += 1

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
