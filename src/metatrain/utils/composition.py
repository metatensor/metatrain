from typing import Dict, List, Optional, Union

import torch
from metatensor.torch import Labels, TensorBlock, TensorMap
from metatensor.torch.atomistic import ModelOutput, System
from metatensor.torch.operations import sum_over_samples_block

from .data import Dataset, DatasetInfo, get_all_targets, get_atomic_types
from .jsonschema import validate


class CompositionModel(torch.nn.Module):
    """A simple model that calculates the energy based on the stoichiometry in a system.

    :param model_hypers: A dictionary of model hyperparameters. The paramater is ignored
        and is only present to be consistent with the general model API.
    :param dataset_info: An object containing information about the dataset, including
        target quantities and atomic types.

    :raises ValueError: If any target quantity in the dataset info is not an energy-like
        quantity.
    """

    def __init__(self, model_hypers: Dict, dataset_info: DatasetInfo):
        super().__init__()

        # `model_hypers` should be an empty dictionary
        validate(
            instance=model_hypers,
            schema={"type": "object", "additionalProperties": False},
        )

        # Check capabilities
        for target in dataset_info.targets.values():
            if target.quantity != "energy":
                raise ValueError(
                    "CompositionModel only supports energy-like outputs, but a "
                    f"{target.quantity} output was provided."
                )

        self.dataset_info = dataset_info
        self.atomic_types = sorted(dataset_info.atomic_types)

        n_types = len(self.atomic_types)
        n_targets = len(dataset_info.targets)

        self.output_to_output_index = {
            target: i for i, target in enumerate(sorted(dataset_info.targets.keys()))
        }

        self.register_buffer(
            "weights", torch.zeros((n_targets, n_types), dtype=torch.float64)
        )

    def train_model(
        self,
        datasets: List[Union[Dataset, torch.utils.data.Subset]],
        fixed_weights: Optional[Dict[str, Dict[int, str]]] = None,
    ) -> None:
        """Train/fit the composition weights for the datasets.

        :param datasets: Dataset(s) to calculate the composition weights for.
        :param fixed_weights: Optional fixed weights to use for the composition model,
            for one or more target quantities.

        :raises ValueError: If the provided datasets contain unknown atomic types.
        :raises RuntimeError: If the linear system to calculate the composition weights
            cannot be solved.
        """
        if not isinstance(datasets, list):
            datasets = [datasets]

        if fixed_weights is None:
            fixed_weights = {}

        missing_types = sorted(get_atomic_types(datasets) - set(self.atomic_types))
        if missing_types:
            raise ValueError(
                f"Provided `datasets` contains unknown atomic types {missing_types}. "
                f"Known types from initilaization are {self.atomic_types}."
            )

        # Fill the weights for each target in the dataset info
        for target_key in self.output_to_output_index.keys():

            if target_key in fixed_weights:
                # The fixed weights are provided for this target. Use them:
                if not sorted(fixed_weights[target_key].keys()) == self.atomic_types:
                    raise ValueError(
                        f"Fixed weights for target {target_key} must contain all "
                        f"atomic types {self.atomic_types}."
                    )

                self.weights[self.output_to_output_index[target_key]] = torch.tensor(
                    [fixed_weights[target_key][i] for i in self.atomic_types],
                    dtype=self.weights.dtype,
                )
            else:
                datasets_with_target = []
                for dataset in datasets:
                    if target_key in get_all_targets(dataset):
                        datasets_with_target.append(dataset)
                if len(datasets_with_target) == 0:
                    raise ValueError(
                        f"Target {target_key} in the model's new capabilities is not "
                        "present in any of the training datasets."
                    )

                targets = torch.stack(
                    [
                        sample[target_key].block().values
                        for dataset in datasets_with_target
                        for sample in dataset
                    ]
                )

                # remove component and property dimensions
                targets = targets.squeeze(dim=(1, 2))

                total_num_structures = sum(
                    [len(dataset) for dataset in datasets_with_target]
                )
                dtype = datasets[0][0]["system"].positions.dtype
                if dtype != torch.float64:
                    raise ValueError(
                        "The composition model only supports float64 during training."
                        f"Got dtype: {dtype}."
                    )

                composition_features = torch.zeros(
                    (total_num_structures, len(self.atomic_types)), dtype=dtype
                )
                structure_index = 0
                for dataset in datasets:
                    for sample in dataset:
                        structure = sample["system"]
                        for j, t in enumerate(self.atomic_types):
                            composition_features[structure_index, j] = torch.sum(
                                structure.types == t
                            )
                        structure_index += 1

                regularizer = 1e-20
                while regularizer:
                    if regularizer > 1e5:
                        raise RuntimeError(
                            "Failed to solve the linear system to calculate the "
                            "composition weights. The dataset is probably too small or "
                            "ill-conditioned."
                        )
                    try:
                        self.weights[self.output_to_output_index[target_key]] = (
                            torch.linalg.solve(
                                composition_features.T @ composition_features
                                + regularizer
                                * torch.eye(
                                    composition_features.shape[1],
                                    dtype=composition_features.dtype,
                                    device=composition_features.device,
                                ),
                                composition_features.T @ targets,
                            ).to(self.weights.dtype)
                        )
                        break
                    except torch._C._LinAlgError:
                        regularizer *= 10.0

    def restart(self, dataset_info: DatasetInfo) -> "CompositionModel":
        """Restart the model with a new dataset info.

        :param dataset_info: New dataset information to be used.
        """
        return self({}, self.dataset_info.union(dataset_info))

    def forward(
        self,
        systems: List[System],
        outputs: Dict[str, ModelOutput],
        selected_atoms: Optional[Labels] = None,
    ) -> Dict[str, TensorMap]:
        """Compute the targets for each system based on the composition weights.

        :param systems: List of systems to calculate the energy per atom.
        :param outputs: Dictionary containing the model outputs.
        :param selected_atoms: Optional selection of atoms for which to compute the
            targets.
        :returns: A dictionary with the computed targets for each system.

        :raises ValueError: If no weights have been computed or if `outputs` keys
            contain unsupported keys.
        :raises NotImplementedError: If `selected_atoms` is provided (not implemented).
        """
        dtype = systems[0].positions.dtype
        device = systems[0].positions.device

        for output_name in outputs:
            if output_name not in self.output_to_output_index:
                raise ValueError(
                    f"output key {output_name} is not supported by this composition "
                    "model."
                )

        # TODO: implement selected_atoms. This is not a big deal because the composition
        # model won't be a bottleneck, so the unwanted atoms can be filtered later by
        # the model that includes the composition model.
        if selected_atoms is not None:
            raise NotImplementedError("`selected_atoms` is not implemented.")

        # Compute the targets for each system by adding the composition weights times
        # number of atoms per atomic type.
        targets_out: Dict[str, TensorMap] = {}
        for target_key, target in outputs.items():
            weights = self.weights[self.output_to_output_index[target_key]]
            targets_list = []
            sample_values: List[List[int]] = []

            for i_system, system in enumerate(systems):
                targets_single = torch.zeros(len(system), dtype=dtype, device=device)

                for i_type, atomic_type in enumerate(self.atomic_types):
                    targets_single[atomic_type == system.types] = weights[i_type]

                targets_list.append(targets_single)
                sample_values += [[i_system, i_atom] for i_atom in range(len(system))]

            targets = torch.concatenate(targets_list)

            block = TensorBlock(
                values=targets.reshape(-1, 1),
                samples=Labels(
                    ["system", "atom"], torch.tensor(sample_values, device=device)
                ),
                components=[],
                properties=Labels(
                    names=["energy"], values=torch.tensor([[0]], device=device)
                ),
            )

            if not target.per_atom:
                block = sum_over_samples_block(block, sample_names="atom")

            targets_out[target_key] = TensorMap(
                keys=Labels(names=["_"], values=torch.tensor([[0]], device=device)),
                blocks=[block],
            )

        return targets_out
