import warnings
from typing import Dict, List, Optional, Union

import metatensor.torch
import torch
from metatensor.torch import Labels, TensorBlock, TensorMap
from metatensor.torch.atomistic import ModelOutput, System

from metatrain.utils.data.target_info import is_auxiliary_output

from ..data import Dataset, DatasetInfo, get_all_targets, get_atomic_types
from ..jsonschema import validate


class CompositionModel(torch.nn.Module):
    """A simple model that calculates the energy based on the stoichiometry in a system.

    :param model_hypers: A dictionary of model hyperparameters. The paramater is ignored
        and is only present to be consistent with the general model API.
    :param dataset_info: An object containing information about the dataset, including
        target quantities and atomic types.

    :raises ValueError: If any target quantity in the dataset info is not an energy-like
        quantity.
    """

    outputs: Dict[str, ModelOutput]
    output_to_output_index: Dict[str, int]

    def __init__(self, model_hypers: Dict, dataset_info: DatasetInfo):
        super().__init__()

        # `model_hypers` should be an empty dictionary
        validate(
            instance=model_hypers,
            schema={"type": "object", "additionalProperties": False},
        )

        self.dataset_info = dataset_info
        self.atomic_types = sorted(dataset_info.atomic_types)

        self.outputs = {
            key: ModelOutput(
                quantity=target_info.quantity,
                unit=target_info.unit,
                per_atom=True,
            )
            for key, target_info in dataset_info.targets.items()
            if target_info.is_scalar and len(target_info.layout.block().properties) == 1
            # important: only scalars can have composition contributions
            # for now, we also require that only one property is present
        }

        n_types = len(self.atomic_types)
        n_targets = len(self.outputs)

        self.output_to_output_index = {
            target: i
            for i, target in enumerate(sorted(dataset_info.targets.keys()))
            if target in self.outputs
        }

        self.register_buffer(
            "weights", torch.zeros((n_targets, n_types), dtype=torch.float64)
        )

        # cache some labels
        self.keys_label = Labels.single()
        self.properties_label = Labels(names=["energy"], values=torch.tensor([[0]]))

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

        additional_types = sorted(
            set(get_atomic_types(datasets)) - set(self.atomic_types)
        )
        if additional_types:
            raise ValueError(
                "Provided `datasets` contains unknown "
                f"atomic types {additional_types}. "
                f"Known types from initialization are {self.atomic_types}."
            )

        missing_types = sorted(set(self.atomic_types) - set(get_atomic_types(datasets)))
        if missing_types:
            warnings.warn(
                f"Provided `datasets` do not contain atomic types {missing_types}. "
                f"Known types from initialization are {self.atomic_types}.",
                stacklevel=2,
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
                    # this is a possibility when transfer learning
                    warnings.warn(
                        f"Target {target_key} in the model's new capabilities is not "
                        "present in any of the training datasets.",
                        stacklevel=2,
                    )
                    continue

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
                        "The composition model only supports float64 during training. "
                        f"Got dtype: {dtype}."
                    )

                composition_features = torch.zeros(
                    (total_num_structures, len(self.atomic_types)), dtype=dtype
                )
                structure_index = 0
                for dataset in datasets_with_target:
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

        :param systems: List of systems to calculate the energy.
        :param outputs: Dictionary containing the model outputs.
        :param selected_atoms: Optional selection of atoms for which to compute the
            predictions.
        :returns: A dictionary with the computed predictions for each system.

        :raises ValueError: If no weights have been computed or if `outputs` keys
            contain unsupported keys.
        """
        dtype = systems[0].positions.dtype
        device = systems[0].positions.device

        # move labels to device (Labels can't be treated as buffers for now)
        if self.keys_label.device != device:
            self.keys_label = self.keys_label.to(device)
        if self.properties_label.values.device != device:
            self.properties_label = self.properties_label.to(device)

        for output_name in outputs:
            # TODO: special case for ensembles
            if is_auxiliary_output(output_name):
                continue  # skip auxiliary outputs
            if output_name not in self.output_to_output_index:
                raise ValueError(
                    f"output key {output_name} is not supported by this composition "
                    "model."
                )

        # Note: atomic types are not checked. At training time, the composition model
        # is initialized with the correct types. At inference time, the checks are
        # performed by MetatensorAtomisticModel.

        # Compute the targets for each system by adding the composition weights times
        # number of atoms per atomic type.
        targets_out: Dict[str, TensorMap] = {}
        for target_key, target in outputs.items():
            if is_auxiliary_output(target_key):
                # TODO: special case for ensembles
                continue  # skip auxiliary outputs
            if target_key not in self.outputs.keys():
                # non-scalar
                continue
            weights = self.weights[self.output_to_output_index[target_key]]

            concatenated_types = torch.concatenate([system.types for system in systems])
            targets = torch.empty(len(concatenated_types), dtype=dtype, device=device)
            for i_type, atomic_type in enumerate(self.atomic_types):
                targets[concatenated_types == atomic_type] = weights[i_type]

            # create sample labels
            sample_values_list = []
            for i_system, system in enumerate(systems):
                system_column = torch.full(
                    (len(system),), i_system, dtype=torch.int, device=device
                )
                atom_column = torch.arange(len(system), device=device)
                samples_values_single_system = torch.stack(
                    [system_column, atom_column], dim=1
                )
                sample_values_list.append(samples_values_single_system)
            sample_values = torch.concatenate(sample_values_list)

            block = TensorBlock(
                values=targets.reshape(-1, 1),
                samples=Labels(["system", "atom"], sample_values),
                components=[],
                properties=self.properties_label,
            )

            targets_out[target_key] = TensorMap(
                keys=self.keys_label,
                blocks=[block],
            )

            # apply selected_atoms to the composition if needed
            if selected_atoms is not None:
                targets_out[target_key] = metatensor.torch.slice(
                    targets_out[target_key], "samples", selected_atoms
                )

            if not target.per_atom:
                targets_out[target_key] = metatensor.torch.sum_over_samples(
                    targets_out[target_key], sample_names="atom"
                )

        return targets_out
