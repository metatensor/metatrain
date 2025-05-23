import logging
from typing import Dict, List, Optional

import metatensor.torch as mts
import torch
from metatensor.torch import Labels, TensorBlock, TensorMap
from metatensor.torch.atomistic import ModelOutput, System

from ..data import DatasetInfo, TargetInfo
from ..jsonschema import validate
from ..sum_over_atoms import sum_over_atoms


class CompositionModel(torch.nn.Module):
    """Fits for a dict of targets"""

    # def __init__(
    #     self,
    #     atomic_types: List[int],
    #     layouts: Dict[str, TensorMap],
    # ) -> None:
    def __init__(
        self,
        model_hypers: Dict,
        dataset_info: DatasetInfo,
    ) -> None:
        super().__init__()

        # `model_hypers` should be an empty dictionary
        validate(
            instance=model_hypers,
            schema={"type": "object", "additionalProperties": False},
        )

        self.dataset_info = dataset_info
        self.atomic_types = sorted(dataset_info.atomic_types)

        self.register_buffer(
            "type_to_index", torch.empty(max(self.atomic_types) + 1, dtype=torch.long)
        )
        for i, atomic_type in enumerate(self.atomic_types):
            self.type_to_index[atomic_type] = i

        target_names = []
        sample_kinds = {}
        for target_name, target_info in dataset_info.targets.items():
            if not self.is_valid_target(target_name, target_info):
                raise ValueError(
                    f"Composition model does not support target quantity "
                    f"{target_info.quantity}. This is an architecture bug. "
                    "Please report this issue and help us improve!"
                )
            target_names.append(target_name)
            if target_info.layout.sample_names == ["system"]:
                sample_kinds[target_name] = "per_structure"
            elif target_info.layout.sample_names == ["system", "atom"]:
                sample_kinds[target_name] = "per_atom"
            elif target_info.layout.sample_names == [
                "system",
                "first_atom",
                "second_atom",
                "cell_shift_a",
                "cell_shift_b",
                "cell_shift_c",
            ]:
                sample_kinds[target_name] = "per_pair"
            else:
                raise ValueError(
                    f"Composition model does not support target layout "
                    f"{target_info.layout}. This is an architecture bug. "
                    "Please report this issue and help us improve!"
                )

        # target_names = []
        # sample_kinds = {}
        # for target_name, layout in layouts.items():  # identify target_type

        #     target_names.append(target_name)
        #     if layout.sample_names == ["system"]:
        #         sample_kinds[target_name] = "per_structure"

        #     elif layout.sample_names == ["system", "atom"]:
        #         sample_kinds[target_name] = "per_atom"

        #     elif layout.sample_names == [
        #         "system",
        #         "first_atom",
        #         "second_atom",
        #         "cell_shift_a",
        #         "cell_shift_b",
        #         "cell_shift_c",
        #     ]:
        #         sample_kinds[target_name] = "per_pair"

        #     else:
        #         raise ValueError

        self.target_names = target_names
        self.sample_kinds = sample_kinds
        self.XTX: Dict[str, TensorMap] = {
            target_name: TensorMap(
                target_info.layout.keys,
                blocks=[
                    TensorBlock(
                        values=torch.zeros(
                            len(self.atomic_types),
                            len(self.atomic_types),
                            dtype=torch.float64,
                        ),
                        samples=Labels(
                            ["first_atom_type"],
                            torch.tensor(self.atomic_types, dtype=torch.int32).reshape(
                                -1, 1
                            ),
                        ),
                        components=[],
                        properties=Labels(
                            ["second_atom_type"],
                            torch.tensor(self.atomic_types, dtype=torch.int32).reshape(
                                -1, 1
                            ),
                        ),
                    )
                    for _ in target_info.layout
                ],
            )
            for target_name, target_info in dataset_info.targets.items()  # layouts.ite
        }
        self.XTY: Dict[str, TensorMap] = {
            target_name: TensorMap(
                target_info.layout.keys,
                blocks=[
                    TensorBlock(
                        values=torch.zeros(
                            len(self.atomic_types),
                            *[len(c) for c in block.components],
                            len(block.properties),
                            dtype=torch.float64,
                        ),
                        samples=Labels(
                            ["center_type"],
                            torch.tensor(self.atomic_types, dtype=torch.int32).reshape(
                                -1, 1
                            ),
                        ),
                        components=block.components,
                        properties=block.properties,
                    )
                    for block in target_info.layout
                ],
            )
            for target_name, target_info in dataset_info.targets.items()  # layouts.it
        }
        self.weights: Dict[str, TensorMap] = {}

        # keeps track of dtype and device of the composition model
        self.register_buffer("dummy_buffer", torch.randn(1))

    def _accumulate(
        self,
        systems: List[System],
        targets: Dict[str, TensorMap],
    ):
        num_atoms = torch.tensor([len(system) for system in systems])

        for target_name, target in targets.items():
            for key, block in target.items():
                # Get the target block values
                values = block.values

                if self.sample_kinds[target_name] == "per_structure":
                    # For per-structure, divide target values by number of atoms
                    values /= num_atoms.reshape(-1, *values.shape[1:])

                    # Compute X
                    X = self._compute_X_per_structure(systems)

                elif self.sample_kinds[target_name] == "per_atom":
                    # if not (key["o3_lambda"] == 0 and key["o3_sigma"] == 1):
                    #     continue

                    # X needs to be sliced based on atom type
                    if "center_type" in key.names:
                        center_types = [key["center_type"]]

                    else:
                        center_types = self.atomic_types

                    # Compute X
                    X = self._compute_X_per_atom(systems, center_types)

                else:
                    assert self.sample_kinds[target_name] == "per_pair"

                    # TODO: assumes coupled basis, perumtationally symmetrized

                    if "o3_lambda" in key.names:  # coupled
                        if not (key["o3_lambda"] == 0 and key["o3_sigma"] == 1):
                            continue

                    # X needs to be sliced based on atom type
                    if (
                        "first_atom_type" in key.names
                        and "second_atom_type" in key.names
                    ):
                        if key["first_atom_type"] != key["second_atom_type"]:
                            continue

                        if key["s2_pi"] != 0:
                            continue

                        center_types = [key["first_atom_type"]]

                    else:
                        center_types = self.atomic_types

                    # Compute X
                    X = self._compute_X_per_atom(systems, center_types)

                # Compute a sparse XTX
                self.XTX[target_name][key].values[:] += X.T @ X

                # Compute XTY
                if len(values.shape) == 2:
                    XTY = torch.einsum("sZ,sP->ZP", X, values)
                    self.XTY[target_name][key].values[:] += XTY

                else:
                    assert len(values.shape) == 3
                    XTY = torch.einsum("sZ,sCP->ZCP", X, values)
                    self.XTY[target_name][key].values[:] += XTY

    def _compute_X_per_structure(self, systems: List[System]) -> torch.Tensor:
        X = []
        for system in systems:
            X_system = torch.tensor(
                [
                    torch.sum(system.types == atom_type)
                    for atom_type in self.atomic_types
                ],
                dtype=torch.float64,
            )
            X.append(X_system / len(system))

        return torch.vstack(X)

    def _compute_X_per_atom(
        self, systems: List[System], center_types: List[int]
    ) -> torch.Tensor:
        X = []

        # TODO: make this one hot encoding quicker

        column_idx_map = {atom_type: i for i, atom_type in enumerate(self.atomic_types)}

        for system in systems:
            for atom_type in system.types:
                if atom_type in center_types:
                    row = torch.zeros(
                        len(self.atomic_types),
                        dtype=torch.float64,
                    )
                    row[column_idx_map[atom_type.item()]] = 1.0
                    X.append(row)

        return torch.vstack(X)

    def fit(self, dataloader, sigma: float = 0.01):
        device = self.dummy_buffer.device

        # acccumulate
        for batch in dataloader:
            self._accumulate(
                batch.systems,
                {target_name: batch[target_name] for target_name in self.target_names},
            )

        # fit
        for target_name in self.target_names:
            if self.sample_kinds[target_name] == "per_structure":
                blocks = []
                for key in self.XTX[target_name].keys:
                    XTX_block = self.XTX[target_name][key]
                    XTY_block = self.XTY[target_name][key]
                    blocks.append(
                        TensorBlock(
                            values=_solve_linear_system(
                                XTX_block.values, XTY_block.values
                            ),
                            samples=XTY_block.samples,
                            components=XTY_block.components,
                            properties=XTY_block.properties,
                        )
                    )

                self.weights[target_name] = TensorMap(
                    self.XTX[target_name].keys, blocks
                )

            elif self.sample_kinds[target_name] in ["per_atom", "per_pair"]:
                blocks = []
                for key in self.XTX[target_name].keys:
                    XTX_block = self.XTX[target_name][key]
                    XTY_block = self.XTY[target_name][key]

                    XTX_values = XTX_block.values
                    XTY_values = XTY_block.values

                    # TODO: should non-invariant keys be even present?
                    weights_are_zero = False
                    if "o3_lambda" in key.names:
                        if not (key["o3_lambda"] == 0 and key["o3_sigma"] == 1):
                            weights_are_zero = True
                        # Weights are zero for off-site blocks
                        if "second_atom_type" in key.names:
                            if not (
                                key["s2_pi"] == 0
                                and key["first_atom_type"] == key["second_atom_type"]
                            ):
                                weights_are_zero = True

                    if weights_are_zero:
                        weight_block = torch.zeros_like(XTY_values)
                    else:
                        XTY_shape = XTY_values.shape
                        if len(XTY_values.shape) != 2:
                            XTY_values = XTY_values.reshape(XTY_values.shape[0], -1)

                        weight_block = _solve_linear_system(XTX_values, XTY_values)
                        weight_block = weight_block.reshape(*XTY_shape)

                    blocks.append(
                        TensorBlock(
                            values=weight_block,
                            samples=XTY_block.samples,
                            components=XTY_block.components,
                            properties=XTY_block.properties,
                        )
                    )

                self.weights[target_name] = TensorMap(
                    self.XTX[target_name].keys, blocks
                )

            # make sure to update the weights buffer with the new weights
            self.register_buffer(
                target_name + "_composition_buffer",
                mts.save_buffer(self.weights[target_name].to("cpu", torch.float64)).to(
                    device
                ),
            )

    def restart(self, dataset_info: DatasetInfo) -> "CompositionModel":
        """Restart the model with a new dataset info.

        :param dataset_info: New dataset information to be used.
        """
        for target_name, target_info in dataset_info.targets.items():
            if not self.is_valid_target(target_name, target_info):
                raise ValueError(
                    f"Composition model does not support target "
                    f"{target_name}. This is an architecture bug. "
                    "Please report this issue and help us improve!"
                )

        # merge old and new dataset info
        merged_info = self.dataset_info.union(dataset_info)
        new_atomic_types = [
            at for at in merged_info.atomic_types if at not in self.atomic_types
        ]

        if len(new_atomic_types) > 0:
            raise ValueError(
                f"New atomic types found in the dataset: {new_atomic_types}. "
                "The composition model does not support adding new atomic types."
            )

        self.new_targets = {
            target_name: target_info
            for target_name, target_info in merged_info.targets.items()
            if target_name not in self.dataset_info.targets
        }

        self.dataset_info = merged_info

        # register new outputs
        for target_name, target in self.new_targets.items():
            self._add_output(target_name, target)

        return self

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

        # move weights (TensorMaps can't be treated as buffers for now)
        self._move_weights_to_device_and_dtype(device, dtype)

        for output_name in outputs:
            if output_name not in self.weights:
                raise ValueError(
                    f"output key {output_name} is not supported by this composition "
                    "model."
                )

        # Note: atomic types are not checked. At training time, the composition model
        # is initialized with the correct types. At inference time, the checks are
        # performed by MetatensorAtomisticModel.

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
        sample_labels = Labels(["system", "atom"], sample_values)

        # concatenate all types for all structures
        concatenated_types = torch.concatenate([system.types for system in systems])

        # Compute the output for each system by adding the composition weights times
        # number of atoms per atomic type.
        composition_result_dict: Dict[str, TensorMap] = {}
        for output_name, output_options in outputs.items():
            blocks: List[TensorBlock] = []
            for weight_key, weight_block in self.weights[output_name].items():
                weights_tensor = self.weights[output_name].block(weight_key).values
                composition_values_per_atom = weights_tensor[
                    self.type_to_index[concatenated_types]
                ]
                blocks.append(
                    TensorBlock(
                        values=composition_values_per_atom,
                        samples=sample_labels,
                        components=weight_block.components,
                        properties=weight_block.properties,
                    )
                )
            composition_result_dict[output_name] = TensorMap(
                keys=self.weights[output_name].keys,
                blocks=blocks,
            )

            # apply selected_atoms to the composition if needed
            if selected_atoms is not None:
                composition_result_dict[output_name] = mts.slice(
                    composition_result_dict[output_name], "samples", selected_atoms
                )

            if not output_options.per_atom:  # sum over atoms if needed
                composition_result_dict[output_name] = sum_over_atoms(
                    composition_result_dict[output_name]
                )

        return composition_result_dict

    def _add_output(self, target_name: str, target_info: TargetInfo) -> None:
        self.outputs[target_name] = ModelOutput(
            quantity=target_info.quantity,
            unit=target_info.unit,
            per_atom=True,
        )
        self.weights[target_name] = TensorMap(
            keys=target_info.layout.keys,
            blocks=[
                TensorBlock(
                    values=torch.zeros(
                        ([len(self.atomic_types)] + b.shape[1:]),
                        dtype=torch.float64,
                    ),
                    samples=Labels(
                        names=["center_type"],
                        values=torch.tensor(self.atomic_types, dtype=torch.int).reshape(
                            -1, 1
                        ),
                    ),
                    components=b.components,
                    properties=b.properties,
                )
                for b in target_info.layout.blocks()
            ],
        )

        # register a buffer to store the weights; this is necessary because the weights
        # are TensorMaps and cannot be stored in the state_dict
        fake_weights = TensorMap(
            keys=self.dataset_info.targets[target_name].layout.keys,
            blocks=[
                TensorBlock(
                    values=torch.zeros(
                        (len(self.atomic_types),) + b.values.shape[1:],
                        dtype=torch.float64,
                    ),
                    samples=Labels(
                        names=["center_type"],
                        values=torch.tensor(self.atomic_types, dtype=torch.int).reshape(
                            -1, 1
                        ),
                    ),
                    components=b.components,
                    properties=b.properties,
                )
                for b in target_info.layout.blocks()
            ],
        )
        self.register_buffer(
            target_name + "_composition_buffer",
            mts.save_buffer(fake_weights),
        )

    def _move_weights_to_device_and_dtype(
        self, device: torch.device, dtype: torch.dtype
    ):
        if len(self.weights) != 0:
            if self.weights[list(self.weights.keys())[0]].device != device:
                self.weights = {k: v.to(device) for k, v in self.weights.items()}
            if self.weights[list(self.weights.keys())[0]].dtype != dtype:
                self.weights = {k: v.to(dtype) for k, v in self.weights.items()}

    @staticmethod
    def is_valid_target(target_name: str, target_info: TargetInfo) -> bool:
        """Finds if a ``TargetInfo`` object is compatible with a composition model.

        :param target_info: The ``TargetInfo`` object to be checked.
        """
        # only scalars can have composition contributions
        if not target_info.is_scalar and not target_info.is_spherical:
            logging.debug(
                f"Composition model does not support target {target_name} "
                "since it is not either scalar or spherical."
            )
            return False
        if (
            target_info.is_spherical
            and len(target_info.layout.blocks({"o3_lambda": 0, "o3_sigma": 1})) == 0
        ):
            logging.debug(
                f"Composition model does not support spherical target {target_name} "
                "since it does not have any invariant blocks."
            )
            return False
        return True

    def sync_tensor_maps(self):
        # Reload the weights of the (old) targets, which are not stored in the model
        # state_dict, from the buffers
        for k in self.dataset_info.targets:
            self.weights[k] = mts.load_buffer(
                self.__getattr__(k + "_composition_buffer")
            )


def _solve_linear_system(compf_t_at_compf, compf_t_at_targets) -> torch.Tensor:
    trace_magnitude = float(torch.diag(compf_t_at_compf).abs().mean())
    regularizer = 1e-14 * trace_magnitude
    return torch.linalg.solve(
        compf_t_at_compf
        + regularizer
        * torch.eye(
            compf_t_at_compf.shape[1],
            dtype=compf_t_at_compf.dtype,
            device=compf_t_at_compf.device,
        ),
        compf_t_at_targets,
    )
