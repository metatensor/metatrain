from typing import Any, Optional

import torch
from metatensor.torch import Labels, TensorBlock, TensorMap
from metatomic.torch import ModelOutput, System

from metatrain.soap_bpnn.documentation import SOAPConfig
from metatrain.soap_bpnn.modules.tensor_basis import TensorBasis as TensorBasisModule
from metatrain.utils.data import DatasetInfo, TargetInfo
from metatrain.utils.data.target_info import get_generic_target_info
from metatrain.utils.hypers import init_with_defaults
from metatrain.utils.sum_over_atoms import sum_over_atoms

from ..abc import HookInterface
from .documentation import Hypers


def concatenate_structures(
    systems: list[System],
) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
    """
    Concatenate a list of systems into a single batch.

    :param systems: List of systems to concatenate.
    :return: A tuple containing the concatenated positions, centers, neighbors,
        species, cells, and cell shifts.
    """
    positions = []
    centers = []
    neighbors = []
    species = []
    cell_shifts = []
    cells = []
    node_counter = 0

    for system in systems:
        positions.append(system.positions)
        species.append(system.types)

        neighbor_list = system.get_neighbor_list(system.known_neighbor_lists()[0])
        nl_values = neighbor_list.samples.values

        centers.append(nl_values[:, 0] + node_counter)
        neighbors.append(nl_values[:, 1] + node_counter)
        cell_shifts.append(nl_values[:, 2:])

        cells.append(system.cell)

        node_counter += len(system.positions)

    positions = torch.cat(positions)
    centers = torch.cat(centers)
    neighbors = torch.cat(neighbors)
    species = torch.cat(species)
    cells = torch.stack(cells)
    cell_shifts = torch.cat(cell_shifts)

    return (
        positions,
        centers,
        neighbors,
        species,
        cells,
        cell_shifts,
    )


class TensorBasis(HookInterface[Hypers]):
    """
    Provides a tensor basis in which to predict spherical tensor targets.

    :param hypers: A dictionary with the hook's hyper-parameters.
    :param dataset_info: Information containing details about the dataset, such as
        target quantities and atomic types.
    """

    __checkpoint_version__ = 1

    def __init__(self, hypers: Hypers, dataset_info: DatasetInfo):
        super().__init__(hypers, dataset_info)

        self.hypers = hypers

        # Helper to map from atomic number to the index of that atomic
        # number in the list of atomic types.
        species_to_species_index = torch.empty(
            max(dataset_info.atomic_types) + 1, dtype=torch.long
        )
        species_to_species_index[dataset_info.atomic_types] = torch.arange(
            len(dataset_info.atomic_types)
        )
        self.register_buffer("species_to_species_index", species_to_species_index)

        # Get the information about the output targets from the dataset info
        outputs = hypers["outputs"]
        if isinstance(outputs, str):
            outputs = [outputs]
        if outputs is None:
            raise ValueError("TensorBasis hook requires at least one output target")
        self.out_targets = {name: dataset_info.targets[name] for name in outputs}

        # Names for the inputs that we will request from the model
        inputs = hypers["inputs"]
        if isinstance(inputs, str):
            inputs = [inputs]
        if inputs is None:
            inputs = [
                f"mtt::aux::scalars::{name.replace('mtt::', '')}" for name in outputs
            ]
        if len(inputs) != len(outputs):
            raise ValueError(
                "TensorBasis hook requires the same number of input and output targets"
            )
        self._input_names = inputs

        self._out_to_input_name = {
            out_name: in_name
            for out_name, in_name in zip(
                self.out_targets, self._input_names, strict=True
            )
        }

        # Build the basis calculators for each target,
        # and the output that we have to request from the model
        soap_hypers = hypers.get("soap", init_with_defaults(SOAPConfig))
        self.basis_calculators = torch.nn.ModuleDict({})
        self._input_target_infos = {}
        for input_name, target_name in zip(
            self._input_names, self.out_targets, strict=True
        ):
            target = self.out_targets[target_name]
            # Get one basis calculator for each block of the target, since each block
            # has different o3_lambda and o3_sigma values.
            self.basis_calculators[target_name] = torch.nn.ModuleList(
                [
                    TensorBasisModule(
                        dataset_info.atomic_types,
                        soap_hypers,
                        o3_lambda=key["o3_lambda"],
                        o3_sigma=key["o3_sigma"],
                        add_lambda_basis=True,
                        legacy=False,
                    )
                    for key in target.layout.keys
                ]
            )

            # Build the input that we will request from the model.
            # We will ask for invariant coefficients. For each block we ask for
            # 2l+1 coefficients for each property, since the basis will have
            # 2l+1 tensors.
            # We ask for all the coefficients in a single block, we will untangle
            # them in the forward pass.
            num_properties = sum(
                block.values.shape[1] * block.values.shape[2] * (2 if i > 1 else 1)
                for i, block in enumerate(target.layout.blocks())
            )
            input_target_info = get_generic_target_info(
                input_name,
                {
                    "quantity": "_",
                    "unit": "",
                    "type": {
                        "spherical": {"irreps": [{"o3_lambda": 0, "o3_sigma": 1}]}
                    },
                    "num_subtargets": num_properties,
                    "sample_kind": "atom",
                },
            )

            # If the input is in the dataset, check if it is compatible with the
            # input that we expect. If so, use it.
            if (
                input_name in dataset_info.targets
                or input_name in dataset_info.extra_data
            ):
                # The input is present in the dataset info, check if
                # the target that we built is compatible with the one
                # in the dataset info. If it is, use it as our input
                target_info = (
                    dataset_info.targets.get(input_name)
                    or dataset_info.extra_data[input_name]
                )

                try:
                    _check_compatible_target_infos(target_info, input_target_info)
                except AssertionError:
                    raise ValueError(
                        f"TensorBasis hook requested input {input_name} but we found"
                        f" '{input_name}' in the dataset and it is not compatible as an"
                        f" input for the hook. "
                    )

                self._input_target_infos[input_name] = target_info
            else:
                self._input_target_infos[input_name] = input_target_info

    def requested_target_infos(self) -> dict[str, TargetInfo]:
        """
        Returns the list of requested target infos for the hook.

        :return: A list of requested target names.
        """
        return self._input_target_infos

    def requested_hook_inputs(
        self, outputs: dict[str, ModelOutput]
    ) -> dict[str, ModelOutput]:
        """
        Returns the list of requested inputs for the hook.

        :param outputs: Dictionary of requested outputs. These can contain outputs that
            are handled by other hooks or the main model. In that case, the hook
            ignores those outputs.
        :return: A list of requested input names.
        """
        req_inputs: dict[str, ModelOutput] = {}
        for out_name in outputs:
            if out_name in self._out_to_input_name:
                input_name = self._out_to_input_name[out_name]
                target_info = self._input_target_infos[input_name]
                req_inputs[input_name] = ModelOutput(
                    quantity=target_info.quantity,
                    unit=target_info.unit,
                    sample_kind="atom",
                )
        return req_inputs

    def supported_outputs(self) -> dict[str, TargetInfo]:
        """
        Returns the supported outputs for the hook.

        :return: A list of supported output names.
        """
        return self.out_targets

    def forward(
        self,
        systems: list[System],
        outputs: dict[str, ModelOutput],
        inputs: dict[str, TensorMap],
        selected_atoms: Optional[Labels] = None,
    ) -> dict[str, TensorMap]:
        requested_outs: list[str] = []
        for out_name in self.out_targets:
            if out_name in outputs:
                requested_outs.append(out_name)
        if not requested_outs:
            # Quick exit if no output of this hook is requested
            return {}

        device = systems[0].positions.device

        # -------------------------------
        #   Get structure information
        # -------------------------------

        system_sizes = [len(system) for system in systems]
        system_sizes_tensor = torch.tensor(system_sizes, device=device)
        system_indices = torch.repeat_interleave(
            torch.arange(len(systems), device=device), system_sizes_tensor
        )
        atom_indices = torch.cat(
            [torch.arange(size, device=device) for size in system_sizes]
        )
        sample_values = torch.stack([system_indices, atom_indices], dim=1)

        (
            positions,
            centers,
            neighbors,
            species,
            cells,
            cell_shifts,
        ) = concatenate_structures(systems)
        species = self.species_to_species_index[species]

        # somehow the backward of this operation is very slow at evaluation,
        # where there is only one cell, therefore we simplify the calculation
        # for that case
        if len(cells) == 1:
            cell_contributions = cell_shifts.to(cells.dtype) @ cells[0]
        else:
            cell_contributions = torch.einsum(
                "ab, abc -> ac",
                cell_shifts.to(cells.dtype),
                cells[system_indices][centers],
            )

        interatomic_vectors = (
            positions[neighbors] - positions[centers] + cell_contributions
        )

        # ------------------------------------
        #   Build the values for each target
        # ------------------------------------

        return_dict: dict[str, TensorMap] = {}
        for target_name, basis_calculators in self.basis_calculators.items():
            if target_name in requested_outs:
                target_info = self.out_targets[target_name].to(device)
                input_name = self._out_to_input_name[target_name]
                target_invariant_coefficients = inputs[input_name].block().values

                offset = 0
                blocks: list[TensorBlock] = []
                for i, basis_calculator in enumerate(basis_calculators):
                    layout_block = target_info.layout.block(i)

                    # Get shapes of the invariant coefficients to retrieve
                    # for this block.
                    n_properties = layout_block.properties.values.shape[0]
                    n_basis = layout_block.values.shape[1] * (2 if i > 1 else 1)
                    count = n_properties * n_basis

                    # Get those invariant coefficients
                    invariant_coefficients = target_invariant_coefficients[
                        :, 0, offset : offset + count
                    ].reshape(-1, n_properties, n_basis)
                    # Update counter for the next block
                    offset += count

                    # Now get the tensor basis.
                    tensor_basis = basis_calculator(
                        interatomic_vectors,
                        centers,
                        neighbors,
                        species,
                        sample_values,
                        selected_atoms=None,
                    )

                    # Multiply the invariant coefficients by the tensor basis
                    # to get the final values for each atom.
                    atomic_property_tensor = torch.einsum(
                        "spb, scb -> scp",
                        invariant_coefficients,
                        tensor_basis,
                    )

                    # Build the tensor block.
                    blocks.append(
                        TensorBlock(
                            values=atomic_property_tensor,
                            samples=Labels(
                                names=["system", "atom"], values=sample_values
                            ),
                            components=layout_block.components,
                            properties=layout_block.properties,
                        )
                    )

                tmap = TensorMap(
                    keys=target_info.layout.keys,
                    blocks=blocks,
                )

                if target_info.sample_kind == "system":
                    tmap = sum_over_atoms(tmap)
                return_dict[target_name] = tmap

        return return_dict

    def get_checkpoint(self, best_state_dict: Optional[dict[str, Any]] = None) -> dict:
        state_dict = self.state_dict()
        return {
            "hook_name": "tensor_basis",
            "hook_ckpt_version": self.__checkpoint_version__,
            "hook_data": {
                "hypers": self.hypers,
                "dataset_info": self.dataset_info,
            },
            "model_state_dict": state_dict,
            "best_model_state_dict": best_state_dict or state_dict,
        }

    @classmethod
    def load_checkpoint(cls, checkpoint: dict[str, Any], context: str) -> "TensorBasis":
        if checkpoint["hook_ckpt_version"] != cls.__checkpoint_version__:
            raise ValueError(
                f"Invalid checkpoint version for the hook: "
                f"expected {cls.__checkpoint_version__}, got "
                f"{checkpoint['hook_ckpt_version']}"
            )

        hook = cls(
            hypers=checkpoint["hook_data"]["hypers"],
            dataset_info=checkpoint["hook_data"]["dataset_info"],
        )

        model_state_dict = checkpoint["model_state_dict"]
        if context in ["finetine", "export"]:
            # Load the best model state dict if available
            model_state_dict = checkpoint.get("best_model_state_dict", model_state_dict)

        hook.load_state_dict(model_state_dict)
        return hook


def _check_compatible_target_infos(
    target_info1: TargetInfo, target_info2: TargetInfo
) -> None:
    """Check that two target infos have compatible layouts.

    :param target_info1: The first target info.
    :param target_info2: The second target info.
    :raises ValueError: If the layouts are not compatible.
    """

    assert target_info1.sample_kind == target_info2.sample_kind
    assert target_info1.is_spherical == target_info2.is_spherical

    layout1 = target_info1.layout
    layout2 = target_info2.layout

    for (key1, block1), (key2, block2) in zip(
        layout1.items(), layout2.items(), strict=True
    ):
        assert key1 == key2

        assert block1.samples.values.shape == block2.samples.values.shape, (
            f"Sample shapes do not match for key {key1}: "
            f"{block1.samples.values.shape} vs {block2.samples.values.shape}"
        )
        assert block1.components == block2.components

        assert block1.properties == block2.properties
