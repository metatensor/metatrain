import torch
from metatensor.torch import TensorBlock, TensorMap, Labels
from metatomic.torch import ModelOutput, System
from typing_extensions import TypedDict

from metatrain.utils.data import DatasetInfo, TargetInfo
from metatrain.utils.data.target_info import get_generic_target_info
from metatrain.utils.sum_over_atoms import sum_over_atoms
from metatrain.utils.hypers import init_with_defaults

from metatrain.soap_bpnn.modules.tensor_basis import TensorBasis as TensorBasisModule
from metatrain.soap_bpnn.documentation import SOAPConfig


class HookHypers(TypedDict):
    """
    Hyperparameters for the tensor basis hook.
    """

    soap: SOAPConfig = init_with_defaults(SOAPConfig)

    inputs: str

    outputs: str | list
    """
    Name or names of the targets to predict through a tensor basis.
    
    A separate tensor basis will be built for each target.
    """

def concatenate_structures(
    systems: list[System],
) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
    """
    Concatenate a list of systems into a single batch.

    :param systems: List of systems to concatenate.
    :param neighbor_list_options: Options for the neighbor list.
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


class TensorBasis(torch.nn.Module):
    """
    Provides a tensor basis in which to predict spherical tensor targets.
    """

    def __init__(self, hypers: HookHypers, dataset_info: DatasetInfo):
        super().__init__()

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
        self.out_targets = {
            name: dataset_info.targets[name] for name in outputs
        }

        # Names for the inputs that we will request from the model
        self._input_names = [
            f"mtt::aux::scalars::{name.replace('mtt::', '')}" for name in self.out_targets
        ]

        # Build the basis calculators for each target,
        # and the output that we have to request from the model
        soap_hypers = hypers.get("soap", init_with_defaults(SOAPConfig))
        self.basis_calculators = torch.nn.ModuleDict({})
        self._input_target_infos = {}
        for input_name, target_name in zip(self._input_names, self.out_targets):

            target = self.out_targets[target_name]
            # Get one basis calculator for each block of the target, since each block
            # has different o3_lambda and o3_sigma values.
            self.basis_calculators[target_name] = torch.nn.ModuleList([
                TensorBasisModule(
                    dataset_info.atomic_types,
                    soap_hypers,
                    o3_lambda=key["o3_lambda"],
                    o3_sigma=key["o3_sigma"],
                    add_lambda_basis=True,
                    legacy=False,
                ) for key in target.layout.keys
            ])
            
            # Build the input that we will request from the model.
            # We will ask for invariant coefficients. For each block we ask for 2l+1 coefficients
            # for each property, since the basis will have 2l+1 tensors.
            # We ask for all the coefficients in a single block, we will untangle
            # them in the forward pass.
            num_properties = sum(block.values.shape[1] * block.values.shape[2] for block in target.layout.blocks())
            self._input_target_infos[target_name] = get_generic_target_info(
                input_name,
                {
                    "quantity": "_",
                    "unit": "",
                    "type": {"spherical": {
                        "irreps": [
                            {"o3_lambda": 0, "o3_sigma": 1}
                        ]
                    }},
                    "num_subtargets": num_properties,
                    "sample_kind": "atom",
                },
            )

    def requested_target_infos(self) -> dict[str, TargetInfo]:
        """
        Returns the list of requested target infos for the hook.

        :return: A list of requested target names.
        """
        return self._input_target_infos

    def requested_inputs(self) -> dict[str, ModelOutput]:
        """
        Returns the list of requested inputs for the hook.

        :return: A list of requested input names.
        """ 
        return {
            name: ModelOutput(
                quantity="",
                unit="",
                sample_kind="atom",
            )
            for name in self._input_target_infos
        }

    def forward(
        self, systems: list[System], inputs: dict[str, TensorMap]
    ) -> dict[str, TensorMap]:
        """
        Computes spherical targets using the tensor basis.
        """
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
            target_info = self.out_targets[target_name]
            target_invariant_coefficients = inputs[target_name].block().values

            offset = 0
            blocks: list[TensorBlock] = []
            for i, basis_calculator in enumerate(basis_calculators):

                layout_block = target_info.layout.block(i)

                # Get shapes of the invariant coefficients to retrieve
                # for this block.
                n_properties = layout_block.properties.values.shape[0]
                n_basis = layout_block.values.shape[1]
                count = n_properties * n_basis

                # Get those invariant coefficients
                invariant_coefficients = target_invariant_coefficients[
                    :, 0, offset : offset + count
                ].reshape(
                    -1, n_properties, n_basis
                )
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
                            names=["system", "atom"],
                            values=sample_values
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
