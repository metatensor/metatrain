from abc import abstractmethod
from typing import Any, Dict, List, Optional

import torch
from metatensor.torch import Labels, TensorBlock, TensorMap
from metatomic.torch import ModelMetadata, ModelOutput, NeighborListOptions, System

from .abc import ModelInterface
from .data.dataset import DatasetInfo


class MLIPModel(ModelInterface):
    __checkpoint_version__ = 1
    __supported_devices__ = ["cuda", "cpu"]
    __supported_dtypes__ = [torch.float64, torch.float32]
    __default_metadata__ = ModelMetadata()

    def __init__(self, hypers: Dict[str, Any], dataset_info: DatasetInfo) -> None:
        super().__init__(hypers, dataset_info, self.__default_metadata__)

        if len(dataset_info.targets) > 1:
            raise ValueError(
                "MLIPModel only supports datasets with a single target. "
                f"Found {len(dataset_info.targets)} targets."
            )
        self.target_name = dataset_info.targets.keys()[0]
        if dataset_info.targets[self.target_name].quantity != "energy":
            raise ValueError(
                "MLIPModel only supports datasets with an energy as target quantity. "
                f"Found '{dataset_info.targets[self.target_name].quantity}'."
            )
        if not dataset_info.targets[self.target_name].is_scalar:
            raise ValueError(
                "MLIPModel only supports datasets with a scalar target. "
                "Found a non-scalar target."
            )
        if dataset_info.targets[self.target_name].per_atom:
            raise ValueError(
                "MLIPModel only supports datasets with a total energy target. "
                "Found a per-atom target."
            )
        if (dataset_info.targets[self.target_name].layout.block().properties) > 1:
            raise ValueError(
                "MLIPModel only supports datasets with a single sub-target. "
                "Found "
                f"{dataset_info.targets[self.target_name].layout.block().properties} "
                "sub-targets."
            )

    def forward(
        self,
        systems: List[System],
        outputs: Dict[str, ModelOutput],
        selected_atoms: Optional[Labels] = None,
    ) -> Dict[str, TensorMap]:
        if len(outputs) > 1:
            raise ValueError(
                "MLIPModel only supports a single output. "
                f"Found {len(outputs)} outputs."
            )
        if self.target_name not in outputs:
            raise ValueError(
                f"MLIPModel only supports the '{self.target_name}' output. "
                f"Found outputs: {list(outputs.keys())}."
            )
        if selected_atoms is not None:
            raise ValueError(
                "MLIPModel does not support the 'selected_atoms' argument."
            )

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
            assert len(system.known_neighbor_lists()) == 1, "no neighbor list found"
            neighbor_list = system.get_neighbor_list(self.nl_options)
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
        system_indices = torch.concatenate(
            [
                torch.full(
                    (len(system),),
                    i_system,
                    device=positions.device,
                )
                for i_system, system in enumerate(systems)
            ],
        )

        # somehow the backward of this operation is very slow at evaluation,
        # where there is only one cell, therefore we simplify the calculation
        # for that case
        if len(cells) == 1:
            cell_contributions = cell_shifts.to(cells.dtype) @ cells[0]
        else:
            cell_contributions = torch.einsum(
                "ab, abc -> ac",
                cell_shifts.to(cells.dtype),
                cells[system_indices[centers]],
            )
        edge_vectors = positions[neighbors] - positions[centers] + cell_contributions

        energy_as_tensor = self.compute_energy(
            edge_vectors, species, centers, neighbors, system_indices
        )

        energy_as_tensor_map = TensorMap(
            keys=Labels(
                ["_"],
                torch.tensor([[0]], dtype=torch.int64, device=energy_as_tensor.device),
            ),
            blocks=[
                TensorBlock(
                    values=energy_as_tensor.unsqueeze(-1),
                    samples=Labels(
                        names=["structure"],
                        values=torch.arange(
                            len(energy_as_tensor),
                            device=energy_as_tensor.device,
                        ).unsqueeze(-1),
                    ),
                    components=[],
                    properties=Labels(
                        names=["energy"],
                        values=torch.tensor(
                            [[0]], dtype=torch.int64, device=energy_as_tensor.device
                        ),
                    ),
                )
            ],
        )

        return {self.target_info.name: energy_as_tensor_map}

    def request_neighbor_list(self, cutoff) -> None:
        self.nl_options = NeighborListOptions(
            cutoff=cutoff,
            full=True,
            strict=True,
        )

        def requested_neighbor_lists():
            return [self.nl_options]

        self.requested_neighbor_lists = requested_neighbor_lists

    @abstractmethod
    def compute_energy(
        self,
        edge_vectors: torch.Tensor,
        species: torch.Tensor,
        centers: torch.Tensor,
        neighbors: torch.Tensor,
        system_indices: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the total energy given the edge vectors and other information.

        :param edge_vectors: Tensor of shape (N_edges, 3) containing the vectors
            between neighboring atoms.
        :param species: Tensor of shape (N_atoms,) containing the atomic species
            indices.
        :param centers: Tensor of shape (N_edges,) containing the indices of the
            center atoms for each edge.
        :param neighbors: Tensor of shape (N_edges,) containing the indices of the
            neighbor atoms for each edge.
        :param system_indices: Tensor of shape (N_atoms,) containing the indices
            of the systems each atom belongs to.

        :return: Tensor of shape (N_systems,) containing the total energy for each
            system.
        """
