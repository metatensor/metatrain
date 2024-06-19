import abc
from collections import defaultdict
from typing import Callable, Dict, List, Tuple, TypeVar

import ase
import numpy as np
import torch


AtomicStructure = TypeVar("AtomicStructure")


def structure_to_torch(
    structure: AtomicStructure, device: torch.device = None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    :returns:
        Tuple of posititions, species, cell and periodic boundary conditions
    """
    if isinstance(structure, ase.Atoms):
        # dtype is automatically referred from the type in the structure object
        positions = torch.tensor(
            structure.positions, device=device, dtype=torch.get_default_dtype()
        )
        species = torch.tensor(structure.numbers, device=device)
        cell = torch.tensor(
            structure.cell.array, device=device, dtype=torch.get_default_dtype()
        )
        pbc = torch.tensor(structure.pbc, device=device)
        return positions, species, cell, pbc
    else:
        raise ValueError("Unknown atom type. We only support ase.Atoms at the moment.")


def build_neighborlist(
    positions: torch.Tensor, cell: torch.Tensor, pbc: torch.Tensor, cutoff: float
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

    assert positions.device == cell.device
    assert positions.device == pbc.device
    device = positions.device
    # will be replaced with something with GPU support
    pairs_i, pairs_j, cell_shifts = ase.neighborlist.primitive_neighbor_list(
        quantities="ijS",
        positions=positions.detach().cpu().numpy(),
        cell=cell.detach().cpu().numpy(),
        pbc=pbc.detach().cpu().numpy(),
        cutoff=cutoff,
        self_interaction=False,
        use_scaled_positions=False,
    )
    pairs_i = torch.tensor(pairs_i, device=device)
    pairs_j = torch.tensor(pairs_j, device=device)
    cell_shifts = torch.tensor(cell_shifts, device=device)

    pairs = torch.vstack([pairs_i, pairs_j]).T
    centers = torch.arange(len(positions), device=device)
    return centers, pairs, cell_shifts


class TransformerBase(metaclass=abc.ABCMeta):
    """
    Abstract class for extracting information of an AtomicStructure objects and processing it
    """

    @abc.abstractmethod
    def __call__(self, structure: AtomicStructure) -> Dict[str, torch.Tensor]:
        pass


class TransformerProperty(TransformerBase):
    """
    Extracts property information out of an AtomicStructure using a function given as input
    """

    def __init__(
        self,
        property_name: str,
        get_property: Callable[[AtomicStructure], Tuple[str, torch.Tensor]],
    ):
        self._get_property = get_property
        self._property_name = property_name

    def __call__(self, structure: AtomicStructure) -> Dict[str, torch.Tensor]:
        return {self._property_name: self._get_property(structure)}


class TransformerNeighborList(TransformerBase):
    """
    Produces a neighbour list and with direction vectors from an AtomicStructure
    """

    def __init__(self, cutoff: float, device=None):
        self._cutoff = cutoff
        self._device = device

    def __call__(self, structure: AtomicStructure) -> Dict[str, torch.Tensor]:
        positions_i, species_i, cell_i, pbc_i = structure_to_torch(
            structure, device=self._device
        )
        centers_i, pairs_ij, cell_shifts_ij = build_neighborlist(
            positions_i, cell_i, pbc_i, self._cutoff
        )

        return {
            "positions": positions_i,
            "species": species_i,
            "cell": cell_i,
            "centers": centers_i,
            "pairs": pairs_ij,
            "cell_shifts": cell_shifts_ij,
        }


# Temporary Dataset until we have an metatensor Dataset
class InMemoryDataset(torch.utils.data.Dataset):
    def __init__(
        self, structures: List[AtomicStructure], transformers: List[TransformerBase]
    ):
        super().__init__()
        self.n_structures = len(structures)
        self._data = defaultdict(list)
        for structure in structures:
            for transformer in transformers:
                data_i = transformer(structure)
                for key in data_i.keys():
                    self._data[key].append(data_i[key])

    def __getitem__(self, idx):
        return {key: self._data[key][idx] for key in self._data.keys()}

    def __len__(self):
        return self.n_structures


def collate_nl(data_list):

    collated = {
        key: torch.concatenate([data[key] for data in data_list], dim=0)
        for key in filter(lambda x: x not in ["positions", "cell"], data_list[0].keys())
    }
    collated["positions"] = torch.concatenate([data["positions"] for data in data_list])
    collated["cells"] = torch.stack([data["cell"] for data in data_list])
    collated["structure_centers"] = torch.concatenate(
        [
            torch.tensor(
                [structure_index] * len(data_list[structure_index]["centers"]),
                device=collated["positions"].device,
            )
            for structure_index in range(len(data_list))
        ]
    )
    collated["structure_pairs"] = torch.concatenate(
        [
            torch.tensor(
                [structure_index] * len(data_list[structure_index]["pairs"]),
                device=collated["positions"].device,
            )
            for structure_index in range(len(data_list))
        ]
    )
    collated["structure_offsets"] = torch.tensor(
        np.cumsum(
            [0]
            + [
                structure_data["positions"].shape[0]
                for structure_data in data_list[:-1]
            ]
        ),
        device=collated["positions"][0].device,
        dtype=torch.long,
    )
    return collated
