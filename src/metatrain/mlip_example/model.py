"""Example MLIP model that always predicts zero energy."""

from typing import Any, Dict

import torch

from metatrain.utils.data.dataset import DatasetInfo
from metatrain.utils.mlip import MLIPModel


class ZeroModel(MLIPModel):
    """
    A minimal example MLIP model that always predicts zero energy.

    This model serves as a simple demonstration of how to use the MLIPModel
    base class. It always predicts zero energy regardless of the input structure.
    """

    __checkpoint_version__ = 1

    def __init__(self, hypers: Dict[str, Any], dataset_info: DatasetInfo) -> None:
        """
        Initialize the ZeroModel.

        :param hypers: Model hyperparameters.
        :param dataset_info: Information about the dataset.
        """
        super().__init__(hypers, dataset_info)

        # Request a neighbor list with the cutoff from hyperparameters
        cutoff = hypers["cutoff"]
        self.request_neighbor_list(cutoff)

        # Add a dummy parameter for the optimizer to work with
        # This is needed because PyTorch optimizers require at least one parameter
        self.dummy_param = torch.nn.Parameter(torch.zeros(1))

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

        This implementation always returns zero energy for each system.

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

        :return: Tensor of shape (N_systems,) containing zero energy for each
            system.
        """
        # Get the number of systems
        n_systems = system_indices.max().item() + 1

        # Return zeros for all systems, but multiply by dummy_param to maintain
        # gradient tracking (dummy_param is always 0, so the result is still 0)
        return (
            torch.zeros(n_systems, device=edge_vectors.device, dtype=edge_vectors.dtype)
            + self.dummy_param.sum()
        )
