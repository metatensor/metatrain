import torch


class AtomicComposition(torch.nn.Module):

    def __init__(self, all_species) -> None:
        super().__init__()
        self.all_species = all_species

    def forward(
        self,
        positions: torch.Tensor,
        cells: torch.Tensor,
        species: torch.Tensor,
        cell_shifts: torch.Tensor,
        centers: torch.Tensor,
        pairs: torch.Tensor,
        structure_centers: torch.Tensor,
        structure_pairs: torch.Tensor,
        structure_offsets: torch.Tensor,
    ) -> torch.Tensor:
        n_structures = cells.shape[0]
        composition_features = torch.zeros(
            (n_structures, len(self.all_species)),
            dtype=positions.dtype,
            device=positions.device,
        )
        for i_structure in range(n_structures):
            if i_structure == n_structures - 1:
                species_structure = species[structure_offsets[i_structure] :]
            else:
                species_structure = species[
                    structure_offsets[i_structure] : structure_offsets[i_structure + 1]
                ]
            for i_species, atomic_number in enumerate(self.all_species):
                composition_features[i_structure, i_species] = len(
                    torch.where(species_structure == atomic_number)[0]
                )
        return composition_features
