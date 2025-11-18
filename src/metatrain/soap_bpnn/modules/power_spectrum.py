import torch
from metatensor.torch import Labels, TensorBlock, TensorMap
from spex.spherical_expansion import SphericalExpansion
from torch.nn import Module
from torch.profiler import record_function


class SoapPowerSpectrum(Module):
    """Class for the SOAP power spectrum.

    Arguments are expected in the form of ``specable``-style dictionaries, i.e.,
    ``{ClassName: {key1: value1, key2: value2, ...}}``.

    :param cutoff: Cutoff radius.
    :param max_angular: Maximum angular momentum to consider.
    :param radial: Radial expansion specification.
    :param angular: Type of angular expansion
        (supported: "SphericalHarmonics", "SolidHarmonics").
    :param species: Species embedding specification.
    :param cutoff_function: Cutoff function specification.
    """

    def __init__(
        self,
        cutoff: float,
        max_angular: int,
        radial: dict,
        angular: str,
        species: dict,
        cutoff_function: dict,
    ) -> None:
        super().__init__()

        self.spec = {
            "cutoff": cutoff,
            "max_angular": max_angular,
            "radial": radial,
            "angular": angular,
            "species": species,
            "cutoff_function": cutoff_function,
        }

        self.calculator = SphericalExpansion(**self.spec)

        l_to_treat = list(range(self.calculator.max_angular + 1))
        self.n_per_l = self.calculator.radial.n_per_l

        n_species = (
            species["Alchemical"]["pseudo_species"]
            if "Alchemical" in species
            else len(species["Orthogonal"]["species"])
        )
        self.shape = sum(self.n_per_l[ell] ** 2 * n_species**2 for ell in l_to_treat)

    def forward(
        self,
        R_ij: torch.Tensor,
        i: torch.Tensor,
        j: torch.Tensor,
        species: torch.Tensor,
        structures: torch.Tensor,
        centers: torch.Tensor,
    ) -> TensorMap:
        """Computes the soap power spectrum.

        Since we don't want to be in charge of computing displacements, we take an
        already-computed graph of ``R_ij``, ``i``, and ``j``, as well as center atom
        ``species``. From this perspective, a batch is just a very big graph with many
        disconnected subgraphs, the spherical expansion doesn't need to know the
        difference.

        However, ``metatensor`` output is expected to contain more information, so if
        our input is a big "batch" graph, we need some additional information to keep
        track of which nodes in the big graph belong to which original structure
        (``structures``) and which atom in each structure is which (``centers``).

        For a single-structure graph, this would be just zeros for ``structures`` and
        ``torch.arange(n_atoms)`` for ``centers``. For a two-structure graph, it would
        be a block of zeros and a block of ones for ``structures``, and then a range
        going up to ``n_atoms_0`` and then a range going up to ``n_atoms_1`` for
        ``centers``.

        Note that we take the center species to consider from the input species, so if
        a given graph doesn't contain a given center species, it will also not appear in
        the output.

        :param R_ij: Interatomic displacements of shape ``[pair, 3]``, using the
            convention ``R_ij = R_j - R_i``.
        :param i: Center atom indices of shape ``[pair]``.
        :param j: Neighbour atom indices of shape ``[pair]``.
        :param species: Atomic species of shape ``[center]``, indicating the species of
            the atoms indexed by ``i`` and ``j``.
        :param structures: Structure indices of shape ``[center]``, indicating which
            structure each atom belongs to.
        :param centers: Center atom indices of shape ``[center]``, indicating which atom
            in each structure a given node in the graph is supposed to be.

        :return: SOAP power spectrum, a ``TensorMap``.
        """
        # R_ij: [pair, 3]
        # i: [pair]
        # j: [pair]
        # species: [center]
        # structures: [center]
        # centers: [center]

        with record_function("calc"):
            spherical_expansion = self.calculator.forward(R_ij, i, j, species)
        with record_function("finalize"):
            blocks_from_single_l: list[torch.Tensor] = []
            for tensor in spherical_expansion:
                tensor = tensor.reshape(
                    tensor.shape[0], tensor.shape[1], tensor.shape[2] * tensor.shape[3]
                )
                n_prop = int(tensor.shape[-1] ** 2)
                values = torch.einsum("smn,smN->snN", tensor, tensor).reshape(
                    tensor.shape[0], n_prop
                )
                blocks_from_single_l.append(values)
        with record_function("keys_to_properties_final"):
            output_tensor = torch.concatenate(blocks_from_single_l, dim=1)

            unique_center_species = torch.unique(species)
            blocks: list[TensorBlock] = []
            for s in unique_center_species:
                mask = species == s
                output_tensor_filtered = output_tensor[mask]
                structures_filtered = structures[mask]
                centers_filtered = centers[mask]
                block = TensorBlock(
                    values=output_tensor_filtered,
                    samples=Labels(
                        names=["system", "atom"],
                        values=torch.stack(
                            [structures_filtered, centers_filtered], dim=1
                        ),
                    ),
                    components=[],
                    properties=Labels(
                        names=["property"],
                        values=torch.arange(
                            output_tensor.shape[1], device=output_tensor.device
                        ).unsqueeze(1),
                    ),
                )
                blocks.append(block)
            output_tensor_map = TensorMap(
                keys=Labels(
                    names=["center_type"],
                    values=unique_center_species.unsqueeze(1),
                ),
                blocks=blocks,
            )
        return output_tensor_map
