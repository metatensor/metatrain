from typing import Dict, List, Optional

import metatensor.torch
import torch
from metatensor.torch import Labels, TensorBlock, TensorMap
from metatensor.torch.atomistic import (
    ModelCapabilities,
    ModelOutput,
    NeighborsListOptions,
    System,
)
from omegaconf import OmegaConf

from ... import ARCHITECTURE_CONFIG_PATH
from ...utils.composition import apply_composition_contribution_samples
from ...utils.scaling import apply_scaling
from .modules.center_embedding import CenterEmbedding
from .modules.cg import get_cg_coefficients
from .modules.cg_iterator import CGIterator
from .modules.generalized_cg_iterator import GeneralizedCGIterator
from .modules.initial_features import InitialFeatures
from .modules.linear_map import LinearMap
from .modules.message_passing import MessagePasser
from .modules.precomputations import Precomputer
from .modules.tensor_sum import TensorAdd
from .utils import systems_to_batch


ARCHITECTURE_NAME = "experimental.soap_bpnn"
DEFAULT_HYPERS = OmegaConf.to_container(
    OmegaConf.load(ARCHITECTURE_CONFIG_PATH / f"{ARCHITECTURE_NAME}.yaml")
)
DEFAULT_MODEL_HYPERS = DEFAULT_HYPERS["model"]


class Model(torch.nn.Module):

    def __init__(
        self, capabilities: ModelCapabilities, hypers: Dict = DEFAULT_MODEL_HYPERS
    ) -> None:
        super().__init__()
        self.name = ARCHITECTURE_NAME

        hypers["normalize"] = True

        self.cutoff_radius = hypers["cutoff"]
        self.capabilities = capabilities
        self.all_species = capabilities.atomic_types
        self.hypers = hypers

        n_channels = hypers["n_element_channels"]

        species_to_species_index = torch.zeros(
            (max(self.all_species) + 1,), dtype=torch.int
        )
        species_to_species_index[self.all_species] = torch.arange(
            len(self.all_species), dtype=torch.int
        )
        self.register_buffer("species_to_species_index", species_to_species_index)
        print("species_to_species_index", self.species_to_species_index)
        self.embeddings = torch.nn.Embedding(len(self.all_species), n_channels)

        self.nu_max = hypers["nu_max"]
        self.n_message_passing_layers = hypers["n_message_passing_layers"]
        if self.n_message_passing_layers < 1:
            raise ValueError("Number of message-passing layers must be at least 1")

        self.invariant_message_passer = MessagePasser(
            hypers, self.all_species, [[(0, 1)]]
        )

        self.all_species = self.all_species
        n_max = self.invariant_message_passer.message_passers[0].n_max_l
        self.l_max = len(n_max) - 1
        self.k_max_l = [n_channels * n_max[l] for l in range(self.l_max + 1)]

        print()
        print("BASELINE")
        print()
        print("l_max", self.l_max)
        print("n_max_l", n_max)
        print("n_element_channels", n_channels)
        print("k_max_l", self.k_max_l)
        print()

        cgs = get_cg_coefficients(self.l_max)
        cgs = {
            str(l1) + "_" + str(l2) + "_" + str(L): tensor
            for (l1, l2, L), tensor in cgs._cgs.items()
        }

        self.adder = TensorAdd()

        self.element_embedding = CenterEmbedding(n_channels)
        self.initial_features = InitialFeatures(self.k_max_l[0])
        self.precomputer = Precomputer(self.l_max, normalize=True)
        self.cg_iterator = CGIterator(
            self.k_max_l,
            self.nu_max - 1,
            cgs,
            irreps_in=[(l, 1) for l in range(self.l_max + 1)],
            requested_LS_string="0_1",
        )

        equivariant_message_passers = []
        generalized_cg_iterators = []
        for idx in range(self.n_message_passing_layers - 1):
            irreps_equiv_mp = (
                [[(0, 1)]] + list(self.cg_iterator.irreps_out.values())[:-1]
                if idx == 0
                else [[(0, 1)]]
                + list(generalized_cg_iterators[-1].irreps_out.values())[:-1]
            )
            equivariant_message_passer = MessagePasser(
                hypers, self.all_species, irreps_equiv_mp, cgs
            )
            equivariant_message_passers.append(equivariant_message_passer)
            generalized_cg_iterator = GeneralizedCGIterator(
                self.k_max_l,
                self.nu_max,
                cgs,
                {
                    idx + 1: mp.irreps_out
                    for idx, mp in enumerate(
                        equivariant_message_passers[-1].message_passers
                    )
                },
            )
            generalized_cg_iterators.append(generalized_cg_iterator)

        self.equivariant_message_passers = torch.nn.ModuleList(
            equivariant_message_passers
        )
        self.generalized_cg_iterators = torch.nn.ModuleList(generalized_cg_iterators)

        self.last_layers = torch.nn.ModuleDict(
            {
                output_name: LinearMap(self.k_max_l[0])
                for output_name in capabilities.outputs.keys()
            }
        )

        # creates a composition weight tensor that can be directly indexed by species,
        # this can be left as a tensor of zero or set from the outside using
        # set_composition_weights (recommended for better accuracy)
        n_outputs = len(capabilities.outputs)
        self.register_buffer(
            "composition_weights", torch.zeros((n_outputs, max(self.all_species) + 1))
        )
        # buffers cannot be indexed by strings (torchscript), so we create a single
        # tensor for all output. Due to this, we need to slice the tensor when we use
        # it and use the output name to select the correct slice via a dictionary
        self.output_to_index = {
            output_name: i for i, output_name in enumerate(capabilities.outputs.keys())
        }

        # we also register a buffer for the shifts:
        # these are meant to be modified from outside
        self.register_buffer("scalings", torch.ones((n_outputs,)))

    def forward(
        self,
        systems: List[System],
        outputs: Dict[str, ModelOutput],
        selected_atoms: Optional[Labels] = None,
    ) -> Dict[str, TensorMap]:
        if selected_atoms is not None:
            raise NotImplementedError("PhACE does not support selected atoms.")

        options = self.requested_neighbors_lists()[0]
        structures = systems_to_batch(systems, options)

        n_atoms = len(structures["positions"])

        r, sh = self.precomputer(
            positions=structures["positions"],
            cells=structures["cells"],
            species=structures["species"],
            cell_shifts=structures["cell_shifts"],
            pairs=structures["pairs"],
            structure_pairs=structures["structure_pairs"],
            structure_offsets=structures["structure_offsets"],
        )

        samples_values = torch.stack(
            (
                structures["structure_centers"],
                structures["centers"],
                structures["species"],
            ),
            dim=1,
        )
        samples = metatensor.torch.Labels(
            names=["system", "atom", "center_type"],
            values=samples_values,
        )
        center_species_indices = self.species_to_species_index[structures["species"]]
        center_embeddings = self.embeddings(center_species_indices)

        initial_features = self.initial_features(
            structures["structure_centers"],
            structures["centers"],
            structures["species"],
            structures["positions"].dtype,
        )
        initial_element_embedding = self.element_embedding(
            initial_features, center_embeddings
        )

        spherical_expansion = self.invariant_message_passer(
            r, sh, structures, n_atoms, [initial_element_embedding], samples
        )
        spherical_expansion = self.adder(spherical_expansion, [initial_features])

        features = self.cg_iterator(spherical_expansion)
        features = self.adder(features, spherical_expansion)

        # message passing
        for message_passer, generalized_cg_iterator in zip(
            self.equivariant_message_passers, self.generalized_cg_iterators
        ):
            embedded_features: List[metatensor.torch.TensorMap] = []
            for nu in range(self.nu_max):
                embedded_features.append(
                    self.element_embedding(features[nu], center_embeddings)
                )
            mp_features = message_passer(
                r, sh, structures, n_atoms, embedded_features, samples
            )
            features = self.adder(mp_features, features)

            iterated_features = generalized_cg_iterator(features)
            features = self.adder(iterated_features, features)

        # center embedding before readout
        for nu in range(self.nu_max + 1):
            features[nu] = self.element_embedding(features[nu], center_embeddings)

        hidden_features = features[self.nu_max]

        atomic_energies: Dict[str, TensorMap] = {}
        for output_name, output_layer in self.last_layers.items():
            if output_name in outputs:
                atomic_energies[output_name] = apply_composition_contribution_samples(
                    apply_scaling(
                        output_layer(hidden_features),
                        self.scalings[self.output_to_index[output_name]].item(),
                    ),
                    self.composition_weights[  # type: ignore
                        self.output_to_index[output_name]
                    ],
                )

        # Sum the atomic energies coming from the BPNN to get the total energy
        total_energies: Dict[str, TensorMap] = {}
        for output_name, atomic_energy in atomic_energies.items():
            total_energies[output_name] = metatensor.torch.sum_over_samples(
                atomic_energy, ["atom", "center_type"]
            )

        return total_energies

    def set_composition_weights(
        self,
        output_name: str,
        input_composition_weights: torch.Tensor,
        species: List[int],
    ) -> None:
        """Set the composition weights for a given output."""
        # all species that are not present retain their weight of zero
        self.composition_weights[self.output_to_index[output_name]][  # type: ignore
            species
        ] = input_composition_weights.to(
            dtype=self.composition_weights.dtype,  # type: ignore
            device=self.composition_weights.device,  # type: ignore
        )

    def add_output(self, output_name: str) -> None:
        """Add a new output to the model."""
        # add a new row to the composition weights tensor
        self.composition_weights = torch.cat(
            [
                self.composition_weights,  # type: ignore
                torch.zeros(
                    1,
                    self.composition_weights.shape[1],  # type: ignore
                    dtype=self.composition_weights.dtype,  # type: ignore
                    device=self.composition_weights.device,  # type: ignore
                ),
            ]
        )  # type: ignore
        self.output_to_index[output_name] = len(self.output_to_index)
        # add a new linear layer to the last layers
        hypers_bpnn = self.hypers["bpnn"]
        if hypers_bpnn["num_hidden_layers"] == 0:
            n_inputs_last_layer = hypers_bpnn["input_size"]
        else:
            n_inputs_last_layer = hypers_bpnn["num_neurons_per_layer"]
        self.last_layers[output_name] = LinearMap(self.all_species, n_inputs_last_layer)

    def requested_neighbors_lists(
        self,
    ) -> List[NeighborsListOptions]:
        return [
            NeighborsListOptions(
                cutoff=self.cutoff_radius,
                full_list=True,
            )
        ]
