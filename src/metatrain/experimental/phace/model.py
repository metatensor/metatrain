from pathlib import Path
from typing import Dict, List, Optional, Union

import metatensor.torch
import torch
from metatensor.torch import Labels, TensorMap
from metatensor.torch.atomistic import (
    MetatensorAtomisticModel,
    ModelOutput,
    NeighborListOptions,
    System,
    ModelCapabilities
)

from ...utils.io import check_file_extension

from ...utils.data.dataset import DatasetInfo

from ...utils.composition import apply_composition_contribution_samples
from ...utils.dtype import dtype_to_str
from ...utils.export import export
from ...utils.scaling import apply_scaling
from .modules.center_embedding import embed_centers
from .modules.cg import get_cg_coefficients
from .modules.cg_iterator import CGIterator
from .modules.initial_features import get_initial_features
from .modules.layers import InvariantLinear, InvariantMLP
from .modules.message_passing import InvariantMessagePasser, EquivariantMessagePasser
from .modules.precomputations import Precomputer
from .utils import systems_to_batch


class PhACE(torch.nn.Module):

    __supported_devices__ = ["cuda", "cpu"]
    __supported_dtypes__ = [torch.float64, torch.float32]

    def __init__(self, model_hypers: Dict, dataset_info: DatasetInfo) -> None:
        super().__init__()
        self.hypers = model_hypers
        self.dataset_info = dataset_info
        self.new_outputs = list(dataset_info.targets.keys())
        self.all_species = sorted(dataset_info.atomic_types)

        self.outputs = {
            key: ModelOutput(
                quantity=value.quantity,
                unit=value.unit,
                per_atom=True,
            )
            for key, value in dataset_info.targets.items()
        }
        # the model is always capable of outputting the last layer features
        self.outputs["mtt::aux::last_layer_features"] = ModelOutput(
            unit="unitless", per_atom=True
        )

        model_hypers["normalize"] = True

        self.cutoff_radius = model_hypers["cutoff"]
        self.dataset_info = dataset_info
        self.model_hypers = model_hypers

        self.nu_scaling = model_hypers["nu_scaling"]
        self.mp_scaling = model_hypers["mp_scaling"]
        self.overall_scaling = model_hypers["overall_scaling"]

        n_channels = model_hypers["n_element_channels"]

        species_to_species_index = torch.zeros(
            (max(self.all_species) + 1,), dtype=torch.int
        )
        species_to_species_index[self.all_species] = torch.arange(
            len(self.all_species), dtype=torch.int
        )
        self.register_buffer("species_to_species_index", species_to_species_index)
        print("species_to_species_index", self.species_to_species_index)
        self.embeddings = torch.nn.Embedding(len(self.all_species), n_channels)

        self.nu_max = model_hypers["nu_max"]
        self.n_message_passing_layers = model_hypers["n_message_passing_layers"]
        if self.n_message_passing_layers < 1:
            raise ValueError("Number of message-passing layers must be at least 1")

        self.invariant_message_passer = InvariantMessagePasser(
            model_hypers, self.all_species, self.mp_scaling
        )

        self.all_species = self.all_species
        n_max = self.invariant_message_passer.n_max_l
        self.l_max = len(n_max) - 1
        self.k_max_l = [n_channels * n_max[l] for l in range(self.l_max + 1)]

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

        self.precomputer = Precomputer(self.l_max)
        self.cg_iterator = CGIterator(
            self.k_max_l,
            self.nu_max - 1,
            cgs,
            irreps_in=[(l, 1) for l in range(self.l_max + 1)],
            # requested_LS_string="0_1",  # For ACE
        )

        equivariant_message_passers = []
        generalized_cg_iterators = []
        for idx in range(self.n_message_passing_layers - 1):
            irreps_equiv_mp = list(self.cg_iterator.irreps_out.values())[-1]
            # irreps_equiv_mp = [(0, 1), (1, 1), (2, 1)]
            equivariant_message_passer = EquivariantMessagePasser(
                model_hypers, self.all_species, irreps_equiv_mp, [1, self.nu_max, self.nu_max + 1], cgs, self.mp_scaling
            )
            equivariant_message_passers.append(equivariant_message_passer)
            generalized_cg_iterator = CGIterator(
                self.k_max_l,
                self.nu_max - 1,
                cgs,
                irreps_in=equivariant_message_passer.irreps_out,
                requested_LS_string="0_1",
            )
            generalized_cg_iterators.append(generalized_cg_iterator)

        self.equivariant_message_passers = torch.nn.ModuleList(
            equivariant_message_passers
        )
        self.generalized_cg_iterators = torch.nn.ModuleList(generalized_cg_iterators)

        self.last_mlp = InvariantMLP(self.k_max_l[0])

        self.last_layers = torch.nn.ModuleDict(
            {
                output_name: InvariantLinear(self.k_max_l[0])
                for output_name in self.outputs.keys()
            }
        )

        # creates a composition weight tensor that can be directly indexed by species,
        # this can be left as a tensor of zero or set from the outside using
        # set_composition_weights (recommended for better accuracy)
        n_outputs = len(self.outputs)
        self.register_buffer(
            "composition_weights", torch.zeros((n_outputs, max(self.all_species) + 1))
        )
        # buffers cannot be indexed by strings (torchscript), so we create a single
        # tensor for all output. Due to this, we need to slice the tensor when we use
        # it and use the output name to select the correct slice via a dictionary
        self.output_to_index = {
            output_name: i for i, output_name in enumerate(self.outputs.keys())
        }

        # we also register a buffer for the shifts:
        # these are meant to be modified from outside
        self.register_buffer("scalings", torch.ones((n_outputs,)))

    def restart(self, dataset_info: DatasetInfo) -> "PhACE":
        # merge old and new dataset info
        merged_info = self.dataset_info.union(dataset_info)
        new_all_species = merged_info.all_species - self.all_species
        new_targets = merged_info.targets - self.dataset_info.targets

        if len(new_all_species) > 0:
            raise ValueError(
                f"New atomic types found in the dataset: {new_all_species}. "
                "The SOAP-BPNN model does not support adding new atomic types."
            )

        # register new outputs as new last layers
        for output_name in new_targets:
            self.add_output(output_name)

        self.dataset_info = merged_info
        self.all_species = sorted(self.all_species)

        for target_name, target in new_targets.items():
            self.outputs[target_name] = ModelOutput(
                quantity=target.quantity,
                unit=target.unit,
                per_atom=True,
            )
        self.new_outputs = list(new_targets.keys())

        return self

    def forward(
        self,
        systems: List[System],
        outputs: Dict[str, ModelOutput],
        selected_atoms: Optional[Labels] = None,
    ) -> Dict[str, TensorMap]:
        if selected_atoms is not None:
            raise NotImplementedError("PhACE does not support selected atoms.")

        options = self.requested_neighbor_lists()[0]
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
        sh = metatensor.torch.multiply(sh, self.nu_scaling)

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

        initial_features = get_initial_features(
            structures["structure_centers"],
            structures["centers"],
            structures["species"],
            structures["positions"].dtype,
            self.k_max_l[0]
        )
        initial_element_embedding = embed_centers(
            initial_features, center_embeddings
        )

        spherical_expansion = self.invariant_message_passer(
            r,
            sh, 
            structures["structure_offsets"][structures["structure_pairs"]]
            + structures["pairs"][:, 0],
            structures["structure_offsets"][structures["structure_pairs"]]
            + structures["pairs"][:, 1],
            n_atoms,
            initial_element_embedding,
            samples
        )

        features = self.cg_iterator(spherical_expansion)

        # message passing
        for message_passer, generalized_cg_iterator in zip(
            self.equivariant_message_passers, self.generalized_cg_iterators
        ):
            embedded_features = embed_centers(features, center_embeddings)
            mp_features = message_passer(
                r,
                sh,
                structures["structure_offsets"][structures["structure_pairs"]]
                + structures["pairs"][:, 0],
                structures["structure_offsets"][structures["structure_pairs"]]
                + structures["pairs"][:, 1],
                n_atoms,
                embedded_features,
                samples
            )
            iterated_features = generalized_cg_iterator(mp_features)
            features = iterated_features

        embed_centers(features, center_embeddings)
        features = self.last_mlp(features)

        return_dict: Dict[str, TensorMap] = {}
        # output the hidden features, if requested:
        if "mtt::aux::last_layer_features" in outputs:
            last_layer_features_options = outputs["mtt::aux::last_layer_features"]
            out_features = features
            if not last_layer_features_options.per_atom:
                out_features = metatensor.torch.sum_over_samples(out_features, ["atom"])
            return_dict["mtt::aux::last_layer_features"] = out_features

        atomic_energies: Dict[str, TensorMap] = {}
        for output_name, output_layer in self.last_layers.items():
            if output_name in outputs:
                atomic_energies[output_name] = apply_composition_contribution_samples(
                    metatensor.torch.multiply(output_layer(features), self.overall_scaling),
                    # apply_scaling(
                    #     output_layer(hidden_features),
                    #     self.scalings[self.output_to_index[output_name]].item(),
                    # ),
                    self.composition_weights[  # type: ignore
                        self.output_to_index[output_name]
                    ],
                )

        # Sum the atomic energies to get the output
        for output_name, atomic_energy in atomic_energies.items():
            return_dict[output_name] = metatensor.torch.sum_over_samples(
                atomic_energy, ["atom", "center_type"]
            )

        return return_dict

    @classmethod
    def load_checkpoint(cls, path: Union[str, Path]) -> "SoapBpnn":

        # Load the checkpoint
        checkpoint = torch.load(path)
        model_hypers = checkpoint["model_hypers"]
        model_state_dict = checkpoint["model_state_dict"]

        # Create the model
        model = cls(**model_hypers)
        dtype = next(iter(model_state_dict.values())).dtype
        model.to(dtype).load_state_dict(model_state_dict)

        return model

    def export(self) -> MetatensorAtomisticModel:
        dtype = next(self.parameters()).dtype
        if dtype not in self.__supported_dtypes__:
            raise ValueError(f"unsupported dtype {self.dtype} for PhACE")

        capabilities = ModelCapabilities(
            outputs=self.outputs,
            atomic_types=self.all_species,
            interaction_range=self.model_hypers["cutoff"],
            length_unit=self.dataset_info.length_unit,
            supported_devices=self.__supported_devices__,
            dtype=dtype_to_str(dtype),
        )

        return export(model=self, model_capabilities=capabilities)

    @torch.jit.export
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
        model_hypers_bpnn = self.model_hypers["bpnn"]
        if model_hypers_bpnn["num_hidden_layers"] == 0:
            n_inputs_last_layer = model_hypers_bpnn["input_size"]
        else:
            n_inputs_last_layer = model_hypers_bpnn["num_neurons_per_layer"]
        self.last_layers[output_name] = InvariantLinear(self.all_species, n_inputs_last_layer)

    def requested_neighbor_lists(
        self,
    ) -> List[NeighborListOptions]:
        return [
            NeighborListOptions(
                cutoff=self.cutoff_radius,
                full_list=True,
            )
        ]

    def save_checkpoint(self, path: Union[str, Path]):
        torch.save(
            {
                "model_model_hypers": {
                    "model_model_hypers": self.model_hypers,
                    "dataset_info": self.dataset_info,
                },
                "model_state_dict": self.state_dict(),
            },
            check_file_extension(path, ".ckpt"),
        )
