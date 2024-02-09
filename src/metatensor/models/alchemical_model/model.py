from typing import Dict, List, Optional, Union

import metatensor.torch
import numpy as np
import torch
from metatensor.torch import Labels, TensorMap
from metatensor.torch.atomistic import ModelCapabilities, ModelOutput, System
from omegaconf import OmegaConf
from torch_alchemical.nn import AlchemicalEmbedding, LayerNorm, MultiChannelLinear, SiLU
from torch_alchemical.nn.power_spectrum import PowerSpectrum
from torch_alchemical.operations import sum_over_components
from torch_spex.spherical_expansions import SphericalExpansion

from .. import ARCHITECTURE_CONFIG_PATH
from ..utils.composition import apply_composition_contribution
from ..utils.normalize import apply_normalization
from .utils import systems_to_torch_spex_dict


DEFAULT_HYPERS = OmegaConf.to_container(
    OmegaConf.load(ARCHITECTURE_CONFIG_PATH / "alchemical_model.yaml")
)

DEFAULT_MODEL_HYPERS = DEFAULT_HYPERS["model"]

ARCHITECTURE_NAME = "alchemical_model"


class AlchemicalSoapCalculator(torch.nn.Module):
    def __init__(
        self,
        all_species: Union[list, np.ndarray],
        cutoff_radius: float,
        basis_cutoff: float,
        radial_basis_type: str = "le",
        basis_scale: float = 3.0,
        trainable_basis: bool = True,
        basis_normalization_factor: Optional[float] = None,
        num_pseudo_species: Optional[int] = None,
    ):
        super().__init__()
        if isinstance(all_species, np.ndarray):
            all_species = all_species.tolist()
        self.all_species = all_species
        self.cutoff_radius = cutoff_radius
        self.basis_cutoff = basis_cutoff
        self.basis_scale = basis_scale
        self.radial_basis_type = radial_basis_type
        self.basis_normalization_factor = basis_normalization_factor
        self.trainable_basis = trainable_basis
        self.num_pseudo_species = num_pseudo_species
        hypers = {
            "cutoff radius": self.cutoff_radius,
            "radial basis": {
                "type": self.radial_basis_type,
                "E_max": self.basis_cutoff,
                "mlp": self.trainable_basis,
                "scale": self.basis_scale,
                "cost_trade_off": False,
            },
        }
        if self.num_pseudo_species is not None:
            hypers["alchemical"] = self.num_pseudo_species
        if self.basis_normalization_factor:
            hypers["normalize"] = self.basis_normalization_factor
        self.spex_calculator = SphericalExpansion(
            hypers=hypers,
            all_species=self.all_species,
        )
        self.l_max = self.spex_calculator.vector_expansion_calculator.l_max
        self.ps_calculator = PowerSpectrum(self.l_max, all_species)

    def forward(self, systems: List[System]):
        batch_dict = systems_to_torch_spex_dict(systems)
        spex = self.spex_calculator(
            positions=batch_dict["positions"],
            cells=batch_dict["cells"],
            species=batch_dict["species"],
            cell_shifts=batch_dict["cell_shifts"],
            centers=batch_dict["centers"],
            pairs=batch_dict["pairs"],
            structure_centers=batch_dict["structure_centers"],
            structure_pairs=batch_dict["structure_pairs"],
            structure_offsets=batch_dict["structure_offsets"],
        )
        power_spectrum = self.ps_calculator(spex)
        return power_spectrum

    @property
    def num_features(self):
        vex_calculator = self.spex_calculator.vector_expansion_calculator
        n_max = vex_calculator.radial_basis_calculator.n_max_l
        l_max = len(n_max) - 1
        n_feat = sum(
            [n_max[l_ch] ** 2 * self.num_pseudo_species**2 for l_ch in range(l_max + 1)]
        )
        return n_feat


class Model(torch.nn.Module):
    def __init__(
        self, capabilities: ModelCapabilities, hypers: Dict = DEFAULT_MODEL_HYPERS
    ) -> None:
        super().__init__()
        self.name = ARCHITECTURE_NAME

        # Check capabilities
        for output in capabilities.outputs.values():
            if output.quantity != "energy":
                raise ValueError(
                    "Alchemical Model only supports energy-like outputs, "
                    f"but a {output.quantity} was provided"
                )
            if output.per_atom:
                raise ValueError(
                    "Alchemical Model only supports per-structure outputs, "
                    "but a per-atom output was provided"
                )

        self.capabilities = capabilities
        self.all_species = capabilities.species
        self.hypers = hypers

        # creates a composition weight tensor that can be directly indexed by species,
        # this can be left as a tensor of zero or set from the outside using
        # set_composition_weights (recommended for better accuracy)
        n_outputs = len(capabilities.outputs)
        self.register_buffer(
            "composition_weights", torch.zeros((n_outputs, max(self.all_species) + 1))
        )
        # creates a normalization factor for energies
        # this can be left as a tensor of 1.0 or set from the outside using
        # set_normalization_factor (recommended for better accuracy)
        self.register_buffer(
            "normalization_factor", torch.tensor(1.0, dtype=torch.float32)
        )
        # buffers cannot be indexed by strings (torchscript), so we create a single
        # tensor for all output. Due to this, we need to slice the tensor when we use
        # it and use the output name to select the correct slice via a dictionary
        self.output_to_index = {
            output_name: i for i, output_name in enumerate(capabilities.outputs.keys())
        }

        # TODO Inject basis_normalization_factor and device into the hypers
        self.soap_features_layer = AlchemicalSoapCalculator(
            all_species=self.all_species, **hypers["soap"]
        )

        self.num_pseudo_species = hypers["soap"]["num_pseudo_species"]
        ps_input_size = self.soap_features_layer.num_features

        self.layer_norm = LayerNorm(ps_input_size)
        vex_calculator = (
            self.soap_features_layer.spex_calculator.vector_expansion_calculator
        )
        contraction_layer = vex_calculator.radial_basis_calculator.combination_matrix
        self.embedding = AlchemicalEmbedding(
            unique_numbers=self.all_species,
            num_pseudo_species=self.num_pseudo_species,
            contraction_layer=contraction_layer,
        )

        num_hidden_layers = hypers["bpnn"]["num_hidden_layers"]
        num_neurons_per_layer = hypers["bpnn"]["num_neurons_per_layer"]
        activation_function = hypers["bpnn"]["activation_function"]
        if activation_function == "SiLU":
            self.activation_function = SiLU()
        else:
            raise ValueError(
                f"Activation function {activation_function} not supported."
            )

        layer_size = [ps_input_size] + [num_neurons_per_layer] * num_hidden_layers
        bpnn_layers = []
        for layer_index in range(1, len(layer_size)):
            bpnn_layers.append(
                MultiChannelLinear(
                    in_features=layer_size[layer_index - 1],
                    out_features=layer_size[layer_index],
                    num_channels=self.num_pseudo_species,
                    bias=False,
                )
            )
            bpnn_layers.append(self.activation_function)

        n_inputs_last_layer = layer_size[-1]
        n_outputs_last_layer = 1

        self.bpnn = torch.nn.ModuleList(bpnn_layers)

        self.last_layers = torch.nn.ModuleDict(
            {
                output_name: MultiChannelLinear(
                    in_features=n_inputs_last_layer,
                    out_features=n_outputs_last_layer,
                    num_channels=self.num_pseudo_species,
                    bias=False,
                )
                for output_name in capabilities.outputs.keys()
            }
        )

    def forward(
        self,
        systems: List[System],
        outputs: Dict[str, ModelOutput],
        selected_atoms: Optional[Labels] = None,
    ) -> Dict[str, TensorMap]:
        if selected_atoms is not None:
            raise NotImplementedError(
                "Alchemical Model does not support selected atoms."
            )

        for requested_output in outputs.keys():
            if requested_output not in self.capabilities.outputs.keys():
                raise ValueError(
                    f"Requested output {requested_output} is not within "
                    "the model's capabilities."
                )
        for system in systems:
            if len(system.known_neighbors_lists()) == 0:
                raise RuntimeError(
                    "Alchemical Model requires neighbor lists to be provided."
                )
            for element in system.species:
                if element not in self.all_species:
                    raise ValueError(
                        f"Current model doesn't support element {element}."
                    )

        soap_features = self.soap_features_layer(systems)
        soap_features = self.layer_norm(soap_features)
        hidden_features = self.embedding(soap_features)

        for layer in self.bpnn:
            hidden_features = layer(hidden_features)

        atomic_energies: Dict[str, TensorMap] = {}
        for output_name, output_layer in self.last_layers.items():
            if output_name in outputs:
                atomic_energies[output_name] = apply_composition_contribution(
                    sum_over_components(output_layer(hidden_features)),
                    self.composition_weights[self.output_to_index[output_name]],
                )

        total_energies: Dict[str, TensorMap] = {}
        for output_name, atomic_energy in atomic_energies.items():
            atomic_energy = atomic_energy.keys_to_samples("species_center")
            total_energies_item = metatensor.torch.sum_over_samples(
                atomic_energy, ["center", "species_center"]
            )
            normalization_factor = self.normalization_factor * torch.sqrt(
                torch.tensor(self.num_pseudo_species)
            )
            total_energies_item = apply_normalization(
                total_energies_item, normalization_factor
            )
            total_energies[output_name] = total_energies_item
            # Change the energy label from _ to (0, 1):
            total_energies[output_name] = TensorMap(
                keys=Labels(
                    names=["lambda", "sigma"],
                    values=torch.tensor([[0, 1]]),
                ),
                blocks=[total_energies[output_name].block()],
            )

        return total_energies

    def set_composition_weights(
        self, output_name: str, input_composition_weights: torch.Tensor
    ) -> None:
        """Set the composition weights for a given output."""
        # all species that are not present retain their weight of zero
        self.composition_weights[self.output_to_index[output_name]][
            self.all_species
        ] = input_composition_weights

    def set_normalization_factor(self, normalization_factor: torch.Tensor) -> None:
        """Set the normalization factor for output of the model."""
        self.normalization_factor = normalization_factor

    def set_basis_normalization_factor(self, basis_normalization_factor: torch.Tensor):
        """Set the normalization factor for the basis functions of the model."""
        self.soap_features_layer.spex_calculator.normalization_factor = (
            1.0 / torch.sqrt(basis_normalization_factor)
        )
        self.soap_features_layer.spex_calculator.normalization_factor_0 = (
            1.0 / basis_normalization_factor ** (3 / 4)
        )
