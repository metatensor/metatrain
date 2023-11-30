import torch
from typing import Dict, List

import metatensor.torch
from metatensor.torch import Labels, TensorBlock, TensorMap
import rascaline.torch


class MLPMap(torch.nn.Module):
    def __init__(self, all_species: List[int], hypers: dict) -> None:
        super().__init__()

        activation_function_name = hypers["activation_function"]
        if activation_function_name == "SiLU":
            self.activation_function = torch.nn.SiLU()
        else:
            raise ValueError(f"Unsupported activation function: {activation_function_name}")

        # Build a neural network for each species
        nns_per_species = []
        for _ in all_species:
            module_list = [
                torch.nn.Linear(hypers["input_size"], hypers["num_neurons_per_layers"]),
                torch.nn.SiLU(),
            ]
            for _ in range(hypers["num_hidden_layers"]):
                module_list.append(torch.nn.Linear(hypers["num_neurons_per_layers"], hypers["num_neurons_per_layers"]))
                module_list.append(torch.nn.SiLU())

            # If there are no hidden layers, the number of inputs for the last layer is the input size 
            n_inputs_last_layer = hypers["num_neurons_per_layers"] if hypers["num_hidden_layers"] > 0 else hypers["input_size"]
            
            module_list.append(torch.nn.Linear(n_inputs_last_layer, hypers["output_size"]))
            nns_per_species.append(torch.nn.Sequential(*module_list))

        # Create a module dict to store the neural networks
        self.layers = torch.nn.ModuleDict({
            str(species): nn for species, nn in zip(all_species, nns_per_species)
        })

    def forward(self, features: TensorMap) -> TensorMap:
        new_blocks: List[TensorBlock] = []
        for species_str, network in self.layers.items():
            species = int(species_str)
            # Here, do we have to check that the species is actually present in the system?
            block = features.block({"species_center": species})
            output_values = network(block.values)
            new_blocks.append(
                TensorBlock(
                    values=output_values,
                    samples=block.samples,
                    components=block.components,
                    properties=Labels.range("properties", output_values.shape[-1])
                )
            )
        return TensorMap(
            keys=features.keys,
            blocks=new_blocks
        )
            

class SoapBPNN(torch.nn.Module):
    def __init__(self, all_species, hypers) -> None:
        super().__init__()
        self.soap_calculator = rascaline.torch.PowerSpectrum(
            hypers["soap"]
        )
        hypers_bpnn = hypers["bpnn"]
        hypers_bpnn["input_size"] = hypers["soap"]["max_radial"]**2 * (hypers["soap"]["max_angular"] + 1)
        hypers_bpnn["output_size"] = 1
        self.bpnn = MLPMap(all_species, hypers["bpnn"])
        self.neighbor_species_1_labels = Labels(
            names=["species_neighbor_1"],
            values=torch.tensor(all_species).reshape(-1, 1)
        )
        self.neighbor_species_2_labels = Labels(
            names=["species_neighbor_2"],
            values=torch.tensor(all_species).reshape(-1, 1)
        )

    def forward(self, systems: List[rascaline.torch.System]) -> Dict[str, TensorMap]:
        
        soap_features = self.soap_calculator(systems)

        device = soap_features.block(0).values.device
        soap_features = soap_features.keys_to_properties(self.neighbor_species_1_labels.to(device))
        soap_features = soap_features.keys_to_properties(self.neighbor_species_2_labels.to(device))

        atomic_energies = self.bpnn(soap_features)
        atomic_energies = atomic_energies.keys_to_samples("species_center")

        # Sum the atomic energies coming from the BPNN to get the total energy
        total_energies = metatensor.torch.sum(atomic_energies, ["center", "species_center"])

        return {"energy": total_energies}


