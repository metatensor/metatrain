import copy
from typing import Dict, List, Optional

import metatensor.torch
import rascaline.torch
import torch
from metatensor.learn.data import DataLoader
from metatensor.torch import Labels, TensorBlock, TensorMap
from metatensor.torch.atomistic import ModelCapabilities, ModelOutput, System

from ...utils.composition import apply_composition_contribution
from ...utils.output_gradient import compute_gradient


class Identity(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: TensorMap) -> TensorMap:
        return x


class MLPMap(torch.nn.Module):
    def __init__(self, all_species: List[int], hypers: dict) -> None:
        super().__init__()

        self.activation_function = torch.nn.SiLU()

        # Build a neural network for each species
        nns_per_species = []
        for _ in all_species:
            module_list: List[torch.nn.Module] = []
            for _ in range(hypers["num_hidden_layers"]):
                if len(module_list) == 0:
                    module_list.append(
                        torch.nn.Linear(
                            hypers["input_size"], hypers["num_neurons_per_layer"]
                        )
                    )
                else:
                    module_list.append(
                        torch.nn.Linear(
                            hypers["num_neurons_per_layer"],
                            hypers["num_neurons_per_layer"],
                        )
                    )
                module_list.append(self.activation_function)

            nns_per_species.append(torch.nn.Sequential(*module_list))

        # Create a module dict to store the neural networks
        self.layers = torch.nn.ModuleDict(
            {str(species): nn for species, nn in zip(all_species, nns_per_species)}
        )

    def forward(self, features: TensorMap) -> TensorMap:
        # Create a list of the blocks that are present in the features:
        present_blocks = [
            int(features.keys.entry(i).values.item())
            for i in range(features.keys.values.shape[0])
        ]

        new_keys: List[int] = []
        new_blocks: List[TensorBlock] = []
        for species_str, network in self.layers.items():
            species = int(species_str)
            if species in present_blocks:
                new_keys.append(species)
                block = features.block({"center_type": species})
                output_values = network(block.values)
                new_blocks.append(
                    TensorBlock(
                        values=output_values,
                        samples=block.samples,
                        components=block.components,
                        # cannot use Labels.range() here because of torch.jit.save
                        properties=Labels(
                            names=["properties"],
                            values=torch.arange(
                                output_values.shape[1], device=output_values.device
                            ).reshape(-1, 1),
                        ),
                    )
                )
        new_keys_labels = Labels(
            names=["center_type"],
            values=torch.tensor(new_keys, device=new_blocks[0].values.device).reshape(
                -1, 1
            ),
        )

        return TensorMap(keys=new_keys_labels, blocks=new_blocks)


class LayerNormMap(torch.nn.Module):
    def __init__(self, all_species: List[int], n_layer: int) -> None:
        super().__init__()

        # Initialize a layernorm for each species
        layernorm_per_species = []
        for _ in all_species:
            layernorm_per_species.append(torch.nn.LayerNorm((n_layer,)))

        # Create a module dict to store the neural networks
        self.layernorms = torch.nn.ModuleDict(
            {
                str(species): layer
                for species, layer in zip(all_species, layernorm_per_species)
            }
        )

    def forward(self, features: TensorMap) -> TensorMap:
        # Create a list of the blocks that are present in the features:
        present_blocks = [
            int(features.keys.entry(i).values.item())
            for i in range(features.keys.values.shape[0])
        ]

        new_keys: List[int] = []
        new_blocks: List[TensorBlock] = []
        for species_str, layer in self.layernorms.items():
            species = int(species_str)
            if species in present_blocks:
                new_keys.append(species)
                block = features.block({"center_type": species})
                output_values = layer(block.values)
                new_blocks.append(
                    TensorBlock(
                        values=output_values,
                        samples=block.samples,
                        components=block.components,
                        properties=block.properties,
                    )
                )
        new_keys_labels = Labels(
            names=["center_type"],
            values=torch.tensor(new_keys, device=new_blocks[0].values.device).reshape(
                -1, 1
            ),
        )

        return TensorMap(keys=new_keys_labels, blocks=new_blocks)


class LinearMap(torch.nn.Module):
    def __init__(self, all_species: List[int], n_inputs: int) -> None:
        super().__init__()

        # Build a neural network for each species
        layer_per_species = []
        for _ in all_species:
            layer_per_species.append(torch.nn.Linear(n_inputs, 1))

        # Create a module dict to store the neural networks
        self.layers = torch.nn.ModuleDict(
            {
                str(species): layer
                for species, layer in zip(all_species, layer_per_species)
            }
        )

    def forward(self, features: TensorMap) -> TensorMap:
        # Create a list of the blocks that are present in the features:
        present_blocks = [
            int(features.keys.entry(i).values.item())
            for i in range(features.keys.values.shape[0])
        ]

        new_keys: List[int] = []
        new_blocks: List[TensorBlock] = []
        for species_str, layer in self.layers.items():
            species = int(species_str)
            if species in present_blocks:
                new_keys.append(species)
                block = features.block({"center_type": species})
                output_values = layer(block.values)
                new_blocks.append(
                    TensorBlock(
                        values=output_values,
                        samples=block.samples,
                        components=block.components,
                        properties=Labels(
                            names=["energy"],
                            values=torch.zeros(
                                (1, 1), dtype=torch.int32, device=block.values.device
                            ),
                        ),
                    )
                )
        new_keys_labels = Labels(
            names=["center_type"],
            values=torch.tensor(new_keys, device=new_blocks[0].values.device).reshape(
                -1, 1
            ),
        )

        return TensorMap(keys=new_keys_labels, blocks=new_blocks)


class Identity(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: TensorMap) -> TensorMap:
        return x


class Model(torch.nn.Module):
    def __init__(
        self, capabilities: ModelCapabilities, hypers: Dict = DEFAULT_MODEL_HYPERS
    ) -> None:
        super().__init__()
        self.name = ARCHITECTURE_NAME

        self.capabilities = capabilities
        self.all_species = capabilities.atomic_types
        self.neighbor_combinations = torch.combinations(
            torch.Tensor(self.all_species),
            with_replacement=True,
        ).to(torch.int)
        self.hypers = hypers

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

        self.soap_calculator = rascaline.torch.SoapPowerSpectrum(
            radial_basis={"Gto": {}}, **hypers["soap"]
        )
        soap_size = (
            (len(self.all_species) * (len(self.all_species) + 1) // 2)
            * hypers["soap"]["max_radial"] ** 2
            * (hypers["soap"]["max_angular"] + 1)
        )

        hypers_bpnn = hypers["bpnn"]
        hypers_bpnn["input_size"] = soap_size

        if hypers_bpnn["layernorm"]:
            self.layernorm = LayerNormMap(self.all_species, soap_size)
        else:
            self.layernorm = Identity()

        self.bpnn = MLPMap(self.all_species, hypers_bpnn)

        self.neighbor_species_labels = Labels(
            names=["neighbor_1_type", "neighbor_2_type"],
            values=torch.combinations(
                torch.tensor(self.all_species, dtype=torch.int),
                with_replacement=True,
            ),
        )
        self.center_type_labels = Labels(
            names=["center_type"],
            values=torch.tensor(self.all_species).reshape(-1, 1),
        )
        self.center_type_labels = Labels(
            names=["center_type"],
            values=torch.tensor(self.all_species).reshape(-1, 1),
        )

        if hypers_bpnn["num_hidden_layers"] == 0:
            self.n_inputs_last_layer = hypers_bpnn["input_size"]
        else:
            self.n_inputs_last_layer = hypers_bpnn["num_neurons_per_layer"]

        self.last_layers = torch.nn.ModuleDict(
            {
                output_name: LinearMap(self.all_species, self.n_inputs_last_layer)
                for output_name in capabilities.outputs.keys()
                if "mtm::aux::" not in output_name
            }
        )

        # attribute for wrapper models using the last layer features
        self.ll_feat_size = self.n_inputs_last_layer * len(self.all_species)

    def forward(
        self,
        systems: List[System],
        outputs: Dict[str, ModelOutput],
        selected_atoms: Optional[Labels] = None,
    ) -> Dict[str, TensorMap]:
        # initialize the return dictionary
        return_dict: Dict[str, TensorMap] = {}

        soap_features = self.soap_calculator(systems, selected_samples=selected_atoms)
        device = soap_features.block(0).values.device
        soap_features = soap_features.keys_to_properties(
            self.neighbor_species_labels.to(device)
        )

        soap_features = self.layernorm(soap_features)
        last_layer_features = self.bpnn(soap_features)

        # output the hidden features, if requested:
        if "mtm::aux::last_layer_features" in outputs:
            last_layer_features_options = outputs["mtm::aux::last_layer_features"]
            out_features = last_layer_features.keys_to_properties(
                self.center_type_labels.to(device)
            )
            if not last_layer_features_options.per_atom:
                out_features = metatensor.torch.sum_over_samples(out_features, ["atom"])
            return_dict["mtm::aux::last_layer_features"] = (
                _remove_center_type_from_properties(out_features)
            )

        atomic_energies: Dict[str, TensorMap] = {}
        for output_name, output_layer in self.last_layers.items():
            if output_name in outputs:
                atomic_energies[output_name] = apply_composition_contribution(
                    output_layer(last_layer_features),
                    self.composition_weights[  # type: ignore
                        self.output_to_index[output_name]
                    ],
                )

        # Sum the atomic energies coming from the BPNN to get the total energy
        for output_name, atomic_energy in atomic_energies.items():
            atomic_energy = atomic_energy.keys_to_samples("center_type")
            if outputs[output_name].per_atom:
                # this operation should just remove the center_type label
                return_dict[output_name] = metatensor.torch.remove_dimension(
                    atomic_energy, axis="samples", name="center_type"
                )
            else:
                return_dict[output_name] = metatensor.torch.sum_over_samples(
                    atomic_energy, ["atom", "center_type"]
                )

        return return_dict

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


def _remove_center_type_from_properties(tensor_map: TensorMap) -> TensorMap:
    new_blocks: List[TensorBlock] = []
    for block in tensor_map.blocks():
        new_blocks.append(
            TensorBlock(
                values=block.values,
                samples=block.samples,
                components=block.components,
                properties=Labels(
                    names=["properties"],
                    values=torch.arange(
                        block.values.shape[-1], device=block.values.device
                    ).reshape(-1, 1),
                ),
            )
        )
    return TensorMap(keys=tensor_map.keys, blocks=new_blocks)


class LLPRModel(torch.nn.Module):
    def __init__(
        self,
        model: Model,
    ) -> None:

        super().__init__()
        self.orig_model = copy.deepcopy(model)

        # initialize (inv_)covariance matrices
        self.ll_feat_size = self.orig_model.ll_feat_size
        self.register_buffer(
            "covariance",
            torch.zeros(
                (self.ll_feat_size, self.ll_feat_size),
                device=next(self.orig_model.parameters()).device,
            ),
        )
        self.register_buffer(
            "inv_covariance",
            torch.zeros(
                (self.ll_feat_size, self.ll_feat_size),
                device=next(self.orig_model.parameters()).device,
            ),
        )

        self.covariance_computed = False
        self.covariance_gradients_computed = False
        self.inv_covariance_computed = False

    def forward(
        self,
        systems: List[System],
        outputs: Dict[str, ModelOutput],
        selected_atoms: Optional[Labels] = None,
    ) -> Dict[str, TensorMap]:

        return_dict = self.orig_model(systems, outputs, selected_atoms)
        last_layer_features_options = outputs["last_layer_features"]

        # automatically provide uncertainties if inv_covariance is computed
        if self.inv_covariance_computed:

            cur_ll_feat_map = return_dict["last_layer_features"]

            if last_layer_features_options.per_atom:
                lpr_values = torch.einsum(
                    "ij, jk, ik -> i",
                    cur_ll_feat_map.block().values,
                    self.inv_covariance,
                    cur_ll_feat_map.block().values,
                )
                lpr_values = 1 / lpr_values.unsqueeze(1)

                lpr_map = TensorMap(
                    keys=Labels.single(),
                    blocks=[
                        TensorBlock(
                            values=lpr_values,
                            samples=cur_ll_feat_map.block().samples,
                            components=cur_ll_feat_map.block().components,
                            properties=Labels.single(),
                        )
                    ],
                )
                return_dict["LPR"] = lpr_map

                # modify ll feats to sums to output uncertainties
                cur_ll_feat_map = metatensor.torch.sum_over_samples(
                    cur_ll_feat_map, ["atom"]
                )

            else:
                num_atoms = torch.Tensor([len(sys) for sys in systems]).unsqueeze(-1)
                pr_values = torch.einsum(
                    "ij, jk, ik -> i",
                    cur_ll_feat_map.block().values / num_atoms,
                    self.inv_covariance,
                    cur_ll_feat_map.block().values / num_atoms,
                )
                pr_values = 1 / pr_values.unsqueeze(1)

                pr_map = TensorMap(
                    keys=Labels.single(),
                    blocks=[
                        TensorBlock(
                            values=pr_values,
                            samples=cur_ll_feat_map.block().samples,
                            components=cur_ll_feat_map.block().components,
                            properties=Labels.single(),
                        )
                    ],
                )
                return_dict["PR"] = pr_map

            # output EXTENSIVE uncertainties
            # (computed with the sum of atomic ll feats)
            uncertainty_values = torch.einsum(
                "ij, jk, ik -> i",
                cur_ll_feat_map.block().values,
                self.inv_covariance,
                cur_ll_feat_map.block().values,
            )
            uncertainty_values = uncertainty_values.unsqueeze(1)

            uncertainty_map = TensorMap(
                keys=Labels.single(),
                blocks=[
                    TensorBlock(
                        values=uncertainty_values,
                        samples=cur_ll_feat_map.block().samples,
                        components=cur_ll_feat_map.block().components,
                        properties=Labels.single(),
                    )
                ],
            )
            return_dict["energy_uncertainty"] = uncertainty_map

        return return_dict

    def compute_covariance(self, systems: List[System]) -> None:
        # Utility function to compute the covariance matrix for a training set.
        device = self.covariance.device
        systems = [system.to(device=device) for system in systems]
        output_dict = {
            "last_layer_features": ModelOutput(
                quantity="",
                unit="",
                per_atom=True,
            )
        }
        output = self.forward(systems, output_dict)
        ll_featmap = output["last_layer_features"]
        ll_featmap = metatensor.torch.mean_over_samples(ll_featmap, ["atom"])
        ll_feats = ll_featmap.block().values
        self.covariance += ll_feats.T @ ll_feats
        self.covariance_computed = True

    def compute_covariance_dataloader(self, train_loader: DataLoader) -> None:
        # Utility function to compute the covariance matrix for a training set.
        for batch in train_loader:
            systems, _ = batch
            self.compute_covariance(systems)

    def add_gradients_to_covariance(
        self,
        systems: List[System],
        F_E_loss_ratio: float = 1,
        training: bool = True,
    ) -> None:

        device = self.covariance.device
        systems = [system.to(device=device) for system in systems]
        for system in systems:
            system.positions.requires_grad_(True)
        output_dict = {"last_layer_features": ModelOutput(quantity="", unit="")}
        output = self.forward(systems, output_dict)

        # Compute gradients, feature by feature
        ll_feats = output["last_layer_features"].block().values
        ll_feat_grad_list = []
        for i in range(ll_feats.shape[-1]):
            ll_feat_grad = compute_gradient(
                ll_feats[:, i],
                [system.positions for system in systems],
                is_training=training,
            )
            ll_feat_grad_list.append(torch.cat(ll_feat_grad, dim=0).detach())
        ll_feat_grads = torch.stack(ll_feat_grad_list)
        ll_feat_grads = ll_feat_grads.permute(1, 2, 0)  # (n_atoms, 3, n_feats)

        # Reshape and add the gradients to covariance matrix,
        # Multiply by the force-to-energy loss ratio
        ll_feat_grads = ll_feat_grads.reshape(-1, ll_feats.shape[-1]) * F_E_loss_ratio
        self.covariance += ll_feat_grads.T @ ll_feat_grads

        self.covariance_gradients_computed = True
        self.inv_covariance_computed = False  # Force re-calibration for uncertainties

    def add_gradients_to_covariance_dataloader(
        self,
        train_loader: DataLoader,
        F_E_loss_ratio: float = 1,
    ) -> None:
        # Utility function to compute the covariance matrix for a training set.
        for batch in train_loader:
            systems, _ = batch
            self.add_gradients_to_covariance(systems)

    def compute_inv_covariance(self, C: float, sigma: float) -> None:
        # Utility function to set the hyperparameters of the uncertainty model.
        if not self.covariance_computed:
            raise RuntimeError(
                "You must compute the covariance matrix before "
                "computing the inverse covariance matrix!"
            )
        cov_prime = self.covariance + sigma**2 * torch.eye(
            self.ll_feat_size, device=self.covariance.device
        )

        if torch.linalg.matrix_rank(cov_prime) < self.covariance.shape[0]:
            raise RuntimeError(
                "Provided sigma value does not result in a full rank matrix!"
            )
        else:
            self.inv_covariance = C * torch.linalg.inv(cov_prime)
            self.inv_covariance_computed = True

    def get_ll_feats_dataloader(self, train_loader: DataLoader) -> TensorMap:
        # Utility function to get the ll feats with a dataloader
        device = self.covariance.device
        ll_featmap_list = []
        for batch in train_loader:
            systems, _ = batch
            systems = [system.to(device=device) for system in systems]
            output_dict = {
                "last_layer_features": ModelOutput(
                    quantity="",
                    unit="",
                    per_atom=True,
                )
            }
            output = self.forward(systems, output_dict)
            ll_featmap = output["last_layer_features"]
            ll_featmap_list.append(ll_featmap)
        return metatensor.torch.join(ll_featmap_list, axis="samples")
