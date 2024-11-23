from pathlib import Path
from typing import Dict, List, Optional, Union

import metatensor.torch
import torch
from metatensor.torch import Labels, TensorMap
from metatensor.torch.atomistic import (
    MetatensorAtomisticModel,
    ModelCapabilities,
    ModelMetadata,
    ModelOutput,
    NeighborListOptions,
    System,
)

from ...utils.additive import ZBL, CompositionModel
from ...utils.data.dataset import DatasetInfo, TargetInfo
from ...utils.dtype import dtype_to_str
from ...utils.io import check_file_extension
from .modules.center_embedding import embed_centers
from .modules.cg import cgs_to_sparse, get_cg_coefficients
from .modules.initial_features import get_initial_features
from .modules.layers import InvariantLinear, InvariantMLP
from .modules.message_passing import InvariantMessagePasser
from .modules.precomputations import Precomputer
from .utils import systems_to_batch
from ...utils.scaler import Scaler


class PhACE(torch.nn.Module):

    __supported_devices__ = ["cuda", "cpu"]
    __supported_dtypes__ = [torch.float64, torch.float32]

    def __init__(self, model_hypers: Dict, dataset_info: DatasetInfo) -> None:
        super().__init__()
        self.hypers = model_hypers
        self.dataset_info = dataset_info
        self.new_outputs = list(dataset_info.targets.keys())
        self.atomic_types = sorted(dataset_info.atomic_types)

        model_hypers["normalize"] = True

        self.cutoff_radius = model_hypers["cutoff"]
        self.dataset_info = dataset_info
        self.model_hypers = model_hypers

        self.nu_scaling = model_hypers["nu_scaling"]
        self.mp_scaling = model_hypers["mp_scaling"]
        self.overall_scaling = model_hypers["overall_scaling"]

        n_channels = model_hypers["n_element_channels"]

        species_to_species_index = torch.zeros(
            (max(self.atomic_types) + 1,), dtype=torch.int
        )
        species_to_species_index[self.atomic_types] = torch.arange(
            len(self.atomic_types), dtype=torch.int
        )
        self.register_buffer("species_to_species_index", species_to_species_index)
        print("species_to_species_index", self.species_to_species_index)
        self.embeddings = torch.nn.Embedding(len(self.atomic_types), n_channels)

        self.nu_max = model_hypers["nu_max"]
        self.n_message_passing_layers = model_hypers["n_message_passing_layers"]
        if self.n_message_passing_layers < 1:
            raise ValueError("Number of message-passing layers must be at least 1")

        self.invariant_message_passer = InvariantMessagePasser(
            model_hypers,
            self.atomic_types,
            self.mp_scaling,
            model_hypers["disable_nu_0"],
        )

        self.atomic_types = self.atomic_types
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
        if model_hypers["use_mops"]:
            cgs = cgs_to_sparse(cgs, self.l_max)

        self.precomputer = Precomputer(
            self.l_max, use_sphericart=model_hypers["use_sphericart"]
        )

        if model_hypers["use_mops"]:
            from .modules.cg_iterator_mops import CGIterator
            from .modules.message_passing_mops import EquivariantMessagePasser
        else:
            from .modules.cg_iterator import CGIterator
            from .modules.message_passing import EquivariantMessagePasser

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
            if idx == 0:
                irreps_equiv_mp = self.cg_iterator.irreps_out
            else:
                irreps_equiv_mp = generalized_cg_iterators[-1].irreps_out
            equivariant_message_passer = EquivariantMessagePasser(
                model_hypers,
                self.atomic_types,
                irreps_equiv_mp,
                [1, self.nu_max, self.nu_max + 1],
                cgs,
                self.mp_scaling,
            )
            equivariant_message_passers.append(equivariant_message_passer)
            generalized_cg_iterator = CGIterator(
                self.k_max_l,
                self.nu_max - 1,
                cgs,
                irreps_in=equivariant_message_passer.irreps_out,
                requested_LS_string=(
                    "0_1" if idx == self.n_message_passing_layers - 2 else None
                ),
            )
            generalized_cg_iterators.append(generalized_cg_iterator)

        self.equivariant_message_passers = torch.nn.ModuleList(
            equivariant_message_passers
        )
        self.generalized_cg_iterators = torch.nn.ModuleList(generalized_cg_iterators)

        self.last_mlp = InvariantMLP(self.k_max_l[0])

        self.last_layers = torch.nn.ModuleDict({})
        self.outputs = {
            "mtt::aux::last_layer_features": ModelOutput(unit="unitless", per_atom=True)
        }  # the model is always capable of outputting the last-layer features
        self.last_layers = torch.nn.ModuleDict({})
        for target_name, target in dataset_info.targets.items():
            self._add_output(target_name, target)

        # additive models: these are handled by the trainer at training
        # time, and they are added to the output at evaluation time
        composition_model = CompositionModel(
            model_hypers={},
            dataset_info=dataset_info,
        )
        additive_models = [composition_model]
        if self.hypers["zbl"]:
            additive_models.append(ZBL(model_hypers, dataset_info))
        self.additive_models = torch.nn.ModuleList(additive_models)

        # scaler: this is also handled by the trainer at training time
        self.scaler = Scaler(model_hypers={}, dataset_info=dataset_info)

    def restart(self, dataset_info: DatasetInfo) -> "PhACE":
        # merge old and new dataset info
        merged_info = self.dataset_info.union(dataset_info)
        new_atomic_types = [
            at for at in merged_info.atomic_types if at not in self.atomic_types
        ]
        new_targets = {
            key: value
            for key, value in merged_info.targets.items()
            if key not in self.dataset_info.targets
        }
        self.has_new_targets = len(new_targets) > 0

        if len(new_atomic_types) > 0:
            raise ValueError(
                f"New atomic types found in the dataset: {new_atomic_types}. "
                "The SOAP-BPNN model does not support adding new atomic types."
            )

        # register new outputs as new last layers
        for target_name, target in new_targets.items():
            self._add_output(target_name, target)

        self.dataset_info = merged_info

        # restart the composition and scaler models
        self.additive_models[0].restart(dataset_info)
        self.scaler.restart(dataset_info)

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
            self.k_max_l[0],
        )
        initial_element_embedding = embed_centers(initial_features, center_embeddings)

        spherical_expansion = self.invariant_message_passer(
            r,
            sh,
            structures["structure_offsets"][structures["structure_pairs"]]
            + structures["pairs"][:, 0],
            structures["structure_offsets"][structures["structure_pairs"]]
            + structures["pairs"][:, 1],
            n_atoms,
            initial_element_embedding,
            samples,
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
                samples,
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
                atomic_energies[output_name] = metatensor.torch.multiply(
                    output_layer(features), self.overall_scaling
                )

        # Sum the atomic energies to get the output
        for output_name, atomic_energy in atomic_energies.items():
            return_dict[output_name] = metatensor.torch.sum_over_samples(
                atomic_energy, ["atom", "center_type"]
            )

        if not self.training:
            # at evaluation, we also introduce the scaler and additive contributions
            return_dict = self.scaler(return_dict)
            for additive_model in self.additive_models:
                # some of the outputs might not be present in the additive model
                # (e.g. the composition model only provides outputs for scalar targets)
                outputs_for_additive_model: Dict[str, ModelOutput] = {}
                for output_name in outputs:
                    if output_name in additive_model.outputs:
                        outputs_for_additive_model[output_name] = outputs[output_name]
                additive_contributions = additive_model(
                    systems, outputs_for_additive_model, selected_atoms
                )
                for name in additive_contributions:
                    if name.startswith("mtt::aux::"):
                        continue  # skip auxiliary outputs (not targets)
                    return_dict[name] = metatensor.torch.add(
                        return_dict[name],
                        additive_contributions[name],
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
            raise ValueError(f"unsupported dtype {self.dtype} for SoapBpnn")

        # Make sure the model is all in the same dtype
        # For example, after training, the additive models could still be in
        # float64
        self.to(dtype)

        interaction_ranges = [
            self.hypers["cutoff"] * self.hypers["n_message_passing_layers"]
        ]
        for additive_model in self.additive_models:
            if hasattr(additive_model, "cutoff_radius"):
                interaction_ranges.append(additive_model.cutoff_radius)
        interaction_range = max(interaction_ranges)

        capabilities = ModelCapabilities(
            outputs=self.outputs,
            atomic_types=self.atomic_types,
            interaction_range=interaction_range,
            length_unit=self.dataset_info.length_unit,
            supported_devices=self.__supported_devices__,
            dtype=dtype_to_str(dtype),
        )

        return MetatensorAtomisticModel(self.eval(), ModelMetadata(), capabilities)

    def _add_output(self, target_name: str, target: TargetInfo) -> None:

        if target.is_scalar:
            self.last_layers[target_name] = InvariantLinear(self.k_max_l[0])
        else:
            raise NotImplementedError("PhACE does not support non-scalar targets.")

        self.outputs[target_name] = ModelOutput(
            quantity=target.quantity,
            unit=target.unit,
            per_atom=True,
        )

    def requested_neighbor_lists(
        self,
    ) -> List[NeighborListOptions]:
        return [
            NeighborListOptions(
                cutoff=self.cutoff_radius,
                full_list=True,
                strict=True,
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
