import logging
from typing import Any, Dict, List, Literal, Optional

import mace.modules as mace_modules
import torch
from e3nn import o3
from e3nn.util import jit
from mace.modules import MACE
from metatensor.torch import Labels, TensorMap
from metatomic.torch import (
    AtomisticModel,
    ModelCapabilities,
    ModelMetadata,
    ModelOutput,
    NeighborListOptions,
    System,
)

from metatrain.utils.abc import ModelInterface
from metatrain.utils.additive import CompositionModel
from metatrain.utils.data import DatasetInfo, TargetInfo
from metatrain.utils.dtype import dtype_to_str
from metatrain.utils.metadata import merge_metadata
from metatrain.utils.scaler import Scaler

from .modules.finetuning import apply_finetuning_strategy
from .modules.heads import NonLinearHead
from .modules.structures import create_batch
from .utils.mts import (
    add_contribution,
    e3nn_to_tensormap,
    get_system_indices_and_labels,
)


class MetaMACE(ModelInterface):
    """Interface of MACE for metatrain."""

    __checkpoint_version__ = 1
    __supported_devices__ = ["cuda", "cpu"]
    __supported_dtypes__ = [torch.float64, torch.float32]
    __default_metadata__ = ModelMetadata(
        references={
            "architecture": [
                "https://arxiv.org/abs/2205.06643",
                "https://openreview.net/forum?id=YPpSngE-ZU",
            ]
        }
    )

    def __init__(self, hypers: Dict, dataset_info: DatasetInfo) -> None:
        super().__init__(hypers, dataset_info, self.__default_metadata__)

        self.requested_nl = NeighborListOptions(
            cutoff=self.hypers["cutoff"],
            full_list=True,
            strict=True,
        )

        self.cutoff = float(self.hypers["cutoff"])

        if self.hypers["mace_model"] is not None:
            self.mace_model = torch.load(self.hypers["mace_model"], weights_only=False)
        else:
            self.mace_model = MACE(
                r_max=self.cutoff,
                num_bessel=self.hypers["num_radial_basis"],
                num_polynomial_cutoff=self.hypers["num_cutoff_basis"],
                max_ell=self.hypers["max_ell"],
                interaction_cls=mace_modules.interaction_classes[
                    self.hypers["interaction"]
                ],
                num_interactions=self.hypers["num_interactions"],
                num_elements=len(dataset_info.atomic_types),
                hidden_irreps=o3.Irreps(self.hypers["hidden_irreps"]),
                edge_irreps=o3.Irreps(self.hypers["edge_irreps"])
                if "edge_irreps" in self.hypers
                else None,
                atomic_energies=torch.zeros(len(dataset_info.atomic_types)),
                apply_cutoff=self.hypers["apply_cutoff"],
                avg_num_neighbors=self.hypers["avg_num_neighbors"],
                atomic_numbers=dataset_info.atomic_types,
                pair_repulsion=self.hypers["pair_repulsion"],
                distance_transform=self.hypers["distance_transform"],
                correlation=self.hypers["correlation"],
                gate=mace_modules.gate_dict[self.hypers["gate"]]
                if self.hypers["gate"] is not None
                else None,
                interaction_cls_first=mace_modules.interaction_classes[
                    self.hypers["interaction_first"]
                ],
                MLP_irreps=o3.Irreps(self.hypers["MLP_irreps"]),
                radial_MLP=self.hypers["radial_MLP"],
                radial_type=self.hypers["radial_type"],
                use_embedding_readout=self.hypers["use_embedding_readout"],
                use_last_readout_only=self.hypers["use_last_readout_only"],
                use_agnostic_product=self.hypers["use_agnostic_product"],
            )

        self.mace_head_target = str(self.hypers["mace_head_target"])

        self.atomic_types = self.mace_model.atomic_numbers.tolist()

        self.register_buffer(
            "atomic_types_to_species_index",
            torch.zeros(max(self.atomic_types) + 1, dtype=torch.int64),
        )
        for i, atomic_type in enumerate(self.atomic_types):
            self.atomic_types_to_species_index[atomic_type] = i

        self.outputs = {"features": ModelOutput(unit="", per_atom=True)}
        self.heads = torch.nn.ModuleDict()
        for target_name, target_info in dataset_info.targets.items():
            self._add_output(target_name, target_info)

        composition_model = CompositionModel(
            hypers={},
            dataset_info=DatasetInfo(
                length_unit=dataset_info.length_unit,
                atomic_types=self.atomic_types,
                targets={
                    target_name: target_info
                    for target_name, target_info in dataset_info.targets.items()
                    if CompositionModel.is_valid_target(target_name, target_info)
                },
            ),
        )
        additive_models = [composition_model]
        self.additive_models = torch.nn.ModuleList(additive_models)

        # scaler: this is also handled by the trainer at training time
        self.scaler = Scaler(hypers={}, dataset_info=dataset_info)

        self.finetune_config: Dict[str, Any] = {}

    def restart(self, dataset_info: DatasetInfo) -> "MetaMACE":
        # Check that the new dataset info does not contain new atomic types
        if new_atomic_types := set(dataset_info.atomic_types) - set(
            self.dataset_info.atomic_types
        ):
            raise ValueError(
                f"New atomic types found in the dataset: {new_atomic_types}. "
                "The MACE model does not support adding new atomic types."
            )

        # Merge the old dataset info with the new one
        merged_info = self.dataset_info.union(dataset_info)

        # Check if there are new targets
        new_targets = {
            key: value
            for key, value in merged_info.targets.items()
            if key not in self.dataset_info.targets
        }
        self.has_new_targets = len(new_targets) > 0

        # Add extra heads for the new targets
        for target_name, target in new_targets.items():
            self._add_output(target_name, target)

        self.dataset_info = merged_info

        # restart the composition and scaler models
        self.additive_models[0] = self.additive_models[0].restart(
            dataset_info=DatasetInfo(
                length_unit=dataset_info.length_unit,
                atomic_types=self.dataset_info.atomic_types,
                targets={
                    target_name: target_info
                    for target_name, target_info in dataset_info.targets.items()
                    if CompositionModel.is_valid_target(target_name, target_info)
                },
            ),
        )
        self.scaler = self.scaler.restart(dataset_info)

        return self

    def forward(
        self,
        systems: List[System],
        outputs: Dict[str, ModelOutput],
        selected_atoms: Optional[Labels] = None,
    ) -> Dict[str, TensorMap]:
        if selected_atoms is not None:
            raise NotImplementedError(
                "selected_atoms is not supported in MetaMACE for now. "
            )

        # Move everything to the same device
        device = systems[0].device
        self.dataset_info = self.dataset_info.to(device=device)

        # Create the batch to pass as input for MACE.
        # THIS PROBABLY SHOULD BE MOVED OUTSIDE THE MODEL!!
        # (But I don't know if this would affect the interfaces e.g. with
        # ASE, LAMMPS, etc.)
        data = create_batch(
            systems=systems,
            neighbor_list_options=self.requested_nl,
            atomic_types_to_species_index=self.atomic_types_to_species_index,
            n_types=len(self.atomic_types),
            device=device,
        )

        # Change coordinates to YZX
        data["positions"] = data["positions"][:, [1, 2, 0]]

        # Run MACE and extract the node features.
        mace_output = self.mace_model(data, training=self.training, compute_force=False)
        node_features = mace_output["node_feats"]
        assert node_features is not None  # For torchscript

        # Get the labels for the samples (system and atom of each value)
        _, sample_labels = get_system_indices_and_labels(systems, device)

        # Run all heads and collect outputs as TensorMaps
        return_dict: Dict[str, TensorMap] = {}
        for output_name, head in self.heads.items():
            # Get the per node target values
            if output_name == self.mace_head_target:
                # Use the internal MACE head
                node_energy = mace_output["node_energy"]
                assert node_energy is not None  # For torchscript
                node_target = node_energy.reshape(-1, 1)
            else:
                node_target = head.forward(node_features)

            # Convert to TensorMap and store
            return_dict[output_name] = e3nn_to_tensormap(
                node_target,
                sample_labels=sample_labels,
                target_info=self.dataset_info.targets[output_name],
                output_name=output_name,
                outputs=outputs,
            )

        if not self.training:
            # at evaluation, we also introduce the scaler and additive contributions
            return_dict = self.scaler(systems, return_dict)
            for additive_model in self.additive_models:
                add_contribution(
                    return_dict, systems, outputs, additive_model, selected_atoms
                )

        return return_dict

    def supported_outputs(self) -> Dict[str, ModelOutput]:
        return self.outputs

    def requested_neighbor_lists(
        self,
    ) -> List[NeighborListOptions]:
        return [self.requested_nl]

    @classmethod
    def load_checkpoint(
        cls,
        checkpoint: Dict[str, Any],
        context: Literal["restart", "finetune", "export"],
    ) -> "MetaMACE":
        if context == "restart":
            logging.info(f"Using latest model from epoch {checkpoint['epoch']}")
            model_state_dict = checkpoint["model_state_dict"]
        elif context in {"finetune", "export"}:
            logging.info(f"Using best model from epoch {checkpoint['best_epoch']}")
            model_state_dict = checkpoint["best_model_state_dict"]
            if model_state_dict is None:
                model_state_dict = checkpoint["model_state_dict"]
        else:
            raise ValueError("Unknown context tag for checkpoint loading!")

        # Create the model
        model_data = checkpoint["model_data"]
        model = cls(**model_data)
        dtype = None
        for k, v in model_state_dict.items():
            if k.endswith(".weight"):
                dtype = v.dtype
                break
        else:
            raise ValueError("Couldn't infer dtype from the checkpoint file")
        finetune_config = model_state_dict.pop("finetune_config", {})
        if finetune_config:
            # Apply the finetuning strategy
            model = apply_finetuning_strategy(model, finetune_config)
        model.to(dtype).load_state_dict(model_state_dict)
        model.additive_models[0].sync_tensor_maps()
        model.scaler.sync_tensor_maps()

        # Loading the metadata from the checkpoint
        model.metadata = merge_metadata(model.metadata, checkpoint.get("metadata"))

        return model

    def export(self, metadata: Optional[ModelMetadata] = None) -> AtomisticModel:
        dtype = next(self.parameters()).dtype
        if dtype not in self.__supported_dtypes__:
            raise ValueError(f"unsupported dtype {dtype} for MACE")

        # Make sure the model is all in the same dtype
        # For example, after training, the additive models could still be in
        # float64
        self.to(dtype)

        # Additionally, the composition model contains some `TensorMap`s that cannot
        # be registered correctly with Pytorch. This function moves them:
        self.additive_models[0].weights_to(torch.device("cpu"), torch.float64)

        interaction_ranges = [self.hypers["num_interactions"] * self.hypers["cutoff"]]
        interaction_range = max(interaction_ranges)

        capabilities = ModelCapabilities(
            outputs=self.outputs,
            atomic_types=self.atomic_types,
            interaction_range=interaction_range,
            length_unit=self.dataset_info.length_unit,
            supported_devices=self.__supported_devices__,
            dtype=dtype_to_str(dtype),
        )

        metadata = merge_metadata(self.metadata, metadata)

        return AtomisticModel(jit.compile(self.eval()), metadata, capabilities)

    def _add_output(self, target_name: str, target_info: TargetInfo) -> None:
        """
        Register a new output target by creating corresponding heads and last layers.

        :param target_name: Name of the target to add.
        :param target_info: TargetInfo object containing details about the target.
        """
        # We don't support Cartesian tensors with rank > 1
        if target_info.is_cartesian:
            if len(target_info.layout.block().components) > 1:
                raise ValueError(
                    "MetaMACE does not support Cartesian tensors with rank > 1."
                )

        # Get the multiplicity and irrep for each target block
        irreps = []
        for key, block in target_info.layout.items():
            multiplicity = len(block.properties.values)

            if target_info.is_scalar:
                irreps.append((multiplicity, (0, 1)))
            elif target_info.is_spherical:
                ell = int(key["o3_lambda"])
                irreps.append((multiplicity, (ell, (-1) ** ell)))
            elif target_info.is_cartesian:
                ell = 1
                irreps.append((multiplicity, (ell, (-1) ** ell)))

        self.outputs[target_name] = ModelOutput(
            quantity=target_info.quantity,
            unit=target_info.unit,
            per_atom=True,
        )

        hidden_irreps = o3.Irreps(self.hypers["hidden_irreps"])
        n_scalars = hidden_irreps.count((0, 1))
        mace_out_irreps = hidden_irreps * (
            self.hypers["num_interactions"] - 1
        ) + o3.Irreps([(n_scalars, (0, 1))])

        if target_name == self.mace_head_target:
            # Dummy head so that torchscript loops through this target_name
            # when doing self.heads.items(). In reality we use the internal
            # MACE head for this target
            self.heads[target_name] = torch.nn.Identity()
        else:
            self.heads[target_name] = NonLinearHead(
                irreps_in=mace_out_irreps,
                irreps_out=o3.Irreps(irreps),
                MLP_irreps=o3.Irreps(self.hypers["MLP_irreps"]),
                gate=mace_modules.gate_dict.get(self.hypers["gate"], None),
            )

            self.heads[target_name].to(torch.float64)

        ll_features_name = (
            f"mtt::aux::{target_name.replace('mtt::', '')}_last_layer_features"
        )
        self.outputs[ll_features_name] = ModelOutput(per_atom=True)

    @classmethod
    def upgrade_checkpoint(cls, checkpoint: Dict) -> Dict:
        if checkpoint["model_ckpt_version"] != cls.__checkpoint_version__:
            raise RuntimeError(
                f"Unable to upgrade the checkpoint: the checkpoint is using model "
                f"version {checkpoint['model_ckpt_version']}, while the current model "
                f"version is {cls.__checkpoint_version__}."
            )

        return checkpoint

    def get_checkpoint(self) -> Dict:
        model_state_dict = self.state_dict()
        model_state_dict["finetune_config"] = self.finetune_config
        checkpoint = {
            "architecture_name": "experimental.mace",
            "model_ckpt_version": self.__checkpoint_version__,
            "metadata": self.metadata,
            "model_data": {
                "hypers": self.hypers,
                "dataset_info": self.dataset_info.to(device="cpu"),
            },
            "epoch": None,
            "best_epoch": None,
            "model_state_dict": model_state_dict,
            "best_model_state_dict": self.state_dict(),
        }
        return checkpoint
