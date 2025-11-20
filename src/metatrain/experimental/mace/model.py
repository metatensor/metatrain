import logging
from typing import Any, Dict, List, Literal, Optional

import mace.modules as mace_modules
import metatensor.torch as mts
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
from metatrain.utils.sum_over_atoms import sum_over_atoms

from .documentation import ModelHypers
from .modules.finetuning import apply_finetuning_strategy
from .modules.heads import NonLinearHead
from .modules.scale_shift import FakeScaleShift
from .modules.structures import create_batch
from .utils.mts import (
    add_contribution,
    e3nn_to_tensormap,
    get_e3nn_target_info,
    get_system_indices_and_labels,
    target_info_to_e3nn_irreps,
)
from .utils.llf import LinearReadoutLLFExtractor, NonLinearReadoutLLFExtractor, readout_is_linear
from .modules.heads import MACEHeadWrapper


class MetaMACE(ModelInterface[ModelHypers]):
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

    def __init__(self, hypers: ModelHypers, dataset_info: DatasetInfo) -> None:
        super().__init__(hypers, dataset_info, self.__default_metadata__)

        self.requested_nl = NeighborListOptions(
            cutoff=self.hypers["cutoff"],
            full_list=True,
            strict=True,
        )

        self.cutoff = float(self.hypers["cutoff"])
        self.loaded_mace = self.hypers["mace_model"] is not None

        if self.loaded_mace:
            # MACE model provided, load it in case it's a path or use it directly
            if isinstance(self.hypers["mace_model"], str):
                self.mace_model = torch.load(
                    self.hypers["mace_model"], weights_only=False
                )
            elif isinstance(self.hypers["mace_model"], torch.nn.Module):
                self.mace_model = self.hypers["mace_model"]
            else:
                raise ValueError(
                    "The 'mace_model' hyper must be a path or a torch.nn.Module"
                )

            # Remove scale and shift if present
            if self.hypers["mace_model_remove_scale_shift"] and hasattr(
                self.mace_model, "scale_shift"
            ):
                self.mace_model.scale_shift = FakeScaleShift()

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
        self.per_layer_irreps = [product.linear.irreps_out for product in self.mace_model.products]
        self.per_layer_dims = [ir.dim for ir in self.per_layer_irreps]
        self.features_irreps = sum(self.per_layer_irreps, o3.Irreps())

        self.mace_llf_extractors = torch.nn.ModuleList()
        for i, readout in enumerate(self.mace_model.readouts):
            n_scalars = self.per_layer_irreps[i].count((0, 1))
            if readout_is_linear(readout):
                self.mace_llf_extractors.append(
                    LinearReadoutLLFExtractor(readout, n_scalars)
                )
            else:
                self.mace_llf_extractors.append(
                    NonLinearReadoutLLFExtractor(readout, n_scalars)
                )

        self.mace_head_wrapper = MACEHeadWrapper(self.mace_model.readouts)

        self.mace_readouts_are_linear = [
            readout_is_linear(readout) for readout in self.mace_model.readouts
        ]

        self.register_buffer(
            "atomic_types_to_species_index",
            torch.zeros(max(self.atomic_types) + 1, dtype=torch.int64),
        )
        for i, atomic_type in enumerate(self.atomic_types):
            self.atomic_types_to_species_index[atomic_type] = i

        self.heads = torch.nn.ModuleDict()
        self.target_infos: Dict[str, TargetInfo] = {}
        for target_name, target_info in dataset_info.targets.items():
            self._add_output(target_name, target_info)

        # self.target_infos["features"] = get_e3nn_target_info(
        #     "features", {"irreps": self.features_irreps, "per_atom": True}
        # )

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

    def to(self, *args: Any, **kwargs: Any) -> "MetaMACE":
        super().to(*args, **kwargs)
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype
        # Move dataset info to the correct device
        self.dataset_info = self.dataset_info.to(device=device)
        # If the MACE model was loaded as part of the hypers, it is probably
        # a RecursiveScriptModule, which seems to not get moved by super().to()
        # So we move it here manually.
        if self.loaded_mace:
            self.mace_model = self.mace_model.to(device=device, dtype=dtype)

        return self

    def forward(
        self,
        systems: List[System],
        outputs: Dict[str, ModelOutput],
        selected_atoms: Optional[Labels] = None,
    ) -> Dict[str, TensorMap]:
        # Create the batch to pass as input for MACE.
        # THIS PROBABLY SHOULD BE MOVED OUTSIDE THE MODEL!!
        # (But I don't know if this would affect the interfaces e.g. with
        # ASE, LAMMPS, etc.)
        data = create_batch(
            systems=systems,
            neighbor_list_options=self.requested_nl,
            atomic_types_to_species_index=self.atomic_types_to_species_index,
            n_types=len(self.atomic_types),
        )

        self.mace_head_wrapper()

        # Change coordinates to YZX
        data["positions"] = data["positions"][:, [1, 2, 0]]

        # Run MACE and extract the node features.
        mace_output = self.mace_model(data, training=self.training, compute_force=False)
        node_features = mace_output["node_feats"]
        assert node_features is not None  # For torchscript

        # We have ran MACE, now we will simply collect the requested outputs
        model_outputs: dict[str, torch.Tensor] = {}

        # Add features if requested
        if "features" in outputs:
            raise NotImplementedError("Asking for 'features' is not supported yet.")
            model_outputs["features"] = node_features

        # Run heads
        for output_name, head in self.heads.items():
            ll_features_name = self._llf_name(output_name)
            requested_target = output_name in outputs
            requested_llf = ll_features_name in outputs

            # Only use this head if its output or its last layer features were requested
            if requested_target or requested_llf:
                # Get the per-atom target, as well as the per-atom last layer features
                if output_name == self.mace_head_target:
                    # Use the internal MACE head
                    node_energy = mace_output["node_energy"]
                    assert node_energy is not None  # For torchscript
                    node_target = node_energy.to(dtype=node_features.dtype).reshape(
                        -1, 1
                    )
                    if requested_llf:
                        per_layer_features = torch.split(
                            node_features, self.per_layer_dims, dim=-1
                        )

                        ll_feats_list = [
                            extractor(per_layer_features[i])
                            for i, extractor in enumerate(self.mace_llf_extractors)
                        ]

                        # Aggregate node features
                        ll_features = torch.cat(ll_feats_list, dim=-1)
                    else:
                        ll_features = torch.empty(
                            (0, 0), dtype=node_features.dtype, device=node_features.device
                        )
                else:
                    node_target = head.forward(node_features)
                    ll_features = head.last_layer_features

                # Store whatever was requested by the user
                if requested_target:
                    model_outputs[output_name] = node_target
                if requested_llf:
                    model_outputs[ll_features_name] = ll_features

                    print(self.target_infos[ll_features_name])
                    print(ll_features.shape)

        # At this point, we have a dictionary of all outputs as normal torch tensors.
        # Now, we simply convert to TensorMaps.

        # Get the labels for the samples (system and atom of each value)
        _, samples = get_system_indices_and_labels(systems)

        return_dict: Dict[str, TensorMap] = {}
        for output_name, model_output in model_outputs.items():
            per_atom_output = e3nn_to_tensormap(
                model_output,
                samples=samples,
                target_info=self.target_infos[output_name],
            )

            if selected_atoms is not None:
                per_atom_output = mts.slice(
                    per_atom_output, axis="samples", selection=selected_atoms
                )

            return_dict[output_name] = (
                per_atom_output
                if outputs[output_name].per_atom
                else sum_over_atoms(per_atom_output)
            )
        
        # At evaluation, we also introduce the scaler and additive contributions
        if not self.training:
            return_dict = self.scaler(systems, return_dict)
            for additive_model in self.additive_models:
                add_contribution(
                    return_dict, systems, outputs, additive_model, selected_atoms
                )

        return return_dict


    @property
    def outputs(self) -> Dict[str, ModelOutput]:
        return {
            k: ModelOutput(
                quantity=target_info.quantity,
                unit=target_info.unit,
                per_atom=True,
            )
            for k, target_info in self.target_infos.items()
        }

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
        # Infer dtype
        dtype = None
        has_stored_mace = model_data["hypers"]["mace_model"] is not None
        if has_stored_mace:
            # If the model was part of the hypers, get the dtype from the model
            # itself (its parameters are not in the state_dict)
            dtype = list(model.mace_model.parameters())[0].dtype
        else:
            # Otherwise, just look at the weights in the state dict
            for k, v in model_state_dict.items():
                if k.endswith(".weight"):
                    dtype = v.dtype
                    break
            else:
                raise ValueError("Couldn't infer dtype from the checkpoint file")
        # Set up finetuning if needed
        finetune_config = model_state_dict.pop("finetune_config", {})
        if finetune_config:
            # Apply the finetuning strategy
            model = apply_finetuning_strategy(model, finetune_config)

        # Load the state dict. In the case of having stored the MACE model
        # (see get_checkpoint), its parameters are not in the state dict. Therefore
        # we allow the state dict having missing keys that start with "mace_model".
        missing_keys, unexpected_keys = model.to(dtype).load_state_dict(
            model_state_dict, strict=not has_stored_mace
        )
        if len(unexpected_keys) > 0 or any(
            not k.startswith("mace_model") for k in missing_keys
        ):
            raise ValueError(
                f"Error loading the checkpoint: missing keys {missing_keys}, "
                f"unexpected keys {unexpected_keys}."
            )
        # Set up composition and scaler models
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

        interaction_range = self.hypers["num_interactions"] * self.hypers["cutoff"]

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

        self.target_infos[target_name] = target_info
        # Get the multiplicity and irrep for each target block
        target_irreps = target_info_to_e3nn_irreps(target_info)

        if target_name == self.mace_head_target:
            # Dummy head so that torchscript loops through this target_name
            # when doing self.heads.items(). In reality we use the internal
            # MACE head for this target
            self.heads[target_name] = torch.nn.Identity()
            llf_irreps = self.features_irreps.count((0, 1)) * o3.Irrep(0, 1)
        else:
            head = NonLinearHead(
                irreps_in=self.features_irreps,
                irreps_out=target_irreps,
                MLP_irreps=o3.Irreps(self.hypers["MLP_irreps"]),
                gate=mace_modules.gate_dict.get(self.hypers["gate"], None),
            )

            self.heads[target_name] = head.to(torch.float64)
            llf_irreps = head.last_layer_features_irreps

        self.target_infos[self._llf_name(target_name)] = get_e3nn_target_info(
            f"{target_name}_last_layer_features",
            {"irreps": llf_irreps, "per_atom": True},
        )

    def _llf_name(self, target_name: str) -> str:
        """Get the name of the last layer features corresponding to a target.

        :param target_name: Name of the target.
        :return: Name of the last layer features corresponding to the target.
        """
        return f"mtt::aux::{target_name.replace('mtt::', '')}_last_layer_features"

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

        # If the MACE model was passed as part of the hypers, we store it
        # again as part of the hypers.
        hypers = self.hypers.copy()
        if hypers["mace_model"] is not None:
            hypers["mace_model"] = self.mace_model.to(device="cpu")

            # Remove mace_model from state dict to avoid redundancy
            for k in list(model_state_dict.keys()):
                if k.startswith("mace_model."):
                    model_state_dict.pop(k)

        checkpoint = {
            "architecture_name": "experimental.mace",
            "model_ckpt_version": self.__checkpoint_version__,
            "metadata": self.metadata,
            "model_data": {
                "hypers": hypers,
                "dataset_info": self.dataset_info.to(device="cpu"),
            },
            "epoch": None,
            "best_epoch": None,
            "model_state_dict": model_state_dict,
            "best_model_state_dict": model_state_dict,
        }
        return checkpoint
