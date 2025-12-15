import logging
import warnings
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import mace.modules as mace_modules
import metatensor.torch as mts
import torch
from e3nn import o3
from e3nn.util import jit
from mace.modules import MACE
from metatensor.torch import Labels, TensorBlock, TensorMap
from metatensor.torch.operations._add import _add_block_block
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
from .modules.heads import MACEHeadWrapper, NonLinearHead
from .modules.scale_shift import FakeScaleShift
from .utils.mts import (
    e3nn_to_tensormap,
    get_e3nn_mts_layout,
    get_samples_labels,
    target_info_to_e3nn_irreps,
)
from .utils.structures import create_batch


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

    # Attributes of the model. We can't uncomment these descriptions because
    # torchscript complains.
    # cutoff: float
    # """Cutoff radius used in the interactions of the MACE model."""
    # requested_nl: NeighborListOptions
    # """Neighbor list options requested by the model."""
    # mace_model: MACE
    # """The MACE model instance."""
    # loaded_mace: bool
    # """Whether the MACE model was loaded from a MACE model file.

    # This will happen if the 'mace_model' hyperparameter is not None.
    # """
    # atomic_types: list[int]
    # """List of atomic types (atomic numbers) known by the model."""
    # atomic_species_to_index: torch.Tensor
    # """Mapping from atomic type (atomic number) to species index.

    # The species index is simply an index going from 0 to N-1, where
    # N is the number of unique atomic types in the model.
    # """
    # per_layer_irreps: list[o3.Irreps]
    # """Irreps of the hidden features after each MACE message passing step."""
    # features_irreps: o3.Irreps
    # """Irreps of the concatenated features from all MACE message passing steps."""
    # heads: torch.nn.ModuleDict
    # """Dictionary of output heads for each target."""
    # layouts: Dict[str, TensorMap]
    # """Dictionary of TensorMap for each supported output of the model.

    # This includes targets, features and last layer features.

    # Each TensorMap contains the layout needed to build the tensormap
    # corresponding to that output, from the raw torch tensor produced by the model.
    # """
    # additive_models: torch.nn.ModuleList
    # """List of additive models to compute additive contributions."""
    # scaler: Scaler
    # """Scaler to bring all targets to a scale that is optimal for training."""

    def __init__(self, hypers: ModelHypers, dataset_info: DatasetInfo) -> None:
        super().__init__(hypers, dataset_info, self.__default_metadata__)

        # ---------------------------
        # Get the MACE model instance
        # ---------------------------

        self.loaded_mace = self.hypers["mace_model"] is not None
        # Atomic baselines and scale extracted from the loaded MACE model (if any).
        self._loaded_atomic_baseline = None
        self._loaded_scale = 1.0

        if self.loaded_mace:
            # MACE model provided, load it in case it's a path or use it directly
            if isinstance(self.hypers["mace_model"], (Path, str)):
                self.mace_model = torch.load(
                    self.hypers["mace_model"], weights_only=False
                )
            elif isinstance(self.hypers["mace_model"], torch.nn.Module):
                self.mace_model = self.hypers["mace_model"]
            else:
                raise ValueError(
                    "The 'mace_model' hyper must be a path or a torch.nn.Module"
                )

            # If this is the first time we load this model,
            # extract atomic baselines and scales from the loaded model,
            # and set them to zero / identity respectively, since these
            # will be handled by metatrain's scaler and composition model.
            if not getattr(self.mace_model, "_metatrain_extracted_scaleshift", False):
                if hasattr(self.mace_model, "atomic_energies_fn"):
                    self._loaded_atomic_baseline = (
                        self.mace_model.atomic_energies_fn.atomic_energies.clone()
                    )

                    self.mace_model.atomic_energies_fn.atomic_energies[:] = 0.0

                if hasattr(self.mace_model, "scale_shift"):
                    self._loaded_scale = self.mace_model.scale_shift.scale.item()
                    added_baseline = self.mace_model.scale_shift.shift.item()
                    if self._loaded_atomic_baseline is not None:
                        self._loaded_atomic_baseline = (
                            self._loaded_atomic_baseline + added_baseline
                        )
                    else:
                        self._loaded_atomic_baseline = torch.full(
                            (len(self.mace_model.atomic_numbers),),
                            added_baseline,
                        )

                    self.mace_model.scale_shift = FakeScaleShift()

                # Signal that we have already extracted the scale and shift
                # from this model. When this model is stored in a checkpoint
                # and loaded again, metatrain will not try to extract the
                # scale and shift again.
                self.mace_model._metatrain_extracted_scaleshift = True

        else:
            # No MACE model provided, create a new one from hypers
            with warnings.catch_warnings():
                # Don't show warnings from e3nn to user (these warnings
                # only appear in old versions of e3nn)
                warnings.filterwarnings(
                    "ignore", "To copy construct from a tensor", UserWarning
                )
                warnings.filterwarnings(
                    "ignore",
                    "The TorchScript type system",
                    UserWarning,
                )
                self.mace_model = MACE(
                    r_max=self.hypers["r_max"],
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
                    if self.hypers["edge_irreps"] is not None
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

        # ---------------------------
        #  Neighbor list information
        # ---------------------------
        self.cutoff = float(self.mace_model.r_max)
        self.requested_nl = NeighborListOptions(
            cutoff=self.cutoff,
            full_list=True,
            strict=True,
        )

        # ---------------------------
        #   Store info about MACE
        # ---------------------------

        # Atomic species information
        self.atomic_types = self.mace_model.atomic_numbers.tolist()
        self.register_buffer(
            "atomic_types_to_species_index",
            torch.zeros(max(self.atomic_types) + 1, dtype=torch.int64),
        )
        for i, atomic_type in enumerate(self.atomic_types):
            self.atomic_types_to_species_index[atomic_type] = i

        # Information about the irreps of MACE features
        self.per_layer_irreps = [
            product.linear.irreps_out for product in self.mace_model.products
        ]
        self.features_irreps = sum(self.per_layer_irreps, o3.Irreps())

        # ---------------------------
        #    Add heads for targets
        # ---------------------------

        # Create heads for each target, store the layout for each of them.
        self.heads = torch.nn.ModuleDict()
        self.layouts: Dict[str, TensorMap] = {}
        for target_name, target_info in dataset_info.targets.items():
            self._add_output(target_name, target_info)

        self.layouts["features"] = get_e3nn_mts_layout(
            "features",
            {
                "type": {"spherical": {"irreps": self.features_irreps}},
                "per_atom": True,
                "properties_name": "feature",
            },
        )

        targets = dataset_info.targets
        self.outputs = {
            k: ModelOutput(
                quantity=targets[k].quantity if k in targets else "",
                unit=targets[k].unit if k in targets else "",
                per_atom=True,
            )
            for k in self.layouts
        }

        # ---------------------------
        # Data preprocessing modules
        # ---------------------------

        # The composition model and scaler are handled by the trainer during training.
        # Their purpose is to adapt the data for optimal training.
        # At evaluation time, the model applies them on forward.
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
        self.additive_models = torch.nn.ModuleList([composition_model])

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
        # --------------------------
        # Moving to device and dtype
        # --------------------------
        # We can't overwrite the to() method because this does not work with
        # torchscript, so we do the necessary operations here.
        # Get device and dtype from the first system
        device = systems[0].device
        # Move layouts to the correct device
        self.layouts = {k: v.to(device=device) for k, v in self.layouts.items()}

        # --------------------------
        #  Prepare inputs for MACE
        # --------------------------

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

        # Change coordinates to YZX
        data["positions"] = data["positions"][:, [1, 2, 0]]
        data["cell"] = data["cell"][:, [1, 2, 0]]
        data["shifts"] = data["shifts"][:, [1, 2, 0]]

        # --------------------------
        #        Run MACE
        # --------------------------

        # Run MACE and extract the node features.
        mace_output = self.mace_model(data, training=self.training, compute_force=False)
        node_features = mace_output["node_feats"]
        assert node_features is not None  # For torchscript
        node_energy = mace_output["node_energy"]
        assert node_energy is not None  # For torchscript

        # ---------------------------------
        #   Run heads and collect outputs
        # ---------------------------------

        # We have ran MACE, now we will simply collect the requested outputs
        model_outputs: dict[str, torch.Tensor] = {}

        # Add features if requested
        if "features" in outputs:
            model_outputs["features"] = node_features

        # Run heads
        for output_name, head in self.heads.items():
            ll_features_name = self._llf_name(output_name)
            requested_target = output_name in outputs
            requested_llf = ll_features_name in outputs

            # Only use this head if its output or its last layer features were requested
            if requested_target or requested_llf:
                # Get the per-atom target, as well as the per-atom last layer features
                node_target = head.forward(
                    node_features, node_energy, compute_llf=requested_llf
                )
                ll_features = head.last_layer_features

                # Store whatever was requested by the user
                if requested_target:
                    model_outputs[output_name] = node_target
                if requested_llf:
                    model_outputs[ll_features_name] = ll_features

        # -----------------------------------
        #   Convert outputs to TensorMaps
        # -----------------------------------

        # At this point, we have a dictionary of all outputs as normal torch tensors.
        # Now, we simply convert to TensorMaps.

        # Get the labels for the samples (system and atom of each value)
        samples = get_samples_labels(systems)

        return_dict: Dict[str, TensorMap] = {}
        for output_name, model_output in model_outputs.items():
            per_atom_output = e3nn_to_tensormap(
                model_output,
                samples=samples,
                layout=self.layouts[output_name],
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

        # -----------------------------------------
        #   Undo data preprocessing (eval only)
        # -----------------------------------------

        # At evaluation, we also introduce the scaler and additive contributions
        if not self.training:
            return_dict = self.scaler(systems, return_dict)
            self.add_additive_contributions(
                return_dict, systems, outputs, selected_atoms
            )

        return return_dict

    def add_additive_contributions(
        self,
        values: Dict[str, TensorMap],
        systems: List[System],
        outputs: Dict[str, ModelOutput],
        selected_atoms: Optional[Labels] = None,
    ) -> None:
        """Adds the contributions from all additive models to the passed values.

        :param values: Dictionary of TensorMaps containing the current outputs
          (without additive contributions). The additive contributions will be added
          in place to this dictionary.
        :param systems: List of systems that have been evaluated to produce the outputs.
        :param outputs: Dictionary of requested ModelOutputs.
        :param selected_atoms: Optional Labels selecting a subset of atoms.
        """
        for additive_model in self.additive_models:
            outputs_for_additive_model: Dict[str, ModelOutput] = {}
            for name, output in outputs.items():
                if name in additive_model.outputs:
                    outputs_for_additive_model[name] = output
            additive_contributions = additive_model.forward(
                systems,
                outputs_for_additive_model,
                selected_atoms,
            )
            for name in additive_contributions:
                # # TODO: uncomment this after metatensor.torch.add is updated to
                # # handle sparse sums
                # return_dict[name] = metatensor.torch.add(
                #     return_dict[name],
                #     additive_contributions[name].to(
                #         device=return_dict[name].device,
                #         dtype=return_dict[name].dtype
                #         ),
                # )

                # TODO: "manual" sparse sum: update to metatensor.torch.add after
                # sparse sum is implemented in metatensor.operations
                output_blocks: List[TensorBlock] = []
                for k, b in values[name].items():
                    if k in additive_contributions[name].keys:
                        output_blocks.append(
                            _add_block_block(
                                b,
                                additive_contributions[name]
                                .block(k)
                                .to(device=b.device, dtype=b.dtype),
                            )
                        )
                    else:
                        output_blocks.append(b)
                values[name] = TensorMap(values[name].keys, output_blocks)

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

        interaction_range = self.hypers["num_interactions"] * self.cutoff

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

        self.layouts[target_name] = target_info.layout

        if target_name == self.hypers["mace_head_target"]:
            # Fake head that will not compute the target, but will help
            # us extract the last layer features from MACE internal head.
            self.heads[target_name] = MACEHeadWrapper(
                self.mace_model.readouts, self.per_layer_irreps
            )
        else:
            head = NonLinearHead(
                irreps_in=self.features_irreps,
                irreps_out=target_info_to_e3nn_irreps(target_info),
                MLP_irreps=o3.Irreps(self.hypers["MLP_irreps"]),
                gate=mace_modules.gate_dict.get(self.hypers["gate"], None),
            )

            self.heads[target_name] = head.to(torch.float64)

        llf_irreps = self.heads[target_name].last_layer_features_irreps

        self.layouts[self._llf_name(target_name)] = get_e3nn_mts_layout(
            f"{target_name}_last_layer_features",
            {
                "type": {"spherical": {"irreps": llf_irreps}},
                "per_atom": True,
                "properties_name": "feature",
            },
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

    def get_fixed_composition_weights(self) -> dict[str, dict[int, float]]:
        """Get composition weights from the loaded MACE model.

        :return: Tensor of shape (N,) with the atomic baselines for each
          atomic type known by the model, or None if no MACE model was loaded.
        """
        if self._loaded_atomic_baseline is None:
            return {}
        else:
            return {
                self.hypers["mace_head_target"]: {
                    k: v
                    for k, v in zip(
                        self.atomic_types,
                        self._loaded_atomic_baseline.tolist(),
                        strict=True,
                    )
                }
            }

    def get_fixed_scaling_weights(self) -> dict[str, float | dict[int, float]]:
        """Get scaling weights from the loaded MACE model.

        :return: Scale factor used in the loaded MACE model, or None
          if no MACE model was loaded.
        """
        if self._loaded_scale == 1.0:
            return {}
        else:
            # Get info about the mace head target
            mace_head_target = self.hypers["mace_head_target"]
            per_atom = self.dataset_info.targets[mace_head_target].per_atom

            # Define scaling weights for the target
            weights = (
                {k: self._loaded_scale for k in self.atomic_types}
                if per_atom
                else self._loaded_scale
            )

            # Return dictionary of fixed scaling weights
            return {mace_head_target: weights}
