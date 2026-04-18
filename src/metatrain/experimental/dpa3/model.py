import contextlib
import copy
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

import ase.data
import metatensor.torch as mts
import torch
from deepmd.pt.model.model import get_standard_model
from metatensor.torch import Labels, TensorBlock, TensorMap
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
from metatrain.utils.data import TargetInfo
from metatrain.utils.data.dataset import DatasetInfo
from metatrain.utils.dtype import dtype_to_str
from metatrain.utils.metadata import merge_metadata
from metatrain.utils.scaler import Scaler
from metatrain.utils.sum_over_atoms import sum_over_atoms

from . import checkpoints
from .documentation import ModelHypers
from .modules.structures import concatenate_structures


_PRECISION_INT_TO_DTYPE = {32: torch.float32, 64: torch.float64}
_INT_TO_DEEPMD_PREC = {32: "float32", 64: "float64"}


@contextlib.contextmanager
def _build_on_cpu():
    """Force deepmd-kit module-level device constants to CPU during model
    construction, ensuring deterministic (CPU) RNG for weight initialization
    regardless of CUDA availability."""
    cpu = torch.device("cpu")
    saved = {}
    for name, mod in sys.modules.items():
        if mod is None or not name.startswith("deepmd.pt"):
            continue
        for attr in ("device", "DEVICE"):
            val = getattr(mod, attr, None)
            if isinstance(val, torch.device) and val.type == "cuda":
                saved[(name, attr)] = val
                setattr(mod, attr, cpu)
    try:
        yield
    finally:
        for (name, attr), original in saved.items():
            mod = sys.modules.get(name)
            if mod is not None:
                setattr(mod, attr, original)


def _register_untracked_tensors(model: torch.nn.Module) -> None:
    """Register plain tensor attributes as non-persistent buffers.

    deepmd-kit stores some tensors (e.g. type_mask) as plain attributes
    rather than parameters or buffers.  These are invisible to .to(device)
    and .to(dtype).  Converting them to non-persistent buffers makes them
    move correctly without affecting state_dict (checkpoint compatibility)."""
    for module in model.modules():
        registered = set()
        for n, _ in module.named_parameters(recurse=False):
            registered.add(n)
        for n, _ in module.named_buffers(recurse=False):
            registered.add(n)
        for attr_name in list(vars(module).keys()):
            val = getattr(module, attr_name)
            if isinstance(val, torch.Tensor) and attr_name not in registered:
                delattr(module, attr_name)
                module.register_buffer(attr_name, val, persistent=False)


class DPA3(ModelInterface[ModelHypers]):
    __checkpoint_version__ = 1
    __supported_devices__ = ["cuda", "cpu"]
    __supported_dtypes__ = [torch.float32, torch.float64]
    __default_metadata__ = ModelMetadata(
        references={
            "implementation": [
                "https://github.com/deepmodeling/deepmd-kit",
            ],
            "architecture": [
                "DPA3: https://arxiv.org/abs/2506.01686",
            ],
        }
    )

    component_labels: Dict[str, List[List[Labels]]]  # torchscript needs this

    def __init__(self, hypers: ModelHypers, dataset_info: DatasetInfo) -> None:
        super().__init__(hypers, dataset_info, self.__default_metadata__)
        self.atomic_types = dataset_info.atomic_types

        # Resolve precision: descriptor.precision is the authority.
        desc_prec = self.hypers["descriptor"]["precision"]
        fit_prec = self.hypers["fitting_net"]["precision"]
        if desc_prec not in _PRECISION_INT_TO_DTYPE:
            raise ValueError(
                f"Unsupported descriptor precision: {desc_prec}. "
                f"Must be one of {list(_PRECISION_INT_TO_DTYPE.keys())}."
            )
        if fit_prec not in _PRECISION_INT_TO_DTYPE:
            raise ValueError(
                f"Unsupported fitting_net precision: {fit_prec}. "
                f"Must be one of {list(_PRECISION_INT_TO_DTYPE.keys())}."
            )
        self.dtype: torch.dtype = _PRECISION_INT_TO_DTYPE[desc_prec]

        self.requested_nl = NeighborListOptions(
            cutoff=self.hypers["descriptor"]["repflow"]["e_rcut"],
            full_list=True,
            strict=True,
        )
        self.targets_keys = list(dataset_info.targets.keys())[0]

        # Pretrained model loading: if dpa3_model is provided, load it
        # instead of building from scratch with get_standard_model.
        self.loaded_dpa3 = self.hypers.get("dpa3_model") is not None
        self._loaded_out_bias: Optional[torch.Tensor] = None
        self._loaded_out_std: Optional[torch.Tensor] = None

        if self.loaded_dpa3:
            dpa3_model = self.hypers["dpa3_model"]
            if isinstance(dpa3_model, (str, Path)):
                loaded = torch.load(str(dpa3_model), weights_only=False)
            elif isinstance(dpa3_model, torch.nn.Module):
                loaded = dpa3_model
            else:
                raise ValueError(
                    "The 'dpa3_model' hyper must be a file path or a torch.nn.Module."
                )

            # If loaded is a dict (deepmd-kit checkpoint), extract the model.
            if isinstance(loaded, dict):
                if "model" in loaded:
                    loaded = loaded["model"]
                else:
                    raise ValueError(
                        "Cannot find 'model' key in the checkpoint dict. "
                        "Expected a deepmd-kit checkpoint or a saved Module."
                    )

            # Normalize to CPU; .to(device) during training moves to GPU.
            self.model = loaded.cpu()
            _register_untracked_tensors(self.model)

            # Extract output bias and std from the atomic model, then zero
            # them so metatrain's CompositionModel and Scaler handle them.
            if not getattr(self.model, "_metatrain_extracted_scaleshift", False):
                atomic_model = self.model.atomic_model
                if hasattr(atomic_model, "out_bias"):
                    self._loaded_out_bias = atomic_model.out_bias.clone()
                    atomic_model.out_bias.zero_()
                if hasattr(atomic_model, "out_std"):
                    self._loaded_out_std = atomic_model.out_std.clone()
                    atomic_model.out_std.fill_(1.0)
                self.model._metatrain_extracted_scaleshift = True
        else:
            # Build a new model from hypers.  _build_on_cpu() patches
            # deepmd-kit module-level DEVICE constants so weight init
            # uses CPU RNG (deterministic across CUDA/CPU environments).
            type_map = [ase.data.chemical_symbols[z] for z in self.atomic_types]
            # deepmd-kit expects precision as strings; convert at the boundary.
            deepmd_hypers: Dict[str, Any] = copy.deepcopy(dict(hypers))
            deepmd_hypers["type_map"] = type_map
            deepmd_hypers["descriptor"]["precision"] = _INT_TO_DEEPMD_PREC[desc_prec]
            deepmd_hypers["fitting_net"]["precision"] = _INT_TO_DEEPMD_PREC[fit_prec]
            with _build_on_cpu():
                self.model = get_standard_model(deepmd_hypers)
            _register_untracked_tensors(self.model)

        self.scaler = Scaler(hypers={}, dataset_info=dataset_info)
        self.outputs: Dict[str, ModelOutput] = {}
        self.single_label = Labels.single()

        self.num_properties: Dict[str, Dict[str, int]] = {}

        self.key_labels: Dict[str, Labels] = {}
        self.component_labels: Dict[str, List[List[Labels]]] = {}
        self.property_labels: Dict[str, List[Labels]] = {}
        for target_name, target in dataset_info.targets.items():
            self._add_output(target_name, target)

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

    def _add_output(self, target_name: str, target: TargetInfo) -> None:
        if not target.is_scalar:
            raise ValueError("The DPA3 architecture can only predict scalars.")
        self.num_properties[target_name] = {}
        self.key_labels[target_name] = target.layout.keys
        self.component_labels[target_name] = [
            block.components for block in target.layout.blocks()
        ]
        self.property_labels[target_name] = [
            block.properties for block in target.layout.blocks()
        ]
        self.outputs[target_name] = ModelOutput(
            quantity=target.quantity,
            unit=target.unit,
            sample_kind="atom",
        )

    def get_rcut(self):
        return self.model.atomic_model.get_rcut()

    def get_sel(self):
        return self.model.atomic_model.get_sel()

    def requested_neighbor_lists(self) -> List[NeighborListOptions]:
        return [self.requested_nl]

    def _prepare_atype(self, species: torch.Tensor) -> torch.Tensor:
        """Map padded species tensor to deepmd-kit type indices."""
        max_z = max(max(self.atomic_types), 0) + 1
        lookup = torch.full(
            (max_z + 1,), -1, dtype=species.dtype, device=species.device
        )
        for idx, z in enumerate(self.atomic_types):
            lookup[z] = idx
        valid_mask = species >= 0
        safe_species = species.clamp(min=0)
        result = lookup[safe_species]
        result[~valid_mask] = -1
        return result

    def _forward_from_batch(
        self,
        positions: torch.Tensor,
        atype: torch.Tensor,
        box: Optional[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Pure-tensor forward pass for compilation preparation.

        Takes pre-processed batch tensors and returns raw per-atom energies
        as a flat dictionary.  This bypasses System/TensorMap creation and
        the scaler/additive models, making it suitable for FX tracing.

        :param positions: Padded positions [batch, max_atoms, 3].
        :param atype: Deepmd-kit type indices [batch, max_atoms] (already
            remapped via ``_prepare_atype``).
        :param box: Cell tensor [batch, 3, 3] or None for non-periodic.
        :return: Dict with ``"atom_energy"`` (masked flat tensor of real atom
            energies) and ``"energy"`` (per-system total energy).
        """
        model_ret = self.model.forward_common(
            positions,
            atype,
            box,
            fparam=None,
            aparam=None,
            do_atomic_virial=False,
        )

        result: Dict[str, torch.Tensor] = {}
        if self.model.get_fitting_net() is not None:
            result["atom_energy"] = model_ret["energy"]
            result["energy"] = model_ret["energy_redu"]
        else:
            result["atom_energy"] = model_ret.get("energy", model_ret["updated_coord"])
            result["energy"] = model_ret.get("energy_redu", torch.zeros(1))

        return result

    def forward(
        self,
        systems: List[System],
        outputs: Dict[str, ModelOutput],
        selected_atoms: Optional[Labels] = None,
    ) -> Dict[str, TensorMap]:
        if len(outputs) == 0:
            return {}

        device = systems[0].positions.device

        if self.single_label.values.device != device:
            self.single_label = self.single_label.to(device)
            self.key_labels = {
                output_name: label.to(device)
                for output_name, label in self.key_labels.items()
            }
            self.component_labels = {
                output_name: [
                    [labels.to(device) for labels in components_block]
                    for components_block in components_tmap
                ]
                for output_name, components_tmap in self.component_labels.items()
            }
            self.property_labels = {
                output_name: [labels.to(device) for labels in properties_tmap]
                for output_name, properties_tmap in self.property_labels.items()
            }

        return_dict: Dict[str, TensorMap] = {}

        (positions, species, cells, atom_index, system_index) = concatenate_structures(
            systems
        )

        atype = self._prepare_atype(species)

        if torch.all(cells == 0).item():
            box = None
        else:
            box = cells

        raw = self._forward_from_batch(positions, atype, box)

        atomic_properties: Dict[str, TensorMap] = {}
        blocks: List[TensorBlock] = []

        values = torch.stack([system_index, atom_index], dim=0).transpose(0, 1)
        invariant_coefficients = Labels(
            names=["system", "atom"], values=values.to(device)
        )

        # Use species to identify real atoms (>=0) vs padding (-1).
        # This is robust across platforms, unlike energy-magnitude thresholding
        # which can incorrectly mask real atoms with near-zero initial energies.
        batch_size, max_atoms = species.shape
        atom_energy = raw["atom_energy"].reshape(batch_size, max_atoms)
        mask = species >= 0
        atomic_property_tensor = atom_energy[mask].unsqueeze(-1)

        blocks.append(
            TensorBlock(
                values=atomic_property_tensor,
                samples=invariant_coefficients,
                components=self.component_labels[self.targets_keys][0],
                properties=self.property_labels[self.targets_keys][0].to(device),
            )
        )

        atomic_properties[self.targets_keys] = TensorMap(
            self.key_labels[self.targets_keys].to(device), blocks
        )

        if selected_atoms is not None:
            for output_name, tmap in atomic_properties.items():
                atomic_properties[output_name] = mts.slice(
                    tmap, axis="samples", selection=selected_atoms
                )

        for output_name, atomic_property in atomic_properties.items():
            if outputs[output_name].sample_kind == "atom":
                return_dict[output_name] = atomic_property
            else:
                # sum the atomic property to get the total property
                return_dict[output_name] = sum_over_atoms(atomic_property)

        if not self.training:
            # at evaluation, we also introduce the scaler and additive contributions
            return_dict = self.scaler(systems, return_dict)
            for additive_model in self.additive_models:
                outputs_for_additive_model: Dict[str, ModelOutput] = {}
                for name, output in outputs.items():
                    if name in additive_model.outputs:
                        outputs_for_additive_model[name] = output
                additive_contributions = additive_model(
                    systems,
                    outputs_for_additive_model,
                    selected_atoms,
                )
                for name in additive_contributions:
                    return_dict[name] = mts.add(
                        return_dict[name],
                        additive_contributions[name],
                    )

        return return_dict

    def restart(self, dataset_info: DatasetInfo) -> "DPA3":
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
                "The DPA3 model does not support adding new atomic types."
            )

        # register new outputs as new last layers
        for target_name, target in new_targets.items():
            self._add_output(target_name, target)

        self.dataset_info = merged_info

        # restart the composition and scaler models
        self.additive_models[0].restart(
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
        self.scaler.restart(dataset_info)

        return self

    @classmethod
    def load_checkpoint(
        cls,
        checkpoint: Dict[str, Any],
        context: Literal["restart", "finetune", "export"],
    ) -> "DPA3":
        model_data = checkpoint["model_data"]

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

        model = cls(
            hypers=model_data["model_hypers"],
            dataset_info=model_data["dataset_info"],
        )

        # Determine dtype from the deepmd-kit model's construction-time
        # precision (authoritative; .to(dtype) does not update self.prec).
        dtype = next(model.model.parameters()).dtype

        model.to(dtype).load_state_dict(model_state_dict)
        model.additive_models[0].sync_tensor_maps()
        model.scaler.sync_tensor_maps()

        # Loading the metadata from the checkpoint
        metadata = checkpoint.get("metadata", None)
        if metadata is not None:
            model.__default_metadata__ = metadata

        return model

    def export(self, metadata: Optional[ModelMetadata] = None) -> AtomisticModel:
        dtype = next(self.parameters()).dtype
        if dtype not in self.__supported_dtypes__:
            raise ValueError(f"unsupported dtype {dtype} for DPA3")

        # Make sure the model is all in the same dtype
        # For example, after training, the additive models could still be in
        # float64
        self.to(dtype)

        # Additionally, the composition model contains some `TensorMap`s that cannot
        # be registered correctly with Pytorch. This function moves them:

        self.additive_models[0].weights_to(torch.device("cpu"), torch.float64)

        interaction_ranges = [self.hypers["descriptor"]["repflow"]["e_rcut"]]
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
        if metadata is None:
            metadata = self.__default_metadata__
        else:
            metadata = merge_metadata(self.__default_metadata__, metadata)

        return AtomisticModel(self.eval(), metadata, capabilities)

    @classmethod
    def upgrade_checkpoint(cls, checkpoint: Dict) -> Dict:
        for v in range(1, cls.__checkpoint_version__):
            if checkpoint["model_ckpt_version"] == v:
                update = getattr(checkpoints, f"model_update_v{v}_v{v + 1}")
                update(checkpoint)
                checkpoint["model_ckpt_version"] = v + 1

        if checkpoint["model_ckpt_version"] != cls.__checkpoint_version__:
            raise RuntimeError(
                f"Unable to upgrade the checkpoint: the checkpoint is using model "
                f"version {checkpoint['model_ckpt_version']}, while the current model "
                f"version is {cls.__checkpoint_version__}."
            )

        return checkpoint

    def get_checkpoint(self) -> Dict:
        model_state_dict = self.state_dict()
        hypers = dict(self.hypers)

        # deepmd-kit modules contain locally-defined classes (e.g.
        # make_embedding_network.<locals>.EN) that cannot be pickled.
        # Never store a Module in hypers; state_dict captures all weights.
        hypers.pop("dpa3_model", None)

        checkpoint = {
            "architecture_name": "experimental.dpa3",
            "model_ckpt_version": self.__checkpoint_version__,
            "metadata": self.metadata,
            "model_data": {
                "model_hypers": hypers,
                "dataset_info": self.dataset_info,
            },
            "epoch": None,
            "best_epoch": None,
            "model_state_dict": model_state_dict,
            "best_model_state_dict": None,
        }
        return checkpoint

    def supported_outputs(self) -> Dict[str, ModelOutput]:
        return self.outputs

    def get_fixed_composition_weights(self) -> dict[str, dict[int, float]]:
        """Return per-type energy biases extracted from a loaded model.

        These are passed to ``CompositionModel.train_model`` as
        ``fixed_weights`` so the pretrained biases are preserved rather than
        refitted from data.
        """
        if self._loaded_out_bias is None:
            return {}
        # out_bias shape: [n_out, ntypes, max_out_size].  For energy
        # prediction the first output and first property are used.
        bias = self._loaded_out_bias[0, :, 0]  # [ntypes]
        return {
            self.targets_keys: {
                z: bias[i].item() for i, z in enumerate(self.atomic_types)
            }
        }

    def get_fixed_scaling_weights(
        self,
    ) -> dict[str, Union[float, dict[int, float]]]:
        """Return per-type scaling factors extracted from a loaded model.

        These are passed to ``Scaler.train_model`` as ``fixed_weights``.
        """
        if self._loaded_out_std is None:
            return {}
        # out_std shape: [n_out, ntypes, max_out_size]
        std = self._loaded_out_std[0, :, 0]  # [ntypes]
        return {
            self.targets_keys: {
                z: std[i].item() for i, z in enumerate(self.atomic_types)
            }
        }
